// 托盘常驻模式：任务栏托盘图标 + 右键菜单（含二级菜单）+ 后台听写（Windows）
//
// 流程：单实例检查 → 引擎加载（控制台可见）→ FreeConsole 后台化
//      → 听写线程（钩子/录音/识别/上屏）→ 主线程跑托盘消息循环
//      → 菜单事件：切换开关项即时生效，配置修改后通过听写线程重新加载
#![cfg(windows)]

use crate::client;
use crate::config::Config;
use crate::{log_error, log_info};
use anyhow::{bail, Result};
use std::sync::mpsc;
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use std::os::windows::process::CommandExt;
use tray_icon::menu::{CheckMenuItem, Menu, MenuEvent, MenuItem, PredefinedMenuItem, Submenu};
use tray_icon::{Icon, TrayIconBuilder};
use windows::core::{w, PCWSTR};
use windows::Win32::Foundation::{CloseHandle, ERROR_ALREADY_EXISTS, ERROR_SUCCESS, HANDLE};
use windows::Win32::System::Console::FreeConsole;
use windows::Win32::System::Registry::{
    RegCreateKeyExW, RegDeleteValueW, RegOpenKeyExW, RegQueryValueExW, RegSetValueExW, HKEY,
    HKEY_CURRENT_USER, KEY_READ, KEY_SET_VALUE, REG_OPTION_NON_VOLATILE, REG_SZ,
};
use windows::Win32::System::Threading::{CreateMutexW, OpenMutexW, MUTEX_MODIFY_STATE};
use windows::Win32::UI::WindowsAndMessaging::{
    DispatchMessageW, PeekMessageW, TranslateMessage, MSG, PM_REMOVE,
};

const MUTEX_NAME: PCWSTR = w!("CapsWriter-RS::single-instance");
const RUN_KEY: PCWSTR = w!("Software\\Microsoft\\Windows\\CurrentVersion\\Run");
const RUN_VALUE: PCWSTR = w!("CapsWriter-RS");

/// 托盘状态（由听写线程推送）
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Status {
    Idle,        // 就绪
    Recording,   // 按住说话中
    Recognizing, // 识别中
    Paused,      // 已暂停
    Error,       // 听写线程故障
}

static STATUS_TX: OnceLock<mpsc::Sender<Status>> = OnceLock::new();

/// HANDLE 是 *mut c_void，默认不 Send，包一层
struct SendHandle(HANDLE);
unsafe impl Send for SendHandle {}
static MUTEX_HANDLE: Mutex<Option<SendHandle>> = Mutex::new(None);

/// 听写线程 → 托盘状态推送（无接收者时静默丢弃）
pub fn notify(s: Status) {
    if let Some(tx) = STATUS_TX.get() {
        let _ = tx.send(s);
    }
}

// ============ 程序化图标（SDF 绘制麦克风，免外部资源文件） ============

/// 生成 size×size RGBA 麦克风图标：圆角底色块 + 白色麦克风
fn mic_icon(size: u32, bg: [u8; 3]) -> Icon {
    let s = size as f32;
    let scale = s / 2.0; // 单位坐标(0..1) → 像素
    let cov = |d: f32| -> f32 {
        // 距离场 → 像素覆盖度（1px 抗锯齿）
        (0.5 - d * scale).clamp(0.0, 1.0)
    };
    let mut rgba = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let u = (x as f32 + 0.5) / s;
            let v = (y as f32 + 0.5) / s;

            // 底色圆角方块（占满画布 97%）
            let px = (u - 0.5) * 2.0;
            let py = (v - 0.5) * 2.0;
            let b = 0.97;
            let r = 0.42;
            let qx = px.abs() - (b - r);
            let qy = py.abs() - (b - r);
            let d_bg = qx.max(0.0).hypot(qy.max(0.0)) + qx.max(qy).min(0.0) - r;
            let cov_bg = cov(d_bg);

            // 麦克风头部胶囊（竖直，中心 0.345）
            let q2x = (u - 0.5).abs() - 0.0;
            let q2y = (v - 0.345).abs() - 0.06;
            let d_cap = q2x.max(0.0).hypot(q2y.max(0.0)) + q2x.max(q2y).min(0.0) - 0.105;
            // U 形托弧（圆环带下半部分，圆心 0.33 半径 0.28）
            let d_arc = ((u - 0.5).hypot(v - 0.33) - 0.28).abs() - 0.033;
            let cov_arc = if v > 0.38 { cov(d_arc) } else { 0.0 };
            // 支柱
            let d_stem = ((u - 0.5).abs() - 0.04).max((v - 0.66).abs() - 0.06);
            // 底座横条
            let d_base = ((u - 0.5).abs() - 0.17).max((v - 0.755).abs() - 0.035);
            let cov_fg = cov(d_cap).max(cov_arc).max(cov(d_stem)).max(cov(d_base));

            // 白色前景 ⊕ 底色背景（非预乘合成）
            let a = cov_fg + cov_bg * (1.0 - cov_fg);
            let blend = |fg: f32, bgv: u8| -> u8 {
                if a <= 0.0 {
                    0
                } else {
                    ((fg * 255.0 * cov_fg + bgv as f32 * cov_bg * (1.0 - cov_fg)) / a) as u8
                }
            };
            rgba.push(blend(1.0, bg[0]));
            rgba.push(blend(1.0, bg[1]));
            rgba.push(blend(1.0, bg[2]));
            rgba.push((a * 255.0) as u8);
        }
    }
    Icon::from_rgba(rgba, size, size).expect("图标像素尺寸合法")
}

fn status_icon(s: Status) -> Icon {
    match s {
        Status::Idle => mic_icon(32, [64, 132, 244]),     // 蓝
        Status::Recording => mic_icon(32, [232, 70, 70]), // 红
        Status::Recognizing => mic_icon(32, [245, 150, 50]), // 橙
        Status::Paused | Status::Error => mic_icon(32, [128, 128, 128]), // 灰
    }
}

fn status_tooltip(s: Status) -> &'static str {
    match s {
        Status::Idle => "CapsWriter-RS — 就绪（按住左键说话，松开上屏）",
        Status::Recording => "CapsWriter-RS — 录音中…",
        Status::Recognizing => "CapsWriter-RS — 识别中…",
        Status::Paused => "CapsWriter-RS — 已暂停（钩子未生效）",
        Status::Error => "CapsWriter-RS — 听写线程故障，详见日志",
    }
}

// ============ 开机自启（HKCU Run 键） ============

fn to_wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

pub fn autostart_enabled() -> bool {
    unsafe {
        let mut hkey = HKEY::default();
        if RegOpenKeyExW(HKEY_CURRENT_USER, RUN_KEY, None, KEY_READ, &mut hkey) != ERROR_SUCCESS {
            return false;
        }
        let mut cb: u32 = 0;
        RegQueryValueExW(hkey, RUN_VALUE, None, None, None, Some(&mut cb)) == ERROR_SUCCESS
    }
}

/// 返回 是否成功
pub fn set_autostart(on: bool) -> bool {
    unsafe {
        if on {
            let mut hkey = HKEY::default();
            if RegCreateKeyExW(
                HKEY_CURRENT_USER,
                RUN_KEY,
                None,
                PCWSTR::null(),
                REG_OPTION_NON_VOLATILE,
                KEY_SET_VALUE,
                None,
                &mut hkey,
                None,
            ) != ERROR_SUCCESS
            {
                return false;
            }
            let Ok(exe) = std::env::current_exe() else {
                return false;
            };
            let val = format!("\"{}\"", exe.display());
            // REG_SZ：UTF-16 字节流（含结尾 NUL）
            let bytes: Vec<u8> = to_wide(&val).iter().flat_map(|c| c.to_le_bytes()).collect();
            RegSetValueExW(hkey, RUN_VALUE, None, REG_SZ, Some(&bytes)) == ERROR_SUCCESS
        } else {
            let mut hkey = HKEY::default();
            if RegOpenKeyExW(HKEY_CURRENT_USER, RUN_KEY, None, KEY_SET_VALUE, &mut hkey)
                != ERROR_SUCCESS
            {
                return true; // 键不存在，视为已关闭
            }
            RegDeleteValueW(hkey, RUN_VALUE) == ERROR_SUCCESS
        }
    }
}

// ============ 菜单 ID 常量 ============

// 顶层操作
const ID_TOGGLE: &str = "toggle";
const ID_AUTOSTART: &str = "autostart";
const ID_RESTART: &str = "restart";
const ID_QUIT: &str = "quit";

// 引擎
const ID_RELOAD: &str = "reload_config";
const ID_RELOAD_MODEL: &str = "reload_model";
const ID_ENGINE_SENSEVOICE: &str = "engine_sensevoice";
const ID_ENGINE_QWEN3ASR: &str = "engine_qwen3asr";
const ID_PROVIDER_AUTO: &str = "provider_auto";
const ID_PROVIDER_CPU: &str = "provider_cpu";
const ID_PROVIDER_DML: &str = "provider_dml";
const ID_PROVIDER_CUDA: &str = "provider_cuda";
const ID_LANGUAGE_AUTO: &str = "lang_auto";
const ID_LANGUAGE_ZH: &str = "lang_zh";
const ID_LANGUAGE_EN: &str = "lang_en";
const ID_LANGUAGE_JA: &str = "lang_ja";
const ID_LANGUAGE_KO: &str = "lang_ko";
const ID_LANGUAGE_YUE: &str = "lang_yue";
const ID_USE_ITN: &str = "use_itn";
const ID_LLM_USE_GPU: &str = "llm_use_gpu";

// 触发
const ID_HOLD_LEFT: &str = "hold_left";
const ID_FOCUS_CHECK: &str = "focus_check";
const ID_HOLD_X2: &str = "hold_x2";
const ID_HOLD_CAPS: &str = "hold_caps";

// 输出
const ID_PASTE: &str = "paste";
const ID_RESTORE_CLIP: &str = "restore_clip";
const ID_SAVE_AUDIO: &str = "save_audio";
const ID_SAVE_DIARY: &str = "save_diary";

// 热词
const ID_HOT_ENABLED: &str = "hot_enabled";
const ID_RULES_ENABLED: &str = "rules_enabled";

// LLM
const ID_LLM_ENABLED: &str = "llm_enabled";

// 工具
const ID_OPEN_LOGS: &str = "open_logs";
const ID_OPEN_CFG: &str = "open_cfg";

// ============ 主入口 ============

/// 托盘常驻模式：内置服务端 + 后台听写 + 托盘控制
pub fn run_tray(cfg: Config) -> Result<()> {
    // —— 单实例检查 ——
    if unsafe { OpenMutexW(MUTEX_MODIFY_STATE, false, MUTEX_NAME) }.is_ok() {
        bail!("已有 CapsWriter-RS 实例在运行（托盘）");
    }
    let mutex: HANDLE = unsafe { CreateMutexW(None, true, MUTEX_NAME)? };
    if unsafe { windows::Win32::Foundation::GetLastError() } == ERROR_ALREADY_EXISTS {
        bail!("已有 CapsWriter-RS 实例在运行（托盘）");
    }
    // 全局保存 mutex 句柄，重启时需要先 CloseHandle 释放
    {
        let mut g = MUTEX_HANDLE.lock().unwrap();
        *g = Some(SendHandle(mutex));
    }

    // —— 引擎加载 ——
    let engine = crate::init_engine(&cfg)?;

    let (status_tx, status_rx) = mpsc::channel::<Status>();
    STATUS_TX.set(status_tx).ok();

    // —— 听写线程 ——
    let cfg2 = cfg.clone();
    std::thread::Builder::new()
        .name("dictation".into())
        .spawn(move || {
            if let Err(e) = client::run(&cfg2, engine) {
                log_error!("听写线程退出：{e:#}");
                notify(Status::Error);
            }
        })?;

    // —— 后台化 ——
    unsafe { FreeConsole()? };

    // ========== 构建菜单 ==========
    // 顶层操作
    let menu_toggle = MenuItem::with_id(ID_TOGGLE, "暂停听写", true, None);
    let menu_autostart =
        CheckMenuItem::with_id(ID_AUTOSTART, "开机自启", true, autostart_enabled(), None);
    let menu_restart = MenuItem::with_id(ID_RESTART, "⚡ 重启程序", true, None);
    let menu_quit = MenuItem::with_id(ID_QUIT, "退出", true, None);

    // —— 引擎子菜单 ——
    let is_sensevoice = cfg.engine.kind == "sensevoice";
    let is_qwen3 = cfg.engine.kind == "qwen3asr";
    let eng_sv = CheckMenuItem::with_id(
        ID_ENGINE_SENSEVOICE,
        "SenseVoice",
        true,
        is_sensevoice,
        None,
    );
    let eng_q3 = CheckMenuItem::with_id(ID_ENGINE_QWEN3ASR, "Qwen3-ASR", true, is_qwen3, None);
    let eng_sep = PredefinedMenuItem::separator();

    let cur_provider = cfg.engine.provider.to_lowercase();
    let prov_auto = CheckMenuItem::with_id(
        ID_PROVIDER_AUTO,
        "Auto (自动探测)",
        true,
        cur_provider == "auto",
        None,
    );
    let prov_cpu =
        CheckMenuItem::with_id(ID_PROVIDER_CPU, "CPU", true, cur_provider == "cpu", None);
    let prov_dml = CheckMenuItem::with_id(
        ID_PROVIDER_DML,
        "DirectML (GPU)",
        true,
        cur_provider == "dml",
        None,
    );
    let prov_cuda = CheckMenuItem::with_id(
        ID_PROVIDER_CUDA,
        "CUDA (N卡)",
        true,
        cur_provider == "cuda",
        None,
    );

    let prov_sep = PredefinedMenuItem::separator();

    let cur_lang = cfg.engine.sensevoice.language.to_lowercase();
    let lang_auto =
        CheckMenuItem::with_id(ID_LANGUAGE_AUTO, "Auto", true, cur_lang == "auto", None);
    let lang_zh = CheckMenuItem::with_id(ID_LANGUAGE_ZH, "中文", true, cur_lang == "zh", None);
    let lang_en = CheckMenuItem::with_id(ID_LANGUAGE_EN, "English", true, cur_lang == "en", None);
    let lang_ja = CheckMenuItem::with_id(ID_LANGUAGE_JA, "日本語", true, cur_lang == "ja", None);
    let lang_ko = CheckMenuItem::with_id(ID_LANGUAGE_KO, "한국어", true, cur_lang == "ko", None);
    let lang_yue = CheckMenuItem::with_id(ID_LANGUAGE_YUE, "粤语", true, cur_lang == "yue", None);

    let lang_sep = PredefinedMenuItem::separator();

    let use_itn = CheckMenuItem::with_id(
        ID_USE_ITN,
        "数字转换 (itn)",
        true,
        cfg.engine.sensevoice.use_itn,
        None,
    );
    let llm_use_gpu = CheckMenuItem::with_id(
        ID_LLM_USE_GPU,
        "Qwen3 LLM 用 GPU",
        true,
        cfg.engine.qwen3asr.llm_use_gpu,
        None,
    );

    let engine_menu = Submenu::with_id("engine", "引擎与模型", true);
    engine_menu.append(&eng_sv)?;
    engine_menu.append(&eng_q3)?;
    engine_menu.append(&eng_sep)?;
    engine_menu.append(&prov_auto)?;
    engine_menu.append(&prov_cpu)?;
    engine_menu.append(&prov_dml)?;
    engine_menu.append(&prov_cuda)?;
    engine_menu.append(&prov_sep)?;
    engine_menu.append(&lang_auto)?;
    engine_menu.append(&lang_zh)?;
    engine_menu.append(&lang_en)?;
    engine_menu.append(&lang_ja)?;
    engine_menu.append(&lang_ko)?;
    engine_menu.append(&lang_yue)?;
    engine_menu.append(&lang_sep)?;
    engine_menu.append(&use_itn)?;
    engine_menu.append(&llm_use_gpu)?;

    // —— 触发方式子菜单 ——
    let hold_left = CheckMenuItem::with_id(
        ID_HOLD_LEFT,
        "按住左键说话",
        true,
        cfg.client.hold_mouse_left,
        None,
    );
    let focus_check = CheckMenuItem::with_id(
        ID_FOCUS_CHECK,
        "焦点检测 (仅输入框录音)",
        true,
        cfg.client.focus_check,
        None,
    );
    let hold_x2 = CheckMenuItem::with_id(
        ID_HOLD_X2,
        "按住侧键 X2 说话",
        true,
        cfg.client.hold_mouse_x2,
        None,
    );
    let hold_caps = CheckMenuItem::with_id(
        ID_HOLD_CAPS,
        "按住 CapsLock 说话",
        true,
        cfg.client.hold_capslock,
        None,
    );
    let trigger_menu = Submenu::with_id("trigger", "触发方式", true);
    trigger_menu.append(&hold_left)?;
    trigger_menu.append(&focus_check)?;
    trigger_menu.append(&hold_x2)?;
    trigger_menu.append(&hold_caps)?;

    // —— 输出与归档子菜单 ——
    let paste = CheckMenuItem::with_id(ID_PASTE, "粘贴模式 (Ctrl+V)", true, cfg.client.paste, None);
    let restore_clip = CheckMenuItem::with_id(
        ID_RESTORE_CLIP,
        "恢复原剪贴板",
        true,
        cfg.client.restore_clipboard,
        None,
    );
    let clip_sep = PredefinedMenuItem::separator();
    let save_audio =
        CheckMenuItem::with_id(ID_SAVE_AUDIO, "保存录音", true, cfg.client.save_audio, None);
    let save_diary = CheckMenuItem::with_id(
        ID_SAVE_DIARY,
        "按日归档文本",
        true,
        cfg.client.save_diary,
        None,
    );
    let output_menu = Submenu::with_id("output", "输出与归档", true);
    output_menu.append(&paste)?;
    output_menu.append(&restore_clip)?;
    output_menu.append(&clip_sep)?;
    output_menu.append(&save_audio)?;
    output_menu.append(&save_diary)?;

    // —— 热词与 LLM 子菜单 ——
    let hot_enabled = CheckMenuItem::with_id(
        ID_HOT_ENABLED,
        "热词替换",
        true,
        cfg.client.hot_enabled,
        None,
    );
    let rules_enabled = CheckMenuItem::with_id(
        ID_RULES_ENABLED,
        "规则替换",
        true,
        cfg.client.rules_enabled,
        None,
    );
    let hot_sep = PredefinedMenuItem::separator();
    let llm_enabled =
        CheckMenuItem::with_id(ID_LLM_ENABLED, "LLM 角色处理", true, cfg.llm.enabled, None);
    let hotllm_menu = Submenu::with_id("hotllm", "热词与 LLM", true);
    hotllm_menu.append(&hot_enabled)?;
    hotllm_menu.append(&rules_enabled)?;
    hotllm_menu.append(&hot_sep)?;
    hotllm_menu.append(&llm_enabled)?;

    // —— 工具 ——
    let menu_logs = MenuItem::with_id(ID_OPEN_LOGS, "打开日志目录", true, None);
    let menu_cfg_edit = MenuItem::with_id(ID_OPEN_CFG, "编辑配置文件", true, None);

    // —— 组装主菜单 ——
    let menu = Menu::new();
    menu.append(&menu_toggle)?;
    menu.append(&menu_autostart)?;
    menu.append(&PredefinedMenuItem::separator())?;
    menu.append(&engine_menu)?;
    menu.append(&trigger_menu)?;
    menu.append(&output_menu)?;
    menu.append(&hotllm_menu)?;
    menu.append(&PredefinedMenuItem::separator())?;
    menu.append(&menu_restart)?;
    menu.append(&PredefinedMenuItem::separator())?;
    menu.append(&menu_logs)?;
    menu.append(&menu_cfg_edit)?;
    menu.append(&MenuItem::with_id(
        ID_RELOAD,
        "重新加载配置与热词",
        true,
        None,
    ))?;
    menu.append(&MenuItem::with_id(
        ID_RELOAD_MODEL,
        "重新加载配置与模型",
        true,
        None,
    ))?;
    menu.append(&PredefinedMenuItem::separator())?;
    menu.append(&menu_quit)?;

    let tray = TrayIconBuilder::new()
        .with_icon(status_icon(Status::Idle))
        .with_tooltip(status_tooltip(Status::Idle))
        .with_menu(Box::new(menu))
        .build()?;

    let mut cur_status = Status::Idle;
    let mut paused = false;
    let mut quit = false;
    log_info!("托盘常驻模式已启动（后台听写）");

    // 运行时可即时更新的客户端配置项（不需要重启服务端）
    let mut live_cfg = cfg.clone();
    let mut pending: Option<mpsc::Receiver<std::result::Result<Config, String>>> = None;

    // —— 消息循环 ——
    while !quit {
        unsafe {
            let mut msg = MSG::default();
            while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
                if msg.message == 0x0012 {
                    quit = true; // WM_QUIT
                }
                let _ = TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }

        // 听写线程状态 → 图标/提示
        while let Ok(s) = status_rx.try_recv() {
            if s != cur_status {
                cur_status = s;
                let _ = tray.set_icon(Some(status_icon(s)));
                let _ = tray.set_tooltip(Some(status_tooltip(s)));
            }
        }

        if let Some(reply) = &pending {
            match reply.try_recv() {
                Ok(result) => {
                    pending = None;
                    match result {
                        Ok(updated) => {
                            live_cfg = updated;
                            log_info!("配置重载成功，已生效");
                        }
                        Err(e) => {
                            log_error!("重载失败，保留原配置和模型：{e}");
                            let _ = tray.set_tooltip(Some(&format!("重载失败：{e}")));
                        }
                    }
                }
                Err(mpsc::TryRecvError::Disconnected) => {
                    pending = None;
                    log_error!("听写线程已退出，重载失败");
                }
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
        // 菜单事件
        while let Ok(ev) = MenuEvent::receiver().try_recv() {
            let id = ev.id.0.as_str();
            log_info!("菜单事件：{id}");

            if pending.is_some() && id != ID_QUIT && id != ID_OPEN_LOGS {
                log_info!("正在等待听写结束并重新加载，请稍候");
                continue;
            }
            let before = live_cfg.clone();
            let mut force_model = false;
            let mut from_disk = false;
            let mut need_restart = false;

            // ===== 统一 match：即时生效项 + 应用内重载项 + 退出/工具 =====
            match id {
                // —— 退出 ——
                ID_QUIT => {
                    quit = true;
                    log_info!("用户选择退出");
                }

                // —— 重启 ——
                ID_RESTART => {
                    restart_self();
                }
                ID_RELOAD | ID_RELOAD_MODEL => {
                    let path = cfg.base_dir.join("config.toml");
                    match std::fs::read_to_string(&path)
                        .map_err(anyhow::Error::from)
                        .and_then(|_| Config::load_from(&path))
                    {
                        Ok(updated) => {
                            live_cfg = updated;
                            from_disk = true;
                            force_model = id == ID_RELOAD_MODEL;
                            need_restart = true;
                        }
                        Err(e) => {
                            log_error!("配置读取失败，保留原配置：{e:#}");
                            let _ = tray.set_tooltip(Some("配置读取失败，请查看日志"));
                        }
                    }
                }

                // —— 暂停/恢复 ——
                ID_TOGGLE => {
                    paused = !paused;
                    crate::client::hook::set_paused(paused);
                    let _ = menu_toggle.set_text(if paused {
                        "恢复听写"
                    } else {
                        "暂停听写"
                    });
                    cur_status = if paused { Status::Paused } else { Status::Idle };
                    let _ = tray.set_icon(Some(status_icon(cur_status)));
                    let _ = tray.set_tooltip(Some(status_tooltip(cur_status)));
                    log_info!("听写已{}", if paused { "暂停" } else { "恢复" });
                }

                // —— 开机自启 ——
                ID_AUTOSTART => {
                    let want = !autostart_enabled();
                    if set_autostart(want) {
                        let _ = menu_autostart.set_checked(want);
                        log_info!("开机自启已{}", if want { "开启" } else { "关闭" });
                    } else {
                        log_error!("修改开机自启注册表失败");
                        let _ = menu_autostart.set_checked(!want);
                    }
                }

                // —— 工具 ——
                ID_OPEN_LOGS => {
                    let dir = cfg.base_dir.join("logs");
                    let _ = std::fs::create_dir_all(&dir);
                    match std::process::Command::new("explorer").arg(&dir).spawn() {
                        Ok(_) => log_info!("已打开日志目录：{}", dir.display()),
                        Err(e) => log_error!("打开日志目录失败：{e:#}"),
                    }
                }
                ID_OPEN_CFG => {
                    let p = cfg.base_dir.join("config.toml");

                    // 用系统默认关联程序打开；start 第一个 "" 是空窗口标题
                    match std::process::Command::new("cmd")
                        .args(["/c", "start", ""])
                        .arg(&p)
                        .creation_flags(0x08000000) // CREATE_NO_WINDOW
                        .spawn()
                    {
                        Ok(_) => log_info!("已打开配置文件：{}", p.display()),
                        Err(e) => log_error!("打开配置文件失败：{e:#}"),
                    }
                }

                // —— 触发方式（即时生效）——
                ID_HOLD_LEFT => {
                    live_cfg.client.hold_mouse_left = !live_cfg.client.hold_mouse_left;
                    let _ = hold_left.set_checked(live_cfg.client.hold_mouse_left);
                    log_info!(
                        "按住左键说话：{}",
                        if live_cfg.client.hold_mouse_left {
                            "开"
                        } else {
                            "关"
                        }
                    );
                }
                ID_FOCUS_CHECK => {
                    live_cfg.client.focus_check = !live_cfg.client.focus_check;
                    let _ = focus_check.set_checked(live_cfg.client.focus_check);
                    log_info!(
                        "焦点检测：{}",
                        if live_cfg.client.focus_check {
                            "开"
                        } else {
                            "关"
                        }
                    );
                }
                ID_HOLD_X2 => {
                    live_cfg.client.hold_mouse_x2 = !live_cfg.client.hold_mouse_x2;
                    let _ = hold_x2.set_checked(live_cfg.client.hold_mouse_x2);
                    log_info!(
                        "侧键 X2：{}",
                        if live_cfg.client.hold_mouse_x2 {
                            "开"
                        } else {
                            "关"
                        }
                    );
                }
                ID_HOLD_CAPS => {
                    live_cfg.client.hold_capslock = !live_cfg.client.hold_capslock;
                    let _ = hold_caps.set_checked(live_cfg.client.hold_capslock);
                    log_info!(
                        "CapsLock：{}",
                        if live_cfg.client.hold_capslock {
                            "开"
                        } else {
                            "关"
                        }
                    );
                }

                // —— 引擎切换（应用内重载）——
                ID_ENGINE_SENSEVOICE if !is_sensevoice_or_live(&live_cfg, "sensevoice") => {
                    live_cfg.engine.kind = "sensevoice".into();
                    need_restart = true;
                }
                ID_ENGINE_QWEN3ASR if !is_sensevoice_or_live(&live_cfg, "qwen3asr") => {
                    live_cfg.engine.kind = "qwen3asr".into();
                    need_restart = true;
                }

                // —— Provider 切换（应用内重载）——
                ID_PROVIDER_AUTO if live_cfg.engine.provider != "auto" => {
                    live_cfg.engine.provider = "auto".into();
                    need_restart = true;
                }
                ID_PROVIDER_CPU if live_cfg.engine.provider != "cpu" => {
                    live_cfg.engine.provider = "cpu".into();
                    need_restart = true;
                }
                ID_PROVIDER_DML if live_cfg.engine.provider != "dml" => {
                    live_cfg.engine.provider = "dml".into();
                    need_restart = true;
                }
                ID_PROVIDER_CUDA if live_cfg.engine.provider != "cuda" => {
                    live_cfg.engine.provider = "cuda".into();
                    need_restart = true;
                }

                // —— 语言切换（应用内重载）——
                ID_LANGUAGE_AUTO if live_cfg.engine.sensevoice.language != "auto" => {
                    set_language(&mut live_cfg, "auto");
                    need_restart = true;
                }
                ID_LANGUAGE_ZH if live_cfg.engine.sensevoice.language != "zh" => {
                    set_language(&mut live_cfg, "zh");
                    need_restart = true;
                }
                ID_LANGUAGE_EN if live_cfg.engine.sensevoice.language != "en" => {
                    set_language(&mut live_cfg, "en");
                    need_restart = true;
                }
                ID_LANGUAGE_JA if live_cfg.engine.sensevoice.language != "ja" => {
                    set_language(&mut live_cfg, "ja");
                    need_restart = true;
                }
                ID_LANGUAGE_KO if live_cfg.engine.sensevoice.language != "ko" => {
                    set_language(&mut live_cfg, "ko");
                    need_restart = true;
                }
                ID_LANGUAGE_YUE if live_cfg.engine.sensevoice.language != "yue" => {
                    set_language(&mut live_cfg, "yue");
                    need_restart = true;
                }

                // —— 引擎参数（应用内重载）——
                ID_USE_ITN => {
                    live_cfg.engine.sensevoice.use_itn = !live_cfg.engine.sensevoice.use_itn;
                    let _ = use_itn.set_checked(live_cfg.engine.sensevoice.use_itn);
                    need_restart = true;
                }
                ID_LLM_USE_GPU => {
                    live_cfg.engine.qwen3asr.llm_use_gpu = !live_cfg.engine.qwen3asr.llm_use_gpu;
                    let _ = llm_use_gpu.set_checked(live_cfg.engine.qwen3asr.llm_use_gpu);
                    need_restart = true;
                }

                // —— 输出与归档（即时生效）——
                ID_PASTE => {
                    live_cfg.client.paste = !live_cfg.client.paste;
                    let _ = paste.set_checked(live_cfg.client.paste);
                    log_info!(
                        "粘贴模式：{}",
                        if live_cfg.client.paste { "开" } else { "关" }
                    );
                }
                ID_RESTORE_CLIP => {
                    live_cfg.client.restore_clipboard = !live_cfg.client.restore_clipboard;
                    let _ = restore_clip.set_checked(live_cfg.client.restore_clipboard);
                    log_info!(
                        "恢复剪贴板：{}",
                        if live_cfg.client.restore_clipboard {
                            "开"
                        } else {
                            "关"
                        }
                    );
                }
                ID_SAVE_AUDIO => {
                    live_cfg.client.save_audio = !live_cfg.client.save_audio;
                    let _ = save_audio.set_checked(live_cfg.client.save_audio);
                    log_info!(
                        "保存录音：{}",
                        if live_cfg.client.save_audio {
                            "开"
                        } else {
                            "关"
                        }
                    );
                }
                ID_SAVE_DIARY => {
                    live_cfg.client.save_diary = !live_cfg.client.save_diary;
                    let _ = save_diary.set_checked(live_cfg.client.save_diary);
                    log_info!(
                        "按日归档：{}",
                        if live_cfg.client.save_diary {
                            "开"
                        } else {
                            "关"
                        }
                    );
                }

                // —— 热词与 LLM（即时生效）——
                ID_HOT_ENABLED => {
                    live_cfg.client.hot_enabled = !live_cfg.client.hot_enabled;
                    let _ = hot_enabled.set_checked(live_cfg.client.hot_enabled);
                    log_info!(
                        "热词替换：{}",
                        if live_cfg.client.hot_enabled {
                            "开"
                        } else {
                            "关"
                        }
                    );
                }
                ID_RULES_ENABLED => {
                    live_cfg.client.rules_enabled = !live_cfg.client.rules_enabled;
                    let _ = rules_enabled.set_checked(live_cfg.client.rules_enabled);
                    log_info!(
                        "规则替换：{}",
                        if live_cfg.client.rules_enabled {
                            "开"
                        } else {
                            "关"
                        }
                    );
                }
                ID_LLM_ENABLED => {
                    live_cfg.llm.enabled = !live_cfg.llm.enabled;
                    let _ = llm_enabled.set_checked(live_cfg.llm.enabled);
                    log_info!(
                        "LLM 角色：{}",
                        if live_cfg.llm.enabled { "开" } else { "关" }
                    );
                }

                _ => {}
            }

            // 更新引擎单选互斥勾选
            match id {
                ID_ENGINE_SENSEVOICE => {
                    let _ = eng_sv.set_checked(true);
                    let _ = eng_q3.set_checked(false);
                }
                ID_ENGINE_QWEN3ASR => {
                    let _ = eng_sv.set_checked(false);
                    let _ = eng_q3.set_checked(true);
                }
                ID_PROVIDER_AUTO => {
                    for m in [&prov_auto, &prov_cpu, &prov_dml, &prov_cuda] {
                        let _ = m.set_checked(false);
                    }
                    let _ = prov_auto.set_checked(true);
                }
                ID_PROVIDER_CPU => {
                    for m in [&prov_auto, &prov_cpu, &prov_dml, &prov_cuda] {
                        let _ = m.set_checked(false);
                    }
                    let _ = prov_cpu.set_checked(true);
                }
                ID_PROVIDER_DML => {
                    for m in [&prov_auto, &prov_cpu, &prov_dml, &prov_cuda] {
                        let _ = m.set_checked(false);
                    }
                    let _ = prov_dml.set_checked(true);
                }
                ID_PROVIDER_CUDA => {
                    for m in [&prov_auto, &prov_cpu, &prov_dml, &prov_cuda] {
                        let _ = m.set_checked(false);
                    }
                    let _ = prov_cuda.set_checked(true);
                }
                ID_LANGUAGE_AUTO => {
                    uncheck_all_langs(&[
                        &lang_auto, &lang_zh, &lang_en, &lang_ja, &lang_ko, &lang_yue,
                    ]);
                    let _ = lang_auto.set_checked(true);
                }
                ID_LANGUAGE_ZH => {
                    uncheck_all_langs(&[
                        &lang_auto, &lang_zh, &lang_en, &lang_ja, &lang_ko, &lang_yue,
                    ]);
                    let _ = lang_zh.set_checked(true);
                }
                ID_LANGUAGE_EN => {
                    uncheck_all_langs(&[
                        &lang_auto, &lang_zh, &lang_en, &lang_ja, &lang_ko, &lang_yue,
                    ]);
                    let _ = lang_en.set_checked(true);
                }
                ID_LANGUAGE_JA => {
                    uncheck_all_langs(&[
                        &lang_auto, &lang_zh, &lang_en, &lang_ja, &lang_ko, &lang_yue,
                    ]);
                    let _ = lang_ja.set_checked(true);
                }
                ID_LANGUAGE_KO => {
                    uncheck_all_langs(&[
                        &lang_auto, &lang_zh, &lang_en, &lang_ja, &lang_ko, &lang_yue,
                    ]);
                    let _ = lang_ko.set_checked(true);
                }
                ID_LANGUAGE_YUE => {
                    uncheck_all_langs(&[
                        &lang_auto, &lang_zh, &lang_en, &lang_ja, &lang_ko, &lang_yue,
                    ]);
                    let _ = lang_yue.set_checked(true);
                }
                _ => {}
            }

            if need_restart || live_cfg != before {
                match client::request_reload(live_cfg.clone(), force_model, !from_disk) {
                    Ok(reply) => {
                        pending = Some(reply);
                        log_info!("已请求重载，将在当前听写结束后应用");
                        let _ = tray.set_tooltip(Some("正在等待听写结束 / 重新加载…"));
                    }
                    Err(e) => log_error!("无法重载：{e:#}"),
                }
                live_cfg = before;
            }
        }

        // 勾选状态始终反映已成功应用的配置。
        hold_left.set_checked(live_cfg.client.hold_mouse_left);
        hold_x2.set_checked(live_cfg.client.hold_mouse_x2);
        hold_caps.set_checked(live_cfg.client.hold_capslock);
        focus_check.set_checked(live_cfg.client.focus_check);
        paste.set_checked(live_cfg.client.paste);
        restore_clip.set_checked(live_cfg.client.restore_clipboard);
        save_audio.set_checked(live_cfg.client.save_audio);
        save_diary.set_checked(live_cfg.client.save_diary);
        hot_enabled.set_checked(live_cfg.client.hot_enabled);
        rules_enabled.set_checked(live_cfg.client.rules_enabled);
        llm_enabled.set_checked(live_cfg.llm.enabled);
        use_itn.set_checked(live_cfg.engine.sensevoice.use_itn);
        llm_use_gpu.set_checked(live_cfg.engine.qwen3asr.llm_use_gpu);
        eng_sv.set_checked(live_cfg.engine.kind == "sensevoice");
        eng_q3.set_checked(live_cfg.engine.kind == "qwen3asr");
        prov_auto.set_checked(live_cfg.engine.provider == "auto");
        prov_cpu.set_checked(live_cfg.engine.provider == "cpu");
        prov_dml.set_checked(live_cfg.engine.provider == "dml");
        prov_cuda.set_checked(live_cfg.engine.provider == "cuda");
        let language = if live_cfg.engine.kind == "qwen3asr" {
            &live_cfg.engine.qwen3asr.language
        } else {
            &live_cfg.engine.sensevoice.language
        };
        lang_auto.set_checked(language == "auto");
        lang_zh.set_checked(language == "zh");
        lang_en.set_checked(language == "en");
        lang_ja.set_checked(language == "ja");
        lang_ko.set_checked(language == "ko");
        lang_yue.set_checked(language == "yue");
        std::thread::sleep(Duration::from_millis(10));
    }

    drop(tray);
    std::process::exit(0);
}

/// 判断当前引擎 kind 是否已是目标值
fn is_sensevoice_or_live(cfg: &Config, kind: &str) -> bool {
    cfg.engine.kind == kind
}

/// 同时设置 sensevoice 和 qwen3asr 的 language
fn set_language(cfg: &mut Config, lang: &str) {
    cfg.engine.sensevoice.language = lang.into();
    cfg.engine.qwen3asr.language = lang.into();
    log_info!("识别语言：{lang}");
}

fn uncheck_all_langs(items: &[&CheckMenuItem]) {
    for item in items {
        let _ = item.set_checked(false);
    }
}

/// 重启自身：释放 Mutex → spawn 新进程（延迟 500ms 等旧进程退出）→ exit
fn restart_self() {
    let Ok(exe) = std::env::current_exe() else {
        log_error!("无法获取当前 exe 路径，重启取消");
        return;
    };

    let exe_path = exe.to_string_lossy();
    let exe_path: String = exe_path
        .strip_prefix(r"\\?\")
        .unwrap_or(&exe_path)
        .to_string();
    log_info!("重启：{}", exe_path);

    // 1) 先释放单实例 Mutex，让新进程能通过单实例检查
    {
        let mut g = MUTEX_HANDLE.lock().unwrap();
        if let Some(h) = g.take() {
            unsafe {
                let _ = CloseHandle(h.0);
            }
        }
    }

    // 2) 写临时批处理文件再执行
    //    不能直接 cmd /c "ping ... & start \"\" \"path\"" —— Rust 的 Command 会用
    //    CRT 风格转义内部引号（\"），而 cmd.exe 不认 \"，导致路径被引号拆烂。
    //    写 .bat 让 cmd 从磁盘读取脚本，彻底绕过引号转义；del "%~f0" 自删。
    let bat = std::env::temp_dir().join("cwrs_restart.bat");
    let content = format!(
        "@echo off\r\nping 127.0.0.1 -n 2 >nul\r\nstart \"\" \"{exe_path}\"\r\ndel \"%~f0\"\r\n"
    );
    if let Err(e) = std::fs::write(&bat, content.as_bytes()) {
        log_error!("写入重启脚本失败：{e:#}");
        return;
    }
    match std::process::Command::new("cmd")
        .args(["/c", &bat.to_string_lossy()])
        .creation_flags(0x08000000) // CREATE_NO_WINDOW
        .spawn()
    {
        Ok(_child) => {
            log_info!("重启子进程已启动，当前进程退出");
            std::process::exit(0);
        }
        Err(e) => {
            log_error!("重启失败：{e:#}");
        }
    }
}
