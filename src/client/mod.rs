// 客户端控制器：按住说话 → 松开 → 识别 → 后处理 → 上屏 → 归档
//
// 听写模式（钩子/录音触发/上屏）仅 Windows；Linux 上 transcribe 子命令完全可用。

pub mod diary;
#[cfg(windows)]
pub mod focus;
#[cfg(windows)]
pub mod hook;
#[cfg(windows)]
pub mod output;
pub mod recorder;

#[cfg(windows)]
mod run_impl {
    use crate::asr::AsrEngine;
    use crate::config::Config;
    use crate::postprocess::PostProcessor;
    use crate::{log_error, log_info, log_warn};
    use anyhow::Result;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{mpsc, Arc};
    use std::time::{Duration, Instant};

    use super::{diary, focus, hook, output, recorder};

    #[derive(Clone, Copy, PartialEq, Debug)]
    enum HoldKey {
        MouseLeft,
        MouseX2,
        CapsLock,
    }

    struct HoldState {
        key: HoldKey,
        start: Instant,
        start_x: i32,
        start_y: i32,
        moved: bool,
    }

    pub struct ReloadRequest {
        pub cfg: Config,
        pub force_model: bool,
        pub persist: bool,
        pub reply: mpsc::Sender<std::result::Result<Config, String>>,
    }
    static RELOAD: std::sync::OnceLock<mpsc::Sender<ReloadRequest>> = std::sync::OnceLock::new();

    pub fn request_reload(
        cfg: Config,
        force_model: bool,
        persist: bool,
    ) -> Result<mpsc::Receiver<std::result::Result<Config, String>>> {
        let (reply, rx) = mpsc::channel();
        RELOAD
            .get()
            .ok_or_else(|| anyhow::anyhow!("听写线程尚未就绪"))?
            .send(ReloadRequest {
                cfg,
                force_model,
                persist,
                reply,
            })?;
        Ok(rx)
    }

    pub fn run(cfg: &Config, mut engine: Arc<dyn AsrEngine>) -> Result<()> {
        let mut cfg = cfg.clone();
        let (reload_tx, reload_rx) = mpsc::channel::<ReloadRequest>();
        RELOAD
            .set(reload_tx)
            .map_err(|_| anyhow::anyhow!("听写线程已启动"))?;
        println!("─── CapsWriter-RS 语音输入 ───");

        // 后处理器（热词/规则）
        let mut post = PostProcessor::new(&cfg.client, &cfg.base_dir);

        // 归档器
        let mut archiver = diary::Archiver::new(
            &cfg.base_dir,
            &cfg.client.archive_dir,
            cfg.client.save_audio,
            cfg.client.save_diary,
            cfg.client.audio_name_len,
        );

        // 麦克风
        let rec = recorder::Recorder::new()?;

        // 钩子
        let (tx, rx) = mpsc::channel::<hook::HookEvent>();
        let holding = Arc::new(AtomicBool::new(false));
        hook::start(
            tx,
            holding.clone(),
            cfg.client.hold_mouse_left,
            cfg.client.hold_mouse_x2,
            cfg.client.hold_capslock,
        )?;

        println!();
        if cfg.client.hold_mouse_left {
            println!("  · 按住鼠标左键说话，松开上屏（短按/拖拽不受影响）");
        }
        if cfg.client.hold_mouse_x2 {
            println!("  · 按住鼠标侧键 X2 说话，松开上屏");
        }
        if cfg.client.hold_capslock {
            println!("  · 按住 CapsLock 说话，松开上屏（短按恢复大小写切换）");
        }
        println!(
            "  · 短于 {}ms 的按压视为普通操作",
            cfg.client.hold_threshold_ms
        );
        println!();

        let mut threshold = Duration::from_millis(cfg.client.hold_threshold_ms);
        let mut hold: Option<HoldState> = None;

        loop {
            if hold.is_none() {
                if let Ok(request) = reload_rx.try_recv() {
                    hook::set_reloading(true);
                    let result = (|| -> Result<()> {
                        request.cfg.validate()?;
                        anyhow::ensure!(
                            request.cfg.client.tray == cfg.client.tray,
                            "切换托盘/控制台模式需要重启"
                        );
                        let replacement = if request.force_model
                            || request.cfg.engine != cfg.engine
                            || request.cfg.server != cfg.server
                        {
                            Some(crate::init_engine(&request.cfg)?)
                        } else {
                            None
                        };
                        let next_post =
                            PostProcessor::new(&request.cfg.client, &request.cfg.base_dir);
                        let next_archiver = diary::Archiver::new(
                            &request.cfg.base_dir,
                            &request.cfg.client.archive_dir,
                            request.cfg.client.save_audio,
                            request.cfg.client.save_diary,
                            request.cfg.client.audio_name_len,
                        );
                        if request.persist {
                            request.cfg.save()?;
                        }
                        if let Some(next) = replacement {
                            engine = next;
                        }
                        post = next_post;
                        archiver = next_archiver;
                        cfg = request.cfg.clone();
                        threshold = Duration::from_millis(cfg.client.hold_threshold_ms);
                        hook::set_enable_left(cfg.client.hold_mouse_left);
                        hook::set_enable_x2(cfg.client.hold_mouse_x2);
                        hook::set_enable_caps(cfg.client.hold_capslock);
                        Ok(())
                    })();
                    // 加载期间不接受新录音，清理加载前尚未处理的输入事件。
                    while rx.try_recv().is_ok() {}
                    hook::set_reloading(false);
                    let _ = request
                        .reply
                        .send(result.map(|_| cfg.clone()).map_err(|e| format!("{e:#}")));
                }
            }
            let event = match rx.recv_timeout(Duration::from_millis(50)) {
                Ok(event) => event,
                Err(mpsc::RecvTimeoutError::Timeout) => continue,
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            };

            match event {
                hook::HookEvent::LeftDown | hook::HookEvent::X2Down | hook::HookEvent::CapsDown
                    if hold.is_none() =>
                {
                    // 鼠标左键：仅当前焦点为可编辑控件时才进入录音流程
                    //（点在桌面/任务栏/文件列表等非输入区 → 不录音，穿透为普通点击）
                    // focus_check = false 时跳过检测，保持旧的「任何位置按住都录音」行为
                    if matches!(event, hook::HookEvent::LeftDown)
                        && cfg.client.focus_check
                        && !focus::has_editable_focus()
                    {
                        continue;
                    }
                    hold = Some(HoldState {
                        key: match event {
                            hook::HookEvent::LeftDown => HoldKey::MouseLeft,
                            hook::HookEvent::X2Down => HoldKey::MouseX2,
                            _ => HoldKey::CapsLock,
                        },
                        start: Instant::now(),
                        start_x: 0,
                        start_y: 0,
                        moved: false,
                    });
                    rec.start();
                    holding.store(true, Ordering::SeqCst);
                    crate::tray::notify(crate::tray::Status::Recording);
                }

                hook::HookEvent::Move(x, y) => {
                    if let Some(h) = hold.as_mut() {
                        if h.start_x == 0 && h.start_y == 0 && !h.moved {
                            h.start_x = x;
                            h.start_y = y;
                        } else {
                            let dist = ((x - h.start_x).abs() + (y - h.start_y).abs()) as i32;
                            if dist > cfg.client.move_cancel_px {
                                h.moved = true;
                            }
                        }
                    }
                }

                hook::HookEvent::LeftUp | hook::HookEvent::X2Up | hook::HookEvent::CapsUp => {
                    crate::tray::notify(crate::tray::Status::Recognizing);
                    'release: {
                        let Some(h) = hold.as_ref() else {
                            break 'release;
                        };
                        let elapsed = h.start.elapsed();

                        // 键类型不匹配则忽略（如 X2 松开对应左键按住）
                        let matches = match event {
                            hook::HookEvent::LeftUp => h.key == HoldKey::MouseLeft,
                            hook::HookEvent::X2Up => h.key == HoldKey::MouseX2,
                            _ => h.key == HoldKey::CapsLock,
                        };
                        if !matches {
                            // 其他触发键松开不应清除当前录音状态。
                            break 'release;
                        }

                        let h = hold.take().unwrap();
                        holding.store(false, Ordering::SeqCst);

                        // 短按：视为普通按键
                        if elapsed < threshold {
                            let _ = rec.stop(); // 丢弃缓冲
                            match h.key {
                                HoldKey::MouseX2 => {
                                    log_info!("X2 短按，补发原按键");
                                    output::replay_x2_click();
                                }
                                HoldKey::CapsLock => {
                                    log_info!("CapsLock 短按，补发切换");
                                    output::replay_capslock();
                                }
                                HoldKey::MouseLeft => {} // 左键本就穿透，无需处理
                            }
                            break 'release;
                        }

                        // 移动取消：拖拽/框选场景
                        if h.moved {
                            let _ = rec.stop();
                            log_info!("按住期间鼠标移动过大，取消本次识别");
                            break 'release;
                        }

                        // 正常识别流程
                        let samples = rec.stop();
                        let dur = samples.len() as f64 / 16000.0;
                        if dur < 0.15 {
                            log_warn!("录音过短（{dur:.2}s），忽略");
                            break 'release;
                        }

                        // 直接调用引擎识别
                        let t0 = Instant::now();
                        let result = match engine.recognize(&samples) {
                            Ok(r) => r,
                            Err(e) => {
                                log_error!("识别失败：{e:#}");
                                break 'release;
                            }
                        };
                        let reply_text = result.text;
                        let reply_ms = result.ms;
                        if reply_text.trim().is_empty() {
                            log_info!("（未识别到内容，{dur:.1}s 音频）");
                            archiver.save(&samples, "");
                            break 'release;
                        }

                        // 后处理
                        let raw = reply_text.clone();
                        let mut text = post.process(&raw);

                        // LLM 角色
                        if cfg.llm.enabled && !cfg.llm.role.is_empty() {
                            if let Some((role, user_input)) =
                                crate::llm::extract_role(&cfg.llm.role, &text)
                            {
                                log_info!("触发 LLM 角色「{}」", role.name);
                                match crate::llm::complete(role, &user_input) {
                                    Ok(out) => text = out,
                                    Err(e) => log_warn!("LLM 调用失败（{e}），使用原始识别结果"),
                                }
                            }
                        }

                        println!(
                            "▸ {}（音频 {dur:.1}s，引擎 {}ms，全程 {}ms）",
                            text,
                            reply_ms,
                            t0.elapsed().as_millis()
                        );

                        // 上屏
                        output_text(&cfg.client, &text);

                        // 归档（后台线程，不阻塞下一次输入）
                        let arch = archiver.clone();
                        let samples_c = samples.clone();
                        std::thread::spawn(move || arch.save(&samples_c, &text));
                    } // 'release
                    crate::tray::notify(crate::tray::Status::Idle);
                }

                _ => {}
            }
        }
        Ok(())
    }

    fn output_text(cfg: &crate::config::ClientCfg, text: &str) {
        let process = output::active_process_name();
        let force_paste = cfg
            .paste_apps
            .iter()
            .any(|app| app.to_ascii_lowercase() == process);

        if cfg.paste || force_paste {
            let saved = if cfg.restore_clipboard {
                output::clipboard_get_text()
            } else {
                None
            };
            if let Err(e) = output::clipboard_set_text(text) {
                log_error!("写剪贴板失败：{e}，改用模拟输入");
                output::type_text(text, cfg.type_interval_ms);
                return;
            }
            // 稍等剪贴板就绪
            std::thread::sleep(Duration::from_millis(30));
            output::send_ctrl_v();
            if cfg.restore_clipboard {
                if let Some(prev) = saved {
                    std::thread::spawn(move || {
                        std::thread::sleep(Duration::from_millis(150));
                        let _ = output::clipboard_set_text(&prev);
                    });
                }
            }
        } else {
            output::type_text(text, cfg.type_interval_ms);
        }

        // 特定应用输出后自动回车
        for (app, delay) in &cfg.enter_apps {
            if app.to_ascii_lowercase() == process {
                std::thread::sleep(Duration::from_millis((*delay * 1000.0) as u64));
                output::send_enter();
            }
        }
    }
} // mod run_impl

#[cfg(windows)]
pub use run_impl::{request_reload, run};

// 非 Windows：听写（钩子/上屏）不可用，transcribe 子命令正常
#[cfg(not(windows))]
pub fn run(
    _cfg: &crate::config::Config,
    _engine: std::sync::Arc<dyn crate::asr::AsrEngine>,
) -> anyhow::Result<()> {
    anyhow::bail!("听写模式（全局钩子 + 上屏）仅支持 Windows；Linux 上请使用 `transcribe` 子命令")
}
