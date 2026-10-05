// Windows 全局钩子：鼠标左键/X2 按住说话（短按穿透/补发），CapsLock 可选
#![allow(non_snake_case)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::OnceLock;
use windows::core::PCWSTR;
use windows::Win32::Foundation::{LPARAM, LRESULT, WPARAM};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::Input::KeyboardAndMouse::VK_CAPITAL;
use windows::Win32::UI::WindowsAndMessaging::*;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HookEvent {
    LeftDown,
    LeftUp,
    X2Down,
    X2Up,
    CapsDown,
    CapsUp,
    Move(i32, i32),
}

// 钩子回调需要访问的共享状态
static TX: OnceLock<Sender<HookEvent>> = OnceLock::new();
static HOLDING: OnceLock<std::sync::Arc<AtomicBool>> = OnceLock::new();
static ENABLE_LEFT: AtomicBool = AtomicBool::new(true);
static ENABLE_X2: AtomicBool = AtomicBool::new(true);
static ENABLE_CAPS: AtomicBool = AtomicBool::new(false);
/// 托盘「暂停听写」：暂停时不监听也不拦截任何按键
static RELOADING: AtomicBool = AtomicBool::new(false);
pub fn set_reloading(v: bool) {
    RELOADING.store(v, Ordering::SeqCst);
}

static PAUSED: AtomicBool = AtomicBool::new(false);

/// 暂停/恢复钩子（托盘菜单调用）
pub fn set_paused(v: bool) {
    PAUSED.store(v, Ordering::Relaxed);
}

/// 运行时开关：按住左键
pub fn set_enable_left(v: bool) {
    ENABLE_LEFT.store(v, Ordering::Relaxed);
}

/// 运行时开关：侧键 X2
pub fn set_enable_x2(v: bool) {
    ENABLE_X2.store(v, Ordering::Relaxed);
}

/// 运行时开关：CapsLock
pub fn set_enable_caps(v: bool) {
    ENABLE_CAPS.store(v, Ordering::Relaxed);
}

const LLMHF_INJECTED_U32: u32 = 0x0001;
const LLKHF_INJECTED_U32: u32 = 0x0010;
const XBUTTON2_U16: u16 = 0x0002;

fn send_event(e: HookEvent) {
    if let Some(tx) = TX.get() {
        let _ = tx.send(e);
    }
}

fn is_holding() -> bool {
    HOLDING
        .get()
        .map(|h| h.load(Ordering::Relaxed))
        .unwrap_or(false)
}

unsafe extern "system" fn mouse_proc(code: i32, w_param: WPARAM, l_param: LPARAM) -> LRESULT {
    if code == HC_ACTION as i32
        && !PAUSED.load(Ordering::Relaxed)
        && !RELOADING.load(Ordering::SeqCst)
    {
        let info = &*(l_param.0 as *const MSLLHOOKSTRUCT);
        if (info.flags & LLMHF_INJECTED_U32) == 0 {
            let msg = w_param.0 as u32;
            match msg {
                WM_LBUTTONDOWN if ENABLE_LEFT.load(Ordering::Relaxed) => {
                    send_event(HookEvent::LeftDown);
                }
                WM_LBUTTONUP if ENABLE_LEFT.load(Ordering::Relaxed) => {
                    send_event(HookEvent::LeftUp);
                }
                WM_XBUTTONDOWN | WM_XBUTTONUP
                    if ENABLE_X2.load(Ordering::Relaxed)
                        && ((w_param.0 >> 16) as u16 == XBUTTON2_U16
                            || (info.mouseData >> 16) as u16 == XBUTTON2_U16) =>
                {
                    if msg == WM_XBUTTONDOWN {
                        send_event(HookEvent::X2Down);
                        // 吞掉 X2 按键，避免误触发浏览器前进；短按会由控制器补发
                        return LRESULT(1);
                    } else {
                        send_event(HookEvent::X2Up);
                        return LRESULT(1);
                    }
                }
                WM_MOUSEMOVE if is_holding() => {
                    send_event(HookEvent::Move(info.pt.x, info.pt.y));
                }
                _ => {}
            }
        }
    }
    CallNextHookEx(None, code, w_param, l_param)
}

unsafe extern "system" fn keyboard_proc(code: i32, w_param: WPARAM, l_param: LPARAM) -> LRESULT {
    if code == HC_ACTION as i32
        && !PAUSED.load(Ordering::Relaxed)
        && !RELOADING.load(Ordering::SeqCst)
    {
        let info = &*(l_param.0 as *const KBDLLHOOKSTRUCT);
        if (info.flags.0 & LLKHF_INJECTED_U32) == 0
            && info.vkCode == VK_CAPITAL.0 as u32
            && ENABLE_CAPS.load(Ordering::Relaxed)
        {
            match w_param.0 as u32 {
                WM_KEYDOWN | WM_SYSKEYDOWN => {
                    send_event(HookEvent::CapsDown);
                    return LRESULT(1); // 阻断大小写切换
                }
                WM_KEYUP | WM_SYSKEYUP => {
                    send_event(HookEvent::CapsUp);
                    return LRESULT(1);
                }
                _ => {}
            }
        }
    }
    CallNextHookEx(None, code, w_param, l_param)
}

/// 启动钩子线程（含 Win32 消息循环）
pub fn start(
    tx: Sender<HookEvent>,
    holding: std::sync::Arc<AtomicBool>,
    enable_left: bool,
    enable_x2: bool,
    enable_caps: bool,
) -> anyhow::Result<()> {
    TX.set(tx).map_err(|_| anyhow::anyhow!("钩子已启动"))?;
    HOLDING
        .set(holding)
        .map_err(|_| anyhow::anyhow!("钩子已启动"))?;
    ENABLE_LEFT.store(enable_left, Ordering::Relaxed);
    ENABLE_X2.store(enable_x2, Ordering::Relaxed);
    ENABLE_CAPS.store(enable_caps, Ordering::Relaxed);

    std::thread::Builder::new()
        .name("win32-hook".into())
        .spawn(|| unsafe {
            let module = GetModuleHandleW(PCWSTR::null()).unwrap_or_default();
            let mouse_hook =
                SetWindowsHookExW(WH_MOUSE_LL, Some(mouse_proc), Some(module.into()), 0);
            let kb_hook =
                SetWindowsHookExW(WH_KEYBOARD_LL, Some(keyboard_proc), Some(module.into()), 0);
            if mouse_hook.is_err() || kb_hook.is_err() {
                eprintln!(
                    "[hook] 安装钩子失败：mouse={:?} keyboard={:?}",
                    mouse_hook, kb_hook
                );
                return;
            }
            println!(
                "[hook] 全局钩子已安装（左键={} X2={} CapsLock={}）",
                ENABLE_LEFT.load(Ordering::Relaxed),
                ENABLE_X2.load(Ordering::Relaxed),
                ENABLE_CAPS.load(Ordering::Relaxed)
            );
            let mut msg = MSG::default();
            // GetMessageW: >0 继续，0 退出，-1 错误
            loop {
                let r = GetMessageW(&mut msg, None, 0, 0);
                if r.0 <= 0 {
                    break;
                }
                let _ = TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
            if let Ok(h) = mouse_hook {
                let _ = UnhookWindowsHookEx(h);
            }
            if let Ok(h) = kb_hook {
                let _ = UnhookWindowsHookEx(h);
            }
        })?;
    Ok(())
}
