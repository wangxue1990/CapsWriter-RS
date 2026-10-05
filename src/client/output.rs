// 文本上屏：剪贴板 + Ctrl+V 粘贴 / SendInput Unicode 逐字符输入
// 以及：X2/CapsLock 短按补发、当前前台进程探测

use anyhow::Result;
use windows::core::PWSTR;
use windows::Win32::Foundation::{HANDLE, HGLOBAL, HWND};
use windows::Win32::System::DataExchange::{
    CloseClipboard, EmptyClipboard, GetClipboardData, OpenClipboard, SetClipboardData,
};
use windows::Win32::System::Memory::{GlobalAlloc, GlobalLock, GlobalUnlock, GMEM_MOVEABLE};
use windows::Win32::System::Threading::{
    OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION,
};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    SendInput, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBDINPUT, KEYEVENTF_KEYUP, KEYEVENTF_UNICODE,
    MOUSEEVENTF_XDOWN, MOUSEEVENTF_XUP, MOUSEINPUT, VIRTUAL_KEY, VK_CAPITAL, VK_CONTROL, VK_RETURN,
};
use windows::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetWindowThreadProcessId};

const CF_UNICODETEXT: u32 = 13;
const XBUTTON2_U16: u16 = 0x0002;

// ============ 剪贴板 ============

pub fn clipboard_get_text() -> Option<String> {
    unsafe {
        if OpenClipboard(None).is_err() {
            return None;
        }
        let result = (|| {
            let handle = GetClipboardData(CF_UNICODETEXT).ok()?;
            let hglobal = HGLOBAL(handle.0);
            let ptr = GlobalLock(hglobal) as *const u16;
            if ptr.is_null() {
                return None;
            }
            let mut len = 0usize;
            while *ptr.add(len) != 0 {
                len += 1;
            }
            let slice = std::slice::from_raw_parts(ptr, len);
            let _ = GlobalUnlock(hglobal);
            Some(String::from_utf16_lossy(slice))
        })();
        let _ = CloseClipboard();
        result
    }
}

pub fn clipboard_set_text(text: &str) -> Result<()> {
    let wide: Vec<u16> = text.encode_utf16().chain(std::iter::once(0)).collect();
    unsafe {
        OpenClipboard(None).map_err(|e| anyhow::anyhow!("打开剪贴板失败：{e}"))?;
        let result = (|| -> Result<()> {
            EmptyClipboard().map_err(|e| anyhow::anyhow!("清空剪贴板失败：{e}"))?;
            let bytes = wide.len() * 2;
            let hglobal = GlobalAlloc(GMEM_MOVEABLE, bytes)
                .map_err(|e| anyhow::anyhow!("分配内存失败：{e}"))?;
            let ptr = GlobalLock(hglobal) as *mut u16;
            if ptr.is_null() {
                anyhow::bail!("GlobalLock 失败");
            }
            std::ptr::copy_nonoverlapping(wide.as_ptr(), ptr, wide.len());
            let _ = GlobalUnlock(hglobal);
            SetClipboardData(CF_UNICODETEXT, Some(HANDLE(hglobal.0)))
                .map_err(|e| anyhow::anyhow!("写入剪贴板失败：{e}"))?;
            Ok(())
        })();
        let _ = CloseClipboard();
        result
    }
}

// ============ 按键注入 ============

fn send_inputs(inputs: &[INPUT]) {
    if inputs.is_empty() {
        return;
    }
    unsafe {
        SendInput(inputs, std::mem::size_of::<INPUT>() as i32);
    }
}

fn kb_input(vk: u16, scan: u16, up: bool) -> INPUT {
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: VIRTUAL_KEY(vk),
                wScan: scan,
                dwFlags: if up {
                    KEYEVENTF_KEYUP
                } else {
                    Default::default()
                },
                time: 0,
                dwExtraInfo: 0,
            },
        },
    }
}

/// 模拟 Ctrl+V
pub fn send_ctrl_v() {
    let inputs = [
        kb_input(VK_CONTROL.0 as u16, 0, false),
        kb_input('V' as u16, 0, false),
        kb_input('V' as u16, 0, true),
        kb_input(VK_CONTROL.0 as u16, 0, true),
    ];
    send_inputs(&inputs);
}

/// Unicode 逐字符输入（支持中日韩）
pub fn type_text(text: &str, interval_ms: u64) {
    let mut batch: Vec<INPUT> = Vec::with_capacity(64);
    let mut buf = [0u16; 2];
    for ch in text.chars() {
        if ch == '\n' || ch == '\r' {
            batch.push(kb_input(VK_RETURN.0 as u16, 0, false));
            batch.push(kb_input(VK_RETURN.0 as u16, 0, true));
        } else {
            let units: &[u16] = ch.encode_utf16(&mut buf);
            for &u in units {
                batch.push(unicode_input(u, false));
                batch.push(unicode_input(u, true));
            }
        }
        if batch.len() >= 64 {
            send_inputs(&batch);
            batch.clear();
        }
        if interval_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(interval_ms));
        }
    }
    send_inputs(&batch);
}

fn unicode_input(scancode: u16, up: bool) -> INPUT {
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: VIRTUAL_KEY(0),
                wScan: scancode,
                dwFlags: if up {
                    KEYEVENTF_UNICODE | KEYEVENTF_KEYUP
                } else {
                    KEYEVENTF_UNICODE
                },
                time: 0,
                dwExtraInfo: 0,
            },
        },
    }
}

/// 补发鼠标 X2（前进键）短按
pub fn replay_x2_click() {
    let xdata = (XBUTTON2_U16 as u32) << 16;
    let mk = |flags| INPUT {
        r#type: windows::Win32::UI::Input::KeyboardAndMouse::INPUT_MOUSE,
        Anonymous: INPUT_0 {
            mi: MOUSEINPUT {
                dx: 0,
                dy: 0,
                mouseData: xdata,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: 0,
            },
        },
    };
    send_inputs(&[mk(MOUSEEVENTF_XDOWN), mk(MOUSEEVENTF_XUP)]);
}

/// 补发 CapsLock 单击（恢复大小写切换能力）
pub fn replay_capslock() {
    let inputs = [
        kb_input(VK_CAPITAL.0 as u16, 0, false),
        kb_input(VK_CAPITAL.0 as u16, 0, true),
    ];
    send_inputs(&inputs);
}

/// 补发回车
pub fn send_enter() {
    let inputs = [
        kb_input(VK_RETURN.0 as u16, 0, false),
        kb_input(VK_RETURN.0 as u16, 0, true),
    ];
    send_inputs(&inputs);
}

// ============ 前台窗口进程探测 ============

/// 当前前台窗口的进程名（小写，如 "wechat.exe"），失败返回空
pub fn active_process_name() -> String {
    unsafe {
        let hwnd: HWND = GetForegroundWindow();
        if hwnd.0.is_null() {
            return String::new();
        }
        let mut pid = 0u32;
        GetWindowThreadProcessId(hwnd, Some(&mut pid));
        if pid == 0 {
            return String::new();
        }
        let Ok(handle) = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) else {
            return String::new();
        };
        let mut buf = [0u16; 512];
        let mut len = buf.len() as u32;
        let ok = QueryFullProcessImageNameW(
            handle,
            PROCESS_NAME_WIN32,
            PWSTR(buf.as_mut_ptr()),
            &mut len,
        );
        let _ = windows::Win32::Foundation::CloseHandle(handle);
        if ok.is_err() || len == 0 {
            return String::new();
        }
        let path = String::from_utf16_lossy(&buf[..len as usize]);
        path.rsplit(['\\', '/'])
            .next()
            .unwrap_or("")
            .to_ascii_lowercase()
    }
}
