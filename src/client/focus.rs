// 焦点输入框检测（UIAutomation）：鼠标左键按下时判断当前是否有可编辑控件获得焦点
//
// 白名单判定：ControlType == Edit（非只读）或 Document。
// 任何 UIA 故障一律返回 true（fail-open，保持旧的“总是录音”行为）。

#![cfg(windows)]

use std::cell::RefCell;
use windows::Win32::System::Com::{
    CoCreateInstance, CoInitializeEx, CLSCTX_INPROC_SERVER, COINIT_APARTMENTTHREADED,
};
use windows::Win32::UI::Accessibility::{
    CUIAutomation, IUIAutomation, IUIAutomationValuePattern, UIA_DocumentControlTypeId,
    UIA_EditControlTypeId, UIA_ValuePatternId,
};

thread_local! {
    /// 每线程懒初始化（COM 接口指针非 Send/Sync；实际只有听写线程调用）
    static UIA: RefCell<Option<IUIAutomation>> = const { RefCell::new(None) };
}

/// 懒初始化 UIAutomation（失败则永久降级为 fail-open）
fn with_uia<R>(f: impl FnOnce(&IUIAutomation) -> R) -> Option<R> {
    UIA.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            unsafe {
                // COM 初始化；线程已用其它模式初始化过也无妨，继续尝试创建
                let _ = CoInitializeEx(None, COINIT_APARTMENTTHREADED);
                *slot = CoCreateInstance(&CUIAutomation, None, CLSCTX_INPROC_SERVER).ok();
            }
        }
        slot.as_ref().map(f)
    })
}

/// 当前焦点元素是否为可编辑控件
///
/// 判定（白名单，宁缺勿滥）：
///   ControlType == Edit（50004）且非只读 → 输入框（微信/浏览器输入框/记事本…）
///   ControlType == Document（50030）     → Word / 富文本 / 浏览器页面
/// 其余一律 false（曾试过「可写 Value pattern」兜底，但资源管理器文件项因支持
/// 重命名暴露可写 Value pattern 导致误判，已放弃该兜底）
pub fn has_editable_focus() -> bool {
    with_uia(|uia| unsafe {
        let Ok(elem) = uia.GetFocusedElement() else {
            return true;
        };
        let Ok(ct) = elem.CurrentControlType() else {
            return true;
        };

        if ct == UIA_EditControlTypeId {
            // Edit 控件：若支持 Value pattern 且只读 → 不可编辑
            if let Ok(vp) = value_pattern(&elem) {
                if let Ok(ro) = vp.CurrentIsReadOnly() {
                    if ro.as_bool() {
                        return false;
                    }
                }
            }
            return true;
        }
        if ct == UIA_DocumentControlTypeId {
            // Win11 记事本/Word 等富文本编辑区是 Document；
            // Electron/Chromium 整页渲染器也暴露 Document（readonly=true），
            // 有 Value pattern 且只读 → 页面非输入区，不算
            if let Ok(vp) = value_pattern(&elem) {
                if let Ok(ro) = vp.CurrentIsReadOnly() {
                    if ro.as_bool() {
                        return false;
                    }
                }
            }
            return true;
        }
        false
    })
    .unwrap_or(true) // UIA 不可用 → fail-open（保持旧的“总是录音”行为）
}

/// 取焦点元素的 Value pattern（不支持则 Err）
unsafe fn value_pattern(
    elem: &windows::Win32::UI::Accessibility::IUIAutomationElement,
) -> windows::core::Result<IUIAutomationValuePattern> {
    elem.GetCurrentPatternAs::<IUIAutomationValuePattern>(UIA_ValuePatternId)
}

/// 调试：打印当前焦点元素的关键属性
pub fn debug_focused() {
    with_uia(|uia| unsafe {
        let Ok(elem) = uia.GetFocusedElement() else {
            println!("[focus] GetFocusedElement 失败");
            return;
        };
        let ct = elem.CurrentControlType().map(|c| c.0).unwrap_or(-1);
        let name = elem.CurrentName().unwrap_or_default();
        let cls = elem.CurrentClassName().unwrap_or_default();
        let val = value_pattern(&elem)
            .ok()
            .and_then(|vp| vp.CurrentValue().ok())
            .unwrap_or_default();
        let ro = value_pattern(&elem)
            .ok()
            .and_then(|vp| vp.CurrentIsReadOnly().ok());
        println!(
            "[focus] ControlType={ct} name={name:?} class={cls:?} value={val:?} readonly={ro:?}",
        );
    });
}
