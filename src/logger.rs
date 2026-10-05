// 极简日志：控制台 + logs/latest.log
use std::io::Write;
use std::sync::Mutex;
use std::sync::OnceLock;

static LOG_FILE: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();

pub fn init(base_dir: &std::path::Path) {
    let dir = base_dir.join("logs");
    let _ = std::fs::create_dir_all(&dir);
    let file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(dir.join("latest.log"))
        .ok()
        .map(Mutex::new);
    let _ = LOG_FILE.set(file);
}

pub fn emit(level: &str, msg: &str) {
    let ts = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
    let line = format!("[{ts}] [{level}] {msg}");
    println!("{line}");
    if let Some(Some(file)) = LOG_FILE.get().map(|f| f.as_ref()) {
        if let Ok(mut f) = file.lock() {
            let _ = writeln!(f, "{line}");
        }
    }
}

#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => { $crate::logger::emit("INFO", &format!($($arg)*)) };
}
#[macro_export]
macro_rules! log_warn {
    ($($arg:tt)*) => { $crate::logger::emit("WARN", &format!($($arg)*)) };
}
#[macro_export]
macro_rules! log_error {
    ($($arg:tt)*) => { $crate::logger::emit("ERROR", &format!($($arg)*)) };
}
#[macro_export]
macro_rules! log_debug {
    ($($arg:tt)*) => { $crate::logger::emit("DEBUG", &format!($($arg)*)) };
}
