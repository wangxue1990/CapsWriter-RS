// CapsWriter-RS：完全离线语音输入（Rust 实现）
//
// 用法：
//   capswriter-rs.exe                     默认：启动听写（托盘常驻 / 控制台）
//   capswriter-rs.exe transcribe <文件>   转录音视频文件 → srt/txt/json
//   capswriter-rs.exe <文件>              同上（支持拖拽文件到 exe）

mod asr;
mod audio;
mod client;
mod config;
mod decode;
mod llm;
mod logger;
mod postprocess;
mod transcribe;
#[cfg(windows)]
mod tray;

use anyhow::{bail, Result};
use asr::engine::SenseVoiceEngine;
use config::Config;
use std::path::PathBuf;
use std::sync::Arc;

fn main() -> Result<()> {
    let cfg = match Config::load() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("加载配置失败：{e:#}");
            std::process::exit(1);
        }
    };
    logger::init(&cfg.base_dir);

    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("console") => run_dictation(&cfg),
        Some("tray") => run_default(&cfg),
        Some("transcribe") => {
            let Some(path) = args.get(2).map(PathBuf::from) else {
                bail!("用法：capswriter-rs transcribe <文件>");
            };
            run_transcribe(&cfg, path)
        }
        Some("help") | Some("--help") | Some("-h") | Some("version") | Some("--version") => {
            print_help();
            Ok(())
        }
        // 调试：输出当前焦点是否为可编辑控件（focus.rs 判定逻辑）
        #[cfg(windows)]
        Some("focus-check") => {
            println!("焦点可编辑：{}", crate::client::focus::has_editable_focus());
            crate::client::focus::debug_focused();
            Ok(())
        }
        Some(arg) if !arg.starts_with('-') => {
            let path = PathBuf::from(arg);
            if !path.exists() {
                bail!("文件不存在：{}", path.display());
            }
            run_transcribe(&cfg, path)
        }
        _ => run_default(&cfg),
    }
}

/// 默认模式：Windows 优先托盘常驻（后台听写），其余回退控制台听写
#[cfg(windows)]
fn run_default(cfg: &Config) -> Result<()> {
    if cfg.client.tray {
        tray::run_tray(cfg.clone())
    } else {
        run_dictation(cfg)
    }
}

#[cfg(not(windows))]
fn run_default(cfg: &Config) -> Result<()> {
    run_dictation(cfg)
}

fn print_help() {
    println!(
        "CapsWriter-RS v{} — 完全离线语音输入（Rust / SenseVoice / ONNX Runtime）

用法：
  capswriter-rs                    托盘常驻（后台听写，按住左键说话松开上屏）
  capswriter-rs console            控制台听写（不开托盘）
  capswriter-rs transcribe <文件>  转录音视频文件（输出 srt/txt/json）
  capswriter-rs <文件>             同上（可直接拖拽文件到 exe）

托盘模式：图标颜色指示状态（蓝=就绪 红=录音 橙=识别 灰=暂停），
右键菜单：暂停/恢复听写、开机自启、打开日志、退出。
[client] tray = false 可禁用托盘，恢复纯控制台行为。

配置：exe 同目录 config.toml；热词 hot.txt；规则 hot-rule.txt",
        env!("CARGO_PKG_VERSION")
    );
}

/// 加载引擎 + 听写客户端
fn run_dictation(cfg: &Config) -> Result<()> {
    let engine = init_engine(cfg)?;
    client::run(cfg, engine)
}

fn run_transcribe(cfg: &Config, path: PathBuf) -> Result<()> {
    let engine = init_engine(cfg)?;
    transcribe::run(cfg, &path, engine)
}

fn init_engine(cfg: &Config) -> Result<Arc<dyn asr::AsrEngine>> {
    // ort 以 load-dynamic 方式加载 onnxruntime.dll：
    //   cpu  → onnxruntime.dll
    //   dml  → onnxruntime-dml.dll（DirectML 版，配 fp16 模型最佳）
    //   cuda → onnxruntime-cuda.dll（需 CUDA 12 运行库）
    //   auto → 按存在性优先 cuda > dml > cpu
    let (provider, dll) = resolve_ort_setup(cfg)?;
    // 动态库在进程中只初始化一次，不能靠修改环境变量切换。
    static RUNTIME: std::sync::OnceLock<(PathBuf, PathBuf)> = std::sync::OnceLock::new();
    let dll = dll.canonicalize()?;
    let q = &cfg.engine.qwen3asr;
    let llama_dir = if q.llama_bin_dir.is_empty() {
        cfg.rel(&q.model_dir).join("llama/bin")
    } else {
        cfg.rel(&q.llama_bin_dir)
    };
    if let Some((loaded_dll, loaded_llama)) = RUNTIME.get() {
        anyhow::ensure!(
            *loaded_dll == dll,
            "切换 ONNX Runtime 动态库需要重启程序；请编辑配置后使用重启菜单"
        );
        anyhow::ensure!(
            *loaded_llama == llama_dir,
            "切换 llama.cpp 运行库目录需要重启程序"
        );
    } else {
        let _ = RUNTIME.set((dll.clone(), llama_dir));
    }
    std::env::set_var("ORT_DYLIB_PATH", &dll);
    println!("[engine] 执行提供器：{provider}（{}）", dll.display());

    match cfg.engine.kind.as_str() {
        "sensevoice" => {
            println!("[engine] 正在加载 SenseVoice 模型…");
            let engine = SenseVoiceEngine::new(
                &cfg.engine.sensevoice,
                &cfg.model_dir(),
                cfg.server.num_threads,
                &provider,
            )?;
            if cfg.server.warmup {
                engine.warmup();
            }
            Ok(Arc::new(engine))
        }
        "qwen3asr" => {
            println!("[engine] 正在加载 Qwen3-ASR（GGUF）模型…");
            let engine =
                asr::qwen3asr::Qwen3AsrEngine::new(&cfg.engine.qwen3asr, &cfg.base_dir, &provider)?;
            Ok(Arc::new(engine))
        }
        other => bail!("暂不支持引擎类型「{other}」（支持：sensevoice / qwen3asr）"),
    }
}

/// 解析 (生效 provider, onnxruntime dll 路径)
fn resolve_ort_setup(cfg: &Config) -> Result<(String, std::path::PathBuf)> {
    // 显式配置了 dll 路径 → 直接使用
    if !cfg.engine.onnxruntime_dll.is_empty() {
        let p = cfg.rel(&cfg.engine.onnxruntime_dll);
        if !p.exists() {
            bail!("onnxruntime.dll 不存在：{}", p.display());
        }
        return Ok((cfg.engine.provider.to_lowercase(), p));
    }

    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| cfg.base_dir.clone());
    let find = |name: &str| -> Option<std::path::PathBuf> {
        [exe_dir.join(name), cfg.base_dir.join(name)]
            .into_iter()
            .find(|p| p.exists())
    };

    let want = cfg.engine.provider.to_lowercase();
    let candidates: &[(&str, &str)] = match want.as_str() {
        "cpu" => &[("cpu", "onnxruntime.dll")],
        "dml" => &[("dml", "onnxruntime-dml.dll")],
        "cuda" => &[("cuda", "onnxruntime-cuda.dll")],
        // auto：优先 GPU
        _ => &[
            ("cuda", "onnxruntime-cuda.dll"),
            ("dml", "onnxruntime-dml.dll"),
            ("cpu", "onnxruntime.dll"),
        ],
    };
    for (prov, name) in candidates {
        if let Some(p) = find(name) {
            return Ok((prov.to_string(), p));
        }
    }
    bail!(
        "未找到 onnxruntime 运行库（exe 同目录应放置 onnxruntime.dll；provider={want}）。\n\
         GPU 加速需另备 onnxruntime-dml.dll（DirectML 版）+ DirectML.dll，详见 README"
    )
}
