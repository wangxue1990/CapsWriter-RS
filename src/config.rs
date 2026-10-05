// 配置加载/保存：config.toml ↔ Config 结构体
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, PartialEq, Deserialize, Default)]
#[serde(default)]
pub struct ServerCfg {
    pub num_threads: usize,
    pub warmup: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Deserialize)]
#[serde(default)]
pub struct EngineCfg {
    /// 引擎类型：sensevoice | qwen3asr
    pub kind: String,
    /// onnxruntime.dll 路径（空 = 按 provider 自动选择 exe 同目录的 dll）
    pub onnxruntime_dll: String,
    /// 执行提供器：auto | cpu | dml | cuda
    pub provider: String,
    /// SenseVoice 引擎配置（[engine.sensevoice]）
    #[serde(default)]
    pub sensevoice: SenseVoiceCfg,
    /// Qwen3-ASR（GGUF）引擎配置（[engine.qwen3asr]）
    #[serde(default)]
    pub qwen3asr: Qwen3Cfg,
}

/// SenseVoice 引擎配置（[engine.sensevoice]）
#[derive(Debug, Clone, Serialize, PartialEq, Deserialize)]
#[serde(default)]
pub struct SenseVoiceCfg {
    pub model_dir: String,
    pub encoder: String,
    pub decoder: String,
    pub tokenizer: String,
    pub language: String,
    pub use_itn: bool,
    pub seg_duration_s: f64,
    pub seg_overlap_s: f64,
}

impl Default for SenseVoiceCfg {
    fn default() -> Self {
        Self {
            model_dir: "models/SenseVoice-Small/Sensevoice-Small-ONNX".into(),
            encoder: "SenseVoice-Encoder.int8.onnx".into(),
            decoder: "SenseVoice-CTC.int8.onnx".into(),
            tokenizer: "tokenizer.bpe.model".into(),
            language: "auto".into(),
            use_itn: true,
            seg_duration_s: 40.0,
            seg_overlap_s: 5.0,
        }
    }
}

/// Qwen3-ASR 引擎配置（[engine.qwen3asr]）
#[derive(Debug, Clone, Serialize, PartialEq, Deserialize)]
#[serde(default)]
pub struct Qwen3Cfg {
    pub model_dir: String,
    pub llm_gguf: String,
    pub frontend: String,
    pub backend: String,
    /// 识别语言（auto / zh / en / …，拼入 prompt）
    pub language: String,
    /// llama.cpp bin 目录（空 = <model_dir>/llama/bin）
    pub llama_bin_dir: String,
    pub n_ctx: u32,
    pub chunk_size_s: f64,
    pub memory_num: usize,
    pub temperature: f32,
    pub llm_use_gpu: bool,
}

impl Default for Qwen3Cfg {
    fn default() -> Self {
        Self {
            model_dir: "models/Qwen3-ASR".into(),
            llm_gguf: "qwen3_asr_llm.gguf".into(),
            frontend: "qwen3_asr_encoder_frontend.onnx".into(),
            backend: "qwen3_asr_encoder_backend.onnx".into(),
            language: "auto".into(),
            llama_bin_dir: String::new(),
            n_ctx: 2048,
            chunk_size_s: 40.0,
            memory_num: 1,
            temperature: 0.4,
            llm_use_gpu: true,
        }
    }
}

impl Default for EngineCfg {
    fn default() -> Self {
        Self {
            kind: "sensevoice".into(),
            onnxruntime_dll: String::new(),
            provider: "auto".into(),
            sensevoice: SenseVoiceCfg::default(),
            qwen3asr: Qwen3Cfg::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Deserialize)]
#[serde(default)]
pub struct ClientCfg {
    /// 默认模式是否使用托盘常驻（后台听写）
    pub tray: bool,
    pub hold_mouse_left: bool,
    /// 按住左键前是否检测焦点为可编辑控件（false = 不检测，任何地方按住都录音）
    pub focus_check: bool,
    pub hold_mouse_x2: bool,
    pub hold_capslock: bool,
    pub hold_threshold_ms: u64,
    pub move_cancel_px: i32,
    pub paste: bool,
    pub restore_clipboard: bool,
    pub type_interval_ms: u64,
    pub trash_punc: String,
    pub trash_punc_thresh: usize,
    pub paste_apps: Vec<String>,
    pub enter_apps: Vec<(String, f64)>,
    pub hot_enabled: bool,
    pub hotwords_path: String,
    pub hot_thresh: f64,
    pub rules_enabled: bool,
    pub rules_path: String,
    pub save_audio: bool,
    pub audio_name_len: usize,
    pub save_diary: bool,
    pub archive_dir: String,
}

impl Default for ClientCfg {
    fn default() -> Self {
        Self {
            tray: true,
            hold_mouse_left: true,
            focus_check: true,
            hold_mouse_x2: true,
            hold_capslock: false,
            hold_threshold_ms: 250,
            move_cancel_px: 30,
            paste: true,
            restore_clipboard: true,
            type_interval_ms: 0,
            trash_punc: "，。,.".into(),
            trash_punc_thresh: 8,
            paste_apps: vec![],
            enter_apps: vec![],
            hot_enabled: true,
            hotwords_path: "hot.txt".into(),
            hot_thresh: 0.80,
            rules_enabled: true,
            rules_path: "hot-rule.txt".into(),
            save_audio: true,
            audio_name_len: 20,
            save_diary: true,
            archive_dir: "log".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Deserialize, Default)]
#[serde(default)]
pub struct TranscribeCfg {
    pub save_srt: bool,
    pub save_txt: bool,
    pub save_json: bool,
    pub ffmpeg_fallback: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Deserialize, Default)]
#[serde(default)]
pub struct LlmRole {
    pub name: String,
    pub prompt: String,
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub temperature: f64,
}

#[derive(Debug, Clone, Serialize, PartialEq, Deserialize, Default)]
#[serde(default)]
pub struct LlmCfg {
    pub enabled: bool,
    pub role: Vec<LlmRole>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Deserialize, Default)]
#[serde(default)]
pub struct Config {
    pub server: ServerCfg,
    pub engine: EngineCfg,
    pub client: ClientCfg,
    pub transcribe: TranscribeCfg,
    pub llm: LlmCfg,
    /// 配置文件所在目录（所有相对路径的基准）
    #[serde(skip)]
    pub base_dir: PathBuf,
}

impl Config {
    pub fn load() -> Result<Self> {
        let exe_dir = std::env::current_exe()
            .ok()
            .and_then(|p| p.parent().map(|p| p.to_path_buf()))
            .unwrap_or_else(|| PathBuf::from("."));
        Self::load_from(&exe_dir.join("config.toml"))
    }

    pub fn load_from(path: &Path) -> Result<Self> {
        let base_dir = path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        let mut cfg: Config = if path.exists() {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("无法读取配置文件 {}", path.display()))?;
            toml::from_str(&text).with_context(|| format!("配置文件格式错误 {}", path.display()))?
        } else {
            log_warn(&format!("配置文件 {} 不存在，使用默认配置", path.display()));
            Config::default()
        };
        cfg.base_dir = base_dir;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<()> {
        anyhow::ensure!(
            matches!(self.engine.kind.as_str(), "sensevoice" | "qwen3asr"),
            "不支持的引擎类型"
        );
        anyhow::ensure!(
            matches!(
                self.engine.provider.to_lowercase().as_str(),
                "auto" | "cpu" | "dml" | "cuda"
            ),
            "不支持的执行提供器"
        );
        let sv = &self.engine.sensevoice;
        anyhow::ensure!(
            sv.seg_duration_s.is_finite()
                && sv.seg_duration_s >= 0.1
                && sv.seg_overlap_s.is_finite()
                && sv.seg_overlap_s >= 0.0
                && sv.seg_overlap_s < sv.seg_duration_s,
            "分段长度需 >= 0.1 秒，重叠需 >= 0 且小于分段长度"
        );
        let q = &self.engine.qwen3asr;
        anyhow::ensure!(
            q.chunk_size_s.is_finite() && q.chunk_size_s >= 0.1 && q.n_ctx > 0,
            "Qwen3 分段长度需 >= 0.1 秒且 n_ctx > 0"
        );
        anyhow::ensure!(
            self.client.move_cancel_px >= 0
                && self.client.hot_thresh.is_finite()
                && (0.0..=1.0).contains(&self.client.hot_thresh),
            "移动阈值或热词阈值无效"
        );
        Ok(())
    }

    /// 相对路径解析（基于配置目录）
    pub fn rel(&self, p: &str) -> PathBuf {
        let path = Path::new(p);
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.base_dir.join(path)
        }
    }

    pub fn model_dir(&self) -> PathBuf {
        self.rel(&self.engine.sensevoice.model_dir)
    }

    /// 保存配置回 config.toml（序列化所有字段，含注释）
    pub fn save(&self) -> Result<()> {
        let path = self.base_dir.join("config.toml");
        let text = self.to_toml_string();
        std::fs::write(&path, text)
            .with_context(|| format!("无法写入配置文件 {}", path.display()))?;
        Ok(())
    }

    /// 序列化为带注释的 TOML 字符串（手写以保证输出格式友好）
    pub fn to_toml_string(&self) -> String {
        toml::to_string_pretty(self).expect("配置可序列化为 TOML")
    }
}

fn log_warn(msg: &str) {
    println!(
        "[{}] [WARN] {}",
        chrono::Local::now().format("%H:%M:%S"),
        msg
    );
}
