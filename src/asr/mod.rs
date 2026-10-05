pub mod engine;
pub mod gguf;
pub mod llama_ffi;
pub mod mel;
pub mod qwen3asr;
pub mod spm;
pub mod whisper_mel;

pub use engine::{RecognitionResult, SenseVoiceEngine};

use anyhow::Result;

/// 引擎统一接口：server 侧只依赖此 trait
pub trait AsrEngine: Send + Sync {
    fn recognize(&self, samples: &[f32]) -> Result<RecognitionResult>;
    #[allow(dead_code)]
    fn name(&self) -> &'static str;
}

impl AsrEngine for SenseVoiceEngine {
    fn recognize(&self, samples: &[f32]) -> Result<RecognitionResult> {
        SenseVoiceEngine::recognize(self, samples)
    }
    fn name(&self) -> &'static str {
        "sensevoice"
    }
}

impl AsrEngine for qwen3asr::Qwen3AsrEngine {
    fn recognize(&self, samples: &[f32]) -> Result<RecognitionResult> {
        qwen3asr::Qwen3AsrEngine::recognize(self, samples)
    }
    fn name(&self) -> &'static str {
        "qwen3asr"
    }
}
