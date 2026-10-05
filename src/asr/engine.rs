// SenseVoice ONNX 推理引擎：Encoder + CTC Decoder + SentencePiece
// 与原版 CapsWriter 的 sensevoice_onnx/inference 管线逐位对齐

use crate::asr::mel::{MelExtractor, LFR_INPUT};
use crate::asr::spm::SpmModel;
use crate::config::SenseVoiceCfg;
use anyhow::{bail, Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

/// 每秒约 16.67 个 LFR 帧（1 帧 = 0.06s）
const FRAMES_PER_SEC: f64 = 100.0 / 6.0;

#[derive(Debug, Clone)]
pub struct TokenResult {
    pub text: String,
    pub start: f64,
}

#[derive(Debug, Clone, Default)]
pub struct RecognitionResult {
    pub text: String,
    pub tokens: Vec<TokenResult>,
    pub ms: u128,
    pub audio_s: f64,
    pub rtf: f64,
}

pub struct SenseVoiceEngine {
    encoder: Session,
    decoder: Session,
    spm: SpmModel,
    mel: MelExtractor,
    input_is_f16: bool,
    lid_dict: HashMap<String, i64>,
    textnorm_dict: HashMap<String, i64>,
    language: String,
    use_itn: bool,
    chunk_frames: usize,
    overlap_frames: usize,
    enc_out_name: String,
}

impl SenseVoiceEngine {
    pub fn new(
        cfg: &SenseVoiceCfg,
        model_dir: &Path,
        num_threads: usize,
        provider: &str,
    ) -> Result<Self> {
        let encoder_path = model_dir.join(&cfg.encoder);
        let decoder_path = model_dir.join(&cfg.decoder);
        let tokenizer_path = model_dir.join(&cfg.tokenizer);
        for (name, p) in [
            ("encoder", &encoder_path),
            ("decoder", &decoder_path),
            ("tokenizer", &tokenizer_path),
        ] {
            if !p.exists() {
                bail!("模型文件缺失：{name} → {}", p.display());
            }
        }

        let build = |path: &Path| -> Result<Session> {
            let mk = || -> Result<ort::session::builder::SessionBuilder> {
                let b = Session::builder()?;
                let b = b
                    .with_optimization_level(GraphOptimizationLevel::Level3)
                    .with_context(|| format!("配置会话 {}", path.display()))?;
                b.with_intra_threads(num_threads)
                    .map_err(anyhow::Error::from)
            };
            let mut b = mk()?;
            // GPU 执行提供器（load-dynamic：符号不存在 / 驱动缺失时注册失败 → 回退 CPU）
            if matches!(provider, "dml" | "cuda") {
                let eps: Vec<ort::execution_providers::ExecutionProviderDispatch> =
                    if provider == "dml" {
                        vec![ort::execution_providers::DirectMLExecutionProvider::default().build()]
                    } else {
                        vec![ort::execution_providers::CUDAExecutionProvider::default().build()]
                    };
                match b.with_execution_providers(eps) {
                    Ok(b2) => b = b2,
                    Err(e) => {
                        eprintln!("[engine] {provider} EP 注册失败（{e}），回退 CPU");
                        b = mk()?;
                    }
                }
            }
            b.commit_from_file(path)
                .with_context(|| format!("加载 ONNX 模型 {}", path.display()))
        };

        let t0 = Instant::now();
        let encoder = build(&encoder_path)?;
        let decoder = build(&decoder_path)?;
        println!(
            "[engine] 模型加载完成（encoder {:.1}MB / decoder {:.1}MB，耗时 {:.2}s）",
            encoder_path
                .metadata()
                .map(|m| m.len() as f64 / 1e6)
                .unwrap_or(0.0),
            decoder_path
                .metadata()
                .map(|m| m.len() as f64 / 1e6)
                .unwrap_or(0.0),
            t0.elapsed().as_secs_f64()
        );

        // 精度适配：fp16 模型需以 f16 喂数据
        let input_is_f16 = encoder
            .inputs
            .first()
            .map(|i| {
                matches!(
                    i.input_type,
                    ort::value::ValueType::Tensor {
                        ty: ort::tensor::TensorElementType::Float16,
                        ..
                    }
                )
            })
            .unwrap_or(false);

        // 从 ONNX metadata 读取 prompt 字典（缺失则回退内置默认值）
        let mut lid_dict: HashMap<String, i64> = [
            ("auto".to_string(), 4),
            ("zh".to_string(), 3),
            ("en".to_string(), 0),
            ("ja".to_string(), 11),
            ("ko".to_string(), 12),
            ("yue".to_string(), 13),
        ]
        .into_iter()
        .collect();
        let mut textnorm_dict: HashMap<String, i64> =
            [("withitn".to_string(), 14), ("woitn".to_string(), 15)]
                .into_iter()
                .collect();
        if let Ok(meta) = encoder.metadata() {
            if let Ok(Some(v)) = meta.custom("lid_dict") {
                if let Ok(m) = serde_json::from_str::<HashMap<String, i64>>(&v) {
                    lid_dict = m;
                }
            }
            if let Ok(Some(v)) = meta.custom("textnorm_dict") {
                if let Ok(m) = serde_json::from_str::<HashMap<String, i64>>(&v) {
                    textnorm_dict = m;
                }
            }
        }

        let spm_data = std::fs::read(&tokenizer_path)?;
        let spm = SpmModel::parse(&spm_data)
            .with_context(|| format!("解析 tokenizer 失败 {}", tokenizer_path.display()))?;

        let enc_out_name = encoder
            .outputs
            .first()
            .map(|o| o.name.to_string())
            .unwrap_or_else(|| "enc_out".to_string());

        let chunk_frames = (cfg.seg_duration_s * FRAMES_PER_SEC) as usize;
        let overlap_frames = (cfg.seg_overlap_s * FRAMES_PER_SEC) as usize;

        Ok(Self {
            encoder,
            decoder,
            spm,
            mel: MelExtractor::new(),
            input_is_f16,
            lid_dict,
            textnorm_dict,
            language: cfg.language.clone(),
            use_itn: cfg.use_itn,
            chunk_frames,
            overlap_frames,
            enc_out_name,
        })
        .map(|e| {
            println!(
                "[engine] SenseVoice 就绪（vocab={}，f16={}，chunk={}帧/overlap={}帧）",
                e.spm.len(),
                e.input_is_f16,
                e.chunk_frames,
                e.overlap_frames
            );
            e
        })
    }

    /// 预热：跑一次假音频，消除首次推理的初始化开销
    pub fn warmup(&self) {
        let dummy = vec![0f32; 16000]; // 1s 静音
        match self.recognize(&dummy) {
            Ok(_) => println!("[engine] 预热完成"),
            Err(e) => println!("[engine] 预热出现错误（不影响后续使用）：{e}"),
        }
    }

    /// 完整识别：mel → 分段 → 逐段推理 → 重叠拼接
    pub fn recognize(&self, samples: &[f32]) -> Result<RecognitionResult> {
        let t_start = Instant::now();
        let audio_s = samples.len() as f64 / 16000.0;
        let lfr = self.mel.extract(samples);
        let total_frames = lfr.len() / LFR_INPUT;
        if total_frames == 0 {
            return Ok(RecognitionResult {
                audio_s,
                ..Default::default()
            });
        }

        let mut acc_text = String::new();
        let mut acc_tokens: Vec<TokenResult> = Vec::new();
        let stride = self.chunk_frames.saturating_sub(self.overlap_frames).max(1);

        let mut start = 0usize;
        while start < total_frames {
            let end = (start + self.chunk_frames).min(total_frames);
            let chunk = &lfr[start * LFR_INPUT..end * LFR_INPUT];
            let offset_sec = start as f64 / FRAMES_PER_SEC;
            let (text, tokens) = self.recognize_lfr(chunk, offset_sec)?;

            if acc_text.is_empty() {
                acc_text = text;
                acc_tokens = tokens;
            } else {
                // 重叠拼接：最长公共子串对齐（近似 difflib.SequenceMatcher 语义）
                let (a_chars, b_chars): (Vec<char>, Vec<char>) =
                    (acc_text.chars().collect(), text.chars().collect());
                if let Some((ia, _ib, len)) = longest_common_substring(&a_chars, &b_chars) {
                    if len >= 2 {
                        acc_text = a_chars[..ia].iter().collect::<String>() + &text;
                    } else {
                        acc_text.push_str(&text);
                    }
                } else {
                    acc_text.push_str(&text);
                }
                // token 流去重：丢弃时间上与已有尾部重叠的 token
                if let Some(last) = acc_tokens.last().map(|t| t.start) {
                    for tok in tokens {
                        if tok.start >= last {
                            acc_tokens.push(tok);
                        }
                    }
                } else {
                    acc_tokens.extend(tokens);
                }
            }

            if end == total_frames {
                break;
            }
            start += stride;
        }

        let ms = t_start.elapsed().as_millis();
        Ok(RecognitionResult {
            text: acc_text,
            tokens: acc_tokens,
            ms,
            audio_s,
            rtf: if audio_s > 0.0 {
                ms as f64 / 1000.0 / audio_s
            } else {
                0.0
            },
        })
    }

    /// 单段 LFR 特征 → 文本 + token 时间戳
    fn recognize_lfr(&self, lfr: &[f32], offset_sec: f64) -> Result<(String, Vec<TokenResult>)> {
        let t_valid = lfr.len() / LFR_INPUT;
        if t_valid == 0 {
            return Ok((String::new(), Vec::new()));
        }

        // Prompt 构造：[Language, Event, Emo, TextNorm]
        let lid = *self
            .lid_dict
            .get(&self.language)
            .or_else(|| self.lid_dict.get("auto"))
            .unwrap_or(&4);
        let itn = if self.use_itn {
            *self.textnorm_dict.get("withitn").unwrap_or(&14)
        } else {
            *self.textnorm_dict.get("woitn").unwrap_or(&15)
        };

        // Encoder：speech_feat (1,T,560) + mask (1,T) + prompt_ids (1,4)
        let prompt_ids: Vec<i64> = vec![lid, 1, 2, itn];
        let prompt_tensor = Tensor::from_array((vec![1usize, 4], prompt_ids))?;
        let outputs = if self.input_is_f16 {
            let feat: Vec<half::f16> = lfr.iter().map(|&v| half::f16::from_f32(v)).collect();
            let mask: Vec<half::f16> = vec![half::f16::from_f32(1.0); t_valid];
            let feat_tensor = Tensor::from_array((vec![1usize, t_valid, LFR_INPUT], feat))?;
            let mask_tensor = Tensor::from_array((vec![1usize, t_valid], mask))?;
            self.encoder
                .run(ort::inputs![
                    "speech_feat" => feat_tensor,
                    "mask" => mask_tensor,
                    "prompt_ids" => prompt_tensor,
                ]?)
                .context("encoder 推理失败")?
        } else {
            let feat: Vec<f32> = lfr.to_vec();
            let mask: Vec<f32> = vec![1.0; t_valid];
            let feat_tensor = Tensor::from_array((vec![1usize, t_valid, LFR_INPUT], feat))?;
            let mask_tensor = Tensor::from_array((vec![1usize, t_valid], mask))?;
            self.encoder
                .run(ort::inputs![
                    "speech_feat" => feat_tensor,
                    "mask" => mask_tensor,
                    "prompt_ids" => prompt_tensor,
                ]?)
                .context("encoder 推理失败")?
        };

        let enc = if let Ok(t) = outputs[self.enc_out_name.as_str()].try_extract_tensor::<f32>() {
            t.into_owned()
        } else {
            // fp16 模型的 encoder 输出可能保持 f16
            outputs[self.enc_out_name.as_str()]
                .try_extract_tensor::<half::f16>()?
                .mapv(|v| v.to_f32())
        };
        let shape = enc.shape();
        let t_enc = shape[1]; // T + 4
        let d = shape[2]; // 512
                          // 展平为 decoder 输入
        let enc_flat: Vec<f32> = enc.iter().copied().collect();

        // Decoder：输出 top-k 的 log_probs 与 indices
        let dec_out = if self.input_is_f16 {
            let enc16: Vec<half::f16> = enc_flat.iter().map(|&v| half::f16::from_f32(v)).collect();
            let enc_tensor = Tensor::from_array((vec![1usize, t_enc, d], enc16))?;
            self.decoder
                .run(ort::inputs![
                    "enc_out" => enc_tensor,
                ]?)
                .context("decoder 推理失败")?
        } else {
            let enc_tensor = Tensor::from_array((vec![1usize, t_enc, d], enc_flat))?;
            self.decoder
                .run(ort::inputs![
                    "enc_out" => enc_tensor,
                ]?)
                .context("decoder 推理失败")?
        };

        let top_k_name = self.decoder.outputs.first().map(|o| o.name.to_string());
        let idx_name = self.decoder.outputs.get(1).map(|o| o.name.to_string());

        // indices：top-1 即贪心结果（输出可能是 i64 / i32 / f32 / f16）
        let indices = match idx_name {
            Some(name) => {
                let v = &dec_out[name.as_str()];
                if let Ok(t) = v.try_extract_tensor::<i64>() {
                    collect_top1(&t, |x| x as i64)
                } else if let Ok(t) = v.try_extract_tensor::<i32>() {
                    collect_top1(&t, |x| x as i64)
                } else if let Ok(t) = v.try_extract_tensor::<half::f16>() {
                    collect_top1(&t, |x| x.to_f32() as i64)
                } else {
                    let t = v.try_extract_tensor::<f32>()?;
                    collect_top1(&t, |x| x as i64)
                }
            }
            None => bail!("decoder 缺少输出"),
        };
        let _ = top_k_name;

        // CTC 贪心折叠：跳过前 4 个 prompt 帧
        const BLANK: i64 = 0;
        const PROMPT_LEN: usize = 4;
        let mut text = String::new();
        let mut tokens: Vec<TokenResult> = Vec::new();
        let mut prev: i64 = -1;
        for (rel, &id) in indices
            .iter()
            .enumerate()
            .skip(PROMPT_LEN.min(indices.len()))
        {
            let fidx = rel - PROMPT_LEN;
            if id == prev {
                continue;
            }
            prev = id;
            if id == BLANK {
                continue;
            }
            let Some(piece) = self.spm.piece(id) else {
                continue;
            };
            let piece = piece.replace('\u{2581}', " ");
            // 与原版一致：跳过纯空白 piece（保留单个空格）
            if piece.trim().is_empty() && piece != " " {
                continue;
            }
            text.push_str(&piece);
            tokens.push(TokenResult {
                text: piece.clone(),
                start: offset_sec + fidx as f64 * 0.06,
            });
        }
        Ok((text, tokens))
    }
}

fn collect_top1<T: Copy, F: Fn(T) -> i64>(t: &ndarray::ArrayViewD<T>, f: F) -> Vec<i64> {
    // 形状 (1, T, K)：取 [0, f, 0]
    let tt = t.shape()[1];
    let mut out = Vec::with_capacity(tt);
    for fi in 0..tt {
        out.push(f(t[[0, fi, 0]]));
    }
    out
}

/// 最长公共子串：返回 (a 起点, b 起点, 长度)
fn longest_common_substring(a: &[char], b: &[char]) -> Option<(usize, usize, usize)> {
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let mut best = (0usize, 0usize, 0usize);
    let mut prev = vec![0usize; b.len() + 1];
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        for j in 1..=b.len() {
            if a[i - 1] == b[j - 1] {
                cur[j] = prev[j - 1] + 1;
                if cur[j] > best.2 {
                    best = (i - cur[j], j - cur[j], cur[j]);
                }
            } else {
                cur[j] = 0;
            }
        }
        std::mem::swap(&mut prev, &mut cur);
        cur.iter_mut().for_each(|v| *v = 0);
    }
    if best.2 == 0 {
        None
    } else {
        Some(best)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lcs() {
        let a: Vec<char> = "今天天气很好我们出去玩".chars().collect();
        let b: Vec<char> = "很好我们出去玩吧".chars().collect();
        let (ia, _ib, len) = longest_common_substring(&a, &b).unwrap();
        assert_eq!(len, 7);
        assert_eq!(
            &a[ia..ia + len].iter().collect::<String>(),
            "很好我们出去玩"
        );
    }
}
