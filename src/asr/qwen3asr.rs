// Qwen3-ASR 引擎（GGUF 后端）：llama.cpp FFI + Split ONNX 编码器
// 移植自原版 qwen_asr_gguf/inference/{asr.py, encoder.py}
//
// 管线：Whisper128-Mel → Frontend ONNX(每100帧分块) → Backend ONNX(Transformer)
//      → 拼 prompt embedding 序列 → llama.cpp prefill(embd 注入) → 采样生成 → 文本
// 长音频：40s 分块 + 记忆块拼接；解码带熔断（重复检测）+ 加温重试

use crate::asr::engine::RecognitionResult;
use crate::asr::gguf::{EmbeddingTable, GgufFile};
use crate::asr::llama_ffi::{self, LlamaLib, Token};
use crate::asr::whisper_mel::{WhisperMel, N_MELS};
use crate::config::Qwen3Cfg;
use anyhow::{bail, Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::collections::HashSet;
use std::ffi::c_void;
use std::path::Path;
use std::time::Instant;

/// Python floor 除法（负数向下取整）
fn fldiv(a: i64, b: i64) -> i64 {
    let q = a / b;
    if a % b != 0 && ((a < 0) != (b < 0)) {
        q - 1
    } else {
        q
    }
}

/// 完全复刻 Qwen3 前端的有效输出帧数
fn feat_output_len(input_len: usize) -> usize {
    let l = input_len as i64;
    let leave = l % 100;
    let feat = fldiv(leave - 1, 2) + 1;
    (fldiv(fldiv(feat - 1, 2) + 1 - 1, 2) + 1 + (l / 100) * 13) as usize
}

struct QwenInner {
    lib: &'static LlamaLib,
    model: *mut c_void,
    vocab: *const c_void,
    ctx: *mut c_void,
    n_embd: usize,
    table: EmbeddingTable,
    eos: Token,
    id_im_start: Token,
    id_im_end: Token,
    id_audio_start: Token,
    id_audio_end: Token,
    id_asr_text: Token,
}

unsafe impl Send for QwenInner {}

impl Drop for QwenInner {
    fn drop(&mut self) {
        unsafe {
            (self.lib.llama_free)(self.ctx);
            (self.lib.llama_model_free)(self.model);
        }
    }
}

pub struct Qwen3AsrEngine {
    inner: std::sync::Mutex<QwenInner>,
    sess_fe: Session,
    sess_be: Session,
    mel: WhisperMel,
    input_is_f16: bool,
    fe_out_is_f16: bool,
    /// DML 固定形状（30s × 13 帧；0 = 关闭）
    dml_pad_frames: usize,
    language: Option<String>,
    chunk_size_s: f64,
    memory_num: usize,
    temperature: f32,
}

impl Qwen3AsrEngine {
    pub fn new(q: &Qwen3Cfg, base_dir: &Path, provider: &str) -> Result<Self> {
        let model_dir = if Path::new(&q.model_dir).is_absolute() {
            Path::new(&q.model_dir).to_path_buf()
        } else {
            base_dir.join(&q.model_dir)
        };

        // 1. llama.cpp 运行库
        let bin_dir = if q.llama_bin_dir.is_empty() {
            model_dir.join("llama").join("bin")
        } else if Path::new(&q.llama_bin_dir).is_absolute() {
            Path::new(&q.llama_bin_dir).to_path_buf()
        } else {
            base_dir.join(&q.llama_bin_dir)
        };
        let lib = llama_ffi::init(&bin_dir)?;

        // 2. 加载 GGUF 模型
        let gguf_path = model_dir.join(&q.llm_gguf);
        if !gguf_path.exists() {
            bail!("GGUF 模型缺失：{}", gguf_path.display());
        }
        let mut mp = unsafe { (lib.llama_model_default_params)() };
        mp.n_gpu_layers = if q.llm_use_gpu { -1 } else { 0 };
        if !q.llm_use_gpu {
            // devices=[NULL] 强制 CPU（对齐原版）
            mp.devices = Box::into_raw(Box::new(std::ptr::null_mut::<c_void>())) as *mut c_void;
        }
        let model = unsafe {
            let path_str = gguf_path.to_string_lossy().into_owned().into_bytes();
            let mut path_z = path_str;
            path_z.push(0);
            (lib.llama_model_load_from_file)(path_z.as_ptr(), mp)
        };
        if model.is_null() {
            bail!("llama.cpp 加载模型失败：{}", gguf_path.display());
        }
        let vocab = unsafe { (lib.llama_model_get_vocab)(model) };
        let n_embd = unsafe { (lib.llama_model_n_embd)(model) } as usize;
        let eos = unsafe { (lib.llama_vocab_eos)(vocab) };

        // 3. embedding 表（GGUF 直读，免加载整个模型）
        let t0 = Instant::now();
        let gguf = GgufFile::open(&gguf_path)?;
        let table = gguf.embedding_table("token_embd.weight")?;
        println!(
            "[engine] GGUF embedding 视图就绪（{} tokens × {} dims，{}，{:.1}ms）",
            table.vocab_size(),
            table.n_embd(),
            table.quant_type().name(),
            t0.elapsed().as_secs_f64() * 1000.0
        );
        if table.n_embd() != n_embd {
            bail!(
                "embedding 维度不一致：GGUF {} vs llama.cpp {}",
                table.n_embd(),
                n_embd
            );
        }

        // 4. llama context
        let cpu = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let mut cp = unsafe { (lib.llama_context_default_params)() };
        cp.n_ctx = q.n_ctx;
        cp.n_batch = 4096;
        cp.n_ubatch = 512;
        cp.n_seq_max = 1;
        cp.embeddings = false;
        cp.flash_attn_type = 1; // on
        cp.offload_kqv = true;
        cp.no_perf = true;
        cp.n_threads = (cpu / 2).max(1) as i32;
        cp.n_threads_batch = cpu as i32;
        let ctx = unsafe { (lib.llama_init_from_model)(model, cp) };
        if ctx.is_null() {
            bail!("llama 上下文初始化失败");
        }

        // 5. 特殊 token
        let tok_of = |text: &str| -> Token {
            let toks = tokenize(lib, vocab, text, false, true);
            *toks.first().unwrap_or(&(-1))
        };
        let inner = QwenInner {
            lib,
            model,
            vocab,
            ctx,
            n_embd,
            table,
            eos,
            id_im_start: tok_of("<|im_start|>"),
            id_im_end: tok_of("<|im_end|>"),
            id_audio_start: tok_of("<|audio_start|>"),
            id_audio_end: tok_of("<|audio_end|>"),
            id_asr_text: tok_of("<asr_text>"),
        };
        println!(
            "[engine] Qwen3-ASR LLM 就绪（n_embd={n_embd}, ctx={}, 特殊 token 定位完成）",
            q.n_ctx
        );

        // 6. 编码器 ONNX（frontend + backend）
        let fe_path = model_dir.join(&q.frontend);
        let be_path = model_dir.join(&q.backend);
        for (name, p) in [("frontend", &fe_path), ("backend", &be_path)] {
            if !p.exists() {
                bail!("编码器模型缺失：{name} → {}", p.display());
            }
        }
        let build = |path: &Path| -> Result<Session> {
            let mk = || -> Result<ort::session::builder::SessionBuilder> {
                let b = Session::builder()?;
                let b = b
                    .with_optimization_level(GraphOptimizationLevel::Level3)
                    .with_context(|| format!("配置会话 {}", path.display()))?;
                b.with_intra_threads(4).map_err(anyhow::Error::from)
            };
            let mut b = mk()?;
            if provider == "dml" {
                match b.with_execution_providers([
                    ort::execution_providers::DirectMLExecutionProvider::default().build(),
                ]) {
                    Ok(b2) => b = b2,
                    Err(e) => {
                        eprintln!("[engine] 编码器 DML 注册失败（{e}），回退 CPU");
                        b = mk()?;
                    }
                }
            }
            b.commit_from_file(path)
                .with_context(|| format!("加载 {}", path.display()))
        };
        let sess_fe = build(&fe_path)?;
        let sess_be = build(&be_path)?;

        let input_is_f16 = sess_fe
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
        let fe_out_is_f16 = sess_fe
            .outputs
            .first()
            .map(|o| {
                matches!(
                    o.output_type,
                    ort::value::ValueType::Tensor {
                        ty: ort::tensor::TensorElementType::Float16,
                        ..
                    }
                )
            })
            .unwrap_or(false);

        // 7. 预热（短音频）
        let engine = Self {
            inner: std::sync::Mutex::new(inner),
            sess_fe,
            sess_be,
            mel: WhisperMel::new(),
            input_is_f16,
            fe_out_is_f16,
            dml_pad_frames: if provider == "dml" { 30 * 13 } else { 0 },
            language: map_language(&q.language),
            chunk_size_s: q.chunk_size_s,
            memory_num: q.memory_num,
            temperature: q.temperature,
        };
        println!("[engine] 编码器就绪（f16 输入={input_is_f16}），预热中…");
        let dummy = vec![0f32; 16000 * 2];
        let _ = engine.encode(&dummy)?;
        println!("[engine] 预热完成");
        Ok(engine)
    }

    /// 音频 → audio embedding（frames × dim）
    fn encode(&self, audio: &[f32]) -> Result<(Vec<f32>, usize, usize)> {
        // 1. mel (128 × T)
        let mel = self.mel.extract(audio);
        let t = mel.len() / N_MELS;
        if t == 0 {
            bail!("音频过短，无法提取特征");
        }

        // 2. frontend：先 pad 到 100 的倍数（右侧补零），再逐块推理
        let pad = (100 - t % 100) % 100;
        let t_padded = t + pad;
        let num_chunks = t_padded / 100;
        let mut padded = vec![0f32; N_MELS * t_padded];
        for m in 0..N_MELS {
            padded[m * t_padded..m * t_padded + t].copy_from_slice(&mel[m * t..(m + 1) * t]);
        }
        let mut fe_flat: Vec<f32> = Vec::with_capacity(num_chunks * 13 * 1024);
        let mut dim = 0usize;
        for c in 0..num_chunks {
            // 取 padded mel 的第 c 个 100 帧窗口（(128, 100) 行主序）
            let start = c * 100;
            let mut chunk = vec![0f32; N_MELS * 100];
            for m in 0..N_MELS {
                chunk[m * 100..(m + 1) * 100]
                    .copy_from_slice(&padded[m * t_padded + start..m * t_padded + start + 100]);
            }
            let out = if self.input_is_f16 {
                let data: Vec<half::f16> = chunk.iter().map(|&v| half::f16::from_f32(v)).collect();
                let tensor = Tensor::from_array((vec![1usize, N_MELS, 100usize], data))?;
                self.sess_fe
                    .run(ort::inputs!["chunk_mel" => tensor]?)
                    .context("frontend 推理失败")?
            } else {
                let tensor = Tensor::from_array((vec![1usize, N_MELS, 100usize], chunk))?;
                self.sess_fe
                    .run(ort::inputs!["chunk_mel" => tensor]?)
                    .context("frontend 推理失败")?
            };
            // 输出 (1, 13, D)
            let arr = if self.fe_out_is_f16 {
                let a = out[0].try_extract_tensor::<half::f16>()?;
                a.mapv(|v| v.to_f32()).into_owned()
            } else {
                out[0].try_extract_tensor::<f32>()?.into_owned()
            };
            let s = arr.shape();
            if dim == 0 {
                dim = s[2];
            }
            fe_flat.extend(arr.iter().copied());
        }
        if dim == 0 {
            bail!("frontend 无输出");
        }

        // 3. 有效长度切片
        let t_out = feat_output_len(t);
        let frames_per_chunk = 13;
        // fe_flat 布局 (num_chunks*13, dim)，每块 13 帧连续
        let mut hidden = vec![0f32; t_out * dim];
        for f in 0..t_out {
            let src = f * dim;
            hidden[f * dim..(f + 1) * dim].copy_from_slice(&fe_flat[src..src + dim]);
        }
        let _ = frames_per_chunk;

        // 4. backend：hidden (1, T, D) + mask
        let (hidden_in, t_model) = if self.dml_pad_frames > 0 && t_out < self.dml_pad_frames {
            let mut padded = hidden.clone();
            padded.resize(self.dml_pad_frames * dim, 0.0);
            (padded, self.dml_pad_frames)
        } else {
            (hidden.clone(), t_out)
        };

        let audio_embd = if self.input_is_f16 {
            let h: Vec<half::f16> = hidden_in.iter().map(|&v| half::f16::from_f32(v)).collect();
            let h_tensor = Tensor::from_array((vec![1usize, t_model, dim], h))?;
            // mask：无 pad 时全 0；DML pad 时把 [t_out:] 区域屏蔽（对齐原版）
            let mut mask = vec![half::f16::from_f32(0.0); t_model * t_model];
            if t_model > t_out {
                for r in 0..t_model {
                    for c in t_out..t_model {
                        mask[r * t_model + c] = half::f16::from_f32(-10000.0);
                    }
                }
            }
            let m_tensor = Tensor::from_array((vec![1usize, 1usize, t_model, t_model], mask))?;
            let out = self
                .sess_be
                .run(ort::inputs![
                    "hidden_states" => h_tensor,
                    "attention_mask" => m_tensor,
                ]?)
                .context("backend 推理失败")?;
            let a = if self.fe_out_is_f16 {
                out[0]
                    .try_extract_tensor::<half::f16>()?
                    .mapv(|v| v.to_f32())
                    .into_owned()
            } else {
                out[0].try_extract_tensor::<f32>()?.into_owned()
            };
            a.into_raw_vec_and_offset().0
        } else {
            let h_tensor = Tensor::from_array((vec![1usize, t_model, dim], hidden_in))?;
            let mut mask = vec![0.0f32; t_model * t_model];
            if t_model > t_out {
                for r in 0..t_model {
                    for c in t_out..t_model {
                        mask[r * t_model + c] = -10000.0;
                    }
                }
            }
            let m_tensor = Tensor::from_array((vec![1usize, 1usize, t_model, t_model], mask))?;
            let out = self
                .sess_be
                .run(ort::inputs![
                    "hidden_states" => h_tensor,
                    "attention_mask" => m_tensor,
                ]?)
                .context("backend 推理失败")?;
            out[0]
                .try_extract_tensor::<f32>()?
                .into_owned()
                .into_raw_vec_and_offset()
                .0
        };

        // 5. 截断到有效长度（backend 输出维度 2048 = LLM n_embd，与 frontend 的 1024 不同！）
        let dim_be = audio_embd.len() / t_model;
        let valid = t_out * dim_be;
        let embd = if audio_embd.len() > valid {
            audio_embd[..valid].to_vec()
        } else {
            audio_embd
        };
        if std::env::var("CWRS_DEBUG").is_ok() {
            let nz = embd.iter().filter(|v| v.abs() > 1e-6).count();
            let mean = embd.iter().sum::<f32>() / embd.len().max(1) as f32;
            eprintln!(
                "[debug] encode: mel_t={t} audio_embd.len={} frames={t_out} dim={dim_be} 非零={nz} mean={mean:.4} head={:?}",
                embd.len(),
                &embd[..embd.len().min(5)]
            );
            eprintln!("[debug] mel head={:?}", &mel[..mel.len().min(5)]);
        }
        Ok((embd, t_out, dim_be))
    }

    /// 完整识别：分块 + 记忆 + 解码
    pub fn recognize(&self, samples: &[f32]) -> Result<RecognitionResult> {
        let t_start = Instant::now();
        let sr = 16000usize;
        let samples_per_chunk = (self.chunk_size_s * sr as f64) as usize;
        let total = samples.len();
        let num_chunks = total.div_ceil(samples_per_chunk).max(1);

        // 调试：dump 首块 16k 样本 + 全量 mel 供 Python 交叉验证
        if std::env::var("CWRS_DUMP").is_ok() {
            let n = samples.len().min(16000 * 20);
            let spec = hound::WavSpec {
                channels: 1,
                sample_rate: 16000,
                bits_per_sample: 32,
                sample_format: hound::SampleFormat::Float,
            };
            if let Ok(mut w) = hound::WavWriter::create("cwrs_dump_16k.wav", spec) {
                for &s in &samples[..n] {
                    let _ = w.write_sample(s);
                }
                let _ = w.finalize();
                eprintln!("[dump] 16k 样本已写 cwrs_dump_16k.wav（{n} 样本）");
            }
            let mel = self.mel.extract(&samples[..n]);
            let t = mel.len() / N_MELS;
            if let Ok(mut f) = std::fs::File::create("cwrs_dump_mel.bin") {
                use std::io::Write;
                let _ = f.write_all(&(t as u32).to_le_bytes());
                let _ = f.write_all(bytemuck::cast_slice(&mel));
                eprintln!("[dump] mel 已写 cwrs_dump_mel.bin（{t} 帧）");
            }
        }

        // 记忆（audio 特征 + 文本）
        let mut mem_audio: Vec<Vec<f32>> = Vec::new(); // 每块 (frames*dim) 扁平
        let mut mem_frames: Vec<usize> = Vec::new();
        let mut mem_dim: Vec<usize> = Vec::new();
        let mut mem_text: Vec<String> = Vec::new();
        let mut full_text = String::new();

        for i in 0..num_chunks {
            let s = i * samples_per_chunk;
            let e = ((i + 1) * samples_per_chunk).min(total);
            let chunk = if i == num_chunks - 1 && e > s {
                &samples[s..e]
            } else if e <= s {
                &samples[total.saturating_sub(1)..]
            } else {
                &samples[s..e]
            };
            let (audio_embd, frames, dim) = self.encode(chunk)?;

            // 拼接记忆音频 + 当前块
            let mut combined =
                Vec::with_capacity((mem_frames.iter().sum::<usize>() + frames) * dim);
            let mut total_frames = 0usize;
            for (a, &f) in mem_audio.iter().zip(&mem_frames) {
                combined.extend_from_slice(a);
                total_frames += f;
            }
            combined.extend_from_slice(&audio_embd);
            total_frames += frames;

            let prefix_text = mem_text.concat();
            let full_embd = {
                let mut inner = self.inner.lock().unwrap();
                build_prompt_embd(
                    &mut inner,
                    &combined,
                    total_frames,
                    &prefix_text,
                    self.language.as_deref(),
                )?
            };

            let text = {
                let mut inner = self.inner.lock().unwrap();
                safe_decode(&mut inner, &full_embd, self.temperature)?
            };

            // 更新记忆
            mem_audio.push(audio_embd);
            mem_frames.push(frames);
            mem_dim.push(dim);
            mem_text.push(text.clone());
            if mem_audio.len() > self.memory_num {
                mem_audio.remove(0);
                mem_frames.remove(0);
                mem_dim.remove(0);
                mem_text.remove(0);
            }
            full_text.push_str(&text);
        }

        let audio_s = total as f64 / sr as f64;
        let ms = t_start.elapsed().as_millis();
        Ok(RecognitionResult {
            text: full_text,
            tokens: vec![],
            ms,
            audio_s,
            rtf: t_start.elapsed().as_secs_f64() / audio_s.max(1e-6),
        })
    }
}

/// 统一语言代码 → Qwen3 英文名（None = auto 不指定）
fn map_language(code: &str) -> Option<String> {
    match code.to_ascii_lowercase().as_str() {
        "" | "auto" => None,
        "zh" | "chinese" => Some("Chinese".into()),
        "en" | "english" => Some("English".into()),
        "ja" | "japanese" => Some("Japanese".into()),
        "ko" | "korean" => Some("Korean".into()),
        "yue" | "cantonese" => Some("Cantonese".into()),
        other => {
            // 直接当语言名（首字母大写）
            let mut c = other.chars();
            match c.next() {
                Some(f) => Some(f.to_uppercase().collect::<String>() + c.as_str()),
                None => None,
            }
        }
    }
}

fn tokenize(
    lib: &LlamaLib,
    vocab: *const c_void,
    text: &str,
    add_special: bool,
    parse_special: bool,
) -> Vec<Token> {
    let bytes = text.as_bytes();
    let n_max = bytes.len() + 32;
    let mut toks = vec![0i32; n_max];
    let n = unsafe {
        (lib.llama_tokenize)(
            vocab,
            bytes.as_ptr(),
            bytes.len() as i32,
            toks.as_mut_ptr(),
            n_max as i32,
            add_special,
            parse_special,
        )
    };
    if n < 0 {
        Vec::new()
    } else {
        toks.truncate(n as usize);
        toks
    }
}

fn token_bytes(lib: &LlamaLib, vocab: *const c_void, token: Token) -> Vec<u8> {
    let mut buf = [0u8; 256];
    let n = unsafe {
        (lib.llama_token_to_piece)(vocab, token, buf.as_mut_ptr(), buf.len() as i32, 0, true)
    };
    if n > 0 {
        buf[..n as usize].to_vec()
    } else {
        Vec::new()
    }
}

/// 构造 LLM 输入 embedding 序列（区块化打包，对齐原版 _build_prompt_embd）
fn build_prompt_embd(
    inner: &mut QwenInner,
    audio_embd: &[f32],
    audio_frames: usize,
    prefix_text: &str,
    language: Option<&str>,
) -> Result<Vec<f32>> {
    let lib = inner.lib;
    let vocab = inner.vocab;
    let n_embd = inner.n_embd;

    // 区块 A：System + User Header
    let prefix_str = "system\nYou are a helpful assistant.";
    let mut prefix_tokens = vec![inner.id_im_start];
    prefix_tokens.extend(tokenize(lib, vocab, prefix_str, false, true));
    prefix_tokens.push(inner.id_im_end);
    prefix_tokens.push(inner.id_im_start);
    prefix_tokens.extend(tokenize(lib, vocab, "user\n", false, true));
    prefix_tokens.push(inner.id_audio_start);

    // 区块 B：Instruction + Assistant Header + History
    let mut suffix_head = String::from("assistant\n");
    if let Some(l) = language {
        suffix_head.push_str(&format!("language {l}"));
    }
    let mut suffix_tokens = vec![inner.id_audio_end, inner.id_im_end, inner.id_im_start];
    suffix_tokens.extend(tokenize(lib, vocab, &suffix_head, false, true));
    suffix_tokens.push(inner.id_asr_text);
    suffix_tokens.extend(tokenize(lib, vocab, prefix_text, false, true));

    let n_pre = prefix_tokens.len();
    let n_suf = suffix_tokens.len();
    let total = n_pre + audio_frames + n_suf;
    if std::env::var("CWRS_DEBUG").is_ok() {
        eprintln!("[debug] prompt: n_pre={n_pre} n_aud={audio_frames} n_suf={n_suf} total={total}");
        eprintln!("[debug] prefix_tokens={prefix_tokens:?}");
        eprintln!("[debug] suffix_tokens={suffix_tokens:?}");
        let mut row0 = vec![0f32; inner.n_embd];
        if inner.table.row_into(151644, &mut row0).is_ok() {
            eprintln!("[debug] embd[151644] head8={:?}", &row0[..8]);
        }
        let mut row1 = vec![0f32; inner.n_embd];
        if inner.table.row_into(8948, &mut row1).is_ok() {
            eprintln!("[debug] embd[8948] head8={:?}", &row1[..8]);
        }
    }
    let mut out = vec![0f32; total * n_embd];

    let pre_rows = inner.table.rows(&prefix_tokens)?;
    out[..pre_rows.len()].copy_from_slice(&pre_rows);
    out[n_pre * n_embd..n_pre * n_embd + audio_embd.len()].copy_from_slice(audio_embd);
    let suf_rows = inner.table.rows(&suffix_tokens)?;
    let out_len = out.len();
    out[out_len - suf_rows.len()..].copy_from_slice(&suf_rows);
    if std::env::var("CWRS_DUMP").is_ok() {
        let _ = std::fs::write("cwrs_dump_prompt.bin", unsafe {
            std::slice::from_raw_parts(out.as_ptr() as *const u8, out.len() * 4)
        });
        eprintln!("[dump] prompt embd 已写 cwrs_dump_prompt.bin（{total} token × {n_embd}）");
    }
    Ok(out)
}

struct DecodeOutcome {
    text: String,
    aborted: bool,
}

/// 单次 LLM 生成循环（对齐原版 _decode；文本不做流式，直接全量解码）
fn decode_once(
    inner: &mut QwenInner,
    full_embd: &[f32],
    temperature: f32,
) -> Result<DecodeOutcome> {
    let lib = inner.lib;
    // total_len = 完整 prompt 长度（prefix token + 音频帧 + suffix token），从 embd 反推
    let total_len = full_embd.len() / inner.n_embd;
    if total_len == 0 || full_embd.len() % inner.n_embd != 0 {
        bail!("full_embd 长度异常：{} / {}", full_embd.len(), inner.n_embd);
    }
    unsafe {
        let cap = (total_len * 4).max(8192);
        let mut batch = (lib.llama_batch_init)(cap as i32, inner.n_embd as i32, 1);

        // 注入 embedding
        std::ptr::copy_nonoverlapping(full_embd.as_ptr(), batch.embd, full_embd.len());
        // 位置：M-RoPE embd 批次按 4 平面布局（llama.cpp 直接读取 n_tokens×4 个位置）
        for p in 0..3 {
            for i in 0..total_len {
                *batch.pos.add(p * total_len + i) = i as i32;
            }
        }
        for i in 0..total_len {
            *batch.pos.add(3 * total_len + i) = 0;
        }
        batch.n_tokens = total_len as i32;
        for i in 0..total_len {
            *batch.n_seq_id.add(i) = 1;
            **batch.seq_id.add(i) = 0;
            *batch.logits.add(i) = if i == total_len - 1 { 1 } else { 0 };
        }

        // 清空 KV 后 prefill
        let mem = (lib.llama_get_memory)(inner.ctx);
        (lib.llama_memory_clear)(mem, true);
        if (lib.llama_decode)(inner.ctx, batch) != 0 {
            (lib.llama_batch_free)(batch);
            bail!("llama prefill 失败（序列过长？frames={total_len}）");
        }

        // logits 诊断：prefill 后最后位置的 top-5
        if std::env::var("CWRS_DEBUG").is_ok() {
            let n_vocab = (lib.llama_vocab_n_tokens)(inner.vocab) as usize;
            let logits = (lib.llama_get_logits_ith)(inner.ctx, -1);
            if !logits.is_null() {
                let mut idx: Vec<usize> = (0..n_vocab).collect();
                idx.sort_by(|&a, &b| { *logits.add(b) }.partial_cmp(&{ *logits.add(a) }).unwrap());
                let top5: Vec<String> = idx[..5]
                    .iter()
                    .map(|&i| {
                        let piece =
                            String::from_utf8_lossy(&token_bytes(lib, inner.vocab, i as Token))
                                .to_string();
                        format!("{}({:.3})", piece.trim_matches('\0'), { *logits.add(i) })
                    })
                    .collect();
                let norm: f32 =
                    (0..n_vocab).map(|i| (*logits.add(i)).abs()).sum::<f32>() / n_vocab as f32;
                eprintln!(
                    "[debug] prefill logits top5: {} | mean|logit|={norm:.4}",
                    top5.join(", ")
                );
            }
        }

        // 采样链：temp + dist
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos() as u32)
            .unwrap_or(42);
        let chain = (lib.llama_sampler_chain_init)((lib.llama_sampler_chain_default_params)());
        // 对齐原版 LlamaSampler（temperature>0 时）：top_k(50) → temp → dist
        if temperature > 0.0 {
            (lib.llama_sampler_chain_add)(chain, (lib.llama_sampler_init_top_k)(50));
        }
        (lib.llama_sampler_chain_add)(chain, (lib.llama_sampler_init_temp)(temperature));
        if temperature > 0.0 {
            (lib.llama_sampler_chain_add)(chain, (lib.llama_sampler_init_dist)(seed));
        } else {
            (lib.llama_sampler_chain_add)(chain, (lib.llama_sampler_init_greedy)());
        }

        let mut tok = (lib.llama_sampler_sample)(chain, inner.ctx, -1);
        let debug = std::env::var("CWRS_DEBUG").is_ok();
        let mut bytes: Vec<u8> = Vec::new();
        let mut tokens: Vec<Token> = Vec::new();
        let mut aborted = false;
        for it in 0..512 {
            if debug && it < 40 {
                let piece =
                    String::from_utf8_lossy(&token_bytes(lib, inner.vocab, tok)).to_string();
                eprintln!("[debug] iter={it} tok={tok} piece={piece:?}");
            }
            if tok == inner.eos || tok == inner.id_im_end {
                if debug {
                    eprintln!("[debug] 命中结束符：{tok}");
                }
                break;
            }
            // 单 token decode
            let mut t = tok;
            let one = (lib.llama_batch_get_one)(&mut t, 1);
            if (lib.llama_decode)(inner.ctx, one) != 0 {
                break;
            }
            tokens.push(tok);
            bytes.extend(token_bytes(lib, inner.vocab, tok));

            // 熔断：最近 15 个 token 去重后 ≤ 3 → 重复循环
            if tokens.len() > 15 {
                let tail = &tokens[tokens.len() - 15..];
                if tail.iter().collect::<HashSet<_>>().len() <= 3 {
                    aborted = true;
                    break;
                }
            }
            tok = (lib.llama_sampler_sample)(chain, inner.ctx, -1);
        }

        (lib.llama_sampler_free)(chain);
        (lib.llama_batch_free)(batch);
        Ok(DecodeOutcome {
            text: String::from_utf8_lossy(&bytes).to_string(),
            aborted,
        })
    }
}

/// 带熔断加温重试的解码（对齐原版 _safe_decode）
fn safe_decode(inner: &mut QwenInner, full_embd: &[f32], mut temperature: f32) -> Result<String> {
    let mut text = String::new();
    for _ in 0..4 {
        let out = decode_once(inner, full_embd, temperature)?;
        text = out.text;
        if !out.aborted {
            break;
        }
        temperature += 0.3;
        text.push_str("====解码有误，强制熔断====");
        eprintln!("[engine] 触发熔断重试（temp → {temperature:.1}）");
    }
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feat_output_len() {
        // 40s → 4000 帧 → 520
        assert_eq!(feat_output_len(4000), 520);
        // 1s → 100 帧 → 13
        assert_eq!(feat_output_len(100), 13);
        // 250 帧：leave=50 → feat=25 → 7 + 26 = 33
        assert_eq!(feat_output_len(250), 33);
        // 整百边界（Python floor div 语义）
        assert_eq!(feat_output_len(200), 26);
    }

    #[test]
    fn test_fldiv_python_semantics() {
        assert_eq!(fldiv(-1, 2), -1);
        assert_eq!(fldiv(3, 2), 1);
        assert_eq!(fldiv(-3, 2), -2);
    }
}
