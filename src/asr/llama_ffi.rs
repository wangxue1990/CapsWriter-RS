// llama.cpp (b10621 魔改版) 的动态 FFI 绑定
// 结构体布局与原版 core/server/engines/llama/llama.py 的 ctypes 定义逐字段对齐
// 通过 libloading 运行时加载 bin/ 下的 ggml.dll / ggml-base.dll / llama.dll

use anyhow::{bail, Context, Result};
use libloading::os::windows::Library as WinLibrary;
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

pub type Token = i32;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LlamaModelParams {
    pub devices: *mut c_void,
    pub tensor_buft_overrides: *mut c_void,
    pub n_gpu_layers: i32,
    pub split_mode: i32,
    pub load_mode: i32,
    pub main_gpu: i32,
    pub tensor_split: *mut f32,
    pub progress_callback: Option<unsafe extern "C" fn(f32, *mut c_void) -> bool>,
    pub progress_callback_user_data: *mut c_void,
    pub kv_overrides: *mut c_void,
    pub vocab_only: bool,
    pub check_tensors: bool,
    pub use_extra_bufts: bool,
    pub no_host: bool,
    pub no_alloc: bool,
    pub load_mtp: bool,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LlamaContextParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_rs_seq: u32,
    pub n_outputs_max: u32,
    pub n_outputs_max_per_seq: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub ctx_type: i32,
    pub rope_scaling_type: i32,
    pub pooling_type: i32,
    pub attention_type: i32,
    pub flash_attn_type: i32,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: f32,
    pub cb_eval: *mut c_void,
    pub cb_eval_user_data: *mut c_void,
    pub type_k: i32,
    pub type_v: i32,
    pub abort_callback: *mut c_void,
    pub abort_callback_data: *mut c_void,
    pub embeddings: bool,
    pub offload_kqv: bool,
    pub no_perf: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
    pub samplers: *mut c_void,
    pub n_samplers: usize,
    pub ctx_other: *mut c_void,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LlamaBatch {
    pub n_tokens: i32,
    pub token: *mut Token,
    pub embd: *mut f32,
    pub pos: *mut i32,
    pub n_seq_id: *mut i32,
    pub seq_id: *mut *mut Token,
    pub logits: *mut i8,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct LlamaSamplerChainParams {
    pub no_perf: bool,
}

unsafe extern "C" fn log_callback(_level: i32, message: *const u8, _user: *mut c_void) {
    if message.is_null() {
        return;
    }
    unsafe {
        let mut len = 0usize;
        while *message.add(len) != 0 {
            len += 1;
        }
        let bytes = std::slice::from_raw_parts(message, len);
        let s = String::from_utf8_lossy(bytes).trim().to_string();
        if !s.is_empty() && s != "." {
            crate::log_info!("[llama] {s}");
        }
    }
}

/// llama.cpp 函数表（DLL 提取的函数指针，进程内全局一份）
#[allow(non_snake_case)]
pub struct LlamaLib {
    #[allow(dead_code)]
    keep: Vec<WinLibrary>,
    pub llama_vocab_n_tokens: unsafe extern "C" fn(vocab: *const c_void) -> i32,
    pub llama_vocab_eos: unsafe extern "C" fn(vocab: *const c_void) -> Token,
    #[allow(dead_code)]
    pub llama_vocab_bos: unsafe extern "C" fn(vocab: *const c_void) -> Token,
    pub llama_model_default_params: unsafe extern "C" fn() -> LlamaModelParams,
    pub llama_model_load_from_file:
        unsafe extern "C" fn(path: *const u8, params: LlamaModelParams) -> *mut c_void,
    pub llama_model_free: unsafe extern "C" fn(model: *mut c_void),
    pub llama_model_get_vocab: unsafe extern "C" fn(model: *mut c_void) -> *const c_void,
    pub llama_model_n_embd: unsafe extern "C" fn(model: *mut c_void) -> i32,
    pub llama_context_default_params: unsafe extern "C" fn() -> LlamaContextParams,
    pub llama_init_from_model:
        unsafe extern "C" fn(model: *mut c_void, params: LlamaContextParams) -> *mut c_void,
    pub llama_free: unsafe extern "C" fn(ctx: *mut c_void),
    pub llama_batch_init:
        unsafe extern "C" fn(n_tokens: i32, embd: i32, n_seq_max: i32) -> LlamaBatch,
    pub llama_batch_free: unsafe extern "C" fn(batch: LlamaBatch),
    pub llama_batch_get_one: unsafe extern "C" fn(tokens: *mut Token, n_tokens: i32) -> LlamaBatch,
    pub llama_decode: unsafe extern "C" fn(ctx: *mut c_void, batch: LlamaBatch) -> i32,
    pub llama_get_logits_ith: unsafe extern "C" fn(ctx: *mut c_void, i: i32) -> *mut f32,
    pub llama_tokenize: unsafe extern "C" fn(
        vocab: *const c_void,
        text: *const u8,
        text_len: i32,
        tokens: *mut Token,
        n_max: i32,
        add_special: bool,
        parse_special: bool,
    ) -> i32,
    pub llama_token_to_piece: unsafe extern "C" fn(
        vocab: *const c_void,
        token: Token,
        buf: *mut u8,
        len: i32,
        lstrip: i32,
        special: bool,
    ) -> i32,
    pub llama_get_memory: unsafe extern "C" fn(ctx: *mut c_void) -> *mut c_void,
    pub llama_memory_clear: unsafe extern "C" fn(mem: *mut c_void, data: bool),
    pub llama_sampler_chain_default_params: unsafe extern "C" fn() -> LlamaSamplerChainParams,
    pub llama_sampler_chain_init:
        unsafe extern "C" fn(params: LlamaSamplerChainParams) -> *mut c_void,
    pub llama_sampler_chain_add: unsafe extern "C" fn(chain: *mut c_void, sampler: *mut c_void),
    pub llama_sampler_init_greedy: unsafe extern "C" fn() -> *mut c_void,
    pub llama_sampler_init_top_k: unsafe extern "C" fn(i32) -> *mut c_void,
    pub llama_sampler_init_dist: unsafe extern "C" fn(seed: u32) -> *mut c_void,
    pub llama_sampler_init_temp: unsafe extern "C" fn(temp: f32) -> *mut c_void,
    pub llama_sampler_sample:
        unsafe extern "C" fn(sm: *mut c_void, ctx: *mut c_void, idx: i32) -> Token,
    pub llama_sampler_free: unsafe extern "C" fn(sm: *mut c_void),
    #[allow(dead_code)]
    pub llama_sampler_accept: unsafe extern "C" fn(sm: *mut c_void, token: Token),
}

static LIB: OnceLock<&'static LlamaLib> = OnceLock::new();

/// 初始化：加载 bin 目录下的 ggml / ggml-base / llama DLL 并绑定符号（幂等）
pub fn init(bin_dir: &Path) -> Result<&'static LlamaLib> {
    if let Some(lib) = LIB.get() {
        return Ok(lib);
    }

    let ggml = load_dll(bin_dir, "ggml.dll")?;
    let ggml_base = load_dll(bin_dir, "ggml-base.dll")?;
    let llama = load_dll(bin_dir, "llama.dll")?;

    unsafe {
        // 日志回调 + 后端初始化
        let log_set: libloading::os::windows::Symbol<
            unsafe extern "C" fn(
                cb: unsafe extern "C" fn(i32, *const u8, *mut c_void),
                user: *mut c_void,
            ),
        > = llama.get(b"llama_log_set\0").unwrap();
        log_set(log_callback, std::ptr::null_mut());

        let load_all: libloading::os::windows::Symbol<unsafe extern "C" fn()> =
            ggml.get(b"ggml_backend_load_all\0").unwrap();
        // ggml_backend_load_all 从 CWD 扫描后端 DLL —— 对齐原版：临时 chdir 到 bin 目录
        let original_cwd = std::env::current_dir().ok();
        let _ = std::env::set_current_dir(bin_dir);
        load_all();
        if let Some(cwd) = original_cwd {
            let _ = std::env::set_current_dir(cwd);
        }

        let backend_init: libloading::os::windows::Symbol<unsafe extern "C" fn()> =
            llama.get(b"llama_backend_init\0").unwrap();
        backend_init();

        // 先提取全部符号（借用 llama），再 move DLL 进结构体
        macro_rules! typed {
            ($name:literal, $ty:ty) => {{
                let s: libloading::os::windows::Symbol<$ty> = llama.get($name).map_err(|e| {
                    anyhow::anyhow!("符号 {} 缺失：{}", String::from_utf8_lossy($name), e)
                })?;
                *s
            }};
        }

        let llama_vocab_n_tokens = typed!(
            b"llama_vocab_n_tokens\0",
            unsafe extern "C" fn(*const c_void) -> i32
        );
        let llama_vocab_eos = typed!(
            b"llama_vocab_eos\0",
            unsafe extern "C" fn(*const c_void) -> Token
        );
        let llama_vocab_bos = typed!(
            b"llama_vocab_bos\0",
            unsafe extern "C" fn(*const c_void) -> Token
        );
        let llama_model_default_params = typed!(
            b"llama_model_default_params\0",
            unsafe extern "C" fn() -> LlamaModelParams
        );
        let llama_model_load_from_file = typed!(
            b"llama_model_load_from_file\0",
            unsafe extern "C" fn(*const u8, LlamaModelParams) -> *mut c_void
        );
        let llama_model_free = typed!(b"llama_model_free\0", unsafe extern "C" fn(*mut c_void));
        let llama_model_get_vocab = typed!(
            b"llama_model_get_vocab\0",
            unsafe extern "C" fn(*mut c_void) -> *const c_void
        );
        let llama_model_n_embd = typed!(
            b"llama_model_n_embd\0",
            unsafe extern "C" fn(*mut c_void) -> i32
        );
        let llama_context_default_params = typed!(
            b"llama_context_default_params\0",
            unsafe extern "C" fn() -> LlamaContextParams
        );
        let llama_init_from_model = typed!(
            b"llama_init_from_model\0",
            unsafe extern "C" fn(*mut c_void, LlamaContextParams) -> *mut c_void
        );
        let llama_free = typed!(b"llama_free\0", unsafe extern "C" fn(*mut c_void));
        let llama_batch_init = typed!(
            b"llama_batch_init\0",
            unsafe extern "C" fn(i32, i32, i32) -> LlamaBatch
        );
        let llama_batch_free = typed!(b"llama_batch_free\0", unsafe extern "C" fn(LlamaBatch));
        let llama_batch_get_one = typed!(
            b"llama_batch_get_one\0",
            unsafe extern "C" fn(*mut Token, i32) -> LlamaBatch
        );
        let llama_decode = typed!(
            b"llama_decode\0",
            unsafe extern "C" fn(*mut c_void, LlamaBatch) -> i32
        );
        let llama_get_logits_ith = typed!(
            b"llama_get_logits_ith\0",
            unsafe extern "C" fn(*mut c_void, i32) -> *mut f32
        );
        let llama_tokenize = typed!(
            b"llama_tokenize\0",
            unsafe extern "C" fn(*const c_void, *const u8, i32, *mut Token, i32, bool, bool) -> i32
        );
        let llama_token_to_piece = typed!(
            b"llama_token_to_piece\0",
            unsafe extern "C" fn(*const c_void, Token, *mut u8, i32, i32, bool) -> i32
        );
        let llama_get_memory = typed!(
            b"llama_get_memory\0",
            unsafe extern "C" fn(*mut c_void) -> *mut c_void
        );
        let llama_memory_clear = typed!(
            b"llama_memory_clear\0",
            unsafe extern "C" fn(*mut c_void, bool)
        );
        let llama_sampler_chain_default_params = typed!(
            b"llama_sampler_chain_default_params\0",
            unsafe extern "C" fn() -> LlamaSamplerChainParams
        );
        let llama_sampler_chain_init = typed!(
            b"llama_sampler_chain_init\0",
            unsafe extern "C" fn(LlamaSamplerChainParams) -> *mut c_void
        );
        let llama_sampler_chain_add = typed!(
            b"llama_sampler_chain_add\0",
            unsafe extern "C" fn(*mut c_void, *mut c_void)
        );
        let llama_sampler_init_greedy = typed!(
            b"llama_sampler_init_greedy\0",
            unsafe extern "C" fn() -> *mut c_void
        );
        let llama_sampler_init_top_k = typed!(
            b"llama_sampler_init_top_k\0",
            unsafe extern "C" fn(i32) -> *mut c_void
        );
        let llama_sampler_init_dist = typed!(
            b"llama_sampler_init_dist\0",
            unsafe extern "C" fn(u32) -> *mut c_void
        );
        let llama_sampler_init_temp = typed!(
            b"llama_sampler_init_temp\0",
            unsafe extern "C" fn(f32) -> *mut c_void
        );
        let llama_sampler_sample = typed!(
            b"llama_sampler_sample\0",
            unsafe extern "C" fn(*mut c_void, *mut c_void, i32) -> Token
        );
        let llama_sampler_free = typed!(b"llama_sampler_free\0", unsafe extern "C" fn(*mut c_void));
        let llama_sampler_accept = typed!(
            b"llama_sampler_accept\0",
            unsafe extern "C" fn(*mut c_void, Token)
        );

        let lib = LlamaLib {
            keep: vec![ggml, ggml_base, llama],
            llama_vocab_n_tokens,
            llama_vocab_eos,
            llama_vocab_bos,
            llama_model_default_params,
            llama_model_load_from_file,
            llama_model_free,
            llama_model_get_vocab,
            llama_model_n_embd,
            llama_context_default_params,
            llama_init_from_model,
            llama_free,
            llama_batch_init,
            llama_batch_free,
            llama_batch_get_one,
            llama_decode,
            llama_get_logits_ith,
            llama_tokenize,
            llama_token_to_piece,
            llama_get_memory,
            llama_memory_clear,
            llama_sampler_chain_default_params,
            llama_sampler_chain_init,
            llama_sampler_chain_add,
            llama_sampler_init_greedy,
            llama_sampler_init_top_k,
            llama_sampler_init_dist,
            llama_sampler_init_temp,
            llama_sampler_sample,
            llama_sampler_free,
            llama_sampler_accept,
        };

        let leaked: &'static LlamaLib = Box::leak(Box::new(lib));
        let _ = LIB.set(leaked);
        Ok(leaked)
    }
}

fn load_dll(dir: &Path, name: &str) -> Result<WinLibrary> {
    const LOAD_WITH_ALTERED_SEARCH_PATH: u32 = 0x00000008;
    let path: PathBuf = dir.join(name);
    if !path.exists() {
        bail!("llama 运行库缺失：{}", path.display());
    }
    unsafe {
        WinLibrary::load_with_flags(&path, LOAD_WITH_ALTERED_SEARCH_PATH)
            .with_context(|| format!("加载 {}", path.display()))
    }
}
