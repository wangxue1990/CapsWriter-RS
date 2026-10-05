// 文件转录：解码 → 引擎识别 → 后处理 → 写出 srt/txt/json

use crate::asr::engine::TokenResult;
use crate::asr::AsrEngine;
use crate::config::Config;
use crate::decode;
use crate::log_info;
use crate::postprocess::PostProcessor;
use anyhow::{bail, Context, Result};
use std::path::Path;
use std::sync::Arc;

pub fn run(cfg: &Config, path: &Path, engine: Arc<dyn AsrEngine>) -> Result<()> {
    let t0 = std::time::Instant::now();
    println!("─── CapsWriter-RS 文件转录 ───");
    println!("输入：{}", path.display());

    let samples = decode::load_audio_16k(path, cfg.transcribe.ffmpeg_fallback)
        .with_context(|| format!("解码失败 {}", path.display()))?;
    let duration = samples.len() as f64 / 16000.0;
    println!(
        "解码完成：{duration:.1}s 音频，{:.1}MB",
        samples.len() as f64 * 4.0 / 1e6
    );

    println!("已加载引擎，开始转录…");
    let result = engine.recognize(&samples)?;
    if result.text.is_empty() && result.tokens.is_empty() {
        bail!("识别结果为空");
    }
    println!(
        "转录完成：{:.1}s 音频 / {:.1}s 引擎耗时（RTF {:.3}）",
        result.audio_s,
        result.ms as f64 / 1000.0,
        result.rtf
    );

    let post = PostProcessor::new(&cfg.client, &cfg.base_dir);
    let raw_text = result.text.clone();

    // 输出文件
    let stem = path.with_extension("");
    if cfg.transcribe.save_txt {
        let out = stem.with_extension("txt");
        std::fs::write(&out, format!("{}\n", post.process(&raw_text)))?;
        println!("已保存 {}", out.display());
    }
    if cfg.transcribe.save_json {
        let out = stem.with_extension("json");
        let tokens: Vec<serde_json::Value> = result
            .tokens
            .iter()
            .map(|t| serde_json::json!({"t": t.text, "s": t.start}))
            .collect();
        let obj = serde_json::json!({
            "file": path.to_string_lossy(),
            "duration_s": duration,
            "text": raw_text,
            "tokens": tokens,
        });
        std::fs::write(&out, serde_json::to_string_pretty(&obj)?)?;
        println!("已保存 {}", out.display());
    }
    if cfg.transcribe.save_srt {
        let tokens: Vec<TokenResult> = result
            .tokens
            .iter()
            .map(|t| TokenResult {
                text: t.text.clone(),
                start: t.start,
            })
            .collect();
        let srt = build_srt(&tokens, &post);
        let out = stem.with_extension("srt");
        std::fs::write(&out, srt)?;
        println!("已保存 {}", out.display());
    }

    log_info!("文件转录总耗时 {:.1}s", t0.elapsed().as_secs_f64());
    println!("────── 识别文本 ──────");
    println!("{}", post.process(&raw_text));
    println!("──────────────────────");
    Ok(())
}

/// token 流 → SRT 字幕（按标点或每 25 字切行）
fn build_srt(tokens: &[TokenResult], post: &PostProcessor) -> String {
    #[derive(Default)]
    struct Line {
        text: String,
        start: Option<f64>,
        end: Option<f64>,
    }
    let mut lines: Vec<Line> = vec![Line::default()];
    for tok in tokens {
        let line = lines.last_mut().unwrap();
        let is_break = tok.text.contains(['。', '！', '？', '?', '!']);
        if line.start.is_none() {
            line.start = Some(tok.start);
        }
        line.text.push_str(&tok.text);
        line.end = Some(tok.start + 0.06);
        if is_break || line.text.chars().count() >= 25 {
            lines.push(Line::default());
        }
    }
    lines.retain(|l| !l.text.trim().is_empty());

    let mut out = String::new();
    for (i, line) in lines.iter().enumerate() {
        let text = post.process(line.text.trim());
        let start = line.start.unwrap_or(0.0);
        let end = line.end.unwrap_or(start + 0.6);
        out.push_str(&format!("{}\n", i + 1));
        out.push_str(&format!(
            "{} --> {}\n",
            fmt_srt_time(start),
            fmt_srt_time(end)
        ));
        out.push_str(&format!("{text}\n\n"));
    }
    out
}

fn fmt_srt_time(s: f64) -> String {
    let ms = (s * 1000.0).round() as u64;
    format!(
        "{:02}:{:02}:{:02},{:03}",
        ms / 3_600_000,
        (ms % 3_600_000) / 60_000,
        (ms % 60_000) / 1000,
        ms % 1000
    )
}
