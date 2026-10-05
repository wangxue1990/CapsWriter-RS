// 音频/视频文件解码 → 16kHz 单声道 f32
// wav → hound；mp3/flac/ogg/m4a/aac → symphonia；其他（mp4/mkv 等）→ ffmpeg

use crate::audio;
use anyhow::{bail, Result};
use std::path::Path;
use std::process::Command;

pub fn load_audio_16k(path: &Path, ffmpeg_fallback: bool) -> Result<Vec<f32>> {
    if !path.exists() {
        bail!("文件不存在: {}", path.display());
    }
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .unwrap_or_default();

    match ext.as_str() {
        "wav" | "wave" => {
            let (samples, sr) = audio::read_wav(path)?;
            Ok(audio::resample_to_16k(&samples, sr))
        }
        "mp3" | "flac" | "ogg" | "oga" | "opus" | "m4a" | "aac" | "caf" | "mp4" | "mkv"
        | "webm" => match decode_symphonia(path) {
            Ok((samples, sr)) => Ok(audio::resample_to_16k(&samples, sr)),
            Err(_) if ffmpeg_fallback => decode_ffmpeg(path),
            Err(e) => Err(e.context("symphonia 解码失败")),
        },
        _ => {
            if ffmpeg_fallback {
                decode_ffmpeg(path)
            } else {
                bail!("不支持的格式 .{ext}（可在 PATH 安装 ffmpeg 后重试）")
            }
        }
    }
}

fn decode_symphonia(path: &Path) -> Result<(Vec<f32>, u32)> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let file = std::fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe().format(
        &hint,
        mss,
        &FormatOptions::default(),
        &MetadataOptions::default(),
    )?;
    let mut format = probed.format;
    let (track_id, track_params) = {
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .ok_or_else(|| anyhow::anyhow!("未找到音轨"))?;
        (track.id, track.codec_params.clone())
    };
    let sample_rate = track_params.sample_rate.unwrap_or(16000);
    let mut decoder =
        symphonia::default::get_codecs().make(&track_params, &DecoderOptions::default())?;

    let mut all: Vec<f32> = Vec::new();
    let mut channels = 1usize;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(symphonia::core::errors::Error::ResetRequired) => break,
            Err(e) => return Err(e.into()),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(e.into()),
        };
        let spec = *decoded.spec();
        channels = spec.channels.count().max(1);
        let frames = decoded.frames() as u64;
        let mut sampler = SampleBuffer::<f32>::new(frames, spec);
        sampler.copy_interleaved_ref(decoded);
        all.extend_from_slice(sampler.samples());
    }
    let mono = audio::mix_to_mono(&all, channels);
    let sr = if sample_rate == 0 { 16000 } else { sample_rate };
    Ok((mono, sr))
}

fn decode_ffmpeg(path: &Path) -> Result<Vec<f32>> {
    let output = Command::new("ffmpeg")
        .arg("-y")
        .arg("-i")
        .arg(path)
        .arg("-ar")
        .arg("16000")
        .arg("-ac")
        .arg("1")
        .arg("-f")
        .arg("f32le")
        .arg("pipe:1")
        .output()
        .map_err(|e| anyhow::anyhow!("调用 ffmpeg 失败（未安装或不在 PATH？）：{e}"))?;
    if !output.status.success() {
        bail!(
            "ffmpeg 解码失败: {}",
            String::from_utf8_lossy(&output.stderr)
                .chars()
                .take(500)
                .collect::<String>()
        );
    }
    Ok(bytemuck::cast_slice(&output.stdout).to_vec())
}
