// 音频工具：重采样（对齐 scipy.signal.resample_poly）与 WAV 读写

/// 多相 FIR 重采样，精确移植原版 numpy_resample_poly（Kaiser 窗 β=5.0）
pub fn resample_poly(x: &[f32], up: usize, down: usize) -> Vec<f32> {
    if up == 0 || down == 0 || x.is_empty() {
        return x.to_vec();
    }
    let g = gcd(up, down);
    let (up, down) = (up / g, down / g);
    if up == down {
        return x.to_vec();
    }

    const WINDOW_SIZE: usize = 10;
    let max_rate = up.max(down);
    let f_c = 1.0f64 / max_rate as f64;
    let half_len = WINDOW_SIZE * max_rate;
    let n_taps = 2 * half_len + 1;

    // firwin：sinc + Kaiser(5.0)，归一化到 up
    let beta = 5.0f64;
    let i0_beta = bessel_i0(beta);
    let mut h = Vec::with_capacity(n_taps);
    let mut sum = 0.0f64;
    for i in 0..n_taps {
        let t = i as f64 - half_len as f64;
        let mut v = if (f_c * t).abs() < f64::EPSILON {
            f_c
        } else {
            (f_c * t * std::f64::consts::PI).sin() / (std::f64::consts::PI * t)
        };
        // sinc(f_c * t)
        let arg = 1.0 - (2.0 * t / (n_taps as f64 - 1.0)).powi(2);
        let kaiser = if arg <= 0.0 {
            0.0
        } else {
            bessel_i0(beta * arg.sqrt()) / i0_beta
        };
        v *= kaiser;
        sum += v;
        h.push(v);
    }
    let scale = up as f64 / sum;
    for v in h.iter_mut() {
        *v *= scale;
    }

    // upfirdn：上采样零填充 → 卷积 → 抽取
    // 直接法太慢（44.1k→16k 时 taps=8821），改为稀疏迭代：
    // y_full[idx] = Σ_i x[i] * h[idx - i*up]，只遍历非零的输入点
    let length_in = x.len();
    let length_out = (length_in * up + down - 1) / down;
    let offset = (n_taps - 1) / 2;
    let end = offset + length_in * up;
    let mut y = Vec::with_capacity(length_out);
    let mut idx = offset;
    while idx < end && y.len() < length_out {
        // i 的有效范围：idx - i*up ∈ [0, n_taps)
        let i_min = idx.saturating_sub(n_taps - 1).div_ceil(up);
        let i_max = (idx / up).min(length_in.saturating_sub(1));
        let mut acc = 0.0f64;
        for i in i_min..=i_max {
            let k = idx - i * up;
            if k < n_taps {
                acc += x[i] as f64 * h[k];
            }
        }
        y.push(acc as f32);
        idx += down;
    }
    y
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

fn bessel_i0(x: f64) -> f64 {
    // 级数近似，精度足够
    let mut sum = 1.0;
    let mut term = 1.0;
    let half = x / 2.0;
    for k in 1..50 {
        term *= (half / k as f64).powi(2);
        sum += term;
        if term < 1e-18 {
            break;
        }
    }
    sum
}

/// 任意采样率 → 16kHz
pub fn resample_to_16k(x: &[f32], sr: u32) -> Vec<f32> {
    const TARGET: u32 = 16000;
    if sr == TARGET || x.is_empty() {
        return x.to_vec();
    }
    let g = gcd(sr as usize, TARGET as usize) as u32;
    resample_poly(x, (TARGET / g) as usize, (sr / g) as usize)
}

/// 保存 16kHz 单声道 16bit WAV
pub fn save_wav_16k(path: &std::path::Path, samples: &[f32]) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        writer.write_sample(v)?;
    }
    writer.finalize()?;
    Ok(())
}

/// 读取 WAV → f32 单声道（保持原采样率）
pub fn read_wav(path: &std::path::Path) -> anyhow::Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels.max(1) as usize;
    let max_val = 2i64.pow(spec.bits_per_sample.saturating_sub(1).clamp(1, 31) as u32) as f64;
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap_or(0.0)).collect(),
        hound::SampleFormat::Int => reader
            .samples::<i32>()
            .map(|s| s.unwrap_or(0) as f32 / max_val as f32)
            .collect(),
    };
    let mono = mix_to_mono(&samples, channels);
    Ok((mono, spec.sample_rate))
}

pub fn mix_to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    let n = samples.len() / channels;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let frame = &samples[i * channels..(i + 1) * channels];
        out.push(frame.iter().sum::<f32>() / channels as f32);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resample_identity() {
        let x: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let y = resample_poly(&x, 1, 1);
        assert_eq!(x.len(), y.len());
    }

    #[test]
    fn test_resample_48k_to_16k() {
        // 48k 正弦 → 16k，长度应为 1/3
        let x: Vec<f32> = (0..48000).map(|i| (i as f32 * 0.01).sin()).collect();
        let y = resample_poly(&x, 1, 3);
        assert!((y.len() as i64 - 16000).abs() < 3, "len={}", y.len());
    }
}
