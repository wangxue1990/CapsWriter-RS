// Whisper 风格 128 维 Mel 提取器（Qwen3-ASR 编码器前端用）
// 精确移植原版 qwen_asr_gguf/inference/encoder.py::FastWhisperMel
// 差异点（相对 SenseVoice 的 mel.rs）：无预加重、Hann 窗、reflect padding、
// slaney 滤波器（分段线性 mel 刻度 + 面积归一化）、log10、max-8 截断、(x+4)/4 归一化

use rustfft::{num_complex::Complex, FftPlanner};

pub const N_MELS: usize = 128;
const N_FFT: usize = 400;
const HOP: usize = 160;

/// slaney mel 刻度：1000Hz 以下线性（200/3 Hz/mel），以上对数
fn hz_to_mel_slaney(f: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_HZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = MIN_LOG_HZ / F_SP; // 15
    const LOGSTEP: f64 = 0.06875205455918106; // ln(6.4)/27
    let mut mels = f / F_SP;
    if f >= MIN_LOG_HZ {
        mels = MIN_LOG_MEL + (f / MIN_LOG_HZ).ln() / LOGSTEP;
    }
    mels
}

fn mel_to_hz_slaney(mels: f64) -> f64 {
    const F_SP: f64 = 200.0 / 3.0;
    const MIN_LOG_MEL: f64 = 15.0;
    const LOGSTEP: f64 = 0.06875205455918106; // ln(6.4)/27
    let mut freqs = F_SP * mels;
    if mels >= MIN_LOG_MEL {
        freqs = 1000.0 * (LOGSTEP * (mels - MIN_LOG_MEL)).exp();
    }
    freqs
}

pub struct WhisperMel {
    /// (201 × 128) 行主序：filters[freq_bin * 128 + mel]
    filters: Vec<f32>,
    window: Vec<f32>,
}

impl Default for WhisperMel {
    fn default() -> Self {
        Self::new()
    }
}

impl WhisperMel {
    pub fn new() -> Self {
        let sr = 16000.0f64;
        let n_freqs = N_FFT / 2 + 1; // 201

        // torchaudio slaney 滤波器组生成
        let all_freqs: Vec<f64> = (0..n_freqs)
            .map(|i| i as f64 * (sr / 2.0) / (n_freqs - 1) as f64) // linspace(0, 8000, 201)，步长 40Hz
            .collect();
        let m_pts: Vec<f64> = (0..N_MELS + 2)
            .map(|i| {
                hz_to_mel_slaney(0.0)
                    + i as f64 / (N_MELS + 1) as f64
                        * (hz_to_mel_slaney(8000.0) - hz_to_mel_slaney(0.0))
            })
            .collect();
        let f_pts: Vec<f64> = m_pts.iter().map(|m| mel_to_hz_slaney(*m)).collect();
        let f_diff: Vec<f64> = f_pts.windows(2).map(|w| w[1] - w[0]).collect();

        let mut filters = vec![0f32; n_freqs * N_MELS];
        for (i, &af) in all_freqs.iter().enumerate() {
            for j in 0..N_MELS {
                // up = (af - f_pts[j]) / f_diff[j]；down = (f_pts[j+2] - af) / f_diff[j+1]
                let up = (af - f_pts[j]) / f_diff[j];
                let down = (f_pts[j + 2] - af) / f_diff[j + 1];
                let v = up.min(down).max(0.0);
                // slaney 面积归一化
                let enorm = 2.0 / (f_pts[j + 2] - f_pts[j]);
                filters[i * N_MELS + j] = (v * enorm) as f32;
            }
        }

        // Hann 窗
        let window: Vec<f32> = (0..N_FFT)
            .map(|i| {
                (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64).cos()) as f32
            })
            .collect();

        Self { filters, window }
    }

    /// 输入 16kHz f32 采样，输出 (128 × T) 行主序 mel 谱
    /// 与 librosa center=True 一致：两侧 reflect padding n_fft/2
    pub fn extract(&self, audio: &[f32]) -> Vec<f32> {
        if audio.is_empty() {
            return Vec::new();
        }
        let pad = N_FFT / 2; // 200

        // 1. reflect padding（与 np.pad(mode='reflect') 一致）
        let n = audio.len();
        let mut y = vec![0f32; n + 2 * pad];
        y[pad..pad + n].copy_from_slice(audio);
        for i in 0..pad {
            // 左侧镜像：y[pad-1-i] = audio[i+1]（需 n > pad，短音频钳到边界）
            y[pad - 1 - i] = audio[(i + 1).min(n - 1)];
            // 右侧镜像：y[pad+n+i] = audio[n-2-i]
            let idx = n as isize - 2 - i as isize;
            y[pad + n + i] = if idx >= 0 {
                audio[idx as usize]
            } else {
                audio[0]
            };
        }

        // 2. 分帧 + 加窗 + rfft
        let num_frames = 1 + (y.len().saturating_sub(N_FFT)) / HOP;
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(N_FFT);

        // 功率谱矩阵 (num_frames × 201)，避免逐帧分配
        let n_freqs = N_FFT / 2 + 1;
        let mut power = vec![0f64; num_frames * n_freqs];
        let mut buf = vec![Complex::new(0.0, 0.0); N_FFT];
        for f in 0..num_frames {
            let start = f * HOP;
            for (k, w) in self.window.iter().enumerate() {
                buf[k] = Complex::new((y[start + k] * w) as f64, 0.0);
            }
            fft.process(&mut buf);
            let row = &mut power[f * n_freqs..(f + 1) * n_freqs];
            for k in 0..n_freqs {
                row[k] = buf[k].norm_sqr();
            }
        }

        // 3. mel 映射 (128 × num_frames)
        let mut mel = vec![0f64; N_MELS * num_frames];
        for m in 0..N_MELS {
            for f in 0..num_frames {
                let mut acc = 0f64;
                let row = &power[f * n_freqs..(f + 1) * n_freqs];
                for (i, &p) in row.iter().enumerate() {
                    acc += self.filters[i * N_MELS + m] as f64 * p;
                }
                mel[m * num_frames + f] = acc;
            }
        }

        // 4. log10 + 全局 max-8 截断 + (x+4)/4
        let global_max = mel.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mut out = vec![0f32; N_MELS * num_frames];
        for (o, &v) in out.iter_mut().zip(mel.iter()) {
            let l = v.max(1e-10).log10();
            let l = l.max(global_max.max(1e-10).log10() - 8.0);
            *o = ((l + 4.0) / 4.0) as f32;
        }

        // 5. 帧对齐：丢弃多余帧（n_frames_out = len(audio) // hop）
        let n_out = n / HOP;
        if num_frames > n_out {
            // 保留每行前 n_out 帧
            let mut trimmed = vec![0f32; N_MELS * n_out];
            for m in 0..N_MELS {
                trimmed[m * n_out..(m + 1) * n_out]
                    .copy_from_slice(&out[m * num_frames..m * num_frames + n_out]);
            }
            return trimmed;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape() {
        let m = WhisperMel::new();
        // 1 秒音频 → 100 帧
        let audio = vec![0.01f32; 16000];
        let mel = m.extract(&audio);
        assert_eq!(mel.len(), N_MELS * 100);
        // 静音（近似）→ log10(1e-10) = -10 → max-8 截断后可能全为下限
        assert!(mel.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_energy_concentrates() {
        let m = WhisperMel::new();
        // 300Hz 正弦波 → slaney mel ≈ 4.5/45.2 ≈ 10% → 低频带（前 64 带内）
        let audio: Vec<f32> = (0..16000)
            .map(|i| (2.0 * std::f32::consts::PI * 300.0 * i as f32 / 16000.0).sin() * 0.5)
            .collect();
        let mel = m.extract(&audio);
        let low: f32 = mel[..N_MELS * 50].iter().sum();
        let high: f32 = mel[N_MELS * 50..].iter().sum();
        assert!(low > high, "低频能量应高于高频：{low} vs {high}");
    }
}
