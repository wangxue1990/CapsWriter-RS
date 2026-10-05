// Mel 特征提取：精确移植原版 NumPyMelExtractor
// 预加重(0.97) → 汉明窗 STFT(400/160) → 功率谱 → 80 维 mel(20~8000Hz) → log → LFR(7帧拼接/6帧步进)

use rustfft::{num_complex::Complex, FftPlanner};

pub const N_MELS: usize = 80;
pub const LFR_DIM: usize = 7; // 拼接帧数
pub const LFR_STEP: usize = 6; // 步进帧数
pub const LFR_INPUT: usize = N_MELS * LFR_DIM; // 560

pub struct MelExtractor {
    /// (201 × 80) 行主序：filters[freq_bin * 80 + mel]
    filters: Vec<f32>,
    window: Vec<f32>,
    n_fft: usize,
    hop: usize,
}

impl MelExtractor {
    pub fn new() -> Self {
        let sr = 16000.0f64;
        let n_fft = 400usize;
        let n_mels = 80usize;
        let f_min = 20.0f64;
        let f_max = 8000.0f64;

        let hz_to_mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
        let mel_to_hz = |m: f64| 700.0 * (10f64.powf(m / 2595.0) - 1.0);

        let all_freqs: Vec<f64> = (0..=n_fft / 2)
            .map(|i| i as f64 * sr / n_fft as f64)
            .collect(); // linspace(0, sr/2, 201) 与逐点公式等价
        let m_pts: Vec<f64> = (0..n_mels + 2)
            .map(|i| {
                let t = i as f64 / (n_mels + 1) as f64;
                hz_to_mel(f_min) + t * (hz_to_mel(f_max) - hz_to_mel(f_min))
            })
            .collect();
        let f_pts: Vec<f64> = m_pts.iter().map(|m| mel_to_hz(*m)).collect();
        let f_diff: Vec<f64> = f_pts.windows(2).map(|w| w[1] - w[0]).collect();

        let mut filters = vec![0f32; (n_fft / 2 + 1) * n_mels];
        for (i, &af) in all_freqs.iter().enumerate() {
            for j in 0..n_mels {
                // slopes[k] = f_pts[k] - af
                let up = (af - f_pts[j]) / f_diff[j]; // 对应 -slopes[:, j] / f_diff[j]
                let down = (f_pts[j + 2] - af) / f_diff[j + 1]; // 对应 slopes[:, j+2] / f_diff[j+1]
                filters[i * n_mels + j] = up.min(down).max(0.0) as f32;
            }
        }

        let window: Vec<f32> = (0..n_fft)
            .map(|i| {
                (0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / n_fft as f64).cos()) as f32
            })
            .collect();

        Self {
            filters,
            window,
            n_fft,
            hop: 160,
        }
    }

    /// 输入 16kHz f32 采样，输出 (T_lfr × 560) 行主序特征矩阵
    pub fn extract(&self, audio: &[f32]) -> Vec<f32> {
        if audio.is_empty() {
            return Vec::new();
        }
        // 均值归一化
        let mean = audio.iter().sum::<f32>() / audio.len() as f32;

        // 预加重
        const PRE_EMPHASIS: f32 = 0.97;
        let mut pe = Vec::with_capacity(audio.len());
        pe.push(audio[0] - mean);
        for i in 1..audio.len() {
            pe.push((audio[i] - mean) - PRE_EMPHASIS * (audio[i - 1] - mean));
        }

        // 补零 + 分帧
        let half = self.n_fft / 2;
        let padded_len = pe.len() + 2 * half;
        let mut padded = vec![0f32; padded_len];
        padded[half..half + pe.len()].copy_from_slice(&pe);
        let num_frames = 1 + (padded.len().saturating_sub(self.n_fft)) / self.hop;

        // FFT 计划复用
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.n_fft);
        let mut buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); self.n_fft];

        let n_bins = self.n_fft / 2 + 1;
        let mut log_mel = vec![0f32; num_frames * N_MELS];
        let mut power = vec![0f32; n_bins];

        for f in 0..num_frames {
            let start = f * self.hop;
            for (k, b) in buf.iter_mut().enumerate() {
                b.re = padded[start + k] * self.window[k];
                b.im = 0.0;
            }
            fft.process(&mut buf);
            for k in 0..n_bins {
                power[k] = buf[k].norm_sqr();
            }
            // mel 滤波 + log
            for j in 0..N_MELS {
                let mut acc = 0f32;
                for (k, &p) in power.iter().enumerate() {
                    acc += p * self.filters[k * N_MELS + j];
                }
                log_mel[f * N_MELS + j] = (acc + 1e-7).ln();
            }
        }

        // LFR：7 帧拼接 / 6 帧步进（与原版完全一致的填充逻辑）
        let t_mel = num_frames;
        let t_lfr = (t_mel + 5) / 6;
        if t_lfr == 0 {
            return Vec::new();
        }
        let left_pad = 3;
        let right_pad_len = (t_lfr * LFR_STEP + LFR_DIM).saturating_sub(t_mel);
        let total = left_pad + t_mel + right_pad_len;
        let mut padded_mel = vec![0f32; total * N_MELS];
        for r in 0..left_pad {
            padded_mel[r * N_MELS..(r + 1) * N_MELS].copy_from_slice(&log_mel[..N_MELS]);
        }
        padded_mel[left_pad * N_MELS..(left_pad + t_mel) * N_MELS].copy_from_slice(&log_mel);
        for r in 0..right_pad_len {
            let src = (t_mel - 1) * N_MELS;
            let dst = (left_pad + t_mel + r) * N_MELS;
            padded_mel[dst..dst + N_MELS].copy_from_slice(&log_mel[src..src + N_MELS]);
        }

        let mut lfr = vec![0f32; t_lfr * LFR_INPUT];
        for i in 0..LFR_DIM {
            for t in 0..t_lfr {
                let src = (i + t * LFR_STEP) * N_MELS;
                let dst = t * LFR_INPUT + i * N_MELS;
                lfr[dst..dst + N_MELS].copy_from_slice(&padded_mel[src..src + N_MELS]);
            }
        }
        lfr
    }
}

impl Default for MelExtractor {
    fn default() -> Self {
        Self::new()
    }
}
