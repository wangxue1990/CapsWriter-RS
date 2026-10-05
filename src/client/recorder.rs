// 麦克风录音：cpal（WASAPI），按住采集、松开交付（含重采样到 16k）

use anyhow::{bail, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

pub struct Recorder {
    armed: Arc<AtomicBool>,
    buffer: Arc<Mutex<Vec<f32>>>,
    /// 实际采集采样率（回调里转换困难，停止时统一重采样）
    sample_rate: u32,
    channels: usize,
    _stream: cpal::Stream,
}

impl Recorder {
    pub fn new() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| anyhow::anyhow!("未找到默认麦克风设备"))?;

        let armed = Arc::new(AtomicBool::new(false));
        let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));

        // 优先尝试 16kHz 单声道 f32；失败则用设备默认格式，停止时重采样
        let try_config = |cfg: &cpal::StreamConfig,
                          fmt: cpal::SampleFormat|
         -> Result<(cpal::Stream, u32, usize)> {
            let buf = buffer.clone();
            let armed2 = armed.clone();
            let channels = cfg.channels as usize;
            let stream = match fmt {
                cpal::SampleFormat::F32 => device.build_input_stream(
                    cfg,
                    move |data: &[f32], _| on_samples(data, &buf, &armed2),
                    |err| eprintln!("[mic] 音频流错误：{err}"),
                    None,
                ),
                cpal::SampleFormat::I16 => device.build_input_stream(
                    cfg,
                    move |data: &[i16], _| {
                        let f: Vec<f32> = data.iter().map(|&v| v as f32 / 32768.0).collect();
                        on_samples(&f, &buf, &armed2);
                    },
                    |err| eprintln!("[mic] 音频流错误：{err}"),
                    None,
                ),
                cpal::SampleFormat::U16 => device.build_input_stream(
                    cfg,
                    move |data: &[u16], _| {
                        let f: Vec<f32> = data
                            .iter()
                            .map(|&v| (v as f32 - 32768.0) / 32768.0)
                            .collect();
                        on_samples(&f, &buf, &armed2);
                    },
                    |err| eprintln!("[mic] 音频流错误：{err}"),
                    None,
                ),
                _ => bail!("不支持的采样格式 {fmt:?}"),
            }?;
            Ok((stream, cfg.sample_rate.0, channels))
        };

        let desired = cpal::StreamConfig {
            channels: 1,
            sample_rate: cpal::SampleRate(16000),
            buffer_size: cpal::BufferSize::Default,
        };
        let (stream, rate, channels) = match try_config(&desired, cpal::SampleFormat::F32) {
            Ok(v) => v,
            Err(_) => {
                let def = device.default_input_config()?;
                let cfg: cpal::StreamConfig = def.clone().into();
                try_config(&cfg, def.sample_format())?
            }
        };
        stream.play()?;
        println!(
            "[mic] 麦克风已就绪（{}Hz，{} 声道{}）",
            rate,
            channels,
            if rate == 16000 {
                ""
            } else {
                "，将在停止时重采样到 16k"
            }
        );

        Ok(Self {
            armed,
            buffer,
            sample_rate: rate,
            channels,
            _stream: stream,
        })
    }

    /// 开始采集（清空缓冲）
    pub fn start(&self) {
        self.buffer.lock().unwrap().clear();
        self.armed.store(true, Ordering::SeqCst);
    }

    /// 停止采集并取出 16kHz 单声道样本
    pub fn stop(&self) -> Vec<f32> {
        self.armed.store(false, Ordering::SeqCst);
        let raw = {
            let mut buf = self.buffer.lock().unwrap();
            std::mem::take(&mut *buf)
        };
        let mono = if self.channels > 1 {
            crate::audio::mix_to_mono(&raw, self.channels)
        } else {
            raw
        };
        if self.sample_rate == 16000 {
            mono
        } else {
            crate::audio::resample_to_16k(&mono, self.sample_rate)
        }
    }
}

fn on_samples(data: &[f32], buffer: &Mutex<Vec<f32>>, armed: &AtomicBool) {
    if armed.load(Ordering::Relaxed) {
        buffer.lock().unwrap().extend_from_slice(data);
    }
}
