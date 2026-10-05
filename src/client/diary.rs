// 归档：录音 WAV 保存 + 识别文本日记

use crate::audio;
use std::path::PathBuf;

#[derive(Clone)]
pub struct Archiver {
    root: PathBuf,
    save_audio: bool,
    save_diary: bool,
    audio_name_len: usize,
}

impl Archiver {
    pub fn new(
        base_dir: &std::path::Path,
        archive_dir: &str,
        save_audio: bool,
        save_diary: bool,
        audio_name_len: usize,
    ) -> Self {
        Self {
            root: base_dir.join(archive_dir),
            save_audio,
            save_diary,
            audio_name_len,
        }
    }

    /// 保存一次识别：音频 → 年/月/assets/，文本追加到 年/月/diary-MM-DD.txt
    pub fn save(&self, samples: &[f32], text: &str) {
        if !self.save_audio && !self.save_diary {
            return;
        }
        let now = chrono::Local::now();
        let dir = self
            .root
            .join(now.format("%Y").to_string())
            .join(now.format("%m").to_string());
        let ts = now
            .format("%Y%m%d-%H%M%S%.3f")
            .to_string()
            .replace('.', "-");

        if self.save_audio && !samples.is_empty() {
            let prefix: String = text
                .chars()
                .filter(|c| !c.is_whitespace() && !r#"\/:*?"<>|"#.contains(*c))
                .take(self.audio_name_len)
                .collect();
            let name = if prefix.is_empty() {
                format!("{ts}.wav")
            } else {
                format!("{ts}-{prefix}.wav")
            };
            let path = dir.join("assets").join(name);
            if let Err(e) = audio::save_wav_16k(&path, samples) {
                eprintln!("[archive] 保存录音失败：{e}");
            }
        }

        if self.save_diary {
            let diary = dir.join(format!("diary-{}.txt", now.format("%Y-%m-%d")));
            let line = format!("{} {}\n", now.format("%H:%M:%S"), text);
            use std::io::Write;
            if let Err(e) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&diary)
                .and_then(|mut f| f.write_all(line.as_bytes()))
            {
                eprintln!("[archive] 写日记失败：{e}");
            }
        }
    }
}
