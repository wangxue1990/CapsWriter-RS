// GGUF 二进制解析 + token embedding 表按行反量化
// 移植自原版 CapsWriter 的 llama.py::get_token_embeddings_gguf 与 gguf/quants.py
// （仅按需读取 prompt 用到的少量行，无需加载整个模型）

use anyhow::{bail, Context, Result};
use half::f16;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// GGML 量化类型（GGMLQuantizationType 的子集）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
}

impl GgmlType {
    fn from_id(id: u32) -> Result<Self> {
        Ok(match id {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            other => bail!("不支持的 GGML 量化类型 id={other}"),
        })
    }

    /// (block_size, block_bytes)
    fn block_layout(self) -> (usize, usize) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 => (1, 2),
            Self::Q4_0 => (32, 18),
            Self::Q4_1 => (32, 20),
            Self::Q5_0 => (32, 22),
            Self::Q5_1 => (32, 24),
            Self::Q8_0 => (32, 34),
            Self::Q8_1 => (32, 36),
            Self::Q2K => (256, 84),
            Self::Q3K => (256, 110),
            Self::Q4K => (256, 144),
            Self::Q5K => (256, 176),
            Self::Q6K => (256, 210),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Q4_0 => "Q4_0",
            Self::Q4_1 => "Q4_1",
            Self::Q5_0 => "Q5_0",
            Self::Q5_1 => "Q5_1",
            Self::Q8_0 => "Q8_0",
            Self::Q8_1 => "Q8_1",
            Self::Q2K => "Q2_K",
            Self::Q3K => "Q3_K",
            Self::Q4K => "Q4_K",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
        }
    }
}

/// GGUF 文件中的张量条目
struct TensorEntry {
    offset: u64, // 相对数据区起点
    _shape: Vec<u64>,
    ty: GgmlType,
}

/// GGUF 文件只读访问器：定位张量 + 按行反量化
pub struct GgufFile {
    file: std::fs::File,
    data_start: u64,
    tensors: std::collections::HashMap<String, TensorEntry>,
    #[allow(dead_code)]
    alignment: u64,
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<Self> {
        let mut file =
            std::fs::File::open(path).with_context(|| format!("打开 GGUF {}", path.display()))?;

        // 头部：magic(4) + version(4) + tensor_count(u64) + kv_count(u64)
        let mut hdr = [0u8; 8];
        file.read_exact(&mut hdr)?;
        if &hdr[0..4] != b"GGUF" {
            bail!("不是有效的 GGUF 文件：{}", path.display());
        }
        let version = u32::from_le_bytes(hdr[4..8].try_into().unwrap());
        if version < 2 {
            bail!("GGUF 版本过旧：v{version}");
        }
        let mut buf8 = [0u8; 8];
        file.read_exact(&mut buf8)?;
        let tensor_count = u64::from_le_bytes(buf8);
        file.read_exact(&mut buf8)?;
        let kv_count = u64::from_le_bytes(buf8);

        // 跳过 / 扫描 KV 字段
        let mut alignment: u64 = 32;
        for _ in 0..kv_count {
            let key = read_string(&mut file)?;
            let ty = read_u32(&mut file)?;
            if key == "general.alignment" && ty == 4 {
                alignment = read_u32(&mut file)? as u64;
            } else {
                skip_value(&mut file, ty)?;
            }
        }

        // 读张量表
        let mut tensors = std::collections::HashMap::new();
        for _ in 0..tensor_count {
            let name = read_string(&mut file)?;
            let n_dims = read_u32(&mut file)? as usize;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                file.read_exact(&mut buf8)?;
                shape.push(u64::from_le_bytes(buf8));
            }
            let ty = GgmlType::from_id(read_u32(&mut file)?)?;
            file.read_exact(&mut buf8)?;
            let offset = u64::from_le_bytes(buf8);
            tensors.insert(
                name,
                TensorEntry {
                    offset,
                    _shape: shape,
                    ty,
                },
            );
        }

        // 数据区起点：当前偏移按 alignment 对齐
        let mut data_start = file.stream_position()?;
        let pad = data_start % alignment;
        if pad != 0 {
            data_start += alignment - pad;
        }

        Ok(Self {
            file,
            data_start,
            tensors,
            alignment,
        })
    }

    /// 定位 token embedding 表（shape 为 [n_embd, vocab_size]，倒序）
    /// 返回的表自带文件句柄克隆，与 GgufFile 生命周期解耦
    pub fn embedding_table(&self, tensor: &str) -> Result<EmbeddingTable> {
        let entry = self
            .tensors
            .get(tensor)
            .with_context(|| format!("GGUF 中找不到张量 {tensor}"))?;
        // shape = [n_embd, vocab_size]
        let n_embd = entry._shape.first().copied().unwrap_or(0) as usize;
        let vocab = entry._shape.get(1).copied().unwrap_or(0) as usize;
        if n_embd == 0 || vocab == 0 {
            bail!("张量 {tensor} 维度异常：{:?}", entry._shape);
        }
        let (bs, bb) = entry.ty.block_layout();
        if n_embd % bs != 0 {
            bail!("n_embd={n_embd} 不是块大小 {bs} 的倍数");
        }
        Ok(EmbeddingTable {
            file: self.file.try_clone().context("克隆 GGUF 文件句柄失败")?,
            abs_offset: self.data_start + entry.offset,
            row_bytes: (n_embd / bs) * bb,
            ty: entry.ty,
            n_embd,
            vocab,
        })
    }
}

/// token embedding 表视图（按行懒反量化，自带文件句柄）
pub struct EmbeddingTable {
    file: std::fs::File,
    abs_offset: u64,
    row_bytes: usize,
    ty: GgmlType,
    n_embd: usize,
    vocab: usize,
}

impl EmbeddingTable {
    pub fn n_embd(&self) -> usize {
        self.n_embd
    }
    pub fn vocab_size(&self) -> usize {
        self.vocab
    }
    pub fn quant_type(&self) -> GgmlType {
        self.ty
    }

    /// 反量化一行到 out（长度必须为 n_embd）
    pub fn row_into(&self, token_id: usize, out: &mut [f32]) -> Result<()> {
        if token_id >= self.vocab {
            bail!("token id {token_id} 越界（vocab={}）", self.vocab);
        }
        if out.len() != self.n_embd {
            bail!("out 长度 {} != n_embd {}", out.len(), self.n_embd);
        }
        let mut raw = vec![0u8; self.row_bytes];
        let off = self.abs_offset + (token_id as u64) * (self.row_bytes as u64);
        let mut f = self.file.try_clone().context("克隆 GGUF 文件句柄失败")?;
        f.seek(SeekFrom::Start(off))
            .with_context(|| format!("seek 到 token {token_id}"))?;
        f.read_exact(&mut raw)?;
        dequant_row(self.ty, &raw, out);
        Ok(())
    }

    /// 批量取多行（返回 ids.len() * n_embd 扁平数组，行优先）
    pub fn rows(&self, ids: &[i32]) -> Result<Vec<f32>> {
        let mut out = vec![0f32; ids.len() * self.n_embd];
        for (i, &id) in ids.iter().enumerate() {
            if id < 0 {
                bail!("token id 为负：{id}");
            }
            self.row_into(
                id as usize,
                &mut out[i * self.n_embd..(i + 1) * self.n_embd],
            )?;
        }
        Ok(out)
    }
}

fn read_u32(f: &mut std::fs::File) -> Result<u32> {
    let mut b = [0u8; 4];
    f.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_string(f: &mut std::fs::File) -> Result<String> {
    let mut b = [0u8; 8];
    f.read_exact(&mut b)?;
    let len = u64::from_le_bytes(b) as usize;
    let mut s = vec![0u8; len];
    f.read_exact(&mut s)?;
    String::from_utf8(s).context("GGUF 字符串不是 UTF-8")
}

fn skip_value(f: &mut std::fs::File, ty: u32) -> Result<()> {
    const FIXED: [i64; 13] = [1, 1, 2, 2, 4, 4, 4, 1, -1, -2, 8, 8, 8];
    let sz = FIXED.get(ty as usize).copied().unwrap_or(-1);
    match sz {
        n if n > 0 => {
            f.seek(SeekFrom::Current(n))?;
        }
        -1 => {
            // STRING
            let mut b = [0u8; 8];
            f.read_exact(&mut b)?;
            let len = u64::from_le_bytes(b);
            f.seek(SeekFrom::Current(len as i64))?;
        }
        -2 => {
            // ARRAY
            let ity = read_u32(f)? as usize;
            let mut b = [0u8; 8];
            f.read_exact(&mut b)?;
            let alen = u64::from_le_bytes(b);
            let item = FIXED.get(ity).copied().unwrap_or(-1);
            if item > 0 {
                f.seek(SeekFrom::Current(item * alen as i64))?;
            } else if ity == 8 {
                // string 数组
                for _ in 0..alen {
                    let mut b = [0u8; 8];
                    f.read_exact(&mut b)?;
                    let slen = u64::from_le_bytes(b);
                    f.seek(SeekFrom::Current(slen as i64))?;
                }
            } else {
                bail!("GGUF 嵌套数组不支持（type {ity}）");
            }
        }
        _ => bail!("未知 GGUF 值类型 {ty}"),
    }
    Ok(())
}

fn f16_at(b: &[u8], i: usize) -> f32 {
    f32::from(f16::from_le_bytes([b[i], b[i + 1]]))
}

/// 按行反量化（与原版 gguf/quants.py 位级对齐）
pub fn dequant_row(ty: GgmlType, raw: &[u8], out: &mut [f32]) {
    match ty {
        GgmlType::F32 => {
            for (o, chunk) in out.iter_mut().zip(raw.chunks_exact(4)) {
                *o = f32::from_le_bytes(chunk.try_into().unwrap());
            }
        }
        GgmlType::F16 => {
            for (o, chunk) in out.iter_mut().zip(raw.chunks_exact(2)) {
                *o = f32::from(f16::from_le_bytes([chunk[0], chunk[1]]));
            }
        }
        GgmlType::Q8_0 => {
            // 块 34B：d(f16) + 32×i8；每块 32 值
            for (blk, vals) in raw.chunks_exact(34).zip(out.chunks_exact_mut(32)) {
                let d = f16_at(blk, 0);
                for (o, &q) in vals.iter_mut().zip(&blk[2..34]) {
                    *o = d * q as i8 as f32;
                }
            }
        }
        GgmlType::Q4K => {
            // 块 144B：d + dmin + scales(12) + qs(128)；8 个子块 × 32 值
            for (blk, vals) in raw.chunks_exact(144).zip(out.chunks_exact_mut(256)) {
                let d = f16_at(blk, 0);
                let dmin = f16_at(blk, 2);
                let (sc, mn) = unpack_k_scales(&blk[4..16]);
                let qs = &blk[16..144];
                for j in 0..8 {
                    let dl = d * sc[j] as f32;
                    let ml = dmin * mn[j] as f32;
                    // quants.py: qs.reshape(n,4,1,32)>>[0,4] → 子块 j 取字节组 j/2、位移 (j%2)*4
                    let g = (j / 2) * 32;
                    let sh = (j % 2) * 4;
                    for l in 0..32 {
                        let q = ((qs[g + l] >> sh) & 0x0F) as f32;
                        vals[j * 32 + l] = dl * q - ml;
                    }
                }
            }
        }
        GgmlType::Q5K => {
            // 块 176B：d + dmin + scales(12) + qh(32) + qs(128)
            for (blk, vals) in raw.chunks_exact(176).zip(out.chunks_exact_mut(256)) {
                let d = f16_at(blk, 0);
                let dmin = f16_at(blk, 2);
                let (sc, mn) = unpack_k_scales(&blk[4..16]);
                let qh = &blk[16..48];
                let qs = &blk[48..176];
                for j in 0..8 {
                    let dl = d * sc[j] as f32;
                    let ml = dmin * mn[j] as f32;
                    let g = (j / 2) * 32;
                    let sh = (j % 2) * 4;
                    for l in 0..32 {
                        let ql = ((qs[g + l] >> sh) & 0x0F) as f32;
                        let hi = ((qh[l] >> j) & 0x01) as f32;
                        vals[j * 32 + l] = dl * (ql + hi * 16.0) - ml;
                    }
                }
            }
        }
        GgmlType::Q6K => {
            // 块 210B：ql(128) + qh(64) + scales(16×i8) + d(f16)
            for (blk, vals) in raw.chunks_exact(210).zip(out.chunks_exact_mut(256)) {
                let ql = &blk[0..128];
                let qh = &blk[128..192];
                let scales = &blk[192..208];
                let d = f16_at(blk, 208);
                // quants.py: ql.reshape(n,2,1,64)>>[0,4] → (n,8,32)；
                // qh.reshape(n,2,1,32)>>[0,2,4,6] → (n,8,32)
                // 最终按 (16 组 × 16 值) 组织，scales 有 16 个
                // (n,8,32) reshape (n,16,16)：组 k 行元素 = flat[k*16 + m]
                let mut q = [0f32; 256];
                for j in 0..8 {
                    let g = (j / 4) * 64; // 2 组 64 字节
                    let sh = ((j % 4) / 2) * 4;
                    let hg = (j / 4) * 32;
                    let hsh = (j % 4) * 2;
                    for l in 0..32 {
                        let low = ((ql[g + l] >> sh) & 0x0F) as i32;
                        let hi = ((qh[hg + l] >> hsh) & 0x03) as i32;
                        q[j * 32 + l] = ((low | (hi << 4)) as i8 as i32 - 32) as f32;
                    }
                }
                // q 的 (8,32) 布局 reshape 成 (16,16)：flat = k*16+m
                for k in 0..16 {
                    let dl = d * scales[k] as i8 as f32;
                    for m in 0..16 {
                        let flat = k * 16 + m;
                        vals[flat] = dl * q[flat];
                    }
                }
            }
        }
        other => unreachable!("dequant_row 未实现 {other:?}"),
    }
}

/// Q4_K/Q5_K 的 6-bit scale/min 解包（quants.py::Q4_K.get_scale_min）
fn unpack_k_scales(s: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut sc = [0u8; 8];
    let mut mn = [0u8; 8];
    // scales.reshape(n,3,4)：d=bytes[0..4], m=bytes[4..8], m_d=bytes[8..12]
    let d = &s[0..4];
    let m = &s[4..8];
    let md = &s[8..12];
    for j in 0..4 {
        sc[j] = d[j] & 0x3F;
        mn[j] = m[j] & 0x3F;
        sc[4 + j] = (md[j] & 0x0F) | ((d[j] >> 2) & 0x30);
        mn[4 + j] = (md[j] >> 4) | ((m[j] >> 2) & 0x30);
    }
    (sc, mn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q8_0_roundtrip() {
        // 构造一个 2 块 Q8_0
        let mut raw = vec![0u8; 68];
        raw[0] = 0x00;
        raw[1] = 0x40; // f16 2.0
        for i in 0..32 {
            raw[2 + i] = (i as i8) as u8;
        }
        let mut out = [0f32; 64];
        dequant_row(GgmlType::Q8_0, &raw, &mut out);
        assert!((out[5] - 2.0 * 5.0).abs() < 1e-6);
        assert!((out[20] - 2.0 * 20.0).abs() < 1e-6);
    }

    #[test]
    fn test_k_scales_unpack() {
        // d=[0x3F,0,0,0], m=[0x15,0,0,0], md=[0x21,0,0,0]
        let s = [0x3F, 0, 0, 0, 0x15, 0, 0, 0, 0x21, 0, 0, 0];
        let (sc, mn) = unpack_k_scales(&s);
        assert_eq!(sc[0], 63); // d[0] & 0x3F
        assert_eq!(mn[0], 0x15); // m[0] & 0x3F
                                 // sc[4] = (md[0]&0x0F) | ((d[0]>>2)&0x30) = 1 | (0x0F & 0x30) = 1
        assert_eq!(sc[4], 1);
        // mn[4] = (md[0]>>4) | ((m[0]>>2)&0x30) = 2 | (5 & 0x30) = 2
        assert_eq!(mn[4], 2);
    }

    #[test]
    fn test_parse_real_model() {
        let path = std::path::Path::new("models/Qwen3-ASR/qwen3_asr_llm.gguf");
        if !path.exists() {
            eprintln!("跳过：找不到 {}", path.display());
            return;
        }
        let gguf = GgufFile::open(path).unwrap();
        let table = gguf.embedding_table("token_embd.weight").unwrap();
        eprintln!(
            "embedding 表：{} tokens × {} dims（{}）",
            table.vocab_size(),
            table.n_embd(),
            table.quant_type().name()
        );
        let row = table.rows(&[0, 1, 100]).unwrap();
        assert_eq!(row.len(), 3 * table.n_embd());
        // 数值应有限且非全零
        assert!(row.iter().all(|v| v.is_finite()));
        assert!(row.iter().any(|v| v.abs() > 1e-6));
    }
}
