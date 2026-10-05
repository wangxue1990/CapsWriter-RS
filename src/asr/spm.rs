// SentencePiece BPE 模型解析（只需 id → piece 映射，手工解析 protobuf wire format）
//
// model.proto 顶层字段：1 = pieces（repeated SentencePiece），其余跳过
// SentencePiece：1 = piece (string)，2 = score (fixed32)，3 = type (varint)

pub struct SpmModel {
    pieces: Vec<String>,
}

impl SpmModel {
    pub fn parse(data: &[u8]) -> anyhow::Result<Self> {
        let mut pieces = Vec::new();
        let mut pos = 0usize;
        while pos < data.len() {
            let (tag, np) = read_varint(data, pos)?;
            pos = np;
            let field = tag >> 3;
            let wire = tag & 7;
            match (field, wire) {
                (1, 2) => {
                    let (len, np) = read_varint(data, pos)?;
                    pos = np;
                    let end = pos + len as usize;
                    if end > data.len() {
                        anyhow::bail!("spm 模型损坏");
                    }
                    if let Some(piece) = parse_piece(&data[pos..end])? {
                        pieces.push(piece);
                    }
                    pos = end;
                }
                (_, 0) => {
                    let (_, np) = read_varint(data, pos)?;
                    pos = np;
                }
                (_, 1) => pos += 8,
                (_, 2) => {
                    let (len, np) = read_varint(data, pos)?;
                    pos = np + len as usize;
                }
                (_, 5) => pos += 4,
                _ => anyhow::bail!("spm 模型字段格式异常"),
            }
        }
        if pieces.is_empty() {
            anyhow::bail!("spm 模型中未解析到 pieces");
        }
        Ok(Self { pieces })
    }

    /// id → piece（保留 "▁" 原样，由调用方处理）
    pub fn piece(&self, id: i64) -> Option<&str> {
        if id < 0 {
            return None;
        }
        self.pieces.get(id as usize).map(|s| s.as_str())
    }

    pub fn len(&self) -> usize {
        self.pieces.len()
    }
}

fn parse_piece(msg: &[u8]) -> anyhow::Result<Option<String>> {
    let mut piece: Option<String> = None;
    let mut pos = 0usize;
    while pos < msg.len() {
        let (tag, np) = read_varint(msg, pos)?;
        pos = np;
        let field = tag >> 3;
        let wire = tag & 7;
        match (field, wire) {
            (1, 2) => {
                let (len, np) = read_varint(msg, pos)?;
                pos = np;
                let end = pos + len as usize;
                if end > msg.len() {
                    anyhow::bail!("piece 字段越界");
                }
                piece = Some(String::from_utf8_lossy(&msg[pos..end]).into_owned());
                pos = end;
            }
            (2, 5) => pos += 4,
            (_, 0) => {
                let (_, np) = read_varint(msg, pos)?;
                pos = np;
            }
            (_, 1) => pos += 8,
            (_, 2) => {
                let (len, np) = read_varint(msg, pos)?;
                pos = np + len as usize;
            }
            _ => anyhow::bail!("piece 字段格式异常"),
        }
    }
    Ok(piece)
}

fn read_varint(data: &[u8], mut pos: usize) -> anyhow::Result<(u64, usize)> {
    let mut value: u64 = 0;
    let mut shift = 0;
    loop {
        if pos >= data.len() {
            anyhow::bail!("varint 越界");
        }
        let b = data[pos];
        value |= ((b & 0x7f) as u64) << shift;
        pos += 1;
        if b & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift > 63 {
            anyhow::bail!("varint 过长");
        }
    }
    Ok((value, pos))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_if_model_exists() {
        let p = std::path::Path::new(
            "models/SenseVoice-Small/Sensevoice-Small-ONNX/tokenizer.bpe.model",
        );
        if !p.exists() {
            return; // 模型未就位时跳过
        }
        let data = std::fs::read(p).unwrap();
        let m = SpmModel::parse(&data).unwrap();
        assert!(m.len() > 20000, "pieces = {}", m.len());
        // id=0 通常是 blank/<unk>
        println!(
            "pieces={} id0={:?} id1={:?}",
            m.len(),
            m.piece(0),
            m.piece(1)
        );
    }
}
