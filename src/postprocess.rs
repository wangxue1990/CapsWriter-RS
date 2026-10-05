// 文本后处理：热词（拼音模糊匹配）+ 规则替换 + 末尾标点消除

use crate::config::ClientCfg;
use anyhow::Result;
use std::collections::HashSet;
use std::path::Path;

pub struct PostProcessor {
    hotwords: Option<HotwordMatcher>,
    rules: Option<RuleEngine>,
    cfg: ClientCfg,
}

impl PostProcessor {
    pub fn new(cfg: &ClientCfg, base_dir: &Path) -> Self {
        let hotwords = if cfg.hot_enabled {
            match HotwordMatcher::load(&base_dir.join(&cfg.hotwords_path), cfg.hot_thresh) {
                Ok(h) => {
                    println!("[post] 热词加载完成：{} 组", h.len());
                    Some(h)
                }
                Err(e) => {
                    println!("[post] 热词加载失败：{e}");
                    None
                }
            }
        } else {
            None
        };
        let rules = if cfg.rules_enabled {
            match RuleEngine::load(&base_dir.join(&cfg.rules_path)) {
                Ok(r) => {
                    println!("[post] 规则加载完成：{} 条", r.len());
                    Some(r)
                }
                Err(e) => {
                    println!("[post] 规则加载失败：{e}");
                    None
                }
            }
        } else {
            None
        };
        Self {
            hotwords,
            rules,
            cfg: cfg.clone(),
        }
    }

    pub fn process(&self, text: &str) -> String {
        let mut t = text.trim().to_string();
        if t.is_empty() {
            return t;
        }
        if let Some(h) = &self.hotwords {
            t = h.replace(&t);
        }
        if let Some(r) = &self.rules {
            t = r.apply(&t);
        }
        t = strip_trailing_punc(&t, &self.cfg.trash_punc, self.cfg.trash_punc_thresh);
        t
    }
}

// ============ 末尾标点消除（对齐原版 TextOutput.strip_punc）============

/// 语义单元计数：一个汉字/一个英文单词/一段数字各算 1
fn count_semantic_units(text: &str) -> usize {
    let mut count = 0usize;
    let mut prev_is_ascii_word = false;
    for ch in text.chars() {
        if ch.is_ascii_alphanumeric() {
            if !prev_is_ascii_word {
                count += 1;
                prev_is_ascii_word = true;
            }
        } else {
            prev_is_ascii_word = false;
            if is_cjk(ch) {
                count += 1;
            }
        }
    }
    count
}

fn is_cjk(ch: char) -> bool {
    matches!(ch as u32,
        0x4E00..=0x9FFF | 0x3400..=0x4DBF | 0xF900..=0xFAFF | 0x20000..=0x2FA1F)
}

pub fn strip_trailing_punc(text: &str, punc: &str, thresh: usize) -> String {
    if text.is_empty() || punc.is_empty() {
        return text.to_string();
    }
    if thresh > 0 && count_semantic_units(text) > thresh {
        return text.to_string();
    }
    let last = text.chars().last().unwrap();
    if punc.contains(last) {
        text.chars().take(text.chars().count() - 1).collect()
    } else {
        text.to_string()
    }
}

// ============ 规则替换（hot-rule.txt，格式与原版一致）============

pub struct RuleEngine {
    rules: Vec<(fancy_regex::Regex, String)>,
}

impl RuleEngine {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let mut rules = Vec::new();
        for line in text.lines() {
            // 对齐原版 hot_rule.py：不整行 trim（trim 会吃掉空替换的尾随空格），
            // 以 ' = ' 切分且要求恰好两段（正则内含 ' = ' 的整条跳过）
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let parts: Vec<&str> = line.split(" = ").collect();
            if parts.len() != 2 {
                continue;
            }
            let pat = parts[0].trim();
            if pat.is_empty() {
                continue;
            }
            // 替换侧转义：\s → 空格，\n → 换行，\1~\9 → $1~$9（对齐原版 re.sub 模板语义）
            let repl = convert_replacement(parts[1].trim());
            match fancy_regex::Regex::new(pat) {
                Ok(re) => rules.push((re, repl)),
                Err(e) => println!("[post] 规则跳过（正则错误 {e}）：{line}"),
            }
        }
        Ok(Self { rules })
    }

    pub fn len(&self) -> usize {
        self.rules.len()
    }

    pub fn apply(&self, text: &str) -> String {
        let mut out = text.to_string();
        for (re, repl) in &self.rules {
            out = re.replace_all(&out, repl.as_str()).into_owned();
        }
        out
    }
}

fn convert_replacement(repl: &str) -> String {
    let mut out = String::with_capacity(repl.len() + 4);
    let mut chars = repl.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('s') => out.push(' '),
            Some('n') => out.push('\n'),
            Some('t') => out.push('\t'),
            Some('\\') => out.push('\\'),
            Some(d) if d.is_ascii_digit() => {
                out.push('$');
                out.push(d);
            }
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    // fancy-regex 的 $ 需转义为 $$，避免替换串里的裸 $ 出问题
    out
}

// ============ 热词（拼音模糊匹配）============

pub struct HotwordMatcher {
    entries: Vec<Entry>,
    threshold: f64,
}

struct Entry {
    /// 显示词（替换后的结果）
    display: String,
    /// 各别名的音节序列
    alias_syllables: Vec<Vec<String>>,
    /// 黑名单（源文本匹配到这些词时不替换）
    blacklist: HashSet<String>,
}

impl HotwordMatcher {
    pub fn load(path: &Path, threshold: f64) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let mut entries = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (main_part, blacklist) = match line.split_once("~~~") {
                Some((m, b)) => (
                    m.trim(),
                    b.split('|')
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .collect(),
                ),
                None => (line, HashSet::new()),
            };
            let terms: Vec<String> = main_part
                .split('|')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            if terms.is_empty() {
                continue;
            }
            let display = terms[0].clone();
            let alias_syllables = terms
                .iter()
                .map(|t| to_syllables(t).into_iter().map(|(_, py)| py).collect())
                .collect();
            entries.push(Entry {
                display,
                alias_syllables,
                blacklist,
            });
        }
        Ok(Self { entries, threshold })
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 对文本做热词模糊替换
    pub fn replace(&self, text: &str) -> String {
        if self.entries.is_empty() || text.is_empty() {
            return text.to_string();
        }
        let syllables = to_syllables(text);
        if syllables.is_empty() {
            return text.to_string();
        }
        let mut edits: Vec<(usize, usize, String)> = Vec::new(); // (起始音节, 结束音节, 替换词)
        let n = syllables.len();

        for entry in &self.entries {
            for alias in &entry.alias_syllables {
                let alias_py: String = alias.concat();
                if alias_py.is_empty() {
                    continue;
                }
                let al = alias.len();
                // 窗口长度在别名长度 ±2 范围内搜索（从大到小：长匹配优先，避免短窗口误吞）
                for w in (al.saturating_sub(2)..=(al + 2)).rev() {
                    if w == 0 || w > n {
                        continue;
                    }
                    for start in 0..=(n - w) {
                        // 与已有替换重叠则跳过
                        if edits.iter().any(|(s, e, _)| start < *e && *s < start + w) {
                            continue;
                        }
                        let region_py: String = syllables[start..start + w]
                            .iter()
                            .map(|(_, p)| p.as_str())
                            .collect();
                        let sim = similarity(&region_py, &alias_py);
                        if sim >= self.threshold {
                            let src: String = syllables[start..start + w]
                                .iter()
                                .map(|(o, _)| *o)
                                .collect();
                            if entry.blacklist.contains(&src.to_lowercase()) {
                                continue;
                            }
                            edits.push((start, start + w, entry.display.clone()));
                        }
                    }
                }
            }
        }

        if edits.is_empty() {
            return text.to_string();
        }
        edits.sort_by_key(|(s, _, _)| *s);
        let mut out = String::new();
        let mut pos = 0usize;
        for (s, e, rep) in &edits {
            if *s < pos {
                continue;
            }
            for (o, _) in &syllables[pos..*s] {
                out.push(*o);
            }
            out.push_str(rep);
            pos = *e;
        }
        for (o, _) in &syllables[pos..] {
            out.push(*o);
        }
        out
    }
}

/// 文本 → 音节序列 [(原字符, 拼音小写串)]
/// 汉字取无声调拼音；ASCII 字母/数字保留原字符（小写）；其他字符保留
fn to_syllables(text: &str) -> Vec<(char, String)> {
    use pinyin::ToPinyin;
    let mut out = Vec::new();
    for ch in text.chars() {
        if let Some(py) = ch.to_pinyin() {
            out.push((ch, py.plain().to_string()));
        } else if ch.is_ascii_alphanumeric() {
            out.push((ch, ch.to_ascii_lowercase().to_string()));
        } else {
            out.push((ch, ch.to_string()));
        }
    }
    out
}

/// 归一化编辑距离相似度：1 - levenshtein / max_len
fn similarity(a: &str, b: &str) -> f64 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let dist = levenshtein(&a, &b);
    1.0 - dist as f64 / a.len().max(b.len()) as f64
}

fn levenshtein(a: &[char], b: &[char]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syllables() {
        let s = to_syllables("千问ASR");
        let joined: String = s.iter().map(|(_, p)| p.clone()).collect();
        assert_eq!(joined, "qianwenasr");
    }

    #[test]
    fn test_similarity() {
        assert!(similarity("qianwenasr", "qianwenasr") == 1.0);
        assert!(similarity("qianwen", "qianmen") > 0.7);
    }

    #[test]
    fn test_rules() {
        let rules = RuleEngine {
            rules: vec![(
                fancy_regex::Regex::new("毫安时").unwrap(),
                "mAh".to_string(),
            )],
        };
        assert_eq!(rules.apply("五千毫安时电池"), "五千mAh电池");
    }

    #[test]
    fn test_rule_parse_with_equals_in_pattern() {
        // 正则内含 (?= 的规则不应被「=」误切
        let tmp = std::env::temp_dir().join("cwrs_rule_test.txt");
        std::fs::write(
            &tmp,
            "([A-Za-z])\\.(?=[A-Za-z](?![A-Za-z])) = \\1\n毫安时 = mAh\n",
        )
        .unwrap();
        let rules = RuleEngine::load(&tmp).unwrap();
        assert_eq!(rules.len(), 2);
        // 规则语义：删除「后跟单字母（该字母后不是字母）」的点
        // a.b: 点后 b 的后面是空格 → 删；c.d.e: 两个点后都是单字母 → 都删
        assert_eq!(rules.apply("a.b c.d.e"), "ab cde");

        // 空替换规则：= 后带尾随空格应能解析（trim 整行会吃掉这个空格导致漏加载）
        let tmp2 = std::env::temp_dir().join("cwrs_rule_test2.txt");
        std::fs::write(&tmp2, "\\[breath\\]  =    \n/sil   =\n").unwrap();
        let rules2 = RuleEngine::load(&tmp2).unwrap();
        // 第二行 = 结尾无空格，原版 hot_rule.py 也跳过
        assert_eq!(rules2.len(), 1);
        assert_eq!(rules2.apply("嗯[breath]你好"), "嗯你好");
    }

    #[test]
    fn test_strip_punc() {
        assert_eq!(strip_trailing_punc("你好。", "，。,.", 8), "你好");
        assert_eq!(
            strip_trailing_punc(
                "这是一段很长很长很长很长很长很长很长很长的句子。",
                "，。,.",
                8
            ),
            "这是一段很长很长很长很长很长很长很长很长的句子。"
        );
    }

    #[test]
    fn test_hotword_replace() {
        let h = HotwordMatcher {
            entries: vec![Entry {
                display: "Qwen3-ASR".into(),
                alias_syllables: vec![
                    to_syllables("千问ASR")
                        .into_iter()
                        .map(|(_, py)| py)
                        .collect(),
                    to_syllables("Qwen3-ASR")
                        .into_iter()
                        .map(|(_, py)| py)
                        .collect(),
                ],
                blacklist: HashSet::new(),
            }],
            threshold: 0.8,
        };
        // ASR 错识别为同音字也应被替换
        assert_eq!(h.replace("浅问asr很好用"), "Qwen3-ASR很好用");
    }
}
