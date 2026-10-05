// LLM 角色：识别结果以角色名开头时交给 OpenAI 兼容接口处理

use crate::config::LlmRole;
use anyhow::{bail, Result};

/// 从文本中提取角色调用：若以某角色名开头，返回 (角色, 去掉角色名后的用户输入)
pub fn extract_role<'a>(roles: &'a [LlmRole], text: &'a str) -> Option<(&'a LlmRole, String)> {
    for role in roles {
        if let Some(rest) = text.strip_prefix(&role.name) {
            // 允许角色名后跟空格、冒号、中文冒号
            let rest = rest
                .trim_start_matches([' ', ':', '：', '，', ','])
                .trim()
                .to_string();
            return Some((role, rest));
        }
    }
    None
}

pub fn complete(role: &LlmRole, user_text: &str) -> Result<String> {
    if role.base_url.is_empty() || role.model.is_empty() {
        bail!("LLM 角色配置不完整（base_url / model）");
    }
    let url = format!("{}/chat/completions", role.base_url.trim_end_matches('/'));
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;

    let body = serde_json::json!({
        "model": role.model,
        "temperature": role.temperature,
        "messages": [
            {"role": "system", "content": role.prompt},
            {"role": "user", "content": user_text}
        ]
    });

    let mut req = client.post(&url).json(&body);
    if !role.api_key.is_empty() {
        req = req.bearer_auth(&role.api_key);
    }
    let resp = req.send()?;
    let status = resp.status();
    let value: serde_json::Value = resp.json()?;
    if !status.is_success() {
        bail!(
            "LLM 接口返回 {status}: {}",
            serde_json::to_string(&value).unwrap_or_default()
        );
    }
    let content = value
        .pointer("/choices/0/message/content")
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if content.is_empty() {
        bail!("LLM 返回内容为空");
    }
    Ok(content)
}
