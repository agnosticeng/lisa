//! Qwen35 BPE reconstructed from the GGUF vocabulary and merges.
use anyhow::{Context, Result, ensure};
use lisa_gguf::{Header, Value};
pub fn from_header(h: &Header) -> Result<tokenizers::Tokenizer> {
    let text = |key: &str| h.metadata.get(key).and_then(Value::as_str);
    ensure!(
        text("tokenizer.ggml.model") == Some("gpt2")
            && text("tokenizer.ggml.pre") == Some("qwen35"),
        "expected Qwen35 GGUF BPE"
    );
    let arr = |key: &str| {
        h.metadata
            .get(key)
            .and_then(Value::as_array)
            .with_context(|| format!("missing {key}"))
    };
    let tokens = arr("tokenizer.ggml.tokens")?;
    let kinds = arr("tokenizer.ggml.token_type")?;
    ensure!(
        tokens.len() == kinds.len(),
        "GGUF vocabulary/type lengths differ"
    );
    let mut vocab = serde_json::Map::new();
    let mut added = Vec::new();
    for (id, (t, k)) in tokens.iter().zip(kinds).enumerate() {
        let t = t.as_str().context("non-string vocabulary entry")?;
        ensure!(
            vocab.insert(t.into(), serde_json::json!(id)).is_none(),
            "duplicate GGUF vocabulary entry"
        );
        if matches!(k.as_u64(), Some(3 | 4)) {
            added.push(serde_json::json!({"id":id,"content":t,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}));
        }
    }
    let merges: Vec<_> = arr("tokenizer.ggml.merges")?
        .iter()
        .map(|v| v.as_str().context("invalid GGUF merge"))
        .collect::<Result<_>>()?;
    // Original Qwen35 regex, including Unicode combining marks and single digits.
    let regex = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
    let data = serde_json::json!({"version":"1.0","added_tokens":added,"normalizer":null,"post_processor":null,"truncation":null,"padding":null,
        "pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":regex},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false}]},
        "decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":false,"use_regex":false},
        "model":{"type":"BPE","vocab":vocab,"merges":merges,"ignore_merges":false}});
    tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&data)?)
        .map_err(|e| anyhow::anyhow!("GGUF Qwen BPE: {e}"))
}
