//! Tokenizer checkpoint discovery and metadata adaptation.
use crate::core::tokenizer::Tokenizer;
use anyhow::{Context, ensure};
use lisa_gguf::{Header, Value};
use tokenizers::Tokenizer as HfTokenizer;

pub fn load(dir: &std::path::Path) -> anyhow::Result<Tokenizer> {
    if dir.is_file() {
        let header = lisa_gguf::Header::open(dir)?;
        if matches!(header.architecture()?, "qwen35" | "olala" | "inkling2") {
            let inner = if header.architecture()? == "inkling2" {
                crate::checkpoint::qwen35::tokenizer::from_inkling2_header(&header)?
            } else if header.architecture()? == "olala" {
                crate::checkpoint::qwen35::tokenizer::from_qwen2_header(&header)?
            } else {
                crate::checkpoint::qwen35::tokenizer::from_header(&header)?
            };
            let im_end_ids = [
                "<|im_end|>",
                "<|endoftext|>",
                "<|content_model_end_sampling|>",
            ]
            .iter()
            .filter_map(|t| inner.token_to_id(t))
            .collect();
            return Ok(Tokenizer {
                inner,
                im_end_ids,
                native: None,
                lfm3: false,
            });
        }
        anyhow::ensure!(
            matches!(header.architecture()?, "deepseek41" | "glm5-next"),
            "unsupported GGUF tokenizer architecture"
        );
        let native = native_from_header(&header)?;
        let im_end_ids = if header.architecture()? == "glm5-next" {
            crate::checkpoint::glm53::config(&header)?
                .eos_token_ids
                .into_iter()
                .map(|id| id as u32)
                .collect()
        } else {
            vec![native.eos]
        };
        return Ok(Tokenizer {
            inner: native.decoder_view()?,
            im_end_ids,
            native: Some(native),
            lfm3: false,
        });
    }
    load_file(&dir.join("tokenizer.json"))
}

/// Load a `tokenizer.json` from an explicit path (some checkpoints keep it
/// in a `tokenizer/` subdirectory).
pub fn load_file(path: &std::path::Path) -> anyhow::Result<Tokenizer> {
    let is_glm53 = match path.parent() {
        Some(dir) => is_glm53_model_dir(dir)?,
        None => false,
    };
    let inner = if is_glm53 {
        load_glm53_tokenizer(path)?
    } else {
        HfTokenizer::from_file(path).map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?
    };
    // Discover only terminators actually present in this checkpoint.
    let mut im_end_ids = Vec::new();
    for token in [
        "<|im_end|>",
        "<|endoftext|>",
        "<|content_model_end_sampling|>",
    ] {
        if let Some(id) = inner.token_to_id(token) {
            im_end_ids.push(id);
        }
    }
    let lfm3 = path
        .parent()
        .and_then(|dir| std::fs::read(dir.join("config.json")).ok())
        .map(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes))
        .transpose()?
        .is_some_and(|config| config["model_type"] == "lfm3");
    Ok(Tokenizer {
        inner,
        im_end_ids,
        native: None,
        lfm3,
    })
}

/// Embedded GGUF templates and directory checkpoints use one discovery path.
pub fn load_chat_template(model: &std::path::Path) -> anyhow::Result<Option<String>> {
    if model.is_file() {
        let header = lisa_gguf::Header::open(model)?;
        return Ok(header
            .metadata
            .get("tokenizer.chat_template")
            .and_then(lisa_gguf::Value::as_str)
            .map(str::to_owned));
    }
    match std::fs::read_to_string(model.join("chat_template.jinja")) {
        Ok(template) => {
            // Standalone templates still obtain their special tokens from tokenizer_config.json.
            let config_path = model.join("tokenizer_config.json");
            let config: serde_json::Value = if config_path.is_file() {
                serde_json::from_slice(&std::fs::read(config_path)?)?
            } else {
                serde_json::Value::Null
            };
            let mut source = String::new();
            for name in ["bos_token", "eos_token", "pad_token"] {
                let token = config[name]
                    .as_str()
                    .or_else(|| config[name]["content"].as_str());
                if let Some(token) = token {
                    source.push_str(&format!(
                        "{{%- set {name} = {} -%}}",
                        serde_json::to_string(token)?
                    ));
                }
            }
            source.push_str(&template);
            Ok(Some(source))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e.into()),
    }
}

fn is_glm53_model_dir(dir: &std::path::Path) -> anyhow::Result<bool> {
    let config_path = dir.join("config.json");
    if !config_path.is_file() {
        return Ok(false);
    }
    let config: serde_json::Value = serde_json::from_slice(&std::fs::read(&config_path)?)?;
    let model_type = config.get("model_type").and_then(serde_json::Value::as_str);
    let text_model_type = config
        .pointer("/text_config/model_type")
        .and_then(serde_json::Value::as_str);
    Ok(matches!(model_type, Some("glm5_next" | "glm5_next_text"))
        || text_model_type == Some("glm5_next_text"))
}

fn load_glm53_tokenizer(path: &std::path::Path) -> anyhow::Result<HfTokenizer> {
    let bytes = std::fs::read(path)?;
    load_glm53_tokenizer_bytes(&bytes)
}

pub(crate) fn load_glm53_tokenizer_bytes(bytes: &[u8]) -> anyhow::Result<HfTokenizer> {
    let mut document: serde_json::Value = serde_json::from_slice(bytes)?;
    let model = document
        .get_mut("model")
        .and_then(serde_json::Value::as_object_mut)
        .ok_or_else(|| anyhow::anyhow!("GLM-5.3 tokenizer model is missing"))?;
    anyhow::ensure!(
        model.get("type").and_then(serde_json::Value::as_str) == Some("BPE"),
        "GLM-5.3 DS4 parity requires a BPE tokenizer"
    );

    // The MLX tokenizer JSON sets ignore_merges=true, allowing a whole vocab
    // entry to bypass BPE merge ranks. The reference's GGUF tokenizer always applies its
    // merge table. Force the same semantics while keeping the MLX vocab and
    // pre-tokenizer, so raw text maps to the same token IDs as the reference.
    model.insert("ignore_merges".to_string(), serde_json::Value::Bool(false));
    let bytes = serde_json::to_vec(&document)?;
    HfTokenizer::from_bytes(bytes).map_err(|e| anyhow::anyhow!("GLM-5.3 tokenizer: {e}"))
}

pub fn native_from_header(
    header: &Header,
) -> anyhow::Result<crate::core::native_tokenizer::Tokenizer> {
    let pre = header
        .metadata
        .get("tokenizer.ggml.pre")
        .and_then(Value::as_str);
    let glm4 = pre == Some("glm4");
    ensure!(
        header
            .metadata
            .get("tokenizer.ggml.model")
            .and_then(Value::as_str)
            == Some("gpt2")
            && (glm4 || pre == Some("joyai-llm")),
        "unsupported GGUF tokenizer"
    );
    let strings = |key| -> anyhow::Result<Vec<String>> {
        header
            .metadata
            .get(key)
            .and_then(Value::as_array)
            .context("missing GGUF tokenizer array")?
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .context("invalid GGUF tokenizer string")
            })
            .collect()
    };
    let tokens = strings("tokenizer.ggml.tokens")?;
    let native = crate::core::native_tokenizer::Tokenizer::from_vocabulary(
        glm4,
        tokens,
        strings("tokenizer.ggml.merges")?,
    )?;
    for (key, id) in [
        ("tokenizer.ggml.bos_token_id", native.bos),
        ("tokenizer.ggml.eos_token_id", native.eos),
    ] {
        ensure!(
            header.metadata.get(key).and_then(Value::as_u64) == Some(id as u64),
            "DeepSeek special token metadata mismatch"
        );
    }
    Ok(native)
}
