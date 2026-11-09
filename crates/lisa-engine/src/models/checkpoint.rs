//! Checkpoint identity and local/Hub resolution without loading model weights.

use std::path::{Path, PathBuf};

use super::{flux2, flux3, hf};

/// The implemented checkpoint layouts. Keep metadata aliases in one place so
/// the loader and desktop discovery agree about which family a checkpoint uses.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelKind {
    Qwen4,
    Qwen35,
    Glm53,
    Glm53Gguf,
    DeepSeek41,
    Flux2,
    Flux3,
    Laya,
}

impl ModelKind {
    pub fn from_model_type(name: &str) -> anyhow::Result<Self> {
        match name {
            "qwen4_exp" | "qwen4_exp_text" => Ok(Self::Qwen4),
            "qwen3_5" | "qwen3_5_text" => Ok(Self::Qwen35),
            "glm5_next" | "glm5_next_text" => Ok(Self::Glm53),
            "glm5-next" => Ok(Self::Glm53Gguf),
            "deepseek41" => Ok(Self::DeepSeek41),
            "flux2-klein" => Ok(Self::Flux2),
            "flux3-klein" => Ok(Self::Flux3),
            "laya" => Ok(Self::Laya),
            _ => anyhow::bail!("unsupported model type {name:?}"),
        }
    }
}

/// Resolve `target` to a checkpoint directory or a local GGUF file.
///
/// A path that exists is used as-is. Otherwise `target` is treated as a
/// Hugging Face repo id and resolved against the Hugging Face hub cache; if it
/// is not cached, the built-in Hub downloader fetches it into the same cache. These
/// are the only two sources — there is no legacy fallback.
pub fn resolve_model_dir(target: &str) -> anyhow::Result<PathBuf> {
    let direct = Path::new(target);
    if direct.is_dir() {
        return Ok(hf::checkpoint_path(direct.to_path_buf()));
    }
    if direct.is_file()
        && direct
            .extension()
            .is_some_and(|s| s.eq_ignore_ascii_case("gguf"))
    {
        return Ok(direct.to_path_buf());
    }
    if direct.exists() {
        anyhow::bail!("{} exists but is not a directory", direct.display());
    }

    anyhow::ensure!(
        hf::is_repo_id(target),
        "unknown model {target}: pass a local path or a Hugging Face repo id \
         (models are cached in the HF hub cache)"
    );

    if let Some(dir) = hf::cached_snapshot(target) {
        return Ok(hf::checkpoint_path(dir));
    }

    // Not cached: download it into the hub cache with our own curl-based
    // fetcher (no `huggingface_hub` dependency).
    let dir = hf::download_repo(target)?;
    anyhow::ensure!(
        hf::is_model_dir(&dir),
        "downloaded {target} but found no supported checkpoint entry"
    );
    Ok(hf::checkpoint_path(dir))
}

/// Read the GGUF architecture or config model type without loading weights.
/// Nested `text_config.model_type` takes precedence over the root type.
pub fn model_type_of(dir: &Path) -> anyhow::Result<String> {
    let path = hf::checkpoint_path(dir.to_path_buf());
    let dir = path.as_path();
    if dir.is_file() {
        return Ok(lisa_gguf::Header::open(dir)?.architecture()?.to_owned());
    }
    if flux3::is_checkpoint(dir) {
        return Ok("flux3-klein".into());
    }
    if flux2::is_checkpoint(dir) {
        return Ok("flux2-klein".into());
    }
    let path = dir.join("config.json");
    if path.is_file() {
        let data = std::fs::read_to_string(&path)?;
        let v: serde_json::Value = serde_json::from_str(&data)?;
        if let Some(t) = v
            .get("text_config")
            .and_then(|t| t.get("model_type"))
            .and_then(|t| t.as_str())
        {
            return Ok(t.to_string());
        }
        if let Some(t) = v.get("model_type").and_then(|t| t.as_str()) {
            return Ok(t.to_string());
        }
        anyhow::bail!("{} has no model_type", path.display());
    }

    // Configless checkpoints: Laya keeps its encoder config in `encoder/` and
    // its decision-head config in `rl_agent_config.json`.
    let enc = dir.join("encoder/config.json");
    if enc.is_file() {
        if dir.join("rl_agent_config.json").is_file() {
            return Ok("laya".to_string());
        }
        let data = std::fs::read_to_string(&enc)?;
        let v: serde_json::Value = serde_json::from_str(&data)?;
        if let Some(t) = v.get("model_type").and_then(|t| t.as_str()) {
            return Ok(t.to_string());
        }
    }

    anyhow::bail!(
        "no config.json or encoder/config.json under {}",
        dir.display()
    )
}

/// Application defaults come from the checkpoint, before warmup or session
/// creation. Explicit CLI overrides are applied by the caller after loading.
pub(super) fn default_context_capacity(path: &Path) -> anyhow::Result<Option<usize>> {
    if path.is_file() {
        return gguf_context_capacity(&lisa_gguf::Header::open(path)?);
    }
    let config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
    json_context_capacity(&config)
}

fn context_value(value: Option<u64>, key: &str) -> anyhow::Result<usize> {
    let value = value.ok_or_else(|| anyhow::anyhow!("{key} must be a positive integer"))?;
    anyhow::ensure!(value > 0, "{key} must be a positive integer");
    usize::try_from(value)
        .map_err(|_| anyhow::anyhow!("{key} exceeds the platform's context capacity"))
}

fn gguf_context_capacity(header: &lisa_gguf::Header) -> anyhow::Result<Option<usize>> {
    let key = format!("{}.context_length", header.architecture()?);
    header
        .metadata
        .get(&key)
        .map(|value| context_value(value.as_u64(), &key))
        .transpose()
}

fn json_context_capacity(config: &serde_json::Value) -> anyhow::Result<Option<usize>> {
    let key = "max_position_embeddings";
    config
        .get("text_config")
        .and_then(|text| text.get(key))
        .filter(|value| !value.is_null())
        .or_else(|| config.get(key).filter(|value| !value.is_null()))
        .map(|value| context_value(value.as_u64(), key))
        .transpose()
}

#[cfg(test)]
mod context_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn metadata_context_uses_nested_text_config_and_preserves_missing_defaults() {
        assert_eq!(
            json_context_capacity(&json!({"max_position_embeddings":262144})).unwrap(),
            Some(262144)
        );
        assert_eq!(json_context_capacity(&json!({"max_position_embeddings":4096,"text_config":{"max_position_embeddings":131072}})).unwrap(), Some(131072));
        assert_eq!(json_context_capacity(&json!({"max_position_embeddings":32768,"text_config":{"max_position_embeddings":null}})).unwrap(), Some(32768));
        assert_eq!(json_context_capacity(&json!({})).unwrap(), None);
        for value in [json!(0), json!(-1), json!(1.5), json!("4096")] {
            assert!(json_context_capacity(&json!({"max_position_embeddings":value})).is_err());
        }
    }

    #[test]
    fn gguf_context_uses_the_architecture_metadata_without_a_startup_cap() {
        for (architecture, capacity) in [
            ("qwen35", 262144),
            ("glm5-next", 1048576),
            ("deepseek41", 1048576),
        ] {
            let mut header = lisa_gguf::Header {
                metadata: std::collections::BTreeMap::from([
                    (
                        "general.architecture".into(),
                        lisa_gguf::Value::String(architecture.into()),
                    ),
                    (
                        format!("{architecture}.context_length"),
                        lisa_gguf::Value::Unsigned(capacity),
                    ),
                ]),
                tensors: Default::default(),
                data_offset: 0,
                file_bytes: 0,
            };
            assert_eq!(
                gguf_context_capacity(&header).unwrap(),
                Some(capacity as usize)
            );
            header.metadata.insert(
                format!("{architecture}.context_length"),
                lisa_gguf::Value::Unsigned(0),
            );
            assert!(gguf_context_capacity(&header).is_err());
        }
    }
}
