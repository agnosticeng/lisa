//! Desktop model discovery, sizes and curated download metadata.

use std::path::PathBuf;

use crate::catalog;

#[derive(Clone, Debug)]
pub struct Model {
    pub repo_id: String,
    pub model_type: String,
    pub path: String,
    pub local: bool,
    pub size: u64,
    pub resident_bytes: Option<u64>,
    pub too_big: bool,
}

pub(super) fn config_model_type(dir: &std::path::Path) -> Option<String> {
    lisa_engine::models::model_type_of(dir).ok()
}

pub(super) fn display_name(repo_id: &str) -> &str {
    repo_id.rsplit('/').next().unwrap_or(repo_id)
}

/// Scan the local Hugging Face hub cache for checkpoints the engine supports.
pub(super) fn discover_models() -> Vec<Model> {
    let hub = lisa_engine::models::hf::hub_root();
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(&hub) else {
        return out;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let Some(rest) = name.strip_prefix("models--") else {
            continue;
        };
        let repo_id = rest.replacen("--", "/", 1);
        for path in lisa_engine::models::hf::cached_snapshots(&repo_id) {
            let glm = path.join(lisa_engine::models::glm53::gguf::ENTRY);
            if glm.is_file() {
                let info = (|| -> anyhow::Result<u64> {
                    let package = lisa_gguf::Package::open(&glm)?;
                    lisa_engine::models::glm53::gguf::weights::validate(&package)?;
                    Ok(std::fs::metadata(&glm)?.len())
                })();
                if let Ok(size) = info {
                    out.push(Model {
                        repo_id: repo_id.clone(),
                        model_type: "glm5-next".into(),
                        path: path.to_string_lossy().to_string(),
                        local: true,
                        size,
                        resident_bytes: Some(size),
                        too_big: false,
                    });
                    break;
                }
                continue;
            }
            let gguf = path.join(lisa_engine::models::deepseek41::GGUF_ENTRY);
            if gguf.is_file() {
                let info = (|| -> anyhow::Result<(u64, u64)> {
                    use lisa_engine::models::deepseek41::{
                        config::Config, engram::Layout, weights::WeightPlan,
                    };
                    let package = lisa_gguf::Package::open(&gguf)?;
                    let plan = WeightPlan::inspect(
                        &package,
                        &Config::from_header(&package.header)?,
                        &Layout::from_header(&package.header)?,
                    )?;
                    let size = package.shards.iter().try_fold(0u64, |total, file| {
                        Ok::<_, std::io::Error>(total + std::fs::metadata(file)?.len())
                    })?;
                    Ok((size, plan.resident_payload_bytes))
                })();
                if let Ok((size, resident_bytes)) = info {
                    out.push(Model {
                        repo_id: repo_id.clone(),
                        model_type: "deepseek41".into(),
                        path: path.to_string_lossy().to_string(),
                        local: true,
                        size,
                        resident_bytes: Some(resident_bytes),
                        too_big: false,
                    });
                    break;
                }
                continue;
            }
            if let Some(mt) = config_model_type(&path) {
                if lisa_engine::models::ModelKind::from_model_type(&mt)
                    .is_ok_and(|kind| kind != lisa_engine::models::ModelKind::Laya)
                    && has_model_weights(&path, &mt)
                {
                    let size = model_weight_size(&path, &mt);
                    out.push(Model {
                        repo_id: repo_id.clone(),
                        model_type: mt,
                        path: path.to_string_lossy().to_string(),
                        local: true,
                        size,
                        resident_bytes: None,
                        too_big: false,
                    });
                    break;
                }
            }
        }
    }
    out.sort_by(|a, b| a.repo_id.cmp(&b.repo_id));
    out
}

fn has_model_weights(path: &std::path::Path, model_type: &str) -> bool {
    if model_type == "flux3-klein" {
        return path.join("model.safetensors").is_file()
            && path.join("ae.safetensors").is_file()
            && has_model_weights(&path.join("text_encoder"), "qwen3_vl");
    }
    let components: &[&str] = if model_type == "flux2-klein" {
        &["text_encoder", "transformer", "vae"]
    } else {
        &[""]
    };
    components.iter().all(|component| {
        let root = path.join(component);
        let index = root.join("model.safetensors.index.json");
        if index.is_file() {
            return std::fs::read(&index)
                .ok()
                .and_then(|data| serde_json::from_slice::<serde_json::Value>(&data).ok())
                .is_some_and(|value| {
                    value
                        .get("weight_map")
                        .and_then(|map| map.as_object())
                        .is_some_and(|map| {
                            let mut checked = std::collections::HashSet::new();
                            !map.is_empty()
                                && map.values().all(|file| {
                                    file.as_str().is_some_and(|file| {
                                        !checked.insert(file) || root.join(file).is_file()
                                    })
                                })
                        })
                });
        }
        std::fs::read_dir(root).is_ok_and(|files| {
            files.flatten().any(|file| {
                file.path()
                    .extension()
                    .is_some_and(|ext| ext == "safetensors")
                    && file.metadata().is_ok_and(|meta| meta.len() > 0)
            })
        })
    })
}

fn model_weight_size(path: &std::path::Path, model_type: &str) -> u64 {
    let resident = path.join(".lisa-resident");
    if matches!(model_type, "glm5_next" | "glm5_next_text") && resident.join("index.json").is_file()
    {
        // Prepared GLM maps the resident copy. Counting original and prepared
        // weights together would incorrectly disable a model that fits RAM.
        dir_size(&resident)
    } else {
        dir_size(path)
    }
}

/// Count each file once, following Hub blob links without recursing through
/// directory cycles or double-counting hard-linked weights.
fn dir_size(path: &std::path::Path) -> u64 {
    use std::os::unix::fs::MetadataExt;

    let mut total = 0u64;
    let mut pending = vec![path.to_path_buf()];
    let mut seen = std::collections::HashSet::new();
    while let Some(path) = pending.pop() {
        let Ok(metadata) = std::fs::metadata(&path) else {
            continue;
        };
        if !seen.insert((metadata.dev(), metadata.ino())) {
            continue;
        }
        if metadata.is_dir() {
            if let Ok(entries) = std::fs::read_dir(path) {
                pending.extend(entries.flatten().map(|entry| entry.path()));
            }
        } else if metadata.is_file() {
            total = total.saturating_add(metadata.len());
        }
    }
    total
}

pub(super) fn model_ram_budget() -> u64 {
    lisa_engine::core::admission::MemoryBudget::from_host().budget() as u64
}

/// The Hub cache directory for a repo (`models--org--name`).
fn hf_hub_dir(repo: &str) -> PathBuf {
    lisa_engine::models::hf::repo_dir(repo)
}

/// Bytes already written for a repo's blobs (in-progress downloads included).
pub(super) fn hf_blobs_size(repo: &str) -> u64 {
    dir_size(&hf_hub_dir(repo).join("blobs"))
}

/// Refresh sizes for the explicit catalog; network results cannot add models.
pub(super) fn fetch_remote_models() -> Vec<(String, u64)> {
    catalog::MODELS
        .iter()
        .map(|entry| {
            let size = lisa_engine::models::hf::download_size(entry.repo_id)
                .unwrap_or(entry.download_bytes);
            (entry.repo_id.to_string(), size)
        })
        .collect()
}
