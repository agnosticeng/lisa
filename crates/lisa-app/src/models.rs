//! Apple app model discovery, sizes and curated download metadata.

use crate::catalog;

#[derive(Clone, Debug)]
pub struct Model {
    pub repo_id: String,
    pub path: String,
    pub size: u64,
    pub resident_bytes: Option<u64>,
}

pub(super) fn config_model_type(dir: &std::path::Path) -> Option<String> {
    lisa_engine::checkpoint::model_type_of(dir).ok()
}

/// Scan the local Hub cache for supported checkpoints in the app allowlist.
pub(super) fn discover_models() -> Vec<Model> {
    let hub = lisa_engine::checkpoint::hf::hub_root();
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
        if catalog::find(&repo_id).is_none() {
            continue;
        }
        for path in lisa_engine::checkpoint::hf::cached_snapshots(&repo_id) {
            let qwen = path.join(lisa_engine::weights::qwen35::gguf::ENTRY);
            if qwen.is_file() {
                if let Ok(resident_bytes) = lisa_engine::weights::qwen35::gguf::inspect(&qwen) {
                    out.push(Model {
                        repo_id: repo_id.clone(),
                        path: path.to_string_lossy().into(),
                        size: std::fs::metadata(&qwen).map(|m| m.len()).unwrap_or(0),
                        resident_bytes: Some(resident_bytes),
                    });
                    break;
                }
                continue;
            }
            let inkling2 = path.join(lisa_engine::checkpoint::catalog::inkling2::ENTRY);
            if inkling2.is_file()
                && path
                    .join(lisa_engine::checkpoint::catalog::inkling2::PROJECTOR)
                    .is_file()
            {
                let info = (|| -> anyhow::Result<u64> {
                    let header = lisa_gguf::Header::open(&inkling2)?;
                    lisa_engine::checkpoint::inkling2::config(&header)?;
                    Ok(std::fs::metadata(&inkling2)?.len()
                        + std::fs::metadata(
                            path.join(lisa_engine::checkpoint::catalog::inkling2::PROJECTOR),
                        )?
                        .len())
                })();
                if let Ok(size) = info {
                    out.push(Model {
                        repo_id: repo_id.clone(),
                        path: path.to_string_lossy().into(),
                        size,
                        resident_bytes: Some(size),
                    });
                    break;
                }
                continue;
            }
            let olala = path.join(lisa_engine::checkpoint::catalog::olala::ENTRY);
            if olala.is_file() {
                let info = (|| -> anyhow::Result<u64> {
                    let header = lisa_gguf::Header::open(&olala)?;
                    lisa_engine::checkpoint::olala::config(&header)?;
                    Ok(std::fs::metadata(&olala)?.len())
                })();
                if let Ok(size) = info {
                    out.push(Model {
                        repo_id: repo_id.clone(),
                        path: path.to_string_lossy().into(),
                        size,
                        resident_bytes: Some(size),
                    });
                    break;
                }
                continue;
            }
            let glm = path.join(lisa_engine::checkpoint::catalog::glm53::ENTRY);
            if glm.is_file() {
                let info = (|| -> anyhow::Result<u64> {
                    let package = lisa_gguf::Package::open(&glm)?;
                    lisa_engine::checkpoint::glm53::validate(&package)?;
                    Ok(std::fs::metadata(&glm)?.len())
                })();
                if let Ok(size) = info {
                    out.push(Model {
                        repo_id: repo_id.clone(),
                        path: path.to_string_lossy().to_string(),
                        size,
                        resident_bytes: Some(size),
                    });
                    break;
                }
                continue;
            }
            let gguf = path.join(lisa_engine::checkpoint::catalog::deepseek41::GGUF_ENTRY);
            if gguf.is_file() {
                let info = (|| -> anyhow::Result<(u64, u64)> {
                    use lisa_engine::weights::deepseek41::plan::WeightPlan;
                    let package = lisa_gguf::Package::open(&gguf)?;
                    let plan = WeightPlan::inspect(
                        &package,
                        &lisa_engine::checkpoint::deepseek41::config(&package.header)?,
                        &lisa_engine::checkpoint::deepseek41::engram_layout(&package.header)?,
                    )?;
                    let size = package.shards.iter().try_fold(0u64, |total, file| {
                        Ok::<_, std::io::Error>(total + std::fs::metadata(file)?.len())
                    })?;
                    Ok((size, plan.resident_payload_bytes))
                })();
                if let Ok((size, resident_bytes)) = info {
                    out.push(Model {
                        repo_id: repo_id.clone(),
                        path: path.to_string_lossy().to_string(),
                        size,
                        resident_bytes: Some(resident_bytes),
                    });
                    break;
                }
                continue;
            }
            if let Some(mt) = config_model_type(&path)
                && lisa_engine::checkpoint::ModelKind::from_model_type(&mt)
                    .is_ok_and(|kind| kind != lisa_engine::checkpoint::ModelKind::Laya)
                && has_model_weights(&path, &mt)
            {
                let size = model_weight_size(&path, &mt);
                out.push(Model {
                    repo_id: repo_id.clone(),
                    path: path.to_string_lossy().to_string(),
                    size,
                    resident_bytes: lisa_engine::checkpoint::catalog::bonsai::is_checkpoint(&path)
                        .then_some(lisa_engine::checkpoint::catalog::bonsai::REQUIRED_RAM as u64),
                });
                break;
            }
        }
    }
    out.sort_by(|a, b| a.repo_id.cmp(&b.repo_id));
    out
}

fn has_model_weights(path: &std::path::Path, model_type: &str) -> bool {
    if lisa_engine::checkpoint::catalog::bonsai::is_checkpoint(path) {
        return true;
    }
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

/// Load a local checkpoint. A curated snapshot installed without its `mtp/`
/// sidecar completes it first; when that fails (for example offline), the
/// target still loads, without a drafter.
pub(super) fn load_model(
    path: &std::path::Path,
    progress: impl Fn(Option<(u64, u64)>) + Send + Sync + 'static,
) -> anyhow::Result<lisa_serve::application::loading::Loaded> {
    use lisa_engine::checkpoint::hf;
    let missing_sidecar = hf::snapshot_identity(path).is_some_and(|(repo, _)| {
        catalog::ships_mtp_sidecar(&repo) && !path.join("mtp/weights.safetensors").is_file()
    });
    if missing_sidecar {
        let progress = std::sync::Arc::new(progress);
        progress(Some((0, 0)));
        let download_progress = progress.clone();
        let result = hf::complete_snapshot(path, move |done, total| {
            download_progress(Some((done, total)))
        });
        progress(None);
        if let Err(error) = result {
            eprintln!("[load] MTP sidecar unavailable; loading without a drafter: {error:#}");
        }
    }
    warm_loaded_model(lisa_serve::application::loading::load_dir(path)?, path)
}

fn warm_loaded_model(
    mut loaded: lisa_serve::application::loading::Loaded,
    path: &std::path::Path,
) -> anyhow::Result<lisa_serve::application::loading::Loaded> {
    if matches!(
        config_model_type(path).as_deref(),
        Some(
            "olala"
                | "inkling2"
                | "lfm3"
                | "qwen3_5"
                | "qwen3_5_text"
                | "qwen35"
                | "qwen4_exp"
                | "qwen4_exp_text"
        )
    ) && let lisa_serve::application::loading::Loaded::Language(model) = &mut loaded
    {
        // Bounded, model-owned shapes use private caches. Warm once before
        // reporting Ready, without changing the request's prefill law.
        model.warmup(&[])?;
    }
    Ok(loaded)
}

pub(super) fn download_model_with_progress(
    repo: &str,
    progress: impl Fn(u64, u64) + Send + Sync + 'static,
) -> anyhow::Result<std::path::PathBuf> {
    lisa_engine::checkpoint::hf::download_repo_with_progress(repo, progress)
}
