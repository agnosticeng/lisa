//! Apple app model discovery, sizes and curated download metadata.

use crate::catalog;
use lisa_engine::core::config::ReadConfig;

#[derive(Clone, Debug)]
pub struct Model {
    pub repo_id: String,
    pub model_type: String,
    pub path: String,
    pub size: u64,
    pub resident_bytes: Option<u64>,
}

pub(super) fn config_model_type(dir: &std::path::Path) -> Option<String> {
    lisa_engine::checkpoint::model_type_of(dir).ok()
}

pub(super) fn display_name(repo_id: &str) -> &str {
    repo_id.rsplit('/').next().unwrap_or(repo_id)
}

/// Scan the local Hugging Face hub cache for checkpoints the engine supports.
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
        if repo_id == lisa_engine::bindings::qwen35_gguf::REPO {
            continue;
        }
        for path in lisa_engine::checkpoint::hf::cached_snapshots(&repo_id) {
            let qwen = path.join(lisa_engine::bindings::qwen35_gguf::ENTRY);
            if qwen.is_file() {
                if let Ok(resident_bytes) = lisa_engine::bindings::qwen35_gguf::inspect(&qwen) {
                    out.push(Model {
                        repo_id: repo_id.clone(),
                        model_type: "qwen35".into(),
                        path: path.to_string_lossy().into(),
                        size: std::fs::metadata(&qwen).map(|m| m.len()).unwrap_or(0),
                        resident_bytes: Some(resident_bytes),
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
                        model_type: "glm5-next".into(),
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
                    use lisa_engine::bindings::deepseek41_weights::WeightPlan;
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
                        model_type: "deepseek41".into(),
                        path: path.to_string_lossy().to_string(),
                        size,
                        resident_bytes: Some(resident_bytes),
                    });
                    break;
                }
                continue;
            }
            if let Some(mt) = config_model_type(&path) {
                if lisa_engine::checkpoint::ModelKind::from_model_type(&mt)
                    .is_ok_and(|kind| kind != lisa_engine::checkpoint::ModelKind::Laya)
                    && has_model_weights(&path, &mt)
                {
                    let size = model_weight_size(&path, &mt);
                    out.push(Model {
                        repo_id: repo_id.clone(),
                        model_type: mt,
                        path: path.to_string_lossy().to_string(),
                        size,
                        resident_bytes: lisa_engine::checkpoint::catalog::bonsai::is_checkpoint(
                            &path,
                        )
                        .then_some(lisa_engine::checkpoint::catalog::bonsai::REQUIRED_RAM as u64),
                    });
                    break;
                }
            }
        }
    }
    let draft_bytes =
        cached_dspine().map_or(catalog::DSPINE_DOWNLOAD_BYTES, |path| dir_size(&path));
    for model in &mut out {
        if uses_dspine_path(std::path::Path::new(&model.path)).unwrap_or(false) {
            let mtp = dir_size(&std::path::Path::new(&model.path).join("mtp"));
            let target = model.size.saturating_sub(mtp);
            model.resident_bytes = Some(
                model
                    .resident_bytes
                    .unwrap_or(target)
                    .saturating_add(catalog::DSPINE_WEIGHT_BYTES),
            );
            model.size = target.saturating_add(draft_bytes);
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

fn download_profile(repo: &str) -> lisa_engine::checkpoint::hf::DownloadProfile {
    use lisa_engine::checkpoint::hf::DownloadProfile;
    if catalog::uses_dspine(repo) {
        DownloadProfile::WithoutMtp
    } else {
        DownloadProfile::Complete
    }
}

/// Inspect metadata rather than applying the 27B draft to smaller dense Qwens.
pub(super) fn uses_dspine_path(path: &std::path::Path) -> anyhow::Result<bool> {
    use lisa_engine::{bindings::dspine, checkpoint};
    use lisa_models::qwen3_5::config::Qwen35Config;
    let path = checkpoint::hf::checkpoint_path(path.to_path_buf());
    match checkpoint::model_type_of(&path)?.as_str() {
        "qwen3_5" | "qwen3_5_text" => Ok(dspine::supports_target(&Qwen35Config::from_json(
            &path.join("config.json"),
        )?)),
        _ => Ok(false),
    }
}

fn draft_checkpoint_available(path: &std::path::Path) -> bool {
    // Read metadata and mapped shard headers only; this does not initialize Metal
    // or load tensor payloads. Nonempty files alone can still be truncated.
    let config_ok = std::fs::read(path.join("config.json"))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<serde_json::Value>(&bytes).ok())
        .is_some_and(|config| config["model_type"] == "qwen3" && config["num_hidden_layers"] == 5);
    if !config_ok || !has_model_weights(path, "qwen3") {
        return false;
    }
    std::fs::read_dir(path).is_ok_and(|entries| {
        let shards: Vec<_> = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|file| file.extension().is_some_and(|ext| ext == "safetensors"))
            .collect();
        !shards.is_empty()
            && shards.iter().all(|file| {
                lisa_mlx::Shard::open(file).is_ok_and(|shard| shard.tensor_names().next().is_some())
            })
    })
}

fn cached_draft_repo(repo: &str) -> Option<std::path::PathBuf> {
    lisa_engine::checkpoint::hf::cached_snapshots(repo)
        .into_iter()
        .find(|path| draft_checkpoint_available(path))
}

fn cached_dspine() -> Option<std::path::PathBuf> {
    cached_draft_repo(lisa_engine::bindings::dspine::REPO)
}

/// Check locally first so complete cached drafts also work offline. Missing or
/// interrupted downloads use the Hub client's resume/size-repair path.
fn ensure_draft_repo(
    repo: &str,
    progress: impl Fn(u64, u64) + Send + Sync + 'static,
) -> anyhow::Result<std::path::PathBuf> {
    if let Some(path) = cached_draft_repo(repo) {
        let size = dir_size(&path);
        progress(size, size);
        return Ok(path);
    }
    let path = lisa_engine::checkpoint::hf::download_repo_with_progress(repo, progress)?;
    anyhow::ensure!(
        draft_checkpoint_available(&path),
        "downloaded drafter is incomplete or invalid: {}",
        path.display()
    );
    Ok(path)
}

/// Existing target downloads acquire the missing companion on their next load.
/// Progress is emitted only while fetching a missing companion, then cleared
/// before loading the target. Explicit local overrides remain user-owned.
pub(super) fn load_model(
    path: &std::path::Path,
    progress: impl Fn(Option<(u64, u64)>) + Send + Sync + 'static,
) -> anyhow::Result<lisa_serve::application::loading::Loaded> {
    use lisa_engine::{bindings::dspine, checkpoint};
    use lisa_serve::application::loading;
    if !uses_dspine_path(path)? {
        return loading::load_dir(path);
    }
    let selection = match std::env::var("LISA_QWEN_DSPINE_MODEL") {
        Ok(model) => model,
        Err(std::env::VarError::NotPresent) => dspine::REPO.to_owned(),
        Err(error) => return Err(error.into()),
    };
    let draft =
        if checkpoint::hf::is_repo_id(&selection) && !std::path::Path::new(&selection).exists() {
            match cached_draft_repo(&selection) {
                Some(path) => path,
                None => {
                    let progress = std::sync::Arc::new(progress);
                    progress(Some((0, 0)));
                    let download_progress = progress.clone();
                    let result = ensure_draft_repo(&selection, move |done, total| {
                        download_progress(Some((done, total)))
                    });
                    progress(None);
                    result?
                }
            }
        } else {
            checkpoint::resolve_model_dir(&selection)?
        };
    loading::load_dir_with_dspine(path, &draft)
}

/// Download the shared draft first, then the target without optional MTP files.
/// The model row only completes when both packages have succeeded. Hub cache
/// hits are reused and both phases contribute to the same progress indicator.
pub(super) fn download_model_with_progress(
    repo: &str,
    progress: impl Fn(u64, u64) + Send + Sync + 'static,
) -> anyhow::Result<std::path::PathBuf> {
    use lisa_engine::{bindings::dspine, checkpoint::hf};
    use std::sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    };
    let profile = download_profile(repo);
    if !catalog::uses_dspine(repo) {
        return hf::download_repo_with_progress(repo, progress);
    }
    let target_size = hf::download_size_with_profile(repo, profile)?;
    let progress = Arc::new(progress);
    let draft_size = Arc::new(AtomicU64::new(0));
    let draft_completed = draft_size.clone();
    let draft_progress = progress.clone();
    ensure_draft_repo(dspine::REPO, move |done, total| {
        draft_completed.store(total, Ordering::Relaxed);
        draft_progress(done, total.saturating_add(target_size));
    })?;
    let draft_bytes = draft_size.load(Ordering::Relaxed);
    hf::download_repo_with_profile(repo, profile, move |done, total| {
        progress(
            draft_bytes.saturating_add(done),
            draft_bytes.saturating_add(total),
        );
    })
}
