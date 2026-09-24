//! Model implementations and dispatch.
//!
//! The generation / batching runtime drives any model through [`LanguageModel`].
//! A model owns its config, its weights and its caches; the runtime owns
//! sampling, scheduling and the HTTP surface.
//!
//! A model is addressed by a **local directory or a Hugging Face repo id**.
//! [`resolve_model_dir`] maps either to a local directory, [`model_type_of`]
//! reads the structure from its `config.json`, and [`load`] dispatches on that
//! type to the right implementation. Qwen 3.8 Flash-Next ([`qwen4`]) is the
//! first implementation; add a variant by registering its `model_type` in
//! [`load`].
//!
//! Device backends are a separate axis (see `lisa_mlx`): the runtime talks to
//! an `Array`/`Stream` surface, and a backend supplies it.

pub mod laya;
pub mod qwen4;

use std::path::{Path, PathBuf};

use lisa_mlx::Array;

use crate::core::cache::LayerCache;

/// The interface the runtime needs from a decoder-only text model.
pub trait LanguageModel {
    /// Maximum sequence length (prompt + generated).
    fn max_position_embeddings(&self) -> usize;
    /// End-of-sequence token id.
    fn eos_token_id(&self) -> i64;
    /// Fresh per-layer caches for one stream.
    fn new_caches(&self) -> Vec<LayerCache>;
    /// One forward over `ids [B,S]`; returns `(mixed [B,S,H], multi)`.
    fn forward(
        &mut self,
        ids: &Array,
        caches: Option<&mut Vec<LayerCache>>,
    ) -> anyhow::Result<(Array, Array)>;
    /// Forward with the speculative-verify capture enabled.
    fn forward_capture(
        &mut self,
        ids: &Array,
        caches: Option<&mut Vec<LayerCache>>,
        capture: bool,
    ) -> anyhow::Result<(Array, Array)>;
    /// Prefill `tokens`; returns the last row's hidden state.
    fn prefill(&mut self, tokens: &[u32], caches: &mut Vec<LayerCache>) -> anyhow::Result<Array>;
    /// Prefill `tokens`; returns the last hidden state and the multi stream.
    fn prefill_multi(
        &mut self,
        tokens: &[u32],
        caches: &mut Vec<LayerCache>,
    ) -> anyhow::Result<(Array, Array)>;
    /// The output head: hidden `[.., H]` -> logits `[.., vocab]`.
    fn head(&self, mixed: &Array) -> lisa_mlx::error::Result<Array>;
    /// Warm the kernels for the scored shapes.
    fn warmup(&mut self, seed: &[u32]) -> anyhow::Result<()>;

    // --- optional host-side rolling context (e.g. the PLE n-gram window) ---
    /// Rolling context window size (0 = the model carries none).
    fn context_window(&self) -> usize {
        0
    }
    /// The current per-stream context tails.
    fn context_tails(&self) -> Vec<Vec<i64>> {
        Vec::new()
    }
    /// Replace the per-stream context tails.
    fn set_context_tails(&mut self, _tails: Vec<Vec<i64>>) {}
    /// Drop the rolling context.
    fn clear_context(&mut self) {}

    // --- optional speculative drafting (e.g. the MTP head) ---
    /// Whether this model can draft speculative tokens.
    fn has_drafter(&self) -> bool {
        false
    }
    /// Reset the drafter's own caches.
    fn drafter_reset(&mut self) {}
    /// Trim the drafter's caches by `n` rows (rollback).
    fn drafter_trim(&mut self, _n: usize) {}
    /// The drafter's cache length in rows (pairs consumed so far).
    fn drafter_offset(&self) -> usize {
        0
    }
    /// Rewind the drafter's cache to a previously captured offset (prefix
    /// snapshot restore).
    fn drafter_restore_offset(&mut self, _n: usize) {}
    /// One draft step over `tokens [1,S]` and `multi [1,S,hc*H]`: returns the
    /// drafted token id and the last multi-stream row for the next step.
    fn draft_step(&mut self, _tokens: &Array, _multi: &Array) -> anyhow::Result<(Array, Array)> {
        anyhow::bail!("this model has no drafter")
    }
}

/// A model kind the runtime can load.
pub enum Loaded {
    /// A decoder-only text model (generation, batching, speculative decode).
    Language(Box<dyn LanguageModel>),
    /// A non-generative model with its own interface (e.g. typed decisions).
    Decision(Box<dyn DecisionModel>),
}

/// The interface a non-generative (encoder/decision) model exposes.
///
/// These models are not driven by the generation runtime; a caller addresses
/// them through their own API (e.g. `lisa decide`). This trait is the loading
/// seam, so a decision model can be resolved and dispatched like any other.
pub trait DecisionModel {
    /// The model family name (`"laya"`).
    fn name(&self) -> &str;
    /// The resolved model directory.
    fn dir(&self) -> &Path;
    /// A human-readable structural summary for diagnostics.
    fn summary(&self) -> String;
    /// One decision forward: `(logits per marker, action)`.
    fn decide(
        &self,
        ids: &[u32],
        qtype: i32,
        marker_pos: &[i32],
    ) -> anyhow::Result<(Vec<f32>, Vec<f32>)>;
    /// Run a `state` + typed `questions` request and return the answer JSON.
    fn system_one(&self, _state: &serde_json::Value, _questions: &serde_json::Value) -> anyhow::Result<serde_json::Value> {
        anyhow::bail!("this decision model has no typed-question interface")
    }
}

/// Load a model from a local directory or a Hugging Face repo id, dispatching
/// on the structure read from its config.
pub fn load(target: &str) -> anyhow::Result<Loaded> {
    load_dir(&resolve_model_dir(target)?)
}

/// Load a model from an already-resolved directory, on the selected backend.
pub fn load_dir(dir: &Path) -> anyhow::Result<Loaded> {
    let backend = lisa_mlx::backend::Backend::from_env().map_err(|e| anyhow::anyhow!("{e}"))?;
    load_dir_with(dir, backend)
}

/// Load a model from an already-resolved directory on an explicit backend.
pub fn load_dir_with(dir: &Path, backend: lisa_mlx::backend::Backend) -> anyhow::Result<Loaded> {
    let model_type = model_type_of(dir)?;
    match model_type.as_str() {
        // Qwen 3.8 Flash-Next (hybrid attention + GDN, sparse MoE, PLE, MTP).
        "qwen4_exp" | "qwen4_exp_text" => {
            anyhow::ensure!(
                backend == lisa_mlx::backend::Backend::Metal,
                "the qwen4 model requires the metal backend (got {})",
                backend.name()
            );
            let config = qwen4::ModelConfig::from_json(&dir.join("config.json"))?;
            Ok(Loaded::Language(Box::new(qwen4::Tower::load(dir, config)?)))
        }
        // Laya (ModernBERT encoder + typed-decision head; non-generative).
        "laya" => {
            let device = match backend {
                lisa_mlx::backend::Backend::Metal => laya::LayaDevice::Metal,
                lisa_mlx::backend::Backend::Cpu => laya::LayaDevice::Cpu,
            };
            Ok(Loaded::Decision(Box::new(laya::Laya::load_device(dir, device)?)))
        }
        other => anyhow::bail!(
            "unsupported model_type {other:?} in {}; supported: qwen4_exp, laya",
            dir.display()
        ),
    }
}

/// Resolve `target` to a local directory holding `config.json`.
///
/// A path that exists is used as-is. Otherwise `target` is treated as a
/// Hugging Face repo id and resolved against, in order: `$LISA_MODEL_DIR`
/// (else `~/.cache/lisa-models`) by the repo's short name, the Hugging Face
/// hub cache, then `hf download` into `$LISA_MODEL_DIR`.
pub fn resolve_model_dir(target: &str) -> anyhow::Result<PathBuf> {
    let direct = Path::new(target);
    if direct.is_dir() {
        return Ok(direct.to_path_buf());
    }
    if direct.exists() {
        anyhow::bail!("{} exists but is not a directory", direct.display());
    }

    let short = target.rsplit('/').next().unwrap_or(target);
    let lisa_cache = lisa_model_dir();
    let local = lisa_cache.join(short);
    if local.is_dir() {
        return Ok(local);
    }
    if let Some(dir) = hf_snapshot_dir(target) {
        return Ok(dir);
    }

    // Not present locally: fetch it with the Hugging Face CLI.
    if which("hf").is_none() {
        anyhow::bail!(
            "model {target:?} is not local and `hf` is not on PATH; \
             install huggingface_hub or pass a local path"
        );
    }
    std::fs::create_dir_all(&lisa_cache)?;
    let status = std::process::Command::new("hf")
        .args(["download", target, "--local-dir"])
        .arg(&local)
        .status()?;
    anyhow::ensure!(status.success(), "`hf download {target}` failed");
    Ok(local)
}

/// The `model_type` string from `dir/config.json`. The nested
/// `text_config.model_type` wins when present.
pub fn model_type_of(dir: &Path) -> anyhow::Result<String> {
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

/// The lisa model cache root: `$LISA_MODEL_DIR`, else `~/.cache/lisa-models`.
fn lisa_model_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("LISA_MODEL_DIR") {
        return PathBuf::from(dir);
    }
    home_dir().join(".cache").join("lisa-models")
}

/// A snapshot directory in the Hugging Face hub cache that holds a `config.json`.
fn hf_snapshot_dir(repo: &str) -> Option<PathBuf> {
    let hub = match std::env::var_os("HF_HOME") {
        Some(h) => PathBuf::from(h).join("hub"),
        None => home_dir().join(".cache").join("huggingface").join("hub"),
    };
    let slug = format!("models--{}", repo.replace('/', "--"));
    let snapshots = hub.join(slug).join("snapshots");
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&snapshots)
        .ok()?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("config.json").is_file())
        .collect();
    entries.sort();
    entries.pop()
}

fn home_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Whether `name` resolves on `PATH`.
fn which(name: &str) -> Option<PathBuf> {
    let paths = std::env::var_os("PATH")?;
    std::env::split_paths(&paths)
        .map(|p| p.join(name))
        .find(|p| p.is_file())
}