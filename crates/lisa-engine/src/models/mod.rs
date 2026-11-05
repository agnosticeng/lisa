//! Model implementations and dispatch.
//!
//! The generation / batching runtime drives any model through [`LanguageModel`].
//! A model owns its config, its weights and its caches; the runtime owns
//! sampling, scheduling and the HTTP surface.
//!
//! A model is addressed by a local directory, supported GGUF file or Hub repo
//! ID. [`resolve_model_dir`] resolves its checkpoint, [`model_type_of`] reads
//! metadata, and [`ModelKind`] selects the implementation for [`load`].
//!
//! Device backends are a separate axis (see `lisa_mlx`): the runtime talks to
//! an `Array`/`Stream` surface, and a backend supplies it.

mod checkpoint;
pub use checkpoint::{ModelKind, model_type_of, resolve_model_dir};

pub mod deepseek41;
pub mod flux2;
pub mod flux3;
pub mod gdn;
pub mod glm53;
pub mod hf;
pub mod image_generation;
pub mod laya;
mod media_attention;
pub mod qwen3_5;
pub mod qwen4;
pub mod speculate;
pub mod vision;

use std::path::Path;

use lisa_mlx::Array;
use lisa_mlx::ops::indexing::IndexOp;

use crate::core::cache::{IndexerTape, LayerCache};

/// The interface the runtime needs from a decoder-only text model.
pub trait LanguageModel {
    /// Reference sampler order/RNG is independent of whether a draft head is
    /// present in this checkpoint.
    fn uses_ds4_sampling(&self) -> bool {
        self.has_target_mtp()
    }
    /// Whether the checkpoint has a supported native still-image encoder.
    fn supports_images(&self) -> bool {
        false
    }
    /// Prepare one complete image prompt. Callers use a fresh session, without
    /// token-only prefix caching or speculative decoding, then clear this state.
    fn prepare_images(
        &mut self,
        _tokens: &mut Vec<u32>,
        _images: &[image::RgbImage],
    ) -> anyhow::Result<()> {
        anyhow::bail!("this checkpoint does not support image input")
    }
    /// Remove request-local image embeddings and positional context.
    fn clear_images(&mut self) {}
    /// Explicit arithmetic identity for cached model state, available before
    /// any lazy pipeline compilation. None preserves the runtime's legacy law.
    fn cache_numerical_law(&self) -> Option<u64> {
        None
    }
    /// Effective sequence limit (prompt + generated), including runtime capacity.
    fn max_position_embeddings(&self) -> usize;
    /// Configure the runtime context limit before warmup or creating sessions.
    /// Models that support this report the effective limit through
    /// `max_position_embeddings`; other models keep their existing behavior.
    fn set_context_capacity(&mut self, capacity: usize) -> anyhow::Result<()> {
        anyhow::bail!(
            "runtime context capacity {capacity} is not supported by this model; omit --ctx-size"
        )
    }
    /// End-of-sequence token id.
    fn eos_token_id(&self) -> i64;
    /// All token ids that end generation. Models with a multi-token stop set
    /// override this while keeping `eos_token_id` as the legacy primary id.
    fn eos_token_ids(&self) -> Vec<i64> {
        vec![self.eos_token_id()]
    }
    /// Fresh per-layer caches for one stream.
    fn new_caches(&self) -> Vec<LayerCache>;
    /// One forward over `ids [B,S]`; returns `(mixed [B,S,H], multi)`.
    fn forward(
        &mut self,
        ids: &Array,
        caches: Option<&mut Vec<LayerCache>>,
    ) -> anyhow::Result<(Array, Array)>;
    /// Forward one decode token for independently positioned streams that keep
    /// separate per-row caches. Architectures with row-local sparse state may
    /// override this to batch the stateless tower operations while advancing
    /// each row's cache at its own absolute position.
    fn forward_ragged(
        &mut self,
        ids: &Array,
        per_row_caches: &mut [Vec<LayerCache>],
    ) -> anyhow::Result<(Array, Array)> {
        anyhow::ensure!(
            ids.dim(0) as usize == per_row_caches.len() && ids.dim(1) == 1,
            "ragged forward expects one token and cache per row, got ids {:?}, caches {}",
            ids.shape(),
            per_row_caches.len()
        );
        let n = per_row_caches.len();
        let mut mixed_rows = Vec::with_capacity(n);
        let mut multi_rows = Vec::with_capacity(n);
        let tails = self.context_tails();
        let mut next_tails = Vec::with_capacity(per_row_caches.len());
        for (row, caches) in per_row_caches.iter_mut().enumerate() {
            if tails.len() == n {
                self.set_context_tails(vec![tails[row].clone()]);
            }
            let ids_row = ids.index((row as i32..row as i32 + 1, ..));
            let (mixed, multi) = self.forward(&ids_row, Some(caches))?;
            mixed_rows.push(mixed);
            multi_rows.push(multi);
            let updated = self.context_tails();
            if updated.len() == 1 {
                next_tails.push(updated[0].clone());
            }
        }
        if next_tails.len() == n {
            self.set_context_tails(next_tails);
        } else if tails.len() == per_row_caches.len() {
            self.set_context_tails(tails);
        }
        let mixed_refs: Vec<&Array> = mixed_rows.iter().collect();
        let multi_refs: Vec<&Array> = multi_rows.iter().collect();
        Ok((
            Array::concatenate(&mixed_refs, 0)?,
            Array::concatenate(&multi_refs, 0)?,
        ))
    }
    /// Forward with the speculative-verify capture enabled.
    fn forward_capture(
        &mut self,
        ids: &Array,
        caches: Option<&mut Vec<LayerCache>>,
        capture: bool,
    ) -> anyhow::Result<(Array, Array)>;
    /// Install a layer capture for targeted model-parity tests.
    #[cfg(test)]
    fn set_debug_capture_for_test(&mut self, _layers: Vec<usize>) -> anyhow::Result<()> {
        anyhow::bail!("debug capture is not supported by this model")
    }
    /// Install a test capture limited to selected layer keys.
    #[cfg(test)]
    fn set_debug_capture_filtered_for_test(
        &mut self,
        layers: Vec<usize>,
        _key_filters: Vec<String>,
    ) -> anyhow::Result<()> {
        self.set_debug_capture_for_test(layers)
    }
    /// Flush the targeted model-parity capture to an NPZ file.
    #[cfg(test)]
    fn flush_debug_capture_for_test(&mut self, _path: &str) -> anyhow::Result<()> {
        anyhow::bail!("debug capture is not supported by this model")
    }
    /// Record output-head vectors for a diagnostic parity run. Production
    /// dispatch remains active; the caller excludes this run from timings.
    fn enable_logit_capture(&mut self) -> anyhow::Result<()> {
        anyhow::bail!("logit capture is not supported by this model")
    }
    /// Record selected internal stages while retaining production dispatch.
    fn enable_stage_capture(
        &mut self,
        _layers: Vec<usize>,
        _keys: Vec<String>,
    ) -> anyhow::Result<()> {
        anyhow::bail!("stage capture is not supported by this model")
    }
    /// Write the diagnostic vectors after generation has completed.
    fn write_logit_capture(&self, _path: &str) -> anyhow::Result<()> {
        anyhow::bail!("logit capture is not supported by this model")
    }
    /// Prefill `tokens`; returns the last row's hidden state.
    fn prefill(&mut self, tokens: &[u32], caches: &mut Vec<LayerCache>) -> anyhow::Result<Array>;
    /// Commit one generated token; returns its last hidden state. Models with
    /// a distinct generation graph override this without changing prompt or
    /// suffix prefill, including one-token prompt chunks.
    fn decode_token(&mut self, token: u32, caches: &mut Vec<LayerCache>) -> anyhow::Result<Array> {
        self.prefill(&[token], caches)
    }
    /// Prefill `tokens`; returns the last hidden state and the multi stream.
    fn prefill_multi(
        &mut self,
        tokens: &[u32],
        caches: &mut Vec<LayerCache>,
    ) -> anyhow::Result<(Array, Array)>;
    /// The output head: hidden `[.., H]` -> logits `[.., vocab]`.
    fn head(&self, mixed: &Array) -> lisa_mlx::error::Result<Array>;
    /// Apply the output head to independently cached decode rows. A model may
    /// override this when its reference engine falls back from native batch
    /// execution to a row-serial graph at a context or batch-size boundary.
    fn head_ragged(
        &self,
        mixed: &Array,
        _per_row_caches: &[Vec<LayerCache>],
    ) -> lisa_mlx::error::Result<Array> {
        self.head(mixed)
    }
    /// Warm the kernels for the scored shapes.
    fn warmup(&mut self, seed: &[u32]) -> anyhow::Result<()>;

    // --- optional host-side rolling context (e.g. the PLE n-gram window) ---
    /// Whether B>1 batched decode is safe on this tower. `false` clamps the
    /// scheduler to lone streams (the exact single-stream path).
    fn batch_decode_ok(&self) -> bool {
        true
    }
    /// Some batch-capable towers require every live row to share one absolute
    /// cache position. The scheduler then admits only matching prompt lengths.
    fn batch_decode_equal_length_only(&self) -> bool {
        false
    }
    /// Whether independently positioned streams can share a decode batch
    /// while retaining row-local cache state.
    fn batch_decode_ragged_supported(&self) -> bool {
        false
    }
    /// O3: bytes of KV a single token occupies for one stream (0 = unknown, in
    /// which case admission is skipped rather than guessed).
    fn kv_bytes_per_token(&self) -> usize {
        0
    }
    /// O3: fixed per-stream non-KV state bytes (GDN conv/SSM, indexer tape).
    fn stream_state_bytes(&self) -> usize {
        0
    }
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
    /// Whether the model has a target-hidden MTP head (GLM 5.3). Unlike the
    /// independent Qwen draft chain, this head must be advanced alongside
    /// target hidden rows and is driven by `target_mtp_step`.
    fn has_target_mtp(&self) -> bool {
        false
    }
    /// One target-conditioned draft step at an absolute target position.
    fn target_mtp_step(
        &mut self,
        _cache: &mut IndexerTape,
        _target_multi: &Array,
        _next_token: u32,
        _position: usize,
        _min_position: usize,
    ) -> anyhow::Result<Array> {
        anyhow::bail!("this model has no target-hidden MTP head")
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
    /// Take the most recent `draft_step`'s head confidence: log p_head of the
    /// token it proposed, as an UNEVALUATED `[1]` array (the chunk-A sync
    /// evals these — port spec §2.3; reads the exact re-score row the step
    /// already produced, no new buffer). `None` = the drafter does not expose
    /// one; the two-chunk machinery stays off for that model.
    fn take_draft_confidence(&mut self) -> Option<Array> {
        None
    }
    /// Arm/disarm the per-step confidence capture (an enqueue-time cost: only
    /// the two-chunk rounds need it, so capture stays off otherwise and the
    /// certified paths stay byte-identical in dispatch shape).
    fn set_conf_capture(&mut self, _on: bool) {}
    /// Stable per-architecture row key for the round-cost table (shape, never
    /// a repo id or snapshot hash), e.g. `qwen3_5-64L-5120h`.
    fn mtp_cost_key(&self) -> String {
        String::new()
    }
}

/// A model kind the runtime can load.
pub enum Loaded {
    /// A decoder-only text model (generation, batching, speculative decode).
    Language(Box<dyn LanguageModel>),
    /// A separate text-to-image model, driven by image generation requests.
    Image(Box<image_generation::Pipeline>),
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
    fn system_one(
        &self,
        _state: &serde_json::Value,
        _questions: &serde_json::Value,
    ) -> anyhow::Result<serde_json::Value> {
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
    let path = hf::checkpoint_path(dir.to_path_buf());
    let dir = path.as_path();
    let model_type = model_type_of(dir)?;
    let kind = ModelKind::from_model_type(&model_type)?;
    anyhow::ensure!(
        kind == ModelKind::Laya || backend == lisa_mlx::backend::Backend::Metal,
        "{model_type} requires the Metal backend"
    );
    // Model switching drops the previous owner first; release its pooled
    // buffers before the replacement reserves its resident weights. The CPU
    // decision path must not initialize a Metal runtime just to sweep it.
    if backend == lisa_mlx::backend::Backend::Metal {
        lisa_mlx::runtime::check_platform_support()?;
        lisa_mlx::memory::clear_cache();
    }
    Ok(match kind {
        ModelKind::Glm53Gguf | ModelKind::DeepSeek41 => {
            anyhow::ensure!(
                dir.is_file(),
                "{model_type} requires a GGUF checkpoint file"
            );
            Loaded::Language(match kind {
                ModelKind::Glm53Gguf => Box::new(glm53::gguf::tower::Tower::load(dir)?),
                _ => Box::new(deepseek41::tower::Tower::load(dir)?),
            })
        }
        ModelKind::Flux2 | ModelKind::Flux3 => {
            Loaded::Image(Box::new(image_generation::Pipeline::load(dir)?))
        }
        ModelKind::Qwen4 => {
            let config = qwen4::ModelConfig::from_json(&dir.join("config.json"))?;
            Loaded::Language(Box::new(qwen4::Tower::load(dir, config)?))
        }
        ModelKind::Glm53 => {
            let config = glm53::config::ModelConfig::from_json(&dir.join("config.json"))?;
            Loaded::Language(Box::new(glm53::tower::Tower::load(dir, config)?))
        }
        ModelKind::Qwen35 => {
            let config = qwen3_5::Qwen35Config::from_json(&dir.join("config.json"))?;
            Loaded::Language(Box::new(qwen3_5::Qwen35Tower::load(dir, config)?))
        }
        ModelKind::Laya => {
            let device = match backend {
                lisa_mlx::backend::Backend::Metal => laya::LayaDevice::Metal,
                lisa_mlx::backend::Backend::Cpu => laya::LayaDevice::Cpu,
            };
            Loaded::Decision(Box::new(laya::Laya::load_device(dir, device)?))
        }
    })
}
