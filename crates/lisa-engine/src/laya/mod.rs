//! Laya: a non-autoregressive **typed-decision** model.
//!
//! A ModernBERT encoder plus a small decision head (an option-marker scorer and
//! an act head). Given a *state* and *typed questions*, one forward pass returns
//! calibrated answers — there is no generation. The checkpoint keeps the
//! original Hugging Face weights (`model.safetensors`, fp16/fp32) and its
//! encoder/agent configs beside a `tokenizer/` directory.
//!
//! This module is the loading foundation: resolve/parse/validate the original
//! checkpoint (config + weights + tokenizer) and expose it through
//! [`DecisionModel`]. The engine executes plans from `lisa-models` through the explicitly supplied device contract.

pub mod config;
mod execution;
pub mod prompt;

use std::path::{Path, PathBuf};

use crate::core::tokenizer::Tokenizer;
use crate::decision::DecisionModel;
use lisa_mlx::Shard;

pub use config::LayaConfig;
use config::LayaConfigIo;

/// A loaded Laya checkpoint.
pub struct Laya {
    dir: PathBuf,
    config: LayaConfig,
    tokenizer: Tokenizer,
    weights: Shard,
    /// Tensor names in the original `model.safetensors`.
    tensor_names: Vec<String>,

    execution: Box<dyn execution::Execution>,
}

impl Laya {
    /// Bind the same checkpoint and model plans to an explicitly supplied device.
    pub fn load_on<D>(
        dir: &Path,
        device: D,
        arithmetic: lisa_models::laya::plan::Arithmetic,
    ) -> anyhow::Result<Self>
    where
        D: lisa_backend::ReadbackDevice + Clone + 'static,
        D::Tensor: Clone,
    {
        let mut config = LayaConfig::from_dir(dir)?;

        let tokenizer = Tokenizer::load_file(&dir.join("tokenizer/tokenizer.json"))
            .map_err(|e| anyhow::anyhow!("laya tokenizer: {e}"))?;
        config.mask_id = tokenizer
            .inner
            .token_to_id(&config.mask_token)
            .ok_or_else(|| anyhow::anyhow!("laya: token {} not in tokenizer", config.mask_token))?;

        let weights = open_weights(dir)?;
        let mut tensor_names: Vec<String> = weights.tensor_names().cloned().collect();
        tensor_names.sort();
        verify_encoder_tensors(&weights, &config)?;

        Ok(Self {
            dir: dir.to_path_buf(),
            config,
            tokenizer,
            weights,
            tensor_names,
            execution: Box::new(execution::Bound::new(device, arithmetic)),
        })
    }

    /// One forward, dispatched on the device.
    pub fn decide(
        &self,
        ids: &[u32],
        qtype: i32,
        marker_pos: &[i32],
    ) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        self.execution.forward(self, ids, qtype, marker_pos)
    }

    /// Apply CLI/request overrides onto the loaded config (see
    /// [`config::LayaConfig::apply_overrides`]).
    ///
    /// `temperature` (if given) is `[choice, score, noul]`; `by_options` entries
    /// are `(bucket, temp)`. Precedence: `by_options` > `temperature` > config.
    pub fn apply_overrides(
        &mut self,
        head_max_len: Option<usize>,
        max_len: Option<usize>,
        temperature: Option<Vec<f32>>,
        by_options: &[(String, f32)],
    ) -> anyhow::Result<()> {
        self.config
            .apply_overrides(head_max_len, max_len, temperature, by_options)
    }

    /// The parsed configuration.
    pub fn config(&self) -> &LayaConfig {
        &self.config
    }

    /// The tokenizer (Metaspace pretokenizer; special ids resolved).
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// The original-weight shard.
    pub fn weights(&self) -> &Shard {
        &self.weights
    }

    /// Number of tensors in the checkpoint.
    pub fn num_tensors(&self) -> usize {
        self.tensor_names.len()
    }

    /// A human-readable structural summary for diagnostics.
    pub fn summary(&self) -> String {
        format!(
            "{}\n  dir: {}\n  tensors: {}",
            self.config.summary(),
            self.dir.display(),
            self.tensor_names.len()
        )
    }
}

impl DecisionModel for Laya {
    fn name(&self) -> &str {
        "laya"
    }
    fn dir(&self) -> &Path {
        &self.dir
    }
    fn summary(&self) -> String {
        Laya::summary(self)
    }
    fn decide(
        &self,
        ids: &[u32],
        qtype: i32,
        marker_pos: &[i32],
    ) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        Laya::decide(self, ids, qtype, marker_pos)
    }
    fn system_one(
        &self,
        state: &serde_json::Value,
        questions: &serde_json::Value,
    ) -> anyhow::Result<serde_json::Value> {
        Laya::system_one(self, state, questions)
    }
}

/// Open the checkpoint weights: a single `model.safetensors`, else the sharded
/// index (the original HF layout).
fn open_weights(dir: &Path) -> anyhow::Result<Shard> {
    let single = dir.join("model.safetensors");
    if single.is_file() {
        return Shard::open(&single);
    }
    anyhow::bail!(
        "laya: no model.safetensors under {} (sharded checkpoints not handled yet)",
        dir.display()
    );
}

/// Check that the original-name tensors the forward will need are present.
///
/// Original layout: `encoder.*` (ModernBERT), `head.layers.{0,1}.*` (decision
/// head), `scorer.*` (option-marker scorer), `act_head.*`, `type_emb.weight`.
/// Layer 0 shares `encoder.embeddings.norm` as its attention norm; later
/// layers carry `attn_norm`/`mlp_norm`.
fn verify_encoder_tensors(shard: &Shard, cfg: &LayaConfig) -> anyhow::Result<()> {
    let emb = "encoder.embeddings.tok_embeddings.weight";
    let (_, dtype, shape) = shard
        .tensor_bytes(emb)
        .ok_or_else(|| anyhow::anyhow!("laya: missing {emb}"))?;
    anyhow::ensure!(
        shape == [cfg.vocab_size as i32, cfg.hidden_size as i32],
        "laya: {emb} shape {shape:?}, expected [{}, {}]",
        cfg.vocab_size,
        cfg.hidden_size
    );
    anyhow::ensure!(
        matches!(
            dtype,
            lisa_mlx::Dtype::Float16 | lisa_mlx::Dtype::Float32 | lisa_mlx::Dtype::Bfloat16
        ),
        "laya: {emb} has unsupported dtype {dtype:?}"
    );

    let required = lisa_models::laya::weights::required_tensors(cfg);
    let missing: Vec<String> = required
        .into_iter()
        .filter(|n| shard.tensor_bytes(n).is_none())
        .take(5)
        .collect();
    anyhow::ensure!(missing.is_empty(), "laya: missing tensors {missing:?}");
    Ok(())
}

#[cfg(test)]
mod plan_tests;
