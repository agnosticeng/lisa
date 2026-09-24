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
//! [`DecisionModel`]. The encoder forward and the decision head are the next
//! step.

pub mod config;
pub mod cpu;
pub mod forward;
pub mod prompt;

use std::path::{Path, PathBuf};

use crate::core::loader::Shard;
use crate::core::tokenizer::Tokenizer;
use crate::models::DecisionModel;

pub use config::LayaConfig;

/// Where Laya's forward runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayaDevice {
    /// Metal (GPU) via `lisa_mlx`.
    Metal,
    /// Host f32 (portable; no Metal dependency at run time).
    Cpu,
}

/// A loaded Laya checkpoint.
pub struct Laya {
    dir: PathBuf,
    config: LayaConfig,
    tokenizer: Tokenizer,
    weights: Shard,
    /// Tensor names in the original `model.safetensors`.
    tensor_names: Vec<String>,
    device: LayaDevice,
}

impl Laya {
    /// Load a resolved Laya checkpoint directory (original HF weights) on Metal.
    pub fn load(dir: &Path) -> anyhow::Result<Self> {
        Self::load_device(dir, LayaDevice::Metal)
    }

    /// Load a resolved Laya checkpoint on `device`.
    pub fn load_device(dir: &Path, device: LayaDevice) -> anyhow::Result<Self> {
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

        Ok(Self { dir: dir.to_path_buf(), config, tokenizer, weights, tensor_names, device })
    }

    /// The device this instance runs on.
    pub fn device(&self) -> LayaDevice {
        self.device
    }

    /// One forward, dispatched on the device.
    pub fn decide(
        &self,
        ids: &[u32],
        qtype: i32,
        marker_pos: &[i32],
    ) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
        match self.device {
            LayaDevice::Metal => self.forward_metal(ids, qtype, marker_pos),
            LayaDevice::Cpu => self.forward_cpu(ids, qtype, marker_pos),
        }
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
            "{}\n  dir: {}\n  device: {:?}\n  tensors: {}",
            self.config.summary(),
            self.dir.display(),
            self.device,
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
        matches!(dtype, lisa_mlx::Dtype::Float16 | lisa_mlx::Dtype::Float32 | lisa_mlx::Dtype::Bfloat16),
        "laya: {emb} has unsupported dtype {dtype:?}"
    );

    let mut required = vec![
        "encoder.embeddings.norm.weight".to_string(),
        "encoder.final_norm.weight".to_string(),
        "type_emb.weight".to_string(),
        "scorer.0.weight".to_string(),
        "act_head.0.weight".to_string(),
    ];
    for i in 0..cfg.num_layers {
        let l = format!("encoder.layers.{i}");
        required.push(format!("{l}.attn.Wqkv.weight"));
        required.push(format!("{l}.attn.Wo.weight"));
        required.push(format!("{l}.mlp.Wi.weight"));
        required.push(format!("{l}.mlp.Wo.weight"));
        required.push(format!("{l}.mlp_norm.weight"));
        if i > 0 {
            required.push(format!("{l}.attn_norm.weight"));
        }
    }
    for i in 0..cfg.head_layers {
        for t in ["norm1.weight", "norm2.weight", "self_attn.in_proj_weight", "self_attn.out_proj.weight", "linear1.weight", "linear2.weight"] {
            required.push(format!("head.layers.{i}.{t}"));
        }
    }
    let missing: Vec<String> = required
        .into_iter()
        .filter(|n| shard.tensor_bytes(n).is_none())
        .take(5)
        .collect();
    anyhow::ensure!(missing.is_empty(), "laya: missing tensors {missing:?}");
    Ok(())
}