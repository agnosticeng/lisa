//! Laya: a non-autoregressive **typed-decision** model.
//!
//! A ModernBERT encoder plus a small decision head (an option-marker scorer and
//! an act head). Given a *state* and *typed questions*, one forward pass returns
//! calibrated answers — there is no generation. The checkpoint keeps the
//! original Hugging Face weights (`model.safetensors`, fp16/fp32) and its
//! encoder/agent configs beside a `tokenizer/` directory.
//!
//! Execution receives prepared tensors and metadata from a loading adapter.
//! It runs plans from `lisa-models` through the explicitly supplied device contract.

pub mod config;
mod execution;
pub mod prompt;

use std::path::{Path, PathBuf};

use crate::core::tokenizer::Tokenizer;
use crate::decision::DecisionModel;

pub use config::LayaConfig;
use config::LayaOverrides;

/// A loaded Laya checkpoint.
pub struct Laya {
    dir: PathBuf,
    config: LayaConfig,
    tokenizer: Tokenizer,
    /// Diagnostic names supplied by the loading adapter.
    tensor_names: Vec<String>,

    execution: Box<dyn execution::Execution>,
}

impl Laya {
    /// Bind prepared tensor access and model metadata to a device.
    pub fn from_prepared<D>(
        dir: PathBuf,
        config: LayaConfig,
        tokenizer: Tokenizer,
        tensor_names: Vec<String>,
        question_types: i32,
        weights: Box<dyn crate::weights::source::TensorSource<D::Tensor>>,
        device: D,
        arithmetic: lisa_models::laya::plan::Arithmetic,
    ) -> Self
    where
        D: lisa_backend::ReadbackDevice + Clone + 'static,
        D::Tensor: Clone,
    {
        Self {
            dir,
            config,
            tokenizer,
            tensor_names,
            execution: Box::new(execution::Bound::new(
                device,
                arithmetic,
                weights,
                question_types,
            )),
        }
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

#[cfg(test)]
mod plan_tests;
