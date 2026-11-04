//! Native scalar FFN graph for the original mixed GGUF expert formats.
//! The owning model retains ResidentWeights for this object's full lifetime.

use super::{config::Config, projection};
use crate::core::gguf_resident::{PackedTensor, ResidentWeights};
use anyhow::{Context, Result, ensure};
use lisa_mlx::models::deepseek41::{self as gpu, ActivationFormat};
use lisa_mlx::{Array, Stream};

pub struct Moe {
    router: PackedTensor,
    bias: Array,
    gate: PackedTensor,
    up: PackedTensor,
    down: PackedTensor,
    shared_gate: PackedTensor,
    shared_up: PackedTensor,
    shared_down: PackedTensor,
}

pub struct Output {
    pub routed: Array,
    pub shared: Array,
    /// Remains GPU resident; available to explicit numerical captures.
    pub routing: gpu::RouterOutput,
}

impl Moe {
    pub fn load(weights: &ResidentWeights, layer: usize, config: &Config) -> Result<Self> {
        ensure!(layer < config.layers, "invalid DeepSeek FFN layer");
        let get = |suffix: &str| {
            weights
                .tensors
                .get(&format!("blk.{layer}.{suffix}"))
                .cloned()
                .with_context(|| format!("missing DeepSeek FFN tensor {suffix}"))
        };
        let gate = get("ffn_gate_exps.weight")?;
        let up = get("ffn_up_exps.weight")?;
        let down = get("ffn_down_exps.weight")?;
        for weight in [&gate, &up, &down] {
            projection::block_format(weight.info.kind)?;
        }
        ensure!(
            gate.info.dimensions == [5120, 2304, config.experts as u64]
                && up.info.dimensions == gate.info.dimensions
                && down.info.dimensions == [2304, 5120, config.experts as u64],
            "invalid DeepSeek routed expert geometry"
        );
        Ok(Self {
            router: get("ffn_gate_inp.weight")?,
            bias: projection::plain_view(&get("exp_probs_b.bias")?)?,
            gate,
            up,
            down,
            shared_gate: get("ffn_gate_shexp.weight")?,
            shared_up: get("ffn_up_shexp.weight")?,
            shared_down: get("ffn_down_shexp.weight")?,
        })
    }

    /// Scalar decode with DS4's original router, BF16 shared-expert boundaries,
    /// routed activation weighting, and ordered six-expert sum. Formats absent
    /// from DS4 use the adjacent pinned GGML matvec extension.
    pub fn forward(&self, norm: &Array, stream: &Stream) -> Result<Output> {
        ensure!(
            norm.rank() == 2 && (1..=8).contains(&norm.dim(0)) && norm.dim(1) == 5120,
            "DeepSeek FFN expects one to eight normalized rows"
        );
        let rows = norm.dim(0);
        let logits = projection::linear(&self.router, norm, false, stream)?;
        let routing = gpu::router(&logits, &self.bias, 6, 1.5, stream)?;
        let shared_gate = projection::linear(&self.shared_gate, norm, true, stream)?;
        let shared_up = projection::linear(&self.shared_up, norm, true, stream)?;
        // The original flat shared activation is pointwise. Flattening rows
        // retains that kernel's code generation, including its clamp order.
        let mut shared_mid = gpu::shared_swiglu(
            &shared_gate.reshape(&[1, rows * 2304])?,
            &shared_up.reshape(&[1, rows * 2304])?,
            stream,
        )?
        .reshape(&[rows, 2304])?;
        gpu::quantize_in_place(&mut shared_mid, ActivationFormat::Bf16, stream)?;
        let shared = projection::linear(&self.shared_down, &shared_mid, true, stream)?;
        let input = norm.reshape(&[rows, 1, 5120])?;
        let gate = projection::experts(&self.gate, &input, &routing.selected, stream)?;
        let up = projection::experts(&self.up, &input, &routing.selected, stream)?;
        let mid = gpu::swiglu_weight(&gate, &up, &routing.weights, 10.0, stream)?;
        let experts = projection::experts(&self.down, &mid, &routing.selected, stream)?;
        let routed = gpu::sum_six(&experts, stream)?;
        Ok(Output {
            routed,
            shared,
            routing,
        })
    }
}
