#![cfg(test)]
//! Native scalar FFN graph for the original mixed GGUF expert formats.
//! The owning model retains ResidentWeights for this object's full lifetime.

pub(super) use crate::bindings::deepseek41_projection_oracle as projection;
use crate::core::gguf_resident::PackedTensor;
use anyhow::{Result, ensure};
use lisa_metal::kernels::ggml::{self as gpu, ActivationFormat};
use lisa_metal::{Array, Stream};

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
    /// Scalar decode with the reference's original router, BF16 shared-expert boundaries,
    /// routed activation weighting, and ordered six-expert sum. Formats absent
    /// from the reference use the adjacent pinned GGML matvec extension.
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

// Test-only binding constructor; the forward above is frozen before migration.
pub fn from_weights(w: &[PackedTensor]) -> Result<Moe> {
    Ok(Moe {
        router: w[0].clone(),
        bias: w[1].plain_view()?,
        shared_gate: w[2].clone(),
        shared_up: w[3].clone(),
        shared_down: w[4].clone(),
        gate: w[5].clone(),
        up: w[6].clone(),
        down: w[7].clone(),
    })
}
