#![cfg(test)]
#![allow(dead_code)]
//! Original IQ2_XXS gate/up and Q2_K down experts, selected on the GPU.
#[path = "glm53_projection_oracle.rs"]
pub(super) mod projection;
use self::projection::{Projection, plain};
use crate::core::gguf_resident::{PackedTensor, ResidentWeights};
use anyhow::{Context, Result};
use lisa_metal::kernels::{ggml as packed, sparse::kernels as gpu};
use lisa_metal::{Array, Stream};

pub struct Mlp {
    gate: Projection,
    up: Projection,
    down: Projection,
    routed: Option<Routed>,
}
struct Routed {
    router: Array,
    bias: Array,
    gate: PackedTensor,
    up: PackedTensor,
    down: PackedTensor,
}
impl Mlp {
    pub fn load(w: &ResidentWeights, layer: usize) -> Result<Self> {
        let p = |name: &str| format!("blk.{layer}.{name}");
        let packed = |name: &str| {
            w.tensors
                .get(&p(name))
                .cloned()
                .context("missing GLM expert tensor")
        };
        let sparse = layer >= 3;
        let suffix = if sparse { "_shexp" } else { "" };
        Ok(Self {
            gate: Projection::load(w, &p(&format!("ffn_gate{suffix}.weight")))?,
            up: Projection::load(w, &p(&format!("ffn_up{suffix}.weight")))?,
            down: Projection::load(w, &p(&format!("ffn_down{suffix}.weight")))?,
            routed: if sparse {
                Some(Routed {
                    router: plain(w, &p("ffn_gate_inp.weight"))?,
                    bias: plain(w, &p("exp_probs_b.bias"))?,
                    gate: packed("ffn_gate_exps.weight")?,
                    up: packed("ffn_up_exps.weight")?,
                    down: packed("ffn_down_exps.weight")?,
                })
            } else {
                None
            },
        })
    }
    pub fn parts(&self, x: &Array) -> Result<(Option<Array>, Array)> {
        let s = Stream::thread_local_or_default();
        let gate = self.gate.forward(x)?;
        let up = self.up.forward(x)?;
        let activated = gpu::swiglu_fused(&gate, &up, 10.0, &s).context("GLM SwiGLU declined")?;
        let shared = self.down.forward(&activated)?;
        let Some(r) = &self.routed else {
            return Ok((None, shared));
        };
        let input = x.reshape(&[1, 4096])?;
        let logits =
            gpu::router_matvec_f32(&r.router, &input, &s).context("GLM router declined")?;
        let (ids, weights) = gpu::router_select_fused(&logits, &r.bias, 8, 2.5, &s)
            .context("GLM route selection declined")?;
        let input = input.reshape(&[1, 1, 4096])?;
        let gate = packed::expert_matvec(
            packed::BlockFormat::Iq2Xxs,
            r.gate.packed_bytes(),
            &input,
            &ids,
            &s,
        )?;
        let up = packed::expert_matvec(
            packed::BlockFormat::Iq2Xxs,
            r.up.packed_bytes(),
            &input,
            &ids,
            &s,
        )?;
        let activated = gpu::swiglu_fused(&gate, &up, 10.0, &s)
            .context("GLM expert SwiGLU declined")?
            .multiply(&weights.reshape(&[1, 8, 1])?)?
            .contiguous()?;
        let experts = packed::expert_matvec(
            packed::BlockFormat::Q2K,
            r.down.packed_bytes(),
            &activated,
            &ids,
            &s,
        )?;
        let routed = gpu::moe_sum8(&experts.reshape(&[8, 4096])?, 1, 4096, &s)
            .context("GLM ordered expert sum declined")?
            .reshape(&[1, 1, 4096])?;
        Ok((Some(routed), shared))
    }
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let (routed, shared) = self.parts(x)?;
        Ok(match routed {
            Some(r) => r.add(&shared)?,
            None => shared,
        })
    }
}

pub fn from_weights(w: &[PackedTensor], sparse: bool) -> Result<Mlp> {
    let p = |i: usize| -> Result<Projection> {
        Ok(
            if matches!(
                w[i].info.kind,
                lisa_gguf::TensorType::F32
                    | lisa_gguf::TensorType::F16
                    | lisa_gguf::TensorType::BF16
            ) {
                Projection::Plain(w[i].plain_view()?)
            } else {
                Projection::Packed(w[i].clone())
            },
        )
    };
    Ok(Mlp {
        gate: p(0)?,
        up: p(1)?,
        down: p(2)?,
        routed: if sparse {
            Some(Routed {
                router: w[3].plain_view()?,
                bias: w[4].plain_view()?,
                gate: w[5].clone(),
                up: w[6].clone(),
                down: w[7].clone(),
            })
        } else {
            None
        },
    })
}
