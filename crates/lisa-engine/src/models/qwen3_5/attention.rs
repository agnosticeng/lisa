//! Full attention for `qwen3_5`: GQA with a per-head output gate and learned
//! Q/K RMSNorm (no QSA indexer).

use lisa_mlx::ops::indexing::{Ellipsis, IndexOp};
use lisa_mlx::{fast, ops, Array, Dtype};

use crate::core::cache::FullAttentionCache;
use crate::core::loader::TensorSource;
use crate::core::norm::{rope_partial, RmsNorm, Rotary};
use crate::core::quant::QuantizedLinear;

pub struct Qwen35Attention {
    pub q_proj: QuantizedLinear,
    pub k_proj: QuantizedLinear,
    pub v_proj: QuantizedLinear,
    pub o_proj: QuantizedLinear,
    pub q_norm: RmsNorm,
    pub k_norm: RmsNorm,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
    pub gated: bool,
}

impl Qwen35Attention {
    pub fn load<S: TensorSource>(
        src: &mut S,
        prefix: &str,
        eps: f32,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        gated: bool,
        group_size: i32,
        bits: i32,
    ) -> anyhow::Result<Self> {
        let mut q_proj = QuantizedLinear::load(src, prefix, "q_proj")?;
        q_proj.set_quant(group_size, bits);
        let mut k_proj = QuantizedLinear::load(src, prefix, "k_proj")?;
        k_proj.set_quant(group_size, bits);
        let mut v_proj = QuantizedLinear::load(src, prefix, "v_proj")?;
        v_proj.set_quant(group_size, bits);
        let mut o_proj = QuantizedLinear::load(src, prefix, "o_proj")?;
        o_proj.set_quant(group_size, bits);
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: RmsNorm::load(src, &format!("{prefix}.q_norm"), eps, None)?,
            k_norm: RmsNorm::load(src, &format!("{prefix}.k_norm"), eps, None)?,
            heads,
            kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            gated,
        })
    }

    /// x: `[B, S, hidden]`.
    pub fn forward(
        &self,
        x: &Array,
        rope: &Rotary,
        cache: Option<&mut FullAttentionCache>,
        _offset: usize,
        positions: &Array,
    ) -> lisa_mlx::error::Result<Array> {
        let b = x.dim(0);
        let s = x.dim(1);
        let hd = self.head_dim as i32;
        let heads = self.heads as i32;

        // q_proj is [heads * 2 * head_dim], laid out per head as
        // [q_head | gate_head].
        let qg = self.q_proj.forward(x)?;
        let qg = qg.reshape(&[b, s, heads, 2, hd])?;
        let q = qg.index((Ellipsis, 0, ..)).contiguous()?;
        let gate = if self.gated {
            Some(qg.index((Ellipsis, 1, ..)).contiguous()?)
        } else {
            None
        };
        let k = self
            .k_proj
            .forward(x)?
            .reshape(&[b, s, self.kv_heads as i32, hd])?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape(&[b, s, self.kv_heads as i32, hd])?;
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        let (cos, sin) = rope.cos_sin(positions)?;
        let cos = cos.expand_dims(1)?;
        let sin = sin.expand_dims(1)?;
        let q = rope_partial(&q.transpose_axes(&[0, 2, 1, 3])?, &cos, &sin)?;
        let k = rope_partial(&k.transpose_axes(&[0, 2, 1, 3])?, &cos, &sin)?;
        let v = v.transpose_axes(&[0, 2, 1, 3])?;

        let (live_k, live_v) = match cache {
            Some(c) => c.update(&k, &v)?,
            None => (k, v),
        };
        let out = fast::scaled_dot_product_attention(
            &q,
            &live_k,
            &live_v,
            self.scale,
            fast::ScaledDotProductAttentionMask::Causal,
            None,
        )?;
        let out = out.transpose_axes(&[0, 2, 1, 3])?; // [b,s,heads,hd]
        let out = match gate {
            Some(g) => {
                let a = out.reshape(&[b, s, heads, hd])?;
                let sig = ops::sigmoid(&g)?;
                a.multiply(&sig)?
            }
            None => out.reshape(&[b, s, heads, hd])?,
        };
        let out = out.reshape(&[b, s, heads * hd])?;
        let out = out.as_dtype(Dtype::Bfloat16)?;
        self.o_proj.forward(&out)
    }
}
