//! The native qwen3_5 MTP head (`mtp/weights.safetensors`, `mtp.*` prefix):
//! one full-attention decoder layer that rides the target's embedding table
//! and lm_head (no dedicated embedding, `mtp_use_dedicated_embeddings=false`).
//!
//! Input path: the concat projection
//! `mtp.fc [H, 2H]` takes `rms(emb(token)) ‖ rms(hidden)` where `hidden` is
//! the trunk vector the lm_head consumes (our final-norm hidden — exactly
//! what `forward_capture` returns as its `multi`). The layer output + the
//! head's own `mtp.norm` give `hidden_next`, the next chain step's input;
//! `argmax(lm_head(hidden_next))` is the draft token.

use std::collections::HashMap;

use lisa_mlx::ops::indexing::IndexOp;
use lisa_mlx::Array;

use crate::core::cache::FullAttentionCache;
use crate::core::norm::{positions, RmsNorm, Rotary};
use crate::core::quant::{QuantizedEmbedding, QuantizedLinear};
use crate::models::qwen3_5::tower::DecoderLayer;

pub struct MtpHead {
    pre_fc_norm_embedding: RmsNorm,
    pre_fc_norm_hidden: RmsNorm,
    fc: QuantizedLinear,
    layer: DecoderLayer,
    norm: RmsNorm,
    rope: Rotary,
    hidden_size: usize,
    caches: Vec<crate::core::cache::LayerCache>,
}

impl MtpHead {
    /// Present-and-load: `None` when the checkpoint ships no head.
    pub fn load(
        w: &mut HashMap<String, lisa_mlx::Array>,
        config: &crate::models::qwen3_5::config::Qwen35Config,
    ) -> anyhow::Result<Option<Self>> {
        // The concat projection keys off the head's own presence.
        if !w.contains_key("mtp.fc.weight") {
            return Ok(None);
        }
        let eps = config.rms_norm_eps;
        let gs = config.quant_group_size;
        let bits = config.quant_bits;
        let hidden = config.hidden_size;

        let pre_fc_norm_embedding = RmsNorm::load(w, "mtp.pre_fc_norm_embedding", eps, None)?;
        let pre_fc_norm_hidden = RmsNorm::load(w, "mtp.pre_fc_norm_hidden", eps, None)?;
        let mut fc = QuantizedLinear::load(w, "mtp.fc", "")?;
        fc.set_quant(gs, bits);
        let norm = RmsNorm::load(w, "mtp.norm", eps, None)?;

        let prefix = "mtp.layers.0";
        let layer = DecoderLayer::new_mtp(
            w,
            prefix,
            eps,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.attn_output_gate,
            gs,
            bits,
        )?;

        Ok(Some(Self {
            pre_fc_norm_embedding,
            pre_fc_norm_hidden,
            fc,
            layer,
            norm,
            rope: Rotary::new(config.rotary_dimensions(), config.rope_theta),
            hidden_size: hidden,
            caches: vec![crate::core::cache::LayerCache::Full(FullAttentionCache::new(
                config.num_key_value_heads,
                config.head_dim,
            ))],
        }))
    }

    /// One head forward over `L` positions.
    ///
    /// `tokens` `[1, L]`, `hidden` `[1, L, H]` (trunk normed hidden, or the
    /// previous chain step's `hidden_next`). Returns `(hidden_next [1, L, H],
    /// last row of hidden_next)`; `hidden_next` doubles as the lm_head input.
    /// Appends `L` rows to the head's own KV cache at `offset`.
    pub fn forward(
        &mut self,
        tokens: &Array,
        hidden: &Array,
        embed_tokens: &QuantizedEmbedding,
        offset: usize,
    ) -> anyhow::Result<(Array, Array)> {
        let s = tokens.dim(1) as usize;

        let e = self
            .pre_fc_norm_embedding
            .forward(&embed_tokens.forward(tokens)?)?;
        let h = self.pre_fc_norm_hidden.forward(hidden)?;
        anyhow::ensure!(
            e.dim(1) == h.dim(1) && e.dim(-1) == h.dim(-1),
            "mtp head prime shape mismatch: tokens [{}, {}] e [{}, {}, {}] h [{}, {}, {}]",
            tokens.dim(0),
            tokens.dim(1),
            e.dim(0),
            e.dim(1),
            e.dim(-1),
            h.dim(0),
            h.dim(1),
            h.dim(-1)
        );
        let cat = lisa_mlx::ops::concatenate(&[&e, &h], -1).map_err(|e| anyhow::anyhow!("{e}"))?;
        let x = self.fc.forward(&cat)?;

        let pos = positions(offset, s)?;
        let (cos, sin) = self.rope.cos_sin(&pos)?;
        let cos = cos.expand_dims(1)?;
        let sin = sin.expand_dims(1)?;
        // Same hoist as tower.forward_inner (specs/16): bf16 cast once.
        let cos_bf = cos.as_dtype(lisa_mlx::Dtype::Bfloat16)?;
        let sin_bf = sin.as_dtype(lisa_mlx::Dtype::Bfloat16)?;
        let cache = self.caches.first_mut();
        let (sum, m) = self.layer.forward(&x, None, cache, &cos_bf, &sin_bf, offset, false)?;
        let hidden_next = self.norm.forward_residual_add(&sum, &m)?.1;
        let last = hidden_next
            .index((.., hidden_next.dim(1) - 1, ..))
            .contiguous()?;
        Ok((hidden_next, last))
    }

    pub fn reset_caches(&mut self) {
        const KV: usize = 1;
        const HD: usize = 128;
        let (kv_heads, head_dim) = match self.caches.first() {
            Some(crate::core::cache::LayerCache::Full(f)) => (f.kv_heads, f.head_dim),
            _ => (KV, HD),
        };
        self.caches = vec![crate::core::cache::LayerCache::Full(FullAttentionCache::new(
            kv_heads, head_dim,
        ))];
    }

    pub fn cache_offset(&self) -> usize {
        match self.caches.first() {
            Some(crate::core::cache::LayerCache::Full(f)) => f.offset,
            _ => 0,
        }
    }

    pub fn restore_offset(&mut self, n: usize) {
        for c in self.caches.iter_mut() {
            if let crate::core::cache::LayerCache::Full(f) = c {
                f.restore_offset(n);
            }
        }
    }

    pub fn trim_caches(&mut self, n: usize) {
        for c in self.caches.iter_mut() {
            if let crate::core::cache::LayerCache::Full(f) = c {
                f.trim(n);
            }
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}
