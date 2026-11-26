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

use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::TensorAccess;

use crate::bindings::qwen35::decoder::DecoderLayer;
use crate::bindings::shared::normalization::{RmsNorm, Rotary};
use crate::cache::attention::FullAttentionCache;
use crate::core::graph::PreparedGraph;
use crate::weights::affine::{QuantizedEmbedding, QuantizedLinear};
use lisa_models::qwen3_5::mtp::Plan;

pub struct MtpHead<D: BindingDevice> {
    pub context: DeviceContext<D>,
    pub pre_fc_norm_embedding: RmsNorm<D::Tensor>,
    pub pre_fc_norm_hidden: RmsNorm<D::Tensor>,
    pub fc: QuantizedLinear<D::Tensor>,
    pub layer: DecoderLayer<D>,
    pub norm: RmsNorm<D::Tensor>,
    pub rope: Rotary,
    hidden_size: usize,
    pub caches: Vec<crate::cache::LayerCache<D>>,
    plans: Vec<(Plan, PreparedGraph<D>)>,
}

impl<D: BindingDevice> MtpHead<D> {
    /// Present-and-load: `None` when the checkpoint ships no head.
    pub fn load(
        context: DeviceContext<D>,
        w: &mut HashMap<String, D::Tensor>,
        config: &lisa_models::qwen3_5::config::Qwen35Config,
    ) -> anyhow::Result<Option<Self>> {
        // The concat projection keys off the head's own presence.
        if !w.contains_key("mtp.fc.weight") {
            return Ok(None);
        }
        let eps = config.rms_norm_eps;
        let gs = config.quant_group_size;
        let bits = config.quant_bits;
        let hidden = config.hidden_size;

        let pre_fc_norm_embedding =
            RmsNorm::load_on(&context, w, "mtp.pre_fc_norm_embedding", eps, None)?;
        let pre_fc_norm_hidden =
            RmsNorm::load_on(&context, w, "mtp.pre_fc_norm_hidden", eps, None)?;
        let mut fc = QuantizedLinear::load(w, "mtp.fc", "")?;
        fc.set_quant(gs, bits);
        let norm = RmsNorm::load_on(&context, w, "mtp.norm", eps, None)?;

        let prefix = "mtp.layers.0";
        let layer = DecoderLayer::new_mtp(
            context.clone(),
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
            context: context.clone(),
            pre_fc_norm_embedding,
            pre_fc_norm_hidden,
            fc,
            layer,
            norm,
            rope: Rotary::new(config.rotary_dimensions(), config.rope_theta),
            hidden_size: hidden,
            plans: Vec::new(),
            caches: vec![crate::cache::LayerCache::Full(FullAttentionCache::new(
                context.clone(),
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
        tokens: &D::Tensor,
        hidden: &D::Tensor,
        embed_tokens: &QuantizedEmbedding<D::Tensor>,
        offset: usize,
    ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
        anyhow::ensure!(
            tokens.rank() == 2
                && hidden.rank() == 3
                && tokens.dim(0) == hidden.dim(0)
                && tokens.dim(1) == hidden.dim(1)
                && hidden.dim(2) == self.hidden_size as i32,
            "draft head token/hidden geometry differs"
        );
        let cache = match self.caches.first_mut() {
            Some(crate::cache::LayerCache::Full(c)) => c,
            _ => anyhow::bail!("native draft attention cache is absent"),
        };
        let attention = self
            .layer
            .attn
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("draft attention weights are absent"))?;
        let spec = Plan {
            attention: attention.description(
                tokens.dim(0),
                tokens.dim(1),
                self.rope.dimensions as i32,
                true,
                Some(cache),
            ),
            embedding: (embed_tokens.group_size, embed_tokens.bits),
            fc: (self.fc.group_size, self.fc.bits),
            dense: self.layer.mlp.geometry,
            dense_joined: self.layer.mlp.joined,
            epsilon: [
                self.pre_fc_norm_embedding.eps,
                self.pre_fc_norm_hidden.eps,
                self.layer.input_norm.eps,
                self.layer.post_norm.eps,
                self.norm.eps,
            ]
            .map(f32::to_bits),
            position_offset: offset,
            rotary_base: self.rope.base.to_bits(),
        };
        let index = if let Some(i) = self.plans.iter().position(|(p, _)| p == &spec) {
            i
        } else {
            let mut weights = vec![
                self.pre_fc_norm_embedding.weight.clone(),
                self.pre_fc_norm_hidden.weight.clone(),
                self.fc.weight.clone(),
                self.fc.scales.clone(),
                self.fc.biases.clone(),
                self.layer.input_norm.weight.clone(),
            ];
            weights.extend(attention.weights());
            weights.push(self.layer.post_norm.weight.clone());
            weights.extend_from_slice(self.layer.mlp.weights());
            weights.push(self.norm.weight.clone());
            let plan = PreparedGraph::new(self.context.device().clone(), spec.graph()?, weights)?;
            if self.plans.len() == 16 {
                self.plans.remove(0);
            }
            self.plans.push((spec.clone(), plan));
            self.plans.len() - 1
        };
        let mut inputs = vec![
            tokens,
            hidden,
            &embed_tokens.weight,
            &embed_tokens.scales,
            &embed_tokens.biases,
        ];
        if let (Some(k), Some(v)) = (&cache.keys, &cache.values) {
            inputs.extend([k, v]);
        }
        let mut out = self.plans[index].1.run(&inputs, &mut [])?.into_iter();
        let hidden = out.next().unwrap();
        let last = out.next().unwrap();
        let (offset, capacity, _) = spec
            .attention
            .cache
            .as_ref()
            .unwrap()
            .layout()
            .map_err(anyhow::Error::msg)?;
        cache.keys = out.next();
        cache.values = out.next();
        cache.offset = offset;
        cache.capacity = capacity;
        Ok((hidden, last))
    }

    pub fn reset_caches(&mut self) {
        const KV: usize = 1;
        const HD: usize = 128;
        let (kv_heads, head_dim) = match self.caches.first() {
            Some(crate::cache::LayerCache::Full(f)) => (f.kv_heads, f.head_dim),
            _ => (KV, HD),
        };
        self.caches = vec![crate::cache::LayerCache::Full(FullAttentionCache::new(
            self.context.clone(),
            kv_heads,
            head_dim,
        ))];
    }

    pub fn cache_offset(&self) -> usize {
        match self.caches.first() {
            Some(crate::cache::LayerCache::Full(f)) => f.offset,
            _ => 0,
        }
    }

    pub fn restore_offset(&mut self, n: usize) {
        for c in self.caches.iter_mut() {
            if let crate::cache::LayerCache::Full(f) = c {
                f.restore_offset(n);
            }
        }
    }

    pub fn trim_caches(&mut self, n: usize) {
        for c in self.caches.iter_mut() {
            if let crate::cache::LayerCache::Full(f) = c {
                f.trim(n);
            }
        }
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}
