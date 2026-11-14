//! The native MTP head (embedded under `mtp.*`): 1 full-attention layer that
//! rides the target's embedding table and lm head.

use crate::core::{
    graph::PlanCache,
    sequence::{Blocks, Prepared},
};
use crate::{bindings::BindingDevice, execution::DeviceContext};
use lisa_backend::TensorAccess;
use lisa_models::qwen4::mtp::{self as model, Input, Stage};

use super::normalization::{RmsNorm, Rotary};
use crate::attention_cache::FullAttentionCache;
use crate::bindings::qwen4_hyper::GatedResidual;
use crate::weights::affine::{QuantizedEmbedding, QuantizedLinear};
use std::collections::HashMap;

use crate::bindings::qwen4_decoder::{Block, DecoderLayer};
use crate::cache::LayerCache;

pub struct MtpHead<D: BindingDevice> {
    pub context: DeviceContext<D>,
    pub pre_fc_norm_embedding: RmsNorm<D::Tensor>,
    pub pre_fc_norm_hidden: RmsNorm<D::Tensor>,
    pub fc_embedding: QuantizedLinear<D::Tensor>,
    pub fc_hidden: QuantizedLinear<D::Tensor>,
    pub layer: DecoderLayer<D>,
    pub mixer: GatedResidual<D>,
    pub rope: Rotary,
    pub hidden_size: usize,
    pub hc_count: usize,
    pub caches: Vec<LayerCache<D>>,
}

impl<D: BindingDevice> MtpHead<D> {
    pub fn load_on(
        context: DeviceContext<D>,
        w: &mut HashMap<String, D::Tensor>,
        config: &lisa_models::qwen4::config::ModelConfig,
    ) -> anyhow::Result<Self> {
        let n = model::names();
        let eps = config.rms_norm_eps;
        let hidden = config.hidden_size;
        let hc = config.hc_count;

        let pre_fc_norm_embedding = RmsNorm::load_on(&context, w, n.norm_embedding, eps, None)?;
        let pre_fc_norm_hidden = RmsNorm::load_on(&context, w, n.norm_hidden, eps, None)?;
        let fc_embedding = QuantizedLinear::load(w, n.fc_embedding, "")?;
        let fc_hidden = QuantizedLinear::load(w, n.fc_hidden, "")?;
        let mixer = GatedResidual::load_on(context.clone(), w, n.mixer, hidden, hc, false)?;

        // One full-attention decoder layer keyed mtp.layers.0, no PLE.
        let layer_names = lisa_models::qwen4::tower::layer(n.layer, false, None);
        let block = Block::Attention(crate::bindings::qwen4_attention::Attention::load_on(
            context.clone(),
            w,
            &layer_names.mixer,
            eps,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        )?);
        let mlp = crate::bindings::qwen4_moe::SparseMoeBlock::load_on(
            context.clone(),
            w,
            &layer_names.experts,
            config.num_experts_per_tok,
        )?;
        let attn_hc = GatedResidual::load_on(
            context.clone(),
            w,
            &layer_names.attn_hyper,
            hidden,
            hc,
            true,
        )?;
        let mlp_hc =
            GatedResidual::load_on(context.clone(), w, &layer_names.mlp_hyper, hidden, hc, true)?;
        let layer = DecoderLayer {
            context: context.clone(),
            block,
            mlp,
            attn_hc,
            mlp_hc,
            ple: None,
            has_ple: false,
        };

        Ok(Self {
            context: context.clone(),
            pre_fc_norm_embedding,
            pre_fc_norm_hidden,
            fc_embedding,
            fc_hidden,
            layer,
            mixer,
            rope: Rotary::new(config.rotary_dimensions(), config.rope_theta),
            hidden_size: hidden,
            hc_count: hc,
            caches: vec![LayerCache::Full(FullAttentionCache::new(
                context.clone(),
                config.num_key_value_heads,
                config.head_dim,
            ))],
        })
    }

    /// One draft step.
    ///
    /// `next_token_ids`: [B, S]; `multi_stream`: the target's pre-final-mixer
    /// stream [B, S, hc*H]. Returns `(sample [B, S, H], multi_next)`.
    pub fn forward(
        &mut self,
        next_token_ids: &D::Tensor,
        multi_stream: &D::Tensor,
        embed_tokens: &QuantizedEmbedding<D::Tensor>,
        offset: usize,
    ) -> anyhow::Result<(D::Tensor, D::Tensor)> {
        struct Binding<'a, D: BindingDevice> {
            head: &'a mut MtpHead<D>,
            embedding: &'a QuantizedEmbedding<D::Tensor>,
            offset: usize,
            batch: i32,
            rows: i32,
        }
        impl<D: BindingDevice> Blocks<Stage> for Binding<'_, D> {
            type Tensor = D::Tensor;
            fn call(
                &mut self,
                stage: &Stage,
                a: &[Option<&D::Tensor>],
            ) -> anyhow::Result<Vec<Option<D::Tensor>>> {
                let req = |i: usize| a[i].ok_or_else(|| anyhow::anyhow!("draft input is absent"));
                let out = match stage {
                    Stage::Input => {
                        let h = &self.head;
                        let q = |p: &QuantizedLinear<D::Tensor>| (p.group_size, p.bits);
                        let spec = Input {
                            hidden: h.hidden_size as i32,
                            streams: h.hc_count as i32,
                            embedding: (self.embedding.group_size, self.embedding.bits),
                            projection: [q(&h.fc_embedding), q(&h.fc_hidden)],
                            epsilon: [
                                h.pre_fc_norm_embedding.eps.to_bits(),
                                h.pre_fc_norm_hidden.eps.to_bits(),
                            ],
                        };
                        vec![h.context.with_slot(
                            "qwen4.mtp.inputs",
                            || {
                                std::cell::RefCell::new(PlanCache::new(
                                    h.context.device().clone(),
                                    64,
                                ))
                            },
                            |plans| {
                                plans
                                    .borrow_mut()
                                    .try_get_or_prepare((spec, self.batch, self.rows), || {
                                        spec.graph(self.batch, self.rows)
                                    })?
                                    .run(
                                        &[
                                            req(0)?,
                                            req(1)?,
                                            &self.embedding.weight,
                                            &self.embedding.scales,
                                            &self.embedding.biases,
                                            &h.pre_fc_norm_embedding.weight,
                                            &h.pre_fc_norm_hidden.weight,
                                            &h.fc_embedding.weight,
                                            &h.fc_embedding.scales,
                                            &h.fc_embedding.biases,
                                            &h.fc_hidden.weight,
                                            &h.fc_hidden.scales,
                                            &h.fc_hidden.biases,
                                        ],
                                        &mut [],
                                    )?
                                    .into_one()
                                    .map_err(anyhow::Error::msg)
                            },
                        )?]
                    }
                    Stage::Decoder => {
                        let cache = match self.head.caches.first_mut() {
                            Some(c @ LayerCache::Full(_)) => Some(c),
                            _ => None,
                        };
                        let positions = super::normalization::positions(
                            &self.head.context,
                            self.offset,
                            self.rows as usize,
                        )?;
                        let (a, b, c) = self.head.layer.forward(
                            req(0)?,
                            None,
                            None,
                            cache,
                            &self.head.rope,
                            self.offset,
                            &positions,
                            None,
                            false,
                        )?;
                        vec![a, b, c]
                    }
                    Stage::Finish => {
                        let (a, b) = self.head.mixer.finish(
                            req(0)?,
                            Some(req(1)?),
                            Some(req(2)?),
                            model::FINAL_EPSILON,
                        )?;
                        vec![a, b]
                    }
                };
                Ok(out.into_iter().map(Some).collect())
            }
        }
        static PLAN: std::sync::OnceLock<Prepared<Stage, 8>> = std::sync::OnceLock::new();
        let plan =
            PLAN.get_or_init(|| Prepared::new(model::schedule()).expect("valid draft schedule"));
        let mut binding = Binding {
            head: self,
            embedding: embed_tokens,
            offset,
            batch: next_token_ids.dim(0),
            rows: next_token_ids.dim(1),
        };
        let mut out = plan
            .run(&mut binding, &[Some(next_token_ids), Some(multi_stream)])?
            .into_iter();
        Ok((out.next().flatten().unwrap(), out.next().flatten().unwrap()))
    }

    /// The head's attention-cache offset (one row per consumed pair).
    /// Drop the head's KV cache (used to re-prime it per chat turn).
    pub fn reset_caches(&mut self) {
        const KV: usize = 1;
        let (kv_heads, head_dim) = match self.caches.first() {
            Some(LayerCache::Full(f)) => (f.kv_heads, f.head_dim),
            _ => (KV, 128),
        };
        self.caches = vec![LayerCache::Full(FullAttentionCache::new(
            self.context.clone(),
            kv_heads,
            head_dim,
        ))];
    }

    pub fn cache_offset(&self) -> usize {
        match self.caches.first() {
            Some(LayerCache::Full(f)) => f.offset,
            _ => 0,
        }
    }

    /// Rewind the head's KV cache to a captured offset (prefix snapshot).
    pub fn restore_offset(&mut self, n: usize) {
        for c in self.caches.iter_mut() {
            if let LayerCache::Full(f) = c {
                f.restore_offset(n);
            }
        }
    }

    pub fn trim_caches(&mut self, n: usize) {
        for c in self.caches.iter_mut() {
            if let LayerCache::Full(f) = c {
                f.trim(n);
            }
        }
    }
}
