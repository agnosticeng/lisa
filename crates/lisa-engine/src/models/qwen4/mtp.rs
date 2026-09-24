//! The native MTP head (embedded under `mtp.*`): 1 full-attention layer that
//! rides the target's embedding table and lm head.

use lisa_mlx::Array;

use crate::core::cache::FullAttentionCache;
use crate::models::qwen4::hyper::GatedResidual;
use crate::core::norm::{RmsNorm, Rotary};
use crate::core::quant::{QuantizedLinear, QuantizedEmbedding};
use std::collections::HashMap;

use crate::core::cache::LayerCache;
use crate::models::qwen4::tower::{Block, DecoderLayer};

pub struct MtpHead {
    pub pre_fc_norm_embedding: RmsNorm,
    pub pre_fc_norm_hidden: RmsNorm,
    pub fc_embedding: QuantizedLinear,
    pub fc_hidden: QuantizedLinear,
    pub layer: DecoderLayer,
    pub mixer: GatedResidual,
    pub rope: Rotary,
    pub hidden_size: usize,
    pub hc_count: usize,
    pub caches: Vec<LayerCache>,
}

impl MtpHead {
    pub fn load(w: &mut HashMap<String, lisa_mlx::Array>, config: &crate::models::qwen4::config::ModelConfig) -> anyhow::Result<Self> {
        let eps = config.rms_norm_eps;
        let hidden = config.hidden_size;
        let hc = config.hc_count;

        let pre_fc_norm_embedding = RmsNorm::load(w, "mtp.pre_fc_norm_embedding", eps, None)?;
        let pre_fc_norm_hidden = RmsNorm::load(w, "mtp.pre_fc_norm_hidden", eps, None)?;
        let fc_embedding = QuantizedLinear::load(w, "mtp.fc_embedding", "")?;
        let fc_hidden = QuantizedLinear::load(w, "mtp.fc_hidden", "")?;
        let mixer = GatedResidual::load(w, "mtp.hyper_connection_mixer", hidden, hc, false)?;

        // One full-attention decoder layer keyed mtp.layers.0, no PLE.
        let prefix = "mtp.layers.0";
        let block = Block::Attention(crate::models::qwen4::attention::Attention::load(
            w,
            &format!("{prefix}.self_attn"),
            eps,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
        )?);
        let mlp = crate::models::qwen4::moe::SparseMoeBlock::load(w, &format!("{prefix}.mlp"), config.num_experts_per_tok)?;
        let attn_hc = GatedResidual::load(w, &format!("{prefix}.attn_hyper_connection"), hidden, hc, true)?;
        let mlp_hc = GatedResidual::load(w, &format!("{prefix}.mlp_hyper_connection"), hidden, hc, true)?;
        let layer = DecoderLayer {
            block,
            mlp,
            attn_hc,
            mlp_hc,
            ple: None,
            has_ple: false,
        };

        Ok(Self {
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
        next_token_ids: &Array,
        multi_stream: &Array,
        embed_tokens: &QuantizedEmbedding,
        offset: usize,
    ) -> anyhow::Result<(Array, Array)> {
        let b = next_token_ids.dim(0);
        let s = next_token_ids.dim(1);

        let embedded = self.pre_fc_norm_embedding.forward(&embed_tokens.forward(next_token_ids)?)?;
        let e = self.fc_embedding.forward(&embedded)?;
        let h_in = self.pre_fc_norm_hidden.forward(multi_stream)?;
        let h_in = h_in.reshape(&[b, s, self.hc_count as i32, self.hidden_size as i32])?;
        let h = self.fc_hidden.forward(&h_in)?;
        let x = e.expand_dims(-2)?.add(&h)?;
        let hyper = x.reshape(&[b, s, -1])?;


        if std::env::var("LISA_DUMP_DRAFT").is_ok() {
            static FCALL: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let r = FCALL.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if r >= 200 {
                let d = |name: &str, a: &lisa_mlx::Array| {
                    if let Ok(f) = a.as_dtype(lisa_mlx::Dtype::Float32) {
                        {
                                let v = f.as_slice::<f32>();
                            let v: &[f32] = v;
                            let sum: f32 = v.iter().sum();
                            eprintln!("[head] step#{r} {name} n={} sum={:.6e} h={:?}", v.len(), sum, &v[..4.min(v.len())]);
                        }
                    }
                };
                let _ = &r;
                d("embedded", &embedded);
                d("fc_e", &e);
                d("fc_h", &h);
                d("hyper", &hyper);
            }
        }

        let cache: Option<&mut LayerCache> = match self.caches.first_mut() {
            Some(c @ LayerCache::Full(_)) => Some(c),
            _ => None,
        };
        let positions = crate::core::norm::positions(offset, s as usize)?;
        let (st, moe_out, inject_w) =
            self.layer.forward(&hyper, None, None, cache, &self.rope, offset, &positions, None, false)?;

            let r = 200;
            if r >= 200 {
                let d2 = |name: &str, a: &lisa_mlx::Array| {
                    if let Ok(f) = a.as_dtype(lisa_mlx::Dtype::Float32) {
                        {
                                let v = f.as_slice::<f32>();
                            let v: &[f32] = v;
                            let sum: f32 = v.iter().sum();
                            eprintln!("[head] step#{r} {name} n={} sum={:.6e} h={:?}", v.len(), sum, &v[..4.min(v.len())]);
                        }
                    }
                };
                d2("st", &st);
                d2("moe_out", &moe_out);
                d2("inject_w", &inject_w);
            }
        // Final mixer: injectNorm then the inject-less hcMix (the engine's
        // `TrackFastHead` tail). `multi_next` is the new stream for the chain.
        let final_scale = self.mixer.norm_scale_q().map_err(|e| anyhow::anyhow!("{e}"))?;
        let hc = self.hc_count as i32;
        let hidden_i = self.hidden_size as i32;
        let rows = (b * s) as i32;
        let stream = lisa_mlx::Stream::thread_local_or_default();
        let (multi_next, final_normed) = lisa_mlx::kernels::inject_norm(
            &st,
            Some(&moe_out),
            Some(&inject_w),
            &final_scale,
            hc,
            hidden_i,
            rows,
            false,
            1e-6,
            &stream,
        )
        .ok_or_else(|| anyhow::anyhow!("inject_norm kernel unavailable"))?;
        let multi_next = multi_next.reshape(&[b as i32, s as i32, hc * hidden_i])?;
        let final_normed = final_normed.reshape(&[b as i32, s as i32, hc * hidden_i])?;
        let (sample, _) = self.mixer.mix_from_normed(&final_normed, false).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok((sample, multi_next))
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
            kv_heads, head_dim,
        ))];
    }

    pub fn cache_offset(&self) -> usize {
        match self.caches.first() {
            Some(LayerCache::Full(f)) => f.offset,
            _ => 0,
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
