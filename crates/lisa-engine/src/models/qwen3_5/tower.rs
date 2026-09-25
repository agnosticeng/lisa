//! The `qwen3_5` dense hybrid tower: embeddings, 64 pre-norm layers (GDN or
//! full attention), a dense SwiGLU MLP per layer, the final norm and `lm_head`.

use std::collections::HashMap;
use std::path::Path;

use lisa_mlx::ops::indexing::IndexOp;
use lisa_mlx::Array;

use crate::core::cache::{FullAttentionCache, GdnCache, LayerCache};
use crate::core::loader::{sanitize_name, Checkpoint, Shard, Weights};
use crate::core::norm::{bf16_silu, positions, RmsNorm, Rotary};
use crate::core::quant::{get_tensor, QuantizedEmbedding, QuantizedLinear};
use crate::models::qwen3_5::attention::Qwen35Attention;
use crate::models::qwen3_5::config::Qwen35Config;
use crate::models::qwen4::gdn::GatedDeltaNet;

struct DenseMlp {
    gate_proj: QuantizedLinear,
    up_proj: QuantizedLinear,
    down_proj: QuantizedLinear,
}

impl DenseMlp {
    fn load(w: &mut Weights, prefix: &str, gs: i32, bits: i32) -> anyhow::Result<Self> {
        let mut gate_proj = QuantizedLinear::load(w, prefix, "gate_proj")?;
        gate_proj.set_quant(gs, bits);
        let mut up_proj = QuantizedLinear::load(w, prefix, "up_proj")?;
        up_proj.set_quant(gs, bits);
        let mut down_proj = QuantizedLinear::load(w, prefix, "down_proj")?;
        down_proj.set_quant(gs, bits);
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Array) -> lisa_mlx::error::Result<Array> {
        let g = self.gate_proj.forward(x)?;
        let u = self.up_proj.forward(x)?;
        let a = bf16_silu(&g)?.multiply(&u)?;
        self.down_proj.forward(&a)
    }
}

struct DecoderLayer {
    attn: Option<Qwen35Attention>,
    gdn: Option<GatedDeltaNet>,
    input_norm: RmsNorm,
    post_norm: RmsNorm,
    mlp: DenseMlp,
}

impl DecoderLayer {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        x: &Array,
        cache: Option<&mut LayerCache>,
        rope: &Rotary,
        offset: usize,
        pos: &Array,
        capture: bool,
    ) -> anyhow::Result<Array> {
        let h = self.input_norm.forward(x)?;
        let r = if let Some(a) = self.attn.as_ref() {
            let f = match cache {
                Some(LayerCache::Full(f)) => Some(f),
                _ => None,
            };
            a.forward(&h, rope, f, offset, pos)?
        } else if let Some(g) = self.gdn.as_ref() {
            let g_c = match cache {
                Some(LayerCache::Linear(gc)) => Some(gc),
                _ => None,
            };
            g.forward(&h, g_c, capture)?
        } else {
            anyhow::bail!("layer has neither attention nor GDN")
        };
        let h = x.add(&r)?;
        let h2 = self.post_norm.forward(&h)?;
        let m = self.mlp.forward(&h2)?;
        Ok(h.add(&m)?)
    }
}

pub struct Qwen35Tower {
    embed_tokens: QuantizedEmbedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    lm_head: QuantizedLinear,
    rope: Rotary,
    pub config: Qwen35Config,
}

impl Qwen35Tower {
    pub fn load(dir: &Path, config: Qwen35Config) -> anyhow::Result<Self> {
        let checkpoint = Checkpoint::open(dir)?;
        let map = checkpoint.weight_map.clone();
        let mut shard_order: Vec<String> = Vec::new();
        for shard in map.values() {
            if !shard_order.contains(shard) {
                shard_order.push(shard.clone());
            }
        }
        let mut weights: Weights = HashMap::new();
        for shard_name in &shard_order {
            let shard = Shard::open(&dir.join(shard_name))?;
            for name in shard.tensor_names() {
                if let Some(k) = sanitize_name(name) {
                    if !weights.contains_key(&k) {
                        weights.insert(k, get_tensor(&shard, name)?);
                    }
                }
            }
        }
        // MTP head: a sibling `mtp/weights.safetensors` under the `mtp.` prefix.
        let mtp_file = dir.join("mtp").join("weights.safetensors");
        if mtp_file.is_file() {
            let shard = Shard::open(&mtp_file)?;
            for name in shard.tensor_names() {
                let key = format!("mtp.{name}");
                if !weights.contains_key(&key) {
                    weights.insert(key, get_tensor(&shard, name)?);
                }
            }
        }
        Self::from_weights(weights, config)
    }

    pub fn from_weights(mut w: Weights, config: Qwen35Config) -> anyhow::Result<Self> {
        let eps = config.rms_norm_eps;
        let gs = config.quant_group_size;
        let bits = config.quant_bits;

        let mut embed_tokens = QuantizedEmbedding::load(&mut w, "model.embed_tokens")?;
        embed_tokens.set_quant(gs, bits);
        let mut lm_head = QuantizedLinear::load(&mut w, "lm_head", "")?;
        lm_head.set_quant(gs, bits);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("model.layers.{i}");
            let is_full = config.is_full(i);
            let attn = if is_full {
                Some(Qwen35Attention::load(
                    &mut w,
                    &format!("{p}.self_attn"),
                    eps,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.head_dim,
                    config.attn_output_gate,
                    gs,
                    bits,
                )?)
            } else {
                None
            };
            let mut gdn = if is_full {
                None
            } else {
                Some(GatedDeltaNet::load(
                    &mut w,
                    &format!("{p}.linear_attn"),
                    eps,
                    config.linear_num_key_heads,
                    config.linear_num_value_heads,
                    config.linear_key_head_dim,
                    config.linear_value_head_dim,
                    config.linear_conv_kernel_dim,
                    gs,
                    bits,
                )?)
            };
            if let Some(g) = gdn.as_mut() {
                g.l2_norm = true;
                g.output_gate_silu = true;
            }
            layers.push(DecoderLayer {
                attn,
                gdn,
                input_norm: RmsNorm::load(&mut w, &format!("{p}.input_layernorm"), eps, None)?,
                post_norm: RmsNorm::load(
                    &mut w,
                    &format!("{p}.post_attention_layernorm"),
                    eps,
                    None,
                )?,
                mlp: DenseMlp::load(&mut w, &format!("{p}.mlp"), gs, bits)?,
            });
        }
        let norm = RmsNorm::load(&mut w, "model.norm", eps, None)?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rope: Rotary::new(config.rotary_dimensions(), config.rope_theta),
            config,
        })
    }

    fn forward_inner(
        &mut self,
        ids: &Array,
        caches: Option<&mut Vec<LayerCache>>,
        capture: bool,
    ) -> anyhow::Result<Array> {
        let s = ids.dim(1) as usize;
        let mut x = self.embed_tokens.forward(ids)?;
        let offset = caches
            .as_deref()
            .and_then(|c| {
                c.iter().find_map(|l| match l {
                    LayerCache::Full(f) => Some(f.offset),
                    _ => None,
                })
            })
            .unwrap_or(0);
        let pos = positions(offset, s)?;
        let mut caches = caches;
        for i in 0..self.layers.len() {
            let cache = caches.as_deref_mut().map(|c| &mut c[i]);
            x = self.layers[i].forward(&x, cache, &self.rope, offset, &pos, capture)?;
        }
        Ok(self.norm.forward(&x)?)
    }

    pub fn head(&self, mixed: &Array) -> lisa_mlx::error::Result<Array> {
        self.lm_head.forward(mixed)
    }

    pub fn warmup(&mut self, seed: &[u32]) -> anyhow::Result<()> {
        let base: Vec<u32> = if seed.is_empty() {
            (0..64).map(|i| 128 + (i * 37) % 4096).collect()
        } else {
            seed.to_vec()
        };
        for s in [1usize, 9, 40, 2048] {
            let ids: Vec<i32> = (0..s).map(|i| base[i % base.len()] as i32).collect();
            let arr = Array::from_slice(&ids, &[1i32, s as i32]);
            let mut caches = self.new_caches();
            let h = self.forward_inner(&arr, Some(&mut caches), false)?;
            let logits = self.head(&h)?;
            let _ = logits.eval();
        }
        let _ = lisa_mlx::memory::clear_cache();
        Ok(())
    }

    fn new_caches(&self) -> Vec<LayerCache> {
        self.config
            .layer_types
            .iter()
            .map(|t| {
                if t == "full_attention" {
                    LayerCache::Full(FullAttentionCache::new(
                        self.config.num_key_value_heads,
                        self.config.head_dim,
                    ))
                } else {
                    LayerCache::Linear(GdnCache::new())
                }
            })
            .collect()
    }

    pub fn max_position_embeddings(&self) -> usize {
        self.config.max_position_embeddings
    }
    pub fn eos_token_id(&self) -> i64 {
        self.config.eos_token_id
    }
    fn forward_capture(
        &mut self,
        ids: &Array,
        caches: Option<&mut Vec<LayerCache>>,
        capture: bool,
    ) -> anyhow::Result<(Array, Array)> {
        let h = self.forward_inner(ids, caches, capture)?;
        Ok((h.clone(), h))
    }
    fn prefill_multi(
        &mut self,
        tokens: &[u32],
        caches: &mut Vec<LayerCache>,
    ) -> anyhow::Result<(Array, Array)> {
        const CHUNK: usize = 2048;
        let mut last = None;
        for c in tokens.chunks(CHUNK) {
            let ids: Vec<i32> = c.iter().map(|&t| t as i32).collect();
            let arr = Array::from_slice(&ids, &[1i32, c.len() as i32]);
            last = Some(self.forward_capture(&arr, Some(caches), false)?.0);
            lisa_mlx::memory::trim_cache();
        }
        let h = last.expect("non-empty");
        let last_row = h.index((.., h.dim(1) - 1, ..));
        Ok((last_row, h))
    }
}

impl crate::models::LanguageModel for Qwen35Tower {
    fn max_position_embeddings(&self) -> usize {
        self.max_position_embeddings()
    }
    fn eos_token_id(&self) -> i64 {
        self.eos_token_id()
    }
    fn new_caches(&self) -> Vec<LayerCache> {
        self.new_caches()
    }
    fn forward(
        &mut self,
        ids: &Array,
        caches: Option<&mut Vec<LayerCache>>,
    ) -> anyhow::Result<(Array, Array)> {
        self.forward_capture(ids, caches, false)
    }
    fn forward_capture(
        &mut self,
        ids: &Array,
        caches: Option<&mut Vec<LayerCache>>,
        capture: bool,
    ) -> anyhow::Result<(Array, Array)> {
        self.forward_capture(ids, caches, capture)
    }
    fn prefill(&mut self, tokens: &[u32], caches: &mut Vec<LayerCache>) -> anyhow::Result<Array> {
        Ok(self.prefill_multi(tokens, caches)?.0)
    }
    fn prefill_multi(
        &mut self,
        tokens: &[u32],
        caches: &mut Vec<LayerCache>,
    ) -> anyhow::Result<(Array, Array)> {
        self.prefill_multi(tokens, caches)
    }
    fn head(&self, mixed: &Array) -> lisa_mlx::error::Result<Array> {
        self.head(mixed)
    }
    fn warmup(&mut self, seed: &[u32]) -> anyhow::Result<()> {
        self.warmup(seed)
    }
}
