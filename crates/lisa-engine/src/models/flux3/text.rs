//! Qwen3-VL text-only path for FLUX 3. No visual tower is loaded for generation.
//! Hidden-state taps and padding follow BFL text_encoder.py at bfc8889a4497f846416f253a3972a1941ac698b9.
//! Copyright 2026 Black Forest Labs, Inc. Apache-2.0; see LICENSE.
use crate::{
    core::tokenizer::Tokenizer,
    models::{
        flux2::{silu, weights::Weights},
        media_attention,
    },
};
use anyhow::{Context, ensure};
use lisa_mlx::ops::indexing::{Ellipsis, IndexOp};
use lisa_mlx::{Array, Dtype, fast, ops};
use std::path::Path;

const HIDDEN: i32 = 2560;
const HEADS: i32 = 32;
const KV_HEADS: i32 = 8;
const HEAD_DIM: i32 = 128;
const DEPTH: usize = 36;
const TAPS: [usize; 8] = [4, 8, 12, 16, 20, 24, 28, 32];
type Cache = Vec<Option<(Array, Array)>>;

pub struct Encoder {
    weights: Weights,
    tokenizer: Tokenizer,
}
impl Encoder {
    pub fn load(dir: &Path) -> anyhow::Result<Self> {
        let dir = dir.join("text_encoder");
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let c = &config["text_config"];
        ensure!(
            c["hidden_size"] == HIDDEN
                && c["num_hidden_layers"] == DEPTH
                && c["num_attention_heads"] == HEADS
                && c["num_key_value_heads"] == KV_HEADS
                && c["head_dim"] == HEAD_DIM
                && c["rope_theta"].as_f64() == Some(5_000_000.0)
                && c["tie_word_embeddings"] == true,
            "expected FLUX 3's Qwen3-VL-4B-Instruct text encoder"
        );
        let loaded = Weights::load_filtered(&dir, |key| key.starts_with("model.language_model."))?;
        let weights = Weights::from_map(
            loaded
                .0
                .into_iter()
                .map(|(key, value)| {
                    (
                        key.trim_start_matches("model.language_model.").to_owned(),
                        value,
                    )
                })
                .collect(),
        );
        ensure!(
            weights.count("layers", "input_layernorm.weight") == DEPTH
                && weights.get("embed_tokens.weight")?.dim(1) == HIDDEN
                && weights.0.values().all(|w| w.dtype() == Dtype::Bfloat16),
            "expected the original BF16 FLUX 3 text encoder weights"
        );
        Ok(Self {
            weights,
            tokenizer: Tokenizer::load(&dir)?,
        })
    }
    /// Qwen3-VL's four position streams are equal for pure text, so MRoPE reduces
    /// to ordinary split-half RoPE. Padding positions are explicitly 1.
    fn rope(x: &Array, positions: &[i32]) -> anyhow::Result<Array> {
        let n = positions.len() as i32;
        let mut cos = Vec::with_capacity(positions.len() * 64);
        let mut sin = Vec::with_capacity(positions.len() * 64);
        for &p in positions {
            for i in 0..64 {
                let a = p as f32 / 5_000_000f32.powf(2.0 * i as f32 / 128.0);
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
        let c = Array::from_slice(&cos, &[1, 1, n, 64]);
        let s = Array::from_slice(&sin, &[1, 1, n, 64]);
        let x = x.as_dtype(Dtype::Float32)?;
        let a = x.index((Ellipsis, 0..64));
        let b = x.index((Ellipsis, 64..128));
        Ok(ops::concatenate(
            &[
                &a.multiply(&c)?.subtract(&b.multiply(&s)?)?,
                &b.multiply(&c)?.add(&a.multiply(&s)?)?,
            ],
            -1,
        )?
        .as_dtype(Dtype::Bfloat16)?)
    }
    fn norm(&self, x: &Array, key: &str) -> anyhow::Result<Array> {
        Ok(fast::rms_norm(
            x,
            Some(&self.weights.get(&format!("{key}.weight"))?),
            1e-6,
        )?)
    }
    fn forward(
        &self,
        ids: &[u32],
        real: usize,
        capture: bool,
        mut cache: Option<&mut Cache>,
    ) -> anyhow::Result<Array> {
        ensure!(
            !ids.is_empty() && real > 0 && real <= ids.len(),
            "empty encoder input"
        );
        let n = ids.len() as i32;
        let offset = cache
            .as_ref()
            .and_then(|c| c.first())
            .and_then(|c| c.as_ref())
            .map_or(0, |(k, _)| k.dim(2));
        let positions: Vec<i32> = (0..n)
            .map(|i| if i < real as i32 { offset + i } else { 1 })
            .collect();
        let nk = offset + n;
        let mask: Vec<f32> = (0..n)
            .flat_map(|i| {
                (0..nk).map(move |j| {
                    if j > offset + i || j >= offset + real as i32 {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
            })
            .collect();
        let mask = Array::from_slice(&mask, &[1, 1, n, nk]);
        let mut x = self
            .weights
            .get("embed_tokens.weight")?
            .take_axis(&Array::from_slice(ids, &[n]), 0)?
            .reshape(&[1, n, HIDDEN])?;
        let mut taps = Vec::new();
        for i in 0..if capture { 32 } else { DEPTH } {
            let p = format!("layers.{i}");
            let w = &self.weights;
            let normalized = self.norm(&x, &format!("{p}.input_layernorm"))?;
            let q = self
                .norm(
                    &w.lin(&format!("{p}.self_attn.q_proj"), &normalized)?
                        .reshape(&[1, n, HEADS, HEAD_DIM])?,
                    &format!("{p}.self_attn.q_norm"),
                )?
                .transpose_axes(&[0, 2, 1, 3])?;
            let k = self
                .norm(
                    &w.lin(&format!("{p}.self_attn.k_proj"), &normalized)?
                        .reshape(&[1, n, KV_HEADS, HEAD_DIM])?,
                    &format!("{p}.self_attn.k_norm"),
                )?
                .transpose_axes(&[0, 2, 1, 3])?;
            let q = Self::rope(&q, &positions)?;
            let mut k = Self::rope(&k, &positions)?;
            let mut v = w
                .lin(&format!("{p}.self_attn.v_proj"), &normalized)?
                .reshape(&[1, n, KV_HEADS, HEAD_DIM])?
                .transpose_axes(&[0, 2, 1, 3])?;
            if let Some(ref mut cache) = cache {
                if let Some((prev_k, prev_v)) = &cache[i] {
                    k = ops::concatenate(&[prev_k, &k], 2)?;
                    v = ops::concatenate(&[prev_v, &v], 2)?;
                }
                cache[i] = Some((k.clone(), v.clone()));
            }
            let a = media_attention::attention(
                &q,
                &k,
                &v,
                1.0 / (HEAD_DIM as f32).sqrt(),
                Some(&mask),
            )?
            .transpose_axes(&[0, 2, 1, 3])?
            .contiguous()?
            .reshape(&[1, n, HEADS * HEAD_DIM])?;
            x = x.add(&w.lin(&format!("{p}.self_attn.o_proj"), &a)?)?;
            let norm = self.norm(&x, &format!("{p}.post_attention_layernorm"))?;
            let act = silu(&w.lin(&format!("{p}.mlp.gate_proj"), &norm)?)?
                .multiply(&w.lin(&format!("{p}.mlp.up_proj"), &norm)?)?;
            x = x.add(&w.lin(&format!("{p}.mlp.down_proj"), &act)?)?;
            x.eval_enqueue()?;
            if capture && TAPS.contains(&(i + 1)) {
                taps.push(x.clone());
            }
        }
        let out = if capture {
            ops::concatenate(&taps.iter().collect::<Vec<_>>(), -1)?
        } else {
            // Only the final token needs a vocabulary projection during caption generation.
            let x = self.norm(&x.index((.., n - 1..n, ..)), "norm")?;
            self.weights.lin("embed_tokens", &x)?
        };
        out.eval()?;
        Ok(out)
    }
    pub fn encode(&self, caption: &str) -> anyhow::Result<Array> {
        let text = format!("<|im_start|>user\n{caption}<|im_end|>\n<|im_start|>assistant\n");
        let mut ids = self.tokenizer.encode(&text, false)?;
        ids.truncate(4096);
        let real = ids.len();
        let n = real.div_ceil(80).saturating_mul(80).min(4096);
        let pad = self
            .tokenizer
            .inner
            .token_to_id("<|endoftext|>")
            .context("missing Qwen padding token")?;
        ids.resize(n, pad);
        self.forward(&ids, real, true, None)
    }
    /// Uses the bundled instruction model, with a compact adaptation of BFL's
    /// external-LLM caption guide. No second LLM is resident or downloaded.
    pub fn expand(
        &self,
        prompt: &str,
        width: u32,
        height: u32,
        progress: &mut dyn FnMut() -> anyhow::Result<()>,
    ) -> anyhow::Result<String> {
        let system = include_str!("caption_system.txt");
        let input = format!(
            "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\nTARGET CANVAS: {width}:{height}\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        );
        let mut ids = self.tokenizer.encode(&input, false)?;
        ensure!(
            ids.len() <= 8192,
            "prompt is too long for local caption expansion"
        );
        let mut cache: Cache = (0..DEPTH).map(|_| None).collect();
        let mut generated = Vec::new();
        for step in 0..512 {
            if step % 16 == 0 {
                progress()?;
            }
            let logits = self.forward(&ids, ids.len(), false, Some(&mut cache))?;
            let next = ops::indexing::argmax_axis(&logits, -1, false)?;
            next.eval()?;
            let token = next.item::<u32>();
            if self.tokenizer.im_end_ids.contains(&token) {
                break;
            }
            generated.push(token);
            ids.clear();
            ids.push(token);
        }
        let caption = self.tokenizer.decode_clean(&generated)?.trim().to_owned();
        ensure!(
            !caption.is_empty(),
            "local caption expansion returned no text"
        );
        Ok(caption)
    }
}
