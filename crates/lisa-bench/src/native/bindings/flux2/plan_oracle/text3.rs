#![cfg(test)]
use super::{silu, weights::Weights};
use crate::core::media_attention;
use anyhow::ensure;
use lisa_metal::ops::indexing::{Ellipsis, IndexOp};
use lisa_metal::{Array, Dtype, fast, ops};
const HIDDEN: i32 = 128;
const HEADS: i32 = 2;
const KV_HEADS: i32 = 1;
const HEAD_DIM: i32 = 128;
const DEPTH: usize = 36;
const TAPS: [usize; 8] = [4, 8, 12, 16, 20, 24, 28, 32];
pub type Cache = Vec<Option<(Array, Array)>>;
pub struct Encoder {
    pub weights: Weights,
}
impl Encoder {
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
    pub fn forward(
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
}
