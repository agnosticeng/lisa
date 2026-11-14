//! FLUX 3 Qwen3-VL text-only computation, adapted from BFL text_encoder.py.
//! Reference bfc8889a4497f846416f253a3972a1941ac698b9.
//! Copyright 2026 Black Forest Labs, Inc. Apache-2.0; see LICENSE.flux3.
use super::symbolic as media_attention;
use super::symbolic::{
    self as ops, Program, Schema, Tensor as Array, TensorInfo, Trace, Weights, silu,
};
use anyhow::ensure;
use lisa_backend::ScalarType as Dtype;
pub const HIDDEN: i32 = 2560;
pub const HEADS: i32 = 32;
pub const KV_HEADS: i32 = 8;
pub const HEAD_DIM: i32 = 128;
pub const DEPTH: usize = 36;
const TAPS: [usize; 8] = [4, 8, 12, 16, 20, 24, 28, 32];
type Cache = Vec<Option<(Array, Array)>>;
pub type CacheSchema = Vec<Option<(TensorInfo, TensorInfo)>>;
pub struct Encoding {
    pub program: Program,
    pub cache_layers: Vec<usize>,
}
#[derive(Clone, Copy)]
pub struct Geometry {
    pub hidden: i32,
    pub heads: i32,
    pub kv_heads: i32,
    pub depth: usize,
}
impl Default for Geometry {
    fn default() -> Self {
        Self {
            hidden: HIDDEN,
            heads: HEADS,
            kv_heads: KV_HEADS,
            depth: DEPTH,
        }
    }
}
pub fn validate(config: &serde_json::Value, schema: &Schema) -> anyhow::Result<()> {
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
    ensure!(
        ops::count(schema, "layers", "input_layernorm.weight") == DEPTH
            && schema
                .get("embed_tokens.weight")
                .is_some_and(|x| x.shape.len() == 2 && x.shape[1] == HIDDEN)
            && schema.values().all(|w| w.dtype == Dtype::BF16),
        "expected the original BF16 FLUX 3 text encoder weights"
    );
    Ok(())
}
pub fn prompt(text: &str) -> String {
    format!("<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")
}
pub fn pad(ids: &mut Vec<u32>, pad: u32) -> usize {
    ids.truncate(4096);
    let real = ids.len();
    ids.resize(real.div_ceil(80).saturating_mul(80).min(4096), pad);
    real
}
pub fn caption_prompt(prompt: &str, width: u32, height: u32) -> String {
    let system = include_str!("caption_system.txt");
    format!(
        "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\nTARGET CANVAS: {width}:{height}\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    )
}
pub fn program(
    schema: &Schema,
    geometry: Geometry,
    n: usize,
    real: usize,
    capture: bool,
    cache: Option<&CacheSchema>,
) -> anyhow::Result<Encoding> {
    ensure!(
        geometry.hidden > 0
            && geometry.heads > 0
            && geometry.kv_heads > 0
            && geometry.heads % geometry.kv_heads == 0
            && geometry.depth >= 32
            && geometry.depth <= 128,
        "invalid text geometry"
    );
    ensure!(
        n > 0 && n <= i32::MAX as usize && real > 0 && real <= n,
        "invalid text input dimensions"
    );
    let trace = Trace::new();
    let ids = trace.input(&[n as i32], Dtype::U32);
    let mut cache = cache
        .map(|cache| -> anyhow::Result<Cache> {
            ensure!(cache.len() == geometry.depth, "invalid text cache depth");
            let offset = cache
                .first()
                .and_then(|x| x.as_ref())
                .and_then(|(k, _)| k.shape.get(2))
                .copied()
                .unwrap_or(0);
            ensure!(
                offset >= 0 && offset.checked_add(n as i32).is_some(),
                "text cache offset overflow"
            );
            cache
                .iter()
                .map(|entry| match entry {
                    None => {
                        ensure!(offset == 0, "incomplete text cache");
                        Ok(None)
                    }
                    Some((k, v)) => {
                        ensure!(
                            k.shape == [1, geometry.kv_heads, offset, HEAD_DIM]
                                && v.shape == k.shape
                                && k.dtype == Dtype::BF16
                                && v.dtype == Dtype::BF16,
                            "invalid text cache geometry"
                        );
                        Ok(Some((
                            trace.input(&k.shape, k.dtype),
                            trace.input(&v.shape, v.dtype),
                        )))
                    }
                })
                .collect()
        })
        .transpose()?;
    let encoder = Encoder {
        weights: Weights {
            trace: trace.clone(),
            schema,
        },
        geometry,
    };
    let output = encoder.forward(&ids, real, capture, cache.as_mut())?;
    let mut outputs = vec![output];
    let mut cache_layers = vec![];
    if let Some(cache) = cache {
        for (i, entry) in cache.into_iter().enumerate() {
            if let Some((k, v)) = entry {
                cache_layers.push(i);
                outputs.extend([k, v]);
            }
        }
    }
    Ok(Encoding {
        program: trace.finish(&outputs)?,
        cache_layers,
    })
}
struct Encoder<'a> {
    weights: Weights<'a>,
    geometry: Geometry,
}
impl Encoder<'_> {
    /// Qwen3-VL's four position streams are equal for pure text, so MRoPE reduces
    /// to ordinary split-half RoPE. Padding positions are explicitly 1.
    fn rope(trace: &Trace, x: &Array, positions: &[i32]) -> anyhow::Result<Array> {
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
        let c = trace.constant(&cos, &[1, 1, n, 64]);
        let s = trace.constant(&sin, &[1, 1, n, 64]);
        let x = x.as_dtype(Dtype::F32)?;
        let a = x.slice(-1, 0, 64)?;
        let b = x.slice(-1, 64, 128)?;
        Ok(ops::concatenate(
            &[
                &a.multiply(&c)?.subtract(&b.multiply(&s)?)?,
                &b.multiply(&c)?.add(&a.multiply(&s)?)?,
            ],
            -1,
        )?
        .as_dtype(Dtype::BF16)?)
    }
    fn norm(&self, x: &Array, key: &str) -> anyhow::Result<Array> {
        Ok(x.rms(&self.weights.get(&format!("{key}.weight"))?, 1e-6)?)
    }
    fn forward(
        &self,
        ids: &Array,
        real: usize,
        capture: bool,
        mut cache: Option<&mut Cache>,
    ) -> anyhow::Result<Array> {
        ensure!(
            ids.dim(0) > 0 && real > 0 && real <= ids.dim(0) as usize,
            "empty encoder input"
        );
        let n = ids.dim(0);
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
        let mask = self.weights.trace.constant(&mask, &[1, 1, n, nk]);
        let mut x = self
            .weights
            .get("embed_tokens.weight")?
            .take_axis(ids, 0)?
            .reshape(&[1, n, self.geometry.hidden])?;
        let mut taps = Vec::new();
        for i in 0..if capture { 32 } else { self.geometry.depth } {
            let p = format!("layers.{i}");
            let w = &self.weights;
            let normalized = self.norm(&x, &format!("{p}.input_layernorm"))?;
            let q = self
                .norm(
                    &w.lin(&format!("{p}.self_attn.q_proj"), &normalized)?
                        .reshape(&[1, n, self.geometry.heads, HEAD_DIM])?,
                    &format!("{p}.self_attn.q_norm"),
                )?
                .transpose_axes(&[0, 2, 1, 3])?;
            let k = self
                .norm(
                    &w.lin(&format!("{p}.self_attn.k_proj"), &normalized)?
                        .reshape(&[1, n, self.geometry.kv_heads, HEAD_DIM])?,
                    &format!("{p}.self_attn.k_norm"),
                )?
                .transpose_axes(&[0, 2, 1, 3])?;
            let q = Self::rope(&self.weights.trace, &q, &positions)?;
            let mut k = Self::rope(&self.weights.trace, &k, &positions)?;
            let mut v = w
                .lin(&format!("{p}.self_attn.v_proj"), &normalized)?
                .reshape(&[1, n, self.geometry.kv_heads, HEAD_DIM])?
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
            .reshape(&[1, n, self.geometry.heads * HEAD_DIM])?;
            x = x.add(&w.lin(&format!("{p}.self_attn.o_proj"), &a)?)?;
            let norm = self.norm(&x, &format!("{p}.post_attention_layernorm"))?;
            let act = silu(&w.lin(&format!("{p}.mlp.gate_proj"), &norm)?)?
                .multiply(&w.lin(&format!("{p}.mlp.up_proj"), &norm)?)?;
            x = x.add(&w.lin(&format!("{p}.mlp.down_proj"), &act)?)?;
            if capture && TAPS.contains(&(i + 1)) {
                taps.push(x.clone());
            }
        }
        let out = if capture {
            ops::concatenate(&taps.iter().collect::<Vec<_>>(), -1)?
        } else {
            // Only the final token needs a vocabulary projection during caption generation.
            let x = self.norm(&x.slice(1, n - 1, n)?, "norm")?;
            self.weights.lin("embed_tokens", &x)?
        };
        Ok(out)
    }
}

pub fn select_token() -> crate::graph::Graph {
    let mut b = crate::graph::Builder::new(1, 0, 0);
    let x = b.one(
        lisa_backend::Kernel::Tensor(lisa_backend::TensorOperation::Argmax {
            axis: -1,
            keep_dims: false,
        }),
        vec![crate::graph::Value::Input(0)],
    );
    b.finish(vec![x])
}
