//! GLM-5.3 vision tower, following mlx-serve 6c430614's glm5_next/vision.zig.
use super::qwen::{Linear, gelu_exact, get, load_weights};
use crate::core::norm::bf16_scalar;
use anyhow::ensure;
use lisa_mlx::ops::indexing::IndexOp;
use lisa_mlx::{Array, Dtype, fast, ops};
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
pub struct Config {
    pub depth: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub out_hidden_size: usize,
    pub rms_norm_eps: f32,
    pub swiglu_limit: f32,
}
struct Block {
    n1: Array,
    n2: Array,
    qn: Array,
    kn: Array,
    qkv: Linear,
    proj: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
}
pub struct Encoder {
    pub config: Config,
    patch: Linear,
    blocks: Vec<Block>,
    post: Array,
    downsample: Linear,
    proj: Linear,
    nw: Array,
    nb: Array,
    gate: Linear,
    up: Linear,
    down: Linear,
}
impl Encoder {
    pub fn load(dir: &Path) -> anyhow::Result<Self> {
        let root: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let c: Config = serde_json::from_value(root["vision_config"].clone())?;
        ensure!(
            c.depth > 0
                && c.depth <= 128
                && c.num_heads > 0
                && c.hidden_size % c.num_heads == 0
                && c.hidden_size / c.num_heads % 4 == 0,
            "unsupported GLM vision geometry"
        );
        super::admit_scratch(2usize << 30)?;
        let w = load_weights(dir)?;
        let h = c.hidden_size as i32;
        let p = c.patch_size as i32;
        let t = c.temporal_patch_size as i32;
        let m = c.spatial_merge_size as i32;
        let conv = get(&w, "patch_embed.proj.weight")?;
        let conv = if conv.shape() == [h, t, p, p, 3] {
            conv.transpose_axes(&[0, 4, 1, 2, 3])?
        } else {
            ensure!(
                conv.shape() == [h, 3, t, p, p],
                "unsupported GLM patch layout"
            );
            conv
        };
        let patch = Linear {
            weight: conv.contiguous()?.reshape(&[h, 3 * t * p * p])?,
            bias: Some(get(&w, "patch_embed.proj.bias")?),
        };
        let conv = get(&w, "downsample.weight")?;
        let out = c.out_hidden_size as i32;
        let conv = if conv.shape() == [out, m, m, h] {
            conv
        } else {
            ensure!(
                conv.shape() == [out, h, m, m],
                "unsupported GLM downsample layout"
            );
            conv.transpose_axes(&[0, 2, 3, 1])?
        };
        let downsample = Linear {
            weight: conv.contiguous()?.reshape(&[out, m * m * h])?,
            bias: Some(get(&w, "downsample.bias")?),
        };
        let mut blocks = Vec::new();
        for i in 0..c.depth {
            let key = |s: &str| format!("blocks.{i}.{s}");
            blocks.push(Block {
                n1: get(&w, &key("norm1.weight"))?,
                n2: get(&w, &key("norm2.weight"))?,
                qn: get(&w, &key("attn.q_norm.weight"))?,
                kn: get(&w, &key("attn.k_norm.weight"))?,
                qkv: Linear::load(&w, &key("attn.qkv"))?,
                proj: Linear::load(&w, &key("attn.proj"))?,
                gate: Linear::load(&w, &key("mlp.gate_proj"))?,
                up: Linear::load(&w, &key("mlp.up_proj"))?,
                down: Linear::load(&w, &key("mlp.down_proj"))?,
            });
        }
        Ok(Self {
            config: c,
            patch,
            blocks,
            downsample,
            post: get(&w, "post_layernorm.weight")?,
            proj: Linear::load(&w, "merger.proj")?,
            nw: get(&w, "merger.post_projection_norm.weight")?,
            nb: get(&w, "merger.post_projection_norm.bias")?,
            gate: Linear::load(&w, "merger.gate_proj")?,
            up: Linear::load(&w, "merger.up_proj")?,
            down: Linear::load(&w, "merger.down_proj")?,
        })
    }
    fn norm(&self, x: &Array, w: &Array) -> anyhow::Result<Array> {
        Ok(fast::rms_norm(
            &x.contiguous()?,
            Some(w),
            self.config.rms_norm_eps,
        )?)
    }
    fn mlp(&self, x: &Array, g: &Linear, u: &Linear, d: &Linear) -> anyhow::Result<Array> {
        let limit = self.config.swiglu_limit;
        let gate = g.forward(x)?.minimum(&bf16_scalar(limit))?;
        let up = u
            .forward(x)?
            .maximum(&bf16_scalar(-limit))?
            .minimum(&bf16_scalar(limit))?;
        d.forward(&gate.multiply(&ops::sigmoid(&gate)?)?.multiply(&up)?)
    }
    pub fn encode(&self, image: &image::RgbImage) -> anyhow::Result<(Array, usize, usize)> {
        let c = &self.config;
        let pixels = super::processor::glm(
            image,
            c.patch_size,
            c.temporal_patch_size,
            c.spatial_merge_size,
        )?;
        let (gh, gw) = (pixels.height, pixels.width);
        let n = (gh * gw) as i32;
        let heads = c.num_heads as i32;
        let hd = (c.hidden_size / c.num_heads) as i32;
        super::admit_scratch((gh * gw) * (c.hidden_size * 64 + 256 * c.num_heads * 12))?;
        let mut cos = Vec::new();
        let mut sin = Vec::new();
        let m = c.spatial_merge_size;
        for by in 0..gh / m {
            for bx in 0..gw / m {
                for iy in 0..m {
                    for ix in 0..m {
                        for _ in 0..2 {
                            for pos in [by * m + iy, bx * m + ix] {
                                for k in 0..hd / 4 {
                                    let a = pos as f64
                                        * 10000f64.powf(-(2 * k) as f64 / (hd / 2) as f64);
                                    cos.push(a.cos() as f32);
                                    sin.push(a.sin() as f32);
                                }
                            }
                        }
                    }
                }
            }
        }
        let cos = Array::from_slice(&cos, &[n, 1, hd]);
        let sin = Array::from_slice(&sin, &[n, 1, hd]);
        let mut x = self.patch.forward(
            &Array::from_slice(
                &pixels.values,
                &[
                    n,
                    (3 * c.temporal_patch_size * c.patch_size * c.patch_size) as i32,
                ],
            )
            .as_dtype(Dtype::Bfloat16)?,
        )?;
        for b in &self.blocks {
            let r = b
                .qkv
                .forward(&self.norm(&x, &b.n1)?)?
                .reshape(&[n, 3, heads, hd])?;
            let rotate = |x: Array| -> anyhow::Result<Array> {
                let x = x.as_dtype(Dtype::Float32)?;
                let r = ops::concatenate(
                    &[
                        &(-x.index((.., .., hd / 2..hd))),
                        &x.index((.., .., 0..hd / 2)),
                    ],
                    -1,
                )?;
                Ok(x.multiply(&cos)?
                    .add(&r.multiply(&sin)?)?
                    .as_dtype(Dtype::Bfloat16)?)
            };
            let q = rotate(self.norm(
                &r.index((.., 0..1, .., ..)).reshape(&[n, heads, hd])?,
                &b.qn,
            )?)?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;
            let k = rotate(self.norm(
                &r.index((.., 1..2, .., ..)).reshape(&[n, heads, hd])?,
                &b.kn,
            )?)?
            .transpose_axes(&[1, 0, 2])?
            .expand_dims(0)?;
            let v = r
                .index((.., 2..3, .., ..))
                .reshape(&[n, heads, hd])?
                .transpose_axes(&[1, 0, 2])?
                .expand_dims(0)?;
            let attn = crate::models::media_attention::attention(
                &q,
                &k,
                &v,
                1.0 / (hd as f32).sqrt(),
                None,
            )?;
            let attn = attn
                .transpose_axes(&[0, 2, 1, 3])?
                .contiguous()?
                .reshape(&[n, c.hidden_size as i32])?;
            x = x.add(&b.proj.forward(&attn)?)?;
            x = x.add(&self.mlp(&self.norm(&x, &b.n2)?, &b.gate, &b.up, &b.down)?)?;
            x.eval_enqueue()?;
        }
        let x = self
            .norm(&x, &self.post)?
            .reshape(&[n / (m * m) as i32, (m * m * c.hidden_size) as i32])?;
        let x = self.proj.forward(&self.downsample.forward(&x)?)?;
        let x = gelu_exact(&fast::layer_norm(&x, &self.nw, &self.nb, 1e-5)?)?;
        let x = self
            .mlp(&x, &self.gate, &self.up, &self.down)?
            .expand_dims(0)?;
        x.eval()?;
        Ok((x, gh, gw))
    }
}
