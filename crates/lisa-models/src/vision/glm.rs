//! GLM-5.3 vision dataflow, following mlx-serve 6c430614's
//! glm5_next/vision.zig. Device kernels have no knowledge of this model.
use crate::{
    graph::Value,
    media_attention,
    program::{Build, Program, Schema},
};
use lisa_backend::{Binary as B, ScalarType as S, Unary as U};
use serde::Deserialize;
#[derive(Clone, Deserialize)]
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
impl Config {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.depth > 0
                && self.depth <= 128
                && self.hidden_size > 0
                && self.num_heads > 0
                && self.hidden_size % self.num_heads == 0
                && self.hidden_size / self.num_heads % 4 == 0,
            "unsupported GLM vision geometry"
        );
        anyhow::ensure!(
            (1..=32).contains(&self.patch_size)
                && (1..=8).contains(&self.temporal_patch_size)
                && (1..=8).contains(&self.spatial_merge_size)
                && self.out_hidden_size > 0,
            "invalid GLM vision patch dimensions"
        );
        anyhow::ensure!(
            self.out_hidden_size <= i32::MAX as usize
                && self
                    .hidden_size
                    .checked_mul(self.spatial_merge_size)
                    .and_then(|n| n.checked_mul(self.spatial_merge_size))
                    .is_some_and(|n| n <= i32::MAX as usize),
            "vision dimensions exceed tensor index range"
        );
        anyhow::ensure!(
            self.rms_norm_eps.is_finite()
                && self.rms_norm_eps >= 0.
                && self.swiglu_limit.is_finite()
                && self.swiglu_limit > 0.,
            "invalid GLM vision numerical parameters"
        );
        Ok(())
    }
    pub fn geometry(&self, gh: usize, gw: usize, len: usize) -> anyhow::Result<(i32, i32)> {
        self.validate()?;
        let n = gh
            .checked_mul(gw)
            .ok_or_else(|| anyhow::anyhow!("vision patch count overflow"))?;
        let width = 3usize
            .checked_mul(self.temporal_patch_size)
            .and_then(|v| v.checked_mul(self.patch_size))
            .and_then(|v| v.checked_mul(self.patch_size))
            .ok_or_else(|| anyhow::anyhow!("vision patch width overflow"))?;
        anyhow::ensure!(
            gh > 0
                && gw > 0
                && gh % self.spatial_merge_size == 0
                && gw % self.spatial_merge_size == 0
                && n.checked_mul(width) == Some(len),
            "invalid vision patch shape"
        );
        Ok((i32::try_from(n)?, i32::try_from(width)?))
    }
    pub fn scratch_bytes(&self, n: usize) -> anyhow::Result<usize> {
        self.hidden_size
            .checked_mul(64)
            .and_then(|v| self.num_heads.checked_mul(256 * 12)?.checked_add(v))
            .and_then(|v| n.checked_mul(v))
            .ok_or_else(|| anyhow::anyhow!("vision scratch size overflow"))
    }
}
pub fn prepare_weights(c: &Config, schema: &Schema) -> anyhow::Result<Vec<(String, Program)>> {
    c.validate()?;
    anyhow::ensure!(
        schema
            .get("downsample.bias")
            .is_some_and(|shape| shape.as_slice() == [c.out_hidden_size as i32]),
        "missing or invalid vision downsample bias"
    );

    anyhow::ensure!(
        schema
            .get("patch_embed.proj.bias")
            .is_some_and(|shape| shape.as_slice() == [c.hidden_size as i32]),
        "missing or invalid vision patch bias"
    );

    let (h, t, size, m, out) = (
        i32::try_from(c.hidden_size)?,
        i32::try_from(c.temporal_patch_size)?,
        i32::try_from(c.patch_size)?,
        i32::try_from(c.spatial_merge_size)?,
        i32::try_from(c.out_hidden_size)?,
    );
    let mut result = Vec::new();
    for name in ["patch_embed.proj.weight", "downsample.weight"] {
        let mut p = Build::new(0, schema);
        let shape = schema
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing {name}"))?;
        let mut x = p.weight(name);
        let target = if name == "patch_embed.proj.weight" {
            if shape == &[h, t, size, size, 3] {
                x = p.transpose(x, &[0, 4, 1, 2, 3]);
            } else {
                anyhow::ensure!(
                    shape == &[h, 3, t, size, size],
                    "unsupported GLM patch layout"
                );
            }
            vec![
                h,
                3i32.checked_mul(t)
                    .and_then(|v| v.checked_mul(size))
                    .and_then(|v| v.checked_mul(size))
                    .ok_or_else(|| anyhow::anyhow!("patch width overflow"))?,
            ]
        } else {
            if shape != &[out, m, m, h] {
                anyhow::ensure!(
                    shape == &[out, h, m, m],
                    "unsupported GLM downsample layout"
                );
                x = p.transpose(x, &[0, 2, 3, 1]);
            }
            vec![
                out,
                m.checked_mul(m)
                    .and_then(|v| v.checked_mul(h))
                    .ok_or_else(|| anyhow::anyhow!("downsample width overflow"))?,
            ]
        };
        x = p.contiguous(x);
        x = p.reshape(x, &target);
        result.push((name.into(), p.finish(vec![x])?));
    }
    Ok(result)
}
fn mlp(p: &mut Build<'_>, x: Value, prefix: &str, limit: f32) -> Value {
    let limit_value = p.scalar(limit, S::BF16);
    let negative = p.scalar(-limit, S::BF16);
    let gate = p.linear(x, &format!("{prefix}.gate_proj"));
    let gate = p.binary(B::Minimum, gate, limit_value);
    let up = p.linear(x, &format!("{prefix}.up_proj"));
    let up = p.binary(B::Maximum, up, negative);
    let up = p.binary(B::Minimum, up, limit_value);
    let sigmoid = p.unary(U::Sigmoid, gate);
    let gate = p.binary(B::Multiply, gate, sigmoid);
    let hidden = p.binary(B::Multiply, gate, up);
    p.linear(hidden, &format!("{prefix}.down_proj"))
}
fn rotate(p: &mut Build<'_>, x: Value, cos: Value, sin: Value, hd: i32) -> Value {
    let x = p.cast(x, S::F32);
    let right = p.slice(x, -1, hd / 2, hd);
    let right = p.unary(U::Negate, right);
    let left = p.slice(x, -1, 0, hd / 2);
    let swapped = p.concat(vec![right, left], -1);
    let a = p.binary(B::Multiply, x, cos);
    let b = p.binary(B::Multiply, swapped, sin);
    let out = p.binary(B::Add, a, b);
    p.cast(out, S::BF16)
}
pub fn encoder(c: &Config, schema: &Schema, gh: usize, gw: usize) -> anyhow::Result<Program> {
    c.validate()?;
    let n = i32::try_from(
        gh.checked_mul(gw)
            .ok_or_else(|| anyhow::anyhow!("vision patch count overflow"))?,
    )?;
    anyhow::ensure!(
        gh > 0 && gw > 0 && gh % c.spatial_merge_size == 0 && gw % c.spatial_merge_size == 0,
        "invalid vision grid"
    );
    let h = i32::try_from(c.hidden_size)?;
    let heads = i32::try_from(c.num_heads)?;
    let hd = h / heads;
    let m = i32::try_from(c.spatial_merge_size)?;
    let mut p = Build::new(1, schema);
    let (cos, sin) = super::qwen::rotary_tables(gh, gw, c.spatial_merge_size, hd as usize);
    let cos = p.graph.constant_f32(&cos, &[n, 1, hd]);
    let sin = p.graph.constant_f32(&sin, &[n, 1, hd]);
    let input = p.cast(Value::Input(0), S::BF16);
    let mut x = p.linear(input, "patch_embed.proj");
    for i in 0..c.depth {
        let key = |suffix: &str| format!("blocks.{i}.{suffix}");
        let norm = p.rms(x, &key("norm1.weight"), c.rms_norm_eps);
        let qkv = p.linear(norm, &key("attn.qkv"));
        let qkv = p.reshape(qkv, &[n, 3, heads, hd]);
        let mut projected = Vec::new();
        for (index, name) in [(0, "attn.q_norm.weight"), (1, "attn.k_norm.weight")] {
            let v = p.slice(qkv, 1, index, index + 1);
            let v = p.reshape(v, &[n, heads, hd]);
            let v = p.rms(v, &key(name), c.rms_norm_eps);
            let v = rotate(&mut p, v, cos, sin, hd);
            let v = p.transpose(v, &[1, 0, 2]);
            projected.push(p.reshape(v, &[1, heads, n, hd]));
        }
        let v = p.slice(qkv, 1, 2, 3);
        let v = p.reshape(v, &[n, heads, hd]);
        let v = p.transpose(v, &[1, 0, 2]);
        let v = p.reshape(v, &[1, heads, n, hd]);
        let geometry = media_attention::Geometry {
            q: [1, heads, n, hd],
            k: [1, heads, n, hd],
            v: [1, heads, n, hd],
            output_dtype: S::BF16,
            scale: (1.0 / (hd as f32).sqrt()).to_bits(),
            mask: false,
        };
        let attn = media_attention::append(&mut p, &geometry, projected[0], projected[1], v, None)?;
        let attn = p.transpose(attn, &[0, 2, 1, 3]);
        let attn = p.contiguous(attn);
        let attn = p.reshape(attn, &[n, h]);
        let attn = p.linear(attn, &key("attn.proj"));
        x = p.binary(B::Add, x, attn);
        let norm = p.rms(x, &key("norm2.weight"), c.rms_norm_eps);
        let ff = mlp(&mut p, norm, &key("mlp"), c.swiglu_limit);
        x = p.binary(B::Add, x, ff);
    }
    let x = p.rms(x, "post_layernorm.weight", c.rms_norm_eps);
    let x = p.reshape(x, &[n / (m * m), m * m * h]);
    let x = p.linear(x, "downsample");
    let x = p.linear(x, "merger.proj");
    let x = p.layer_norm(x, "merger.post_projection_norm", 1e-5);
    let x = p.gelu_exact(x);
    let x = mlp(&mut p, x, "merger", c.swiglu_limit);
    let width = p.width("merger.down_proj.weight")?;
    let x = p.reshape(x, &[1, n / (m * m), width]);
    p.finish(vec![x])
}
