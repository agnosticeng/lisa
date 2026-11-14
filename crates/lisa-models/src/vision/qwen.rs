//! Qwen3-VL vision dataflow. Reference: mlx-serve 6c430614,
//! sdk/vision/qwen_vision.zig (MIT); original arithmetic order retained.
use crate::{
    graph::Value,
    program::{Build, Program, Schema},
};
use lisa_backend::{Binary as B, ScalarType as S, TensorOperation as T, Unary as U};
use serde::Deserialize;
#[derive(Clone, Deserialize)]
pub struct Config {
    pub depth: usize,
    pub hidden_size: usize,
    pub num_heads: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub num_position_embeddings: usize,
    pub out_hidden_size: usize,
    #[serde(default)]
    pub deepstack_visual_indexes: Vec<usize>,
}

impl Config {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.hidden_size > 0
                && self.num_heads > 0
                && self.hidden_size % self.num_heads == 0
                && self.hidden_size / self.num_heads % 4 == 0
                && self.depth > 0
                && self.depth <= 128,
            "unsupported Qwen vision dimensions"
        );
        anyhow::ensure!(
            (1..=32).contains(&self.patch_size)
                && (1..=8).contains(&self.temporal_patch_size)
                && (1..=8).contains(&self.spatial_merge_size),
            "invalid vision patch dimensions"
        );
        anyhow::ensure!(
            self.out_hidden_size > 0
                && self.out_hidden_size <= i32::MAX as usize
                && self.num_position_embeddings <= i32::MAX as usize
                && self
                    .hidden_size
                    .checked_mul(self.spatial_merge_size)
                    .and_then(|n| n.checked_mul(self.spatial_merge_size))
                    .is_some_and(|n| n <= i32::MAX as usize),
            "vision dimensions exceed tensor index range"
        );
        anyhow::ensure!(
            self.deepstack_visual_indexes.is_empty(),
            "DeepStack vision injection is not supported by this encoder"
        );
        let grid = (self.num_position_embeddings as f64).sqrt() as usize;
        anyhow::ensure!(
            grid > 0 && grid * grid == self.num_position_embeddings,
            "vision position table must be square"
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
        n.checked_mul(n)
            .and_then(|v| v.checked_mul(self.num_heads))
            .and_then(|v| v.checked_mul(12))
            .and_then(|v| {
                n.checked_mul(self.hidden_size)?
                    .checked_mul(64)?
                    .checked_add(v)
            })
            .ok_or_else(|| anyhow::anyhow!("vision scratch size overflow"))
    }
}
/// Load-time layout transformation. The engine executes this once and replaces
/// the named binding with the output; the device sees only shape/permutation ops.
pub fn prepare_weights(c: &Config, schema: &Schema) -> anyhow::Result<Vec<(String, Program)>> {
    c.validate()?;
    anyhow::ensure!(
        schema
            .get("patch_embed.proj.bias")
            .is_some_and(|shape| shape.as_slice() == [c.hidden_size as i32]),
        "missing or invalid vision patch bias"
    );

    let mut p = Build::new(0, schema);
    let name = "patch_embed.proj.weight";
    let (h, t, size) = (
        c.hidden_size as i32,
        c.temporal_patch_size as i32,
        c.patch_size as i32,
    );
    let shape = schema
        .get(name)
        .ok_or_else(|| anyhow::anyhow!("missing {name}"))?;
    let mut x = p.weight(name);
    if shape == &[h, t, size, size, 3] {
        x = p.transpose(x, &[0, 4, 1, 2, 3]);
    } else {
        anyhow::ensure!(
            shape == &[h, 3, t, size, size],
            "unsupported vision patch projection layout {shape:?}"
        );
    }
    x = p.contiguous(x);
    x = p.reshape(x, &[h, 3 * t * size * size]);
    Ok(vec![(name.into(), p.finish(vec![x])?)])
}
fn positions(p: &mut Build<'_>, c: &Config, gh: usize, gw: usize) -> Value {
    let grid = (c.num_position_embeddings as f64).sqrt() as usize;
    let merge = c.spatial_merge_size;
    let mut indices: [Vec<i32>; 4] = std::array::from_fn(|_| Vec::with_capacity(gh * gw));
    let mut weights: [Vec<f32>; 4] = std::array::from_fn(|_| Vec::with_capacity(gh * gw));
    for by in 0..gh / merge {
        for bx in 0..gw / merge {
            for iy in 0..merge {
                for ix in 0..merge {
                    let y = if gh == 1 {
                        0.0
                    } else {
                        (grid - 1) as f64 * (by * merge + iy) as f64 / (gh - 1) as f64
                    };
                    let x = if gw == 1 {
                        0.0
                    } else {
                        (grid - 1) as f64 * (bx * merge + ix) as f64 / (gw - 1) as f64
                    };
                    let (yf, xf) = (y as usize, x as usize);
                    let (yc, xc) = ((yf + 1).min(grid - 1), (xf + 1).min(grid - 1));
                    let (dy, dx) = (y - yf as f64, x - xf as f64);
                    for (k, (index, weight)) in [
                        (yf * grid + xf, (1.0 - dy) * (1.0 - dx)),
                        (yf * grid + xc, (1.0 - dy) * dx),
                        (yc * grid + xf, dy * (1.0 - dx)),
                        (yc * grid + xc, dy * dx),
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        indices[k].push(index as i32);
                        weights[k].push(weight as f32);
                    }
                }
            }
        }
    }
    let pos = p.weight("pos_embed.weight");
    let mut out = None;
    for k in 0..4 {
        let ids = p.graph.constant_i32(&indices[k], &[(gh * gw) as i32]);
        let weights = p.graph.constant_bf16(&weights[k], &[(gh * gw) as i32, 1]);
        let value = p.op(T::Take { axis: 0 }, vec![pos, ids]);
        let value = p.binary(B::Multiply, value, weights);
        out = Some(match out {
            None => value,
            Some(old) => p.binary(B::Add, old, value),
        });
    }
    out.unwrap()
}
/// Host rotary tables; computation uses f64 exactly as the original vision path.
pub fn rotary_tables(gh: usize, gw: usize, merge: usize, hd: usize) -> (Vec<f32>, Vec<f32>) {
    let half = hd / 2;
    let nf = half / 2;
    let mut cos = Vec::with_capacity(gh * gw * hd);
    let mut sin = Vec::with_capacity(gh * gw * hd);
    for by in 0..gh / merge {
        for bx in 0..gw / merge {
            for iy in 0..merge {
                for ix in 0..merge {
                    for _ in 0..2 {
                        for pos in [by * merge + iy, bx * merge + ix] {
                            for k in 0..nf {
                                let angle =
                                    pos as f64 * 10000f64.powf(-((2 * k) as f64) / half as f64);
                                cos.push(angle.cos() as f32);
                                sin.push(angle.sin() as f32);
                            }
                        }
                    }
                }
            }
        }
    }
    (cos, sin)
}
fn rotate(p: &mut Build<'_>, x: Value, cos: Value, sin: Value, hd: i32) -> Value {
    let right = p.slice(x, -1, hd / 2, hd);
    let negative = p.unary(U::Negate, right);
    let left = p.slice(x, -1, 0, hd / 2);
    let swapped = p.concat(vec![negative, left], -1);
    let a = p.binary(B::Multiply, x, cos);
    let b = p.binary(B::Multiply, swapped, sin);
    p.binary(B::Add, a, b)
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
    let h = c.hidden_size as i32;
    let heads = c.num_heads as i32;
    let hd = h / heads;
    let mut p = Build::new(1, schema);
    let patches = p.cast(Value::Input(0), S::BF16);
    let x = p.linear(patches, "patch_embed.proj");
    let positions = positions(&mut p, c, gh, gw);
    let mut x = p.binary(B::Add, x, positions);
    let (cos, sin) = rotary_tables(gh, gw, c.spatial_merge_size, hd as usize);
    let cos = p.graph.constant_bf16(&cos, &[n, 1, hd]);
    let sin = p.graph.constant_bf16(&sin, &[n, 1, hd]);
    for i in 0..c.depth {
        let key = |suffix: &str| format!("blocks.{i}.{suffix}");
        let norm = p.layer_norm(x, &key("norm1"), 1e-6);
        let qkv = p.linear(norm, &key("attn.qkv"));
        let qkv = p.reshape(qkv, &[n, 3, heads, hd]);
        let q = p.slice(qkv, 1, 0, 1);
        let q = p.reshape(q, &[n, heads, hd]);
        let k = p.slice(qkv, 1, 1, 2);
        let k = p.reshape(k, &[n, heads, hd]);
        let v = p.slice(qkv, 1, 2, 3);
        let v = p.reshape(v, &[n, heads, hd]);
        let q = rotate(&mut p, q, cos, sin, hd);
        let q = p.transpose(q, &[1, 0, 2]);
        let q = p.cast(q, S::F32);
        let k = rotate(&mut p, k, cos, sin, hd);
        let k = p.transpose(k, &[1, 2, 0]);
        let k = p.cast(k, S::F32);
        let v = p.transpose(v, &[1, 0, 2]);
        let v = p.cast(v, S::F32);
        let score = p.op(T::Matmul, vec![q, k]);
        let scale = p.scalar(1.0 / (hd as f32).sqrt(), S::F32);
        let score = p.binary(B::Multiply, score, scale);
        let probs = p.op(T::Softmax { axis: -1 }, vec![score]);
        let context = p.op(T::Matmul, vec![probs, v]);
        let context = p.cast(context, S::BF16);
        let context = p.transpose(context, &[1, 0, 2]);
        let context = p.contiguous(context);
        let context = p.reshape(context, &[n, h]);
        let attn = p.linear(context, &key("attn.proj"));
        x = p.binary(B::Add, x, attn);
        let norm = p.layer_norm(x, &key("norm2"), 1e-6);
        let hidden = p.linear(norm, &key("mlp.linear_fc1"));
        let width = p.width(&key("mlp.linear_fc1.weight"))?;
        let hidden = p.gelu_tanh(hidden, &[n, width]);
        let mlp = p.linear(hidden, &key("mlp.linear_fc2"));
        x = p.binary(B::Add, x, mlp);
    }
    let merge = c.spatial_merge_size as i32;
    let hidden = p.layer_norm(x, "merger.norm", 1e-6);
    let hidden = p.reshape(hidden, &[n / (merge * merge), h * merge * merge]);
    let hidden = p.linear(hidden, "merger.linear_fc1");
    let hidden = p.gelu_exact(hidden);
    let out = p.linear(hidden, "merger.linear_fc2");
    let out = p.reshape(out, &[1, n / (merge * merge), c.out_hidden_size as i32]);
    p.finish(vec![out])
}
