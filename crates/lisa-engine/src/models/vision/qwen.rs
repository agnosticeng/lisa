//! Native Qwen3-VL image encoder used by Qwen 3.5/3.8 and Qwen4-Exp.
//! Reference: mlx-serve 6c430614, sdk/vision/qwen_vision.zig (MIT),
//! and Qwen's checkpoint vision configuration. Original dense weights retained.
use crate::core::{
    loader::{Checkpoint, Shard},
    norm::bf16_scalar,
};
use anyhow::{Context, ensure};
use lisa_mlx::ops::indexing::IndexOp;
use lisa_mlx::{Array, Dtype, fast, ops};
use serde::Deserialize;
use serde_json::Value;
use std::{collections::HashMap, path::Path};

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

pub(super) struct Linear {
    pub weight: Array,
    pub bias: Option<Array>,
}
impl Linear {
    pub fn load(weights: &HashMap<String, Array>, name: &str) -> anyhow::Result<Self> {
        Ok(Self {
            weight: get(weights, &format!("{name}.weight"))?,
            bias: weights.get(&format!("{name}.bias")).cloned(),
        })
    }
    pub fn forward(&self, x: &Array) -> anyhow::Result<Array> {
        let x = x.matmul(&self.weight.transpose_axes(&[1, 0])?)?;
        Ok(if let Some(bias) = &self.bias {
            x.add(bias)?
        } else {
            x
        })
    }
}
struct Norm {
    weight: Array,
    bias: Array,
}
impl Norm {
    fn load(weights: &HashMap<String, Array>, name: &str) -> anyhow::Result<Self> {
        Ok(Self {
            weight: get(weights, &format!("{name}.weight"))?,
            bias: get(weights, &format!("{name}.bias"))?,
        })
    }
    fn forward(&self, x: &Array) -> anyhow::Result<Array> {
        Ok(fast::layer_norm(x, &self.weight, &self.bias, 1e-6)?)
    }
}
struct Block {
    norm1: Norm,
    norm2: Norm,
    qkv: Linear,
    proj: Linear,
    fc1: Linear,
    fc2: Linear,
}
pub(super) fn get(w: &HashMap<String, Array>, name: &str) -> anyhow::Result<Array> {
    w.get(name)
        .cloned()
        .with_context(|| format!("missing vision tensor {name}"))
}

pub struct Encoder {
    pub config: Config,
    pub processor: Value,
    patch: Linear,
    pos: Array,
    blocks: Vec<Block>,
    norm: Norm,
    fc1: Linear,
    fc2: Linear,
}

impl Encoder {
    pub fn load(dir: &Path) -> anyhow::Result<Self> {
        let root: Value = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let config: Config = serde_json::from_value(
            root.get("vision_config")
                .context("checkpoint has no vision_config")?
                .clone(),
        )?;
        ensure!(
            config.hidden_size > 0
                && config.num_heads > 0
                && config.hidden_size % config.num_heads == 0
                && config.hidden_size / config.num_heads % 4 == 0
                && config.depth > 0
                && config.depth <= 128,
            "unsupported Qwen vision dimensions"
        );
        ensure!(
            config.deepstack_visual_indexes.is_empty(),
            "DeepStack vision injection is not supported by this encoder"
        );
        super::admit_scratch(2usize << 30)?;
        let weights = load_weights(dir)?;
        let conv = get(&weights, "patch_embed.proj.weight")?;
        let p = config.patch_size as i32;
        let t = config.temporal_patch_size as i32;
        let h = config.hidden_size as i32;
        let conv = if conv.shape() == [h, t, p, p, 3] {
            conv.transpose_axes(&[0, 4, 1, 2, 3])?
        } else {
            ensure!(
                conv.shape() == [h, 3, t, p, p],
                "unsupported vision patch projection layout {:?}",
                conv.shape()
            );
            conv
        };
        let patch = Linear {
            weight: conv.contiguous()?.reshape(&[h, 3 * t * p * p])?,
            bias: Some(get(&weights, "patch_embed.proj.bias")?),
        };
        let mut blocks = Vec::with_capacity(config.depth);
        for i in 0..config.depth {
            let key = |suffix: &str| format!("blocks.{i}.{suffix}");
            blocks.push(Block {
                norm1: Norm::load(&weights, &key("norm1"))?,
                norm2: Norm::load(&weights, &key("norm2"))?,
                qkv: Linear::load(&weights, &key("attn.qkv"))?,
                proj: Linear::load(&weights, &key("attn.proj"))?,
                fc1: Linear::load(&weights, &key("mlp.linear_fc1"))?,
                fc2: Linear::load(&weights, &key("mlp.linear_fc2"))?,
            });
        }
        let processor = match std::fs::read(dir.join("preprocessor_config.json")) {
            Ok(bytes) => serde_json::from_slice(&bytes)?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Value::Null,
            Err(error) => return Err(error.into()),
        };
        Ok(Self {
            config,
            processor,
            patch,
            pos: get(&weights, "pos_embed.weight")?,
            blocks,
            norm: Norm::load(&weights, "merger.norm")?,
            fc1: Linear::load(&weights, "merger.linear_fc1")?,
            fc2: Linear::load(&weights, "merger.linear_fc2")?,
        })
    }

    fn positions(&self, gh: usize, gw: usize) -> anyhow::Result<Array> {
        let grid = (self.config.num_position_embeddings as f64).sqrt() as usize;
        ensure!(
            grid * grid == self.config.num_position_embeddings,
            "vision position table must be square"
        );
        let merge = self.config.spatial_merge_size;
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
        let mut output = None::<Array>;
        for k in 0..4 {
            let ids = Array::from_slice(&indices[k], &[(gh * gw) as i32]);
            let weights =
                Array::from_slice(&weights[k], &[(gh * gw) as i32, 1]).as_dtype(Dtype::Bfloat16)?;
            let value = self.pos.take_axis(&ids, 0)?.multiply(&weights)?;
            output = Some(match output {
                None => value,
                Some(old) => old.add(&value)?,
            });
        }
        Ok(output.unwrap())
    }

    fn rope(&self, gh: usize, gw: usize) -> anyhow::Result<(Array, Array)> {
        let hd = self.config.hidden_size / self.config.num_heads;
        let half = hd / 2;
        let nf = half / 2;
        let merge = self.config.spatial_merge_size;
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
        let shape = [(gh * gw) as i32, 1, hd as i32];
        Ok((
            Array::from_slice(&cos, &shape).as_dtype(Dtype::Bfloat16)?,
            Array::from_slice(&sin, &shape).as_dtype(Dtype::Bfloat16)?,
        ))
    }

    fn attention(&self, x: &Array, b: &Block, cos: &Array, sin: &Array) -> anyhow::Result<Array> {
        let n = x.dim(0);
        let heads = self.config.num_heads as i32;
        let hd = (self.config.hidden_size / self.config.num_heads) as i32;
        let qkv = b.qkv.forward(x)?.reshape(&[n, 3, heads, hd])?;
        let q = qkv.index((.., 0..1, .., ..)).reshape(&[n, heads, hd])?;
        let k = qkv.index((.., 1..2, .., ..)).reshape(&[n, heads, hd])?;
        let v = qkv.index((.., 2..3, .., ..)).reshape(&[n, heads, hd])?;
        let rotate = |x: &Array| -> anyhow::Result<Array> {
            let rotated = ops::concatenate(
                &[
                    &(-x.index((.., .., hd / 2..hd))),
                    &x.index((.., .., 0..hd / 2)),
                ],
                -1,
            )?;
            Ok(x.multiply(cos)?.add(&rotated.multiply(sin)?)?)
        };
        let q = rotate(&q)?
            .transpose_axes(&[1, 0, 2])?
            .as_dtype(Dtype::Float32)?;
        let k = rotate(&k)?
            .transpose_axes(&[1, 2, 0])?
            .as_dtype(Dtype::Float32)?;
        let v = v.transpose_axes(&[1, 0, 2])?.as_dtype(Dtype::Float32)?;
        let scores = q
            .matmul(&k)?
            .multiply(&Array::from_f32(1.0 / (hd as f32).sqrt()))?;
        let context = scores
            .softmax_axis(-1)?
            .matmul(&v)?
            .as_dtype(Dtype::Bfloat16)?
            .transpose_axes(&[1, 0, 2])?
            .contiguous()?
            .reshape(&[n, self.config.hidden_size as i32])?;
        b.proj.forward(&context)
    }

    pub fn encode(&self, image: &image::RgbImage) -> anyhow::Result<(Array, usize, usize)> {
        let pixels = super::processor::qwen(
            image,
            self.config.patch_size,
            self.config.temporal_patch_size,
            self.config.spatial_merge_size,
            &self.processor,
        )?;
        self.encode_pixels(&pixels.values, pixels.height, pixels.width)
    }

    pub fn encode_pixels(
        &self,
        values: &[f32],
        gh: usize,
        gw: usize,
    ) -> anyhow::Result<(Array, usize, usize)> {
        let n = (gh * gw) as i32;
        let width =
            3 * self.config.temporal_patch_size * self.config.patch_size * self.config.patch_size;
        ensure!(
            gh > 0
                && gw > 0
                && gh % self.config.spatial_merge_size == 0
                && gw % self.config.spatial_merge_size == 0
                && values.len() == gh * gw * width,
            "invalid vision patch shape"
        );
        super::admit_scratch(
            (gh * gw) * (gh * gw) * self.config.num_heads * 12
                + (gh * gw) * self.config.hidden_size * 64,
        )?;
        let patches = Array::from_slice(values, &[n, width as i32]).as_dtype(Dtype::Bfloat16)?;
        let mut x = self
            .patch
            .forward(&patches)?
            .add(&self.positions(gh, gw)?)?;
        let (cos, sin) = self.rope(gh, gw)?;
        for b in &self.blocks {
            x = x.add(&self.attention(&b.norm1.forward(&x)?, b, &cos, &sin)?)?;
            let hidden = b.fc1.forward(&b.norm2.forward(&x)?)?;
            x = x.add(&b.fc2.forward(&gelu_tanh(&hidden)?)?)?;
            x.eval_enqueue()?;
        }
        let merge = self.config.spatial_merge_size as i32;
        let hidden = self.norm.forward(&x)?.reshape(&[
            n / (merge * merge),
            self.config.hidden_size as i32 * merge * merge,
        ])?;
        let out = self
            .fc2
            .forward(&gelu_exact(&self.fc1.forward(&hidden)?)?)?;
        ensure!(
            out.dim(-1) as usize == self.config.out_hidden_size,
            "vision merger output width mismatch"
        );
        out.eval()?;
        Ok((
            out.reshape(&[1, n / (merge * merge), self.config.out_hidden_size as i32])?,
            gh,
            gw,
        ))
    }
}

fn gelu_tanh(x: &Array) -> anyhow::Result<Array> {
    let exponent = bf16_scalar(3.0).broadcast_to(x.shape())?;
    let cube = Array::new(lisa_mlx::mlx_rt::binary_op(
        x.t.device(),
        &x.t,
        &exponent.t,
        "Power",
    )?);
    let inner = x
        .add(&cube.multiply(&bf16_scalar(0.044715))?)?
        .multiply(&bf16_scalar(0.7978845608028654))?;
    Ok(x.multiply(&inner.unary("Tanh")?.add(&bf16_scalar(1.0))?)?
        .multiply(&bf16_scalar(0.5))?)
}
pub(super) fn gelu_exact(x: &Array) -> anyhow::Result<Array> {
    let erf = x
        .multiply(&bf16_scalar(std::f32::consts::FRAC_1_SQRT_2))?
        .unary("Erf")?;
    Ok(x.multiply(&erf.add(&bf16_scalar(1.0))?)?
        .multiply(&bf16_scalar(0.5))?)
}

pub(super) fn load_weights(dir: &Path) -> anyhow::Result<HashMap<String, Array>> {
    let checkpoint = Checkpoint::open(dir)?;
    let mut weights = HashMap::new();
    let mut shards = HashMap::<String, Shard>::new();
    for (name, file) in &checkpoint.weight_map {
        let Some(key) = ["vision_tower.", "model.visual.", "visual."]
            .iter()
            .find_map(|p| name.strip_prefix(p))
        else {
            continue;
        };
        if !shards.contains_key(file) {
            shards.insert(file.clone(), Shard::open(&dir.join(file))?);
        }
        let tensor = crate::core::quant::get_tensor(&shards[file], name)?;
        ensure!(
            tensor.dtype() == Dtype::Bfloat16
                || tensor.dtype() == Dtype::Float32
                || tensor.dtype() == Dtype::Float16,
            "quantized vision tensor {name} requires a matching quantized encoder"
        );
        ensure!(
            weights
                .insert(key.to_string(), tensor.as_dtype(Dtype::Bfloat16)?)
                .is_none(),
            "duplicate vision tensor {key}"
        );
    }
    Ok(weights)
}
