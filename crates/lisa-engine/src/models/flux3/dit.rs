//! Rust/Metal adaptation of BFL flux3/dit.py, revision bfc8889a4497f846416f253a3972a1941ac698b9.
//! Copyright 2026 Black Forest Labs, Inc. Apache-2.0; see LICENSE.
//! Text-to-image, one sample: empty conditioning streams have no attended tokens.
use crate::models::{
    flux2::{layer_norm, silu, weights::Weights},
    media_attention,
};
use anyhow::ensure;
use lisa_mlx::ops::indexing::{Ellipsis, IndexOp};
use lisa_mlx::{Array, Dtype, fast, ops};
use std::path::Path;

const HIDDEN: i32 = 3072;
const HEADS: i32 = 24;
const HEAD_DIM: i32 = 128;
type Modulation = [Array; 3];

pub struct Dit {
    weights: Weights,
}
fn slice(x: &Array, a: i32, b: i32) -> Array {
    x.index((Ellipsis, a..b))
}

/// Adjacent pairs, four independent axes, theta 10000. Image coordinates use 0.5.
fn rotations(ids: &[[f32; 4]]) -> (Array, Array) {
    let mut cos = Vec::with_capacity(ids.len() * 64);
    let mut sin = Vec::with_capacity(ids.len() * 64);
    for id in ids {
        for &pos in id {
            for k in 0..16 {
                let a = pos as f64 / 10000f64.powf((2 * k) as f64 / 32.0);
                cos.push(a.cos() as f32);
                sin.push(a.sin() as f32);
            }
        }
    }
    let shape = [1, 1, ids.len() as i32, 64];
    (
        Array::from_slice(&cos, &shape),
        Array::from_slice(&sin, &shape),
    )
}
fn modulate(x: &Array, m: &Modulation) -> anyhow::Result<Array> {
    Ok(layer_norm(x)?
        .multiply(&m[1].add(&Array::from_f32_as(1.0, Dtype::Bfloat16))?)?
        .add(&m[0])?)
}
impl Dit {
    pub fn load(dir: &Path) -> anyhow::Result<Self> {
        // The image_cond tower is unused for text-to-image. Its weights remain on disk.
        let weights = Weights::load_file(&dir.join("model.safetensors"), |key| {
            !key.contains("image_cond")
        })?;
        ensure!(
            weights.get("emb_in.image.weight")?.shape() == [HIDDEN, 128]
                && weights.get("txt_in.weight")?.shape() == [HIDDEN, 20480]
                && weights.get("ref_time_in.in_layer.weight")?.shape() == [HIDDEN, 256]
                && weights.count("content_mode_blocks.image", "q_proj.weight") == 5
                && weights.count("txt_mode_blocks", "q_proj.weight") == 5
                && weights.count("single_blocks", "q_proj.weight") == 28,
            "expected the native BF16 distilled FLUX 3 Image Klein checkpoint"
        );
        ensure!(
            weights.0.values().all(|w| w.dtype() == Dtype::Bfloat16),
            "FLUX 3 transformer requires native BF16 weights; FP8/NVFP4 are not supported"
        );
        Ok(Self { weights })
    }
    fn time(&self, value: f32, prefix: &str) -> anyhow::Result<Array> {
        // The reference first stores time in bf16 and multiplies by 1000 in bf16.
        let t = half::bf16::from_f32(half::bf16::from_f32(value).to_f32() * 1000.0).to_f32();
        let mut features = [0f32; 256];
        for i in 0..128 {
            let a = t * (-10000f32.ln() * i as f32 / 128.0).exp();
            features[i] = a.cos();
            features[i + 128] = a.sin();
        }
        let x = Array::from_slice(&features, &[1, 1, 256]).as_dtype(Dtype::Bfloat16)?;
        self.weights.lin(
            &format!("{prefix}.out_layer"),
            &silu(&self.weights.lin(&format!("{prefix}.in_layer"), &x)?)?,
        )
    }
    fn mods(&self, vec: &Array, prefix: &str) -> anyhow::Result<Modulation> {
        let m = self.weights.lin(&format!("{prefix}.lin"), &silu(vec)?)?;
        Ok([
            slice(&m, 0, HIDDEN),
            slice(&m, HIDDEN, 2 * HIDDEN),
            slice(&m, 2 * HIDDEN, 3 * HIDDEN),
        ])
    }
    fn heads(&self, x: &Array) -> anyhow::Result<Array> {
        Ok(x.reshape(&[1, x.dim(1), HEADS, HEAD_DIM])?
            .transpose_axes(&[0, 2, 1, 3])?)
    }
    fn norm_rope(&self, x: &Array, key: &str, rot: &(Array, Array)) -> anyhow::Result<Array> {
        let x = self.heads(x)?.as_dtype(Dtype::Float32)?.contiguous()?;
        let scale = self.weights.get(key)?.as_dtype(Dtype::Float32)?;
        // Keep norm and rotation in f32, with only one cast after rotation.
        let x = fast::rms_norm(&x, Some(&scale), 1e-6)?;
        let n = x.dim(2);
        let pairs = x.reshape(&[1, HEADS, n, 64, 2])?;
        let re = pairs.index((Ellipsis, 0..1)).reshape(&[1, HEADS, n, 64])?;
        let im = pairs.index((Ellipsis, 1..2)).reshape(&[1, HEADS, n, 64])?;
        let r = re.multiply(&rot.0)?.subtract(&im.multiply(&rot.1)?)?;
        let i = re.multiply(&rot.1)?.add(&im.multiply(&rot.0)?)?;
        Ok(
            ops::concatenate(&[&r.expand_dims(-1)?, &i.expand_dims(-1)?], -1)?
                .reshape(&[1, HEADS, n, HEAD_DIM])?
                .as_dtype(Dtype::Bfloat16)?,
        )
    }
    fn block(
        &self,
        x: &Array,
        prefix: &str,
        m: &Modulation,
        rot: &(Array, Array),
    ) -> anyhow::Result<Array> {
        let w = &self.weights;
        let n = modulate(x, m)?;
        let q = self.norm_rope(
            &w.lin(&format!("{prefix}.q_proj"), &n)?,
            &format!("{prefix}.norm.query_norm.scale"),
            rot,
        )?;
        let k = self.norm_rope(
            &w.lin(&format!("{prefix}.k_proj"), &n)?,
            &format!("{prefix}.norm.key_norm.scale"),
            rot,
        )?;
        let v = self.heads(&w.lin(&format!("{prefix}.v_proj"), &n)?)?;
        let attn = media_attention::attention(&q, &k, &v, 1.0 / (HEAD_DIM as f32).sqrt(), None)?
            .transpose_axes(&[0, 2, 1, 3])?
            .contiguous()?
            .reshape(x.shape())?;
        let ff = w.lin(&format!("{prefix}.mlp_in"), &n)?;
        let half = ff.dim(-1) / 2;
        let ff = silu(&slice(&ff, 0, half))?.multiply(&slice(&ff, half, 2 * half))?;
        let out = w
            .lin(&format!("{prefix}.attn_out"), &attn)?
            .add(&w.lin(&format!("{prefix}.mlp_out"), &ff)?)?;
        let out = x.add(&m[2].multiply(&out)?)?;
        out.eval_enqueue()?;
        Ok(out)
    }
    pub fn forward(
        &self,
        latent: &Array,
        enc: &Array,
        t: f32,
        target: f32,
        h: i32,
        width: i32,
    ) -> anyhow::Result<Array> {
        let w = &self.weights;
        let ni = h * width;
        let nt = enc.dim(1);
        ensure!(
            latent.shape() == [1, ni, 128] && enc.dim(-1) == 20480,
            "invalid FLUX 3 token shapes"
        );
        let reference = self.time(target, "ref_time_in")?;
        let vi = self.time(t, "time_in")?.add(&reference)?;
        let vt = self.time(0.0, "time_in")?.add(&reference)?;
        let mi = self.mods(&vi, "early_stream_modulations.image")?;
        let mt = self.mods(&vt, "early_stream_modulations.txt")?;
        let si = self.mods(&vi, "single_stream_modulations.image")?;
        let st = self.mods(&vt, "single_stream_modulations.txt")?;
        let ii: Vec<[f32; 4]> = (0..ni)
            .map(|p| [0.0, (p / width) as f32 * 0.5, (p % width) as f32 * 0.5, 0.0])
            .collect();
        let ti: Vec<[f32; 4]> = (0..nt).map(|p| [0.0, 0.0, 0.0, p as f32]).collect();
        let ri = rotations(&ii);
        let rt = rotations(&ti);
        let mut image = w.lin("emb_in.image", &latent.as_dtype(Dtype::Bfloat16)?)?;
        let mut text = w.lin("txt_in", enc)?;
        // Independent attention in the early towers; joint attention begins only below.
        for depth in 0..5 {
            image = self.block(
                &image,
                &format!("content_mode_blocks.image.{depth}"),
                &mi,
                &ri,
            )?;
            text = self.block(&text, &format!("txt_mode_blocks.{depth}"), &mt, &rt)?;
        }
        let joined: Vec<_> = ii.into_iter().chain(ti).collect();
        let r = rotations(&joined);
        let broadcast =
            |x: &Array, n| -> anyhow::Result<Array> { Ok(x.broadcast_to(&[1, n, HIDDEN])?) };
        let m: Modulation = [0, 1, 2]
            .map(|i| -> anyhow::Result<Array> {
                Ok(ops::concatenate(
                    &[&broadcast(&si[i], ni)?, &broadcast(&st[i], nt)?],
                    1,
                )?)
            })
            .into_iter()
            .collect::<anyhow::Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| anyhow::anyhow!("expected three modulation components"))?;
        let mut x = ops::concatenate(&[&image, &text], 1)?;
        for depth in 0..28 {
            x = self.block(&x, &format!("single_blocks.{depth}"), &m, &r)?;
        }
        let out = w.lin("final_layer.image.adaLN_modulation.1", &silu(&vi)?)?;
        let x = layer_norm(&x.index((.., 0..ni, ..)))?
            .multiply(
                &slice(&out, HIDDEN, 2 * HIDDEN).add(&Array::from_f32_as(1.0, Dtype::Bfloat16))?,
            )?
            .add(&slice(&out, 0, HIDDEN))?;
        w.lin("final_layer.image.linear", &x)
    }
}
