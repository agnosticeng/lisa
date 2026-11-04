use super::{silu, weights::Weights};
use anyhow::ensure;
use lisa_mlx::{Array, Dtype, ops};
use std::path::Path;

fn norm(w: &Weights, x: &Array, p: &str) -> anyhow::Result<Array> {
    let (h, width, c) = (x.dim(1), x.dim(2), x.dim(3));
    ensure!(c % 32 == 0, "VAE group norm requires 32 channel groups");
    let flat = x
        .as_dtype(Dtype::Float32)?
        .reshape(&[1, h * width, 32, c / 32])?
        .transpose_axes(&[0, 2, 1, 3])?
        .contiguous()?
        .reshape(&[1, 32, h * width * c / 32])?;
    let xc = flat.subtract(&flat.mean_axis(-1, true)?)?;
    let var = xc.multiply(&xc)?.mean_axis(-1, true)?;
    let y = xc
        .multiply(&var.add(&Array::from_f32(1e-6))?.rsqrt()?)?
        .reshape(&[1, 32, h * width, c / 32])?
        .transpose_axes(&[0, 2, 1, 3])?
        .contiguous()?
        .reshape(x.shape())?;
    Ok(
        y.multiply(&w.get(&format!("{p}.weight"))?.as_dtype(Dtype::Float32)?)?
            .add(&w.get(&format!("{p}.bias"))?.as_dtype(Dtype::Float32)?)?
            .as_dtype(x.dtype())?,
    )
}

/// Exact pixel gather into im2col followed by the existing MLX dense matmul.
/// Spatial rows are bounded to 1024 at a time, independent of output resolution.
fn conv(w: &Weights, x: &Array, p: &str) -> anyhow::Result<Array> {
    let weight = w.get(&format!("{p}.weight"))?;
    let bias = w.get(&format!("{p}.bias"))?;
    let (height, width, channels) = (x.dim(1), x.dim(2), x.dim(3));
    let shape = weight.shape();
    ensure!(
        shape.len() == 4
            && shape[3] == channels
            && shape[1] == shape[2]
            && [1, 3].contains(&shape[1]),
        "unsupported VAE convolution {p}"
    );
    let kernel = shape[1];
    let output = shape[0];
    let rows = height * width;
    let flat = x.contiguous()?.reshape(&[rows, channels])?;
    let wt = weight
        .contiguous()?
        .reshape(&[output, kernel * kernel * channels])?
        .transpose_axes(&[1, 0])?;
    if kernel == 1 {
        return Ok(flat
            .matmul(&wt)?
            .add(&bias)?
            .reshape(&[1, height, width, output])?);
    }
    let zero =
        Array::from_slice(&vec![0f32; channels as usize], &[1, channels]).as_dtype(x.dtype())?;
    let source = ops::concatenate(&[&flat, &zero], 0)?;
    let mut pieces = Vec::new();
    for start in (0..rows).step_by(1024) {
        let count = (rows - start).min(1024);
        let mut ids = Vec::with_capacity((count * 9) as usize);
        for row in start..start + count {
            for ky in 0..3 {
                for kx in 0..3 {
                    let y = row / width + ky - 1;
                    let xx = row % width + kx - 1;
                    ids.push(if y < 0 || y >= height || xx < 0 || xx >= width {
                        rows
                    } else {
                        y * width + xx
                    });
                }
            }
        }
        let patches = source
            .take_axis(&Array::from_slice(&ids, &[count * 9]), 0)?
            .reshape(&[count, 9 * channels])?;
        let out = patches.matmul(&wt)?.add(&bias)?;
        out.eval_enqueue()?;
        pieces.push(out);
    }
    Ok(ops::concatenate(&pieces.iter().collect::<Vec<_>>(), 0)?
        .reshape(&[1, height, width, output])?)
}
fn resnet(w: &Weights, x: &Array, p: &str) -> anyhow::Result<Array> {
    let h = conv(
        w,
        &silu(&norm(w, x, &format!("{p}.norm1"))?)?,
        &format!("{p}.conv1"),
    )?;
    let h = conv(
        w,
        &silu(&norm(w, &h, &format!("{p}.norm2"))?)?,
        &format!("{p}.conv2"),
    )?;
    let skip = if w.0.contains_key(&format!("{p}.conv_shortcut.weight")) {
        conv(w, x, &format!("{p}.conv_shortcut"))?
    } else {
        x.clone()
    };
    Ok(h.add(&skip)?)
}
fn upsample(x: &Array) -> anyhow::Result<Array> {
    let (h, w, c) = (x.dim(1), x.dim(2), x.dim(3));
    let y: Vec<i32> = (0..h * 2).map(|i| i / 2).collect();
    let xx: Vec<i32> = (0..w * 2).map(|i| i / 2).collect();
    Ok(x.take_axis(&Array::from_slice(&y, &[h * 2]), 1)?
        .take_axis(&Array::from_slice(&xx, &[w * 2]), 2)?
        .reshape(&[1, h * 2, w * 2, c])?)
}
pub fn decode(dir: &Path, packed: &Array) -> anyhow::Result<Array> {
    let w = Weights::load(&dir.join("vae"))?;
    decode_weights(&w, packed, Dtype::Bfloat16)
}

/// FLUX.2 uses MLX BHWC/bf16 weights; FLUX 3 maps its native names and decodes in f32.
pub(crate) fn decode_weights(w: &Weights, packed: &Array, dtype: Dtype) -> anyhow::Result<Array> {
    let (h, width) = (packed.dim(2), packed.dim(3));
    let mean = w
        .get("bn.running_mean")?
        .as_dtype(Dtype::Float32)?
        .reshape(&[1, 128, 1, 1])?;
    let std = w
        .get("bn.running_var")?
        .as_dtype(Dtype::Float32)?
        .add(&Array::from_f32(1e-4))?
        .sqrt()?
        .reshape(&[1, 128, 1, 1])?;
    let x = packed
        .as_dtype(Dtype::Float32)?
        .multiply(&std)?
        .add(&mean)?;
    let x = x
        .reshape(&[1, 32, 2, 2, h, width])?
        .transpose_axes(&[0, 1, 4, 2, 5, 3])?
        .contiguous()?
        .reshape(&[1, 32, h * 2, width * 2])?
        .transpose_axes(&[0, 2, 3, 1])?
        .as_dtype(dtype)?;
    let mut x = conv(&w, &conv(&w, &x, "post_quant_conv")?, "decoder.conv_in")?;
    x = resnet(&w, &x, "decoder.mid_block.resnets.0")?;
    let p = "decoder.mid_block.attentions.0";
    let n = norm(&w, &x, &format!("{p}.group_norm"))?;
    let c = x.dim(3);
    let shape = [1, 1, x.dim(1) * x.dim(2), c];
    let q = w.lin(&format!("{p}.to_q"), &n)?.reshape(&shape)?;
    let k = w.lin(&format!("{p}.to_k"), &n)?.reshape(&shape)?;
    let v = w.lin(&format!("{p}.to_v"), &n)?.reshape(&shape)?;
    let at = crate::models::media_attention::attention(&q, &k, &v, 1.0 / (c as f32).sqrt(), None)?
        .reshape(x.shape())?;
    x = x.add(&w.lin(&format!("{p}.to_out"), &at)?)?;
    x = resnet(&w, &x, "decoder.mid_block.resnets.1")?;
    for block in 0..4 {
        for layer in 0..3 {
            x = resnet(
                &w,
                &x,
                &format!("decoder.up_blocks.{block}.resnets.{layer}"),
            )?;
            x.eval_enqueue()?;
        }
        if block < 3 {
            x = conv(
                &w,
                &upsample(&x)?,
                &format!("decoder.up_blocks.{block}.upsamplers.0.conv"),
            )?;
        }
    }
    x = conv(
        &w,
        &silu(&norm(&w, &x, "decoder.conv_norm_out")?)?,
        "decoder.conv_out",
    )?;
    x.eval()?;
    Ok(x)
}
