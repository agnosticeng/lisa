#![cfg(test)]
#![allow(dead_code)]

use lisa_metal::kernels::sparse::kernels as kda_kernels;
use lisa_metal::{Array, Dtype};
pub fn l2norm_last(a: &Array, eps: f32) -> lisa_metal::error::Result<Array> {
    let ms = a.multiply(a)?.contiguous()?.sum_axis(3, true)?; // [.., 1] kept dim over the last axis
    a.multiply(&ms.add(&fscalar(&ms, eps)?)?.rsqrt()?)
}

pub fn linear_projection(x: &Array, weight: &Array) -> anyhow::Result<Array> {
    let (b, t) = (x.dim(0) as usize, x.dim(1) as usize);
    let n_rows = b.checked_mul(t).unwrap_or(usize::MAX);
    if n_rows <= 8 && n_rows > 0 && weight.dtype() == Dtype::Bfloat16 {
        let input = x.reshape(&[n_rows as i32, x.dim(2)])?;
        let stream = lisa_metal::Stream::thread_local_or_default();
        let out = kda_kernels::glm53_bf16_gemv(weight, &input, &stream)
            .ok_or_else(|| anyhow::anyhow!("GLM BF16 GEMV failed for a {n_rows}-row projection"))?;
        return Ok(out.reshape(&[b as i32, t as i32, weight.dim(0)])?);
    }
    if n_rows > 8 && weight.dtype() == Dtype::Bfloat16 {
        let input = x.reshape(&[n_rows as i32, x.dim(2)])?;
        let stream = lisa_metal::Stream::thread_local_or_default();
        let out = kda_kernels::glm53_bf16_gemm(weight, &input, &stream)
            .ok_or_else(|| anyhow::anyhow!("GLM BF16 GEMM failed for a {n_rows}-row projection"))?;
        return Ok(out.reshape(&[b as i32, t as i32, weight.dim(0)])?);
    }
    let wt = weight.transpose_axes(&[1, 0])?.as_dtype(Dtype::Float32)?;
    Ok(mm(x, &wt)?)
}

/// Naive gated RMS norm over the last axis: F32 rms, weight, sigmoid gate.
pub fn gated_rms_norm(x: &Array, weight: &Array, gate: &Array, eps: f32) -> anyhow::Result<Array> {
    let xf = x.as_dtype(Dtype::Float32)?;
    let ms = xf.multiply(&xf)?.contiguous()?.mean_axis(3, true)?; // [.., 1]
    let n = xf.multiply(&ms.add(&fscalar(&ms, eps)?)?.rsqrt()?)?;
    let out = n.multiply(&weight.as_dtype(Dtype::Float32)?)?;
    let out = out.multiply(&gate.as_dtype(Dtype::Float32)?.sigmoid()?)?;
    Ok(out)
}

/// Naive RMS norm over the last axis with an F32 weight.
pub fn rms_norm_last(x: &Array, weight: Option<&Array>, eps: f32) -> anyhow::Result<Array> {
    let last = (x.rank() - 1) as i32;
    let xf = x.as_dtype(Dtype::Float32)?;
    let ms = xf.multiply(&xf)?.contiguous()?.mean_axis(last, true)?; // kept dim
    let mut n = xf.multiply(&ms.add(&fscalar(&ms, eps)?)?.rsqrt()?)?;
    if let Some(w) = weight {
        n = n.multiply(&w.as_dtype(Dtype::Float32)?)?;
    }
    Ok(n)
}

pub fn fscalar(a: &Array, v: f32) -> lisa_metal::error::Result<Array> {
    Ok(Array::from_f32_as(v, a.dtype()))
}

pub fn mm(a: &Array, w: &Array) -> lisa_metal::error::Result<Array> {
    let sh = a.shape();
    let rank = a.rank();
    let rows: i32 = sh[..rank - 1].iter().product::<i32>().max(1);
    let k = sh[rank - 1];
    let flat = a.reshape(&[rows, k])?;
    let out = flat.matmul(w)?;
    Ok(out.reshape(&[&sh[..rank - 1], &[w.shape()[1]]].concat())?)
}
