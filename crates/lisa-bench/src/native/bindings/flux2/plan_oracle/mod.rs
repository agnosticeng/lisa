#![cfg(test)]
use lisa_metal::{Array, Dtype};
pub mod dit2;
pub mod dit3;
pub mod text2;
pub mod weights;
pub(crate) fn silu(x: &Array) -> anyhow::Result<Array> {
    Ok(x.multiply(&lisa_metal::ops::sigmoid(x)?)?)
}
pub(crate) fn layer_norm(x: &Array) -> anyhow::Result<Array> {
    let x = x.as_dtype(Dtype::Float32)?;
    let xc = x.subtract(&x.mean_axis(-1, true)?)?;
    let inv = xc
        .multiply(&xc)?
        .mean_axis(-1, true)?
        .add(Array::from_f32(1e-6))?
        .rsqrt()?;
    Ok(xc.multiply(&inv)?.as_dtype(Dtype::Bfloat16)?)
}

pub mod text3;
