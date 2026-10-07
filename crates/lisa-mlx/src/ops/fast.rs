//! `lisa_mlx::fast`-shaped wrappers for the fused libraries.
use super::*;
use crate::jit;

/// Mask modes for scaled dot product attention.
pub enum ScaledDotProductAttentionMask<'a> {
    Array(&'a Array),
    Causal,
}

/// Mirrors `lisa_mlx`'s `IntoOption<ScaledDotProductAttentionMask>` so call
/// sites can pass either `Causal` or `&mask` directly.
pub trait MaskIntoOption<'a> {
    fn into_option(self) -> Option<ScaledDotProductAttentionMask<'a>>;
}

impl<'a> MaskIntoOption<'a> for ScaledDotProductAttentionMask<'a> {
    fn into_option(self) -> Option<ScaledDotProductAttentionMask<'a>> {
        Some(self)
    }
}

impl<'a> MaskIntoOption<'a> for &'a Array {
    fn into_option(self) -> Option<ScaledDotProductAttentionMask<'a>> {
        Some(ScaledDotProductAttentionMask::Array(self))
    }
}

pub fn scaled_dot_product_attention<'a>(
    queries: &Array,
    keys: &Array,
    values: &Array,
    scale: f32,
    mask: impl MaskIntoOption<'a>,
    _sinks: Option<&Array>,
) -> Result<Array> {
    let (do_causal, mask_t) = match mask.into_option() {
        Some(ScaledDotProductAttentionMask::Causal) => (true, None),
        Some(ScaledDotProductAttentionMask::Array(m)) => (false, Some(&m.t)),
        None => (false, None),
    };
    Ok(Array::new(jit::sdpa(
        queries.t.device(),
        &queries.t,
        &keys.t,
        &values.t,
        scale,
        do_causal,
        mask_t,
    )?))
}

pub fn rms_norm(x: &Array, weight: Option<&Array>, eps: f32) -> Result<Array> {
    let w = weight.map(|w| &w.t);
    Ok(Array::new(jit::rms_norm(x.t.device(), &x.t, w, eps)?))
}

/// Fused residual-add + RMSNorm (specs/08 §1): returns `(sum, normed)` where
/// `sum = bf16(x + r)` and `normed = rms_norm(sum, weight, eps)` — one
/// dispatch, bit-identical to the composed pair.
pub fn fused_add_rms_norm(
    x: &Array,
    r: &Array,
    weight: &Array,
    eps: f32,
) -> Result<(Array, Array)> {
    let (sum, normed) = jit::fused_add_rms_norm(x.t.device(), &x.t, &r.t, &weight.t, eps)?;
    Ok((Array::new(sum), Array::new(normed)))
}

pub fn rope(
    x: &Array,
    dimensions: i32,
    traditional: bool,
    base: Option<f32>,
    scale: f32,
    offset: i32,
    _freqs: Option<&Array>,
) -> Result<Array> {
    let off = Array::from_slice(&[offset], &[1]);
    Ok(Array::new(jit::rope(
        x.t.device(),
        &x.t,
        dimensions,
        traditional,
        base.unwrap_or(10000.0),
        scale,
        &off.t,
        true,
    )?))
}
