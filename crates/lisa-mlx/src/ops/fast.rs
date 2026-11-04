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

impl<'a> MaskIntoOption<'a> for Option<ScaledDotProductAttentionMask<'a>> {
    fn into_option(self) -> Option<ScaledDotProductAttentionMask<'a>> {
        self
    }
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

pub fn layer_norm(x: &Array, weight: &Array, bias: &Array, eps: f32) -> Result<Array> {
    let x = x.contiguous()?;
    Ok(Array::new(jit::layer_norm(
        x.t.device(),
        &x.t,
        &weight.t,
        &bias.t,
        eps,
    )?))
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

#[cfg(test)]
mod offset_tests {
    use super::*;
    use crate::ops::{Dtype, indexing::IndexOp};

    fn bits(array: &Array) -> Result<Vec<u32>> {
        Ok(array
            .as_dtype(Dtype::Float32)?
            .reshape(&[-1])?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    }

    #[test]
    fn layer_norm_supports_offset_views_and_both_reduction_families() -> Result<()> {
        for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
            for width in [64, 768, 1152, 4096, 8192] {
                let data: Vec<f32> = (0..3 * width)
                    .map(|i| ((i * 17 % 131) as f32 - 65.) * 0.125)
                    .collect();
                let x = Array::from_slice(&data, &[3, width])
                    .as_dtype(dtype)?
                    .index((1..3, ..));
                let w = Array::from_slice(&vec![1.25f32; 2 * width as usize], &[2 * width])
                    .as_dtype(dtype)?
                    .index(width..2 * width);
                let b = Array::from_slice(&vec![0.125f32; 2 * width as usize], &[2 * width])
                    .as_dtype(dtype)?
                    .index(width..2 * width);
                let actual = layer_norm(&x, &w, &b, 1e-6)?;
                let expected = layer_norm(&x.copied()?, &w.copied()?, &b.copied()?, 1e-6)?;
                assert_eq!(bits(&actual)?, bits(&expected)?, "{dtype:?} width={width}");
                let values = actual
                    .as_dtype(Dtype::Float32)?
                    .reshape(&[-1])?
                    .to_vec1::<f32>()?;
                assert!(values.iter().all(|v| v.is_finite()));
                if dtype == Dtype::Float32 {
                    for (input, output) in data[width as usize..]
                        .chunks(width as usize)
                        .zip(values.chunks(width as usize))
                    {
                        let mean =
                            input.iter().map(|v| f64::from(*v)).sum::<f64>() / f64::from(width);
                        let var = input
                            .iter()
                            .map(|v| (f64::from(*v) - mean).powi(2))
                            .sum::<f64>()
                            / f64::from(width);
                        for (&a, &b) in input.iter().zip(output) {
                            let expected =
                                (f64::from(a) - mean) / (var + 1e-6).sqrt() * 1.25 + 0.125;
                            assert!((f64::from(b) - expected).abs() < 2e-6);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn normalization_weight_views_use_byte_offsets() -> Result<()> {
        for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
            for width in [128, 4096, 5120] {
                let x = Array::from_slice(
                    &(0..3 * width)
                        .map(|i| (i % 97) as f32 * 0.013 - 0.4)
                        .collect::<Vec<_>>(),
                    &[3, width],
                )
                .as_dtype(dtype)?
                .index((1..3, ..));
                let r = Array::from_slice(&vec![0.125f32; 3 * width as usize], &[3, width])
                    .as_dtype(dtype)?
                    .index((1..3, ..));
                let mut weights = vec![f32::NAN; width as usize];
                weights.extend((0..width).map(|i| 0.5 + (i % 31) as f32 * 0.031));
                let w = Array::from_slice(&weights, &[2 * width])
                    .as_dtype(dtype)?
                    .index(width..2 * width);
                let copied = w.copied()?;
                let expected = rms_norm(&x, Some(&copied), 1e-6)?;
                let actual = rms_norm(&x, Some(&w), 1e-6)?;
                assert_eq!(
                    bits(&actual)?,
                    bits(&expected)?,
                    "rms {dtype:?} width={width}"
                );
                let (es, en) = fused_add_rms_norm(&x, &r, &copied, 1e-6)?;
                let (asum, an) = fused_add_rms_norm(&x, &r, &w, 1e-6)?;
                assert_eq!(bits(&asum)?, bits(&es)?);
                assert_eq!(bits(&an)?, bits(&en)?, "add+rms {dtype:?} width={width}");
                assert!(
                    an.as_dtype(Dtype::Float32)?
                        .reshape(&[-1])?
                        .to_vec1::<f32>()?
                        .iter()
                        .all(|v| v.is_finite())
                );
            }
        }
        Ok(())
    }

    #[test]
    fn rotary_views_use_byte_offsets() -> Result<()> {
        for dtype in [Dtype::Float32, Dtype::Float16, Dtype::Bfloat16] {
            for rows in [1, 3] {
                let x = Array::from_slice(
                    &(0..2 * rows * 128)
                        .map(|i| (i % 113) as f32 * 0.017)
                        .collect::<Vec<_>>(),
                    &[2, rows, 128],
                )
                .as_dtype(dtype)?
                .index((1..2, .., ..));
                let offset = Array::from_slice(&[777i32, 11], &[2]).index(1..2);
                let xc = x.copied()?;
                let oc = offset.copied()?;
                for traditional in [false, true] {
                    let apply = |a: &Array, o: &Array| {
                        jit::rope(
                            a.t.device(),
                            &a.t,
                            128,
                            traditional,
                            10000.0,
                            1.0,
                            &o.t,
                            true,
                        )
                        .map(Array::new)
                    };
                    assert_eq!(
                        bits(&apply(&x, &offset)?)?,
                        bits(&apply(&xc, &oc)?)?,
                        "rope {dtype:?} rows={rows}"
                    );
                }
            }
        }
        Ok(())
    }
}
