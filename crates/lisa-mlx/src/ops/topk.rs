//! Exact small candidate selection without sorting the rest of the vocabulary.

use super::{Array, Dtype, Result, Stream};
use crate::ffi::{KernelInput, MetalKernel, OutputArg, TemplateArg};
use std::sync::OnceLock;

const HEADER: &str = r#"
inline bool lisa_topk_before(float a, uint ai, float b, uint bi) {
    if (ai == UINT_MAX) return false;
    if (bi == UINT_MAX) return true;
    const bool an = metal::isnan(a), bn = metal::isnan(b);
    if (an != bn) return !an;
    if (!an && a != b) return a > b;
    return ai < bi;
}
"#;

fn block_kernel() -> Result<&'static MetalKernel> {
    static KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();
    KERNEL
        .get_or_init(|| {
            MetalKernel::new_with_math(
                "stable_top32_block_v1",
                &["scores", "width", "blocks"],
                &["candidates"],
                crate::kernel_source!("common/reduce/topk32_block.metal"),
                HEADER,
                false,
                false,
                crate::runtime::Math::Safe,
            )
            // Every block writes all 32 slots, including padding sentinels.
            .map(MetalKernel::with_full_write_outputs)
        })
        .as_ref()
        .map_err(|error| crate::error::Error::msg(error.to_string()))
}

fn merge_kernel() -> Result<&'static MetalKernel> {
    static KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();
    KERNEL
        .get_or_init(|| {
            MetalKernel::new_with_math(
                "stable_top32_merge_v1",
                &["scores", "candidates", "width", "runs", "out_runs"],
                &["merged"],
                crate::kernel_source!("common/reduce/topk32_merge.metal"),
                HEADER,
                false,
                false,
                crate::runtime::Math::Safe,
            )
            // A pair contains at least one complete 512-element block, so
            // ranks 0..31 are all written. A lone run copies all 32 slots.
            .map(MetalKernel::with_full_write_outputs)
        })
        .as_ref()
        .map_err(|error| crate::error::Error::msg(error.to_string()))
}

/// Last-axis top-32 IDs in descending value order, ties by increasing ID.
/// Matches the first 32 IDs of a stable ascending argsort of the negated row,
/// including signed zero, infinities and NaNs. No input arithmetic is changed.
pub fn top32_indices(scores: &Array) -> Result<Array> {
    let shape = scores.shape();
    let Some(&width) = shape.last() else {
        crate::bail!("top32 requires a non-scalar array");
    };
    if width < 32
        || !matches!(
            scores.dtype(),
            Dtype::Float32 | Dtype::Bfloat16 | Dtype::Float16
        )
    {
        crate::bail!("top32 requires a floating-point row with at least 32 entries");
    }
    let rows = shape[..shape.len() - 1]
        .iter()
        .try_fold(1i32, |n, &d| n.checked_mul(d))
        .ok_or_else(|| crate::error::Error::msg("top32 row count overflow"))?;
    if rows == 0 {
        crate::bail!("top32 requires at least one row");
    }
    let scores = scores.contiguous()?;
    scores.eval_enqueue()?;
    let stream = Stream::thread_local_or_default();
    const NTH: i32 = 512;
    let mut runs = (width as usize).div_ceil(NTH as usize) as i32;
    let grid = runs
        .checked_mul(NTH)
        .ok_or_else(|| crate::error::Error::msg("top32 grid overflow"))?;
    let scratch_width = runs
        .checked_mul(32)
        .ok_or_else(|| crate::error::Error::msg("top32 scratch overflow"))?;
    let mut candidates = block_kernel()?
        .apply_with_inline_inputs(
            &[
                KernelInput::Tensor(&scores.t),
                KernelInput::I32(width),
                KernelInput::I32(runs),
            ],
            &[TemplateArg::Int("NTH", NTH)],
            (grid, rows, 1),
            (NTH, 1, 1),
            &[OutputArg {
                shape: vec![rows, scratch_width],
                dtype: Dtype::Uint32,
            }],
            &stream,
        )?
        .remove(0);
    while runs > 1 {
        let out_runs = (runs + 1) / 2;
        // A lone trailing run may contain fewer than 32 real IDs. Its holes
        // must retain UINT_MAX for the next merge, not recycled buffer bytes.
        let mut outputs = merge_kernel()?.apply_with_inline_inputs(
            &[
                KernelInput::Tensor(&scores.t),
                KernelInput::Tensor(&candidates.t),
                KernelInput::I32(width),
                KernelInput::I32(runs),
                KernelInput::I32(out_runs),
            ],
            &[],
            (out_runs * 64, rows, 1),
            (64, 1, 1),
            &[OutputArg {
                shape: vec![rows, out_runs * 32],
                dtype: Dtype::Uint32,
            }],
            &stream,
        )?;
        candidates = outputs.remove(0);
        runs = out_runs;
    }
    let mut output_shape = shape.to_vec();
    *output_shape.last_mut().expect("non-scalar") = 32;
    candidates.reshape(&output_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top32_matches_stable_sort_and_host() -> Result<()> {
        for dtype in [Dtype::Float32, Dtype::Bfloat16, Dtype::Float16] {
            for width in [32, 33, 511, 512, 513, 1025, 1537, 4097, 248320] {
                let rows = 2;
                let values: Vec<f32> = (0..rows * width)
                    .map(|i| match if i < width { i % 97 } else { 5 } {
                        0 => f32::NAN,
                        1 => f32::INFINITY,
                        2 => f32::NEG_INFINITY,
                        3 => -0.0,
                        4 => 0.0,
                        _ => ((i * 137 + 17) % 1009) as f32 / 19.0 - 20.0,
                    })
                    .collect();
                let x = Array::from_slice(&values, &[rows, width]).as_dtype(dtype)?;
                let actual = top32_indices(&x)?.reshape(&[-1])?.to_vec1::<u32>()?;
                let sorted = super::super::argsort(&(-&x))?
                    .reshape(&[-1])?
                    .to_vec1::<u32>()?;
                let host_values = x
                    .as_dtype(Dtype::Float32)?
                    .reshape(&[-1])?
                    .to_vec1::<f32>()?;
                for row in 0..rows as usize {
                    let mut host: Vec<u32> = (0..width as u32).collect();
                    host.sort_by(|&a, &b| {
                        let av = host_values[row * width as usize + a as usize];
                        let bv = host_values[row * width as usize + b as usize];
                        av.is_nan()
                            .cmp(&bv.is_nan())
                            .then_with(|| bv.partial_cmp(&av).unwrap_or(std::cmp::Ordering::Equal))
                            .then(a.cmp(&b))
                    });
                    let got = &actual[row * 32..(row + 1) * 32];
                    assert_eq!(got, &host[..32], "host {dtype:?} width={width} row={row}");
                    assert_eq!(
                        got,
                        &sorted[row * width as usize..row * width as usize + 32],
                        "sort {dtype:?} width={width} row={row}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn top32_ties_nan_padding_and_offset() -> Result<()> {
        use super::super::indexing::IndexOp;
        let mut values = vec![f32::INFINITY; 37];
        values.extend((0..513).map(|i| i as f32));
        let view = Array::from_slice(&values, &[550]).index(37..550);
        assert_eq!(
            top32_indices(&view)?.to_vec1::<u32>()?,
            (481..513).rev().collect::<Vec<u32>>()
        );
        for value in [0.0, -0.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let x = Array::from_slice(&vec![value; 2048], &[2, 1024]);
            let slice = x.index((.., 3..516));
            let ids = top32_indices(&slice)?.reshape(&[-1])?.to_vec1::<u32>()?;
            assert_eq!(ids, (0..32).chain(0..32).collect::<Vec<u32>>());
        }
        Ok(())
    }
}
