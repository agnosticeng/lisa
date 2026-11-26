//! Provider-native conformance tests for the portable GGUF binding.
#[cfg(test)]
use crate::core::cache::{FullAttentionCache, LayerCache};
#[cfg(test)]
use anyhow::Result;
#[cfg(test)]
use lisa_metal::{Array, Dtype, ops::indexing::IndexOp};
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn f32_attention_keeps_serial_reductions_in_wide_windows() -> Result<()> {
        for prefix in [0, 37, 2048] {
            for rows in [1, 7, 16] {
                let len = prefix + rows;
                let q = Array::from_slice(
                    &(0..24 * rows * 256)
                        .map(|i| (i % 113) as f32 * 0.017 - 0.7)
                        .collect::<Vec<_>>(),
                    &[1, 24, rows, 256],
                );
                let data: Vec<_> = (0..4 * (len + 16) * 256)
                    .map(|i| (i % 131) as f32 * 0.011 - 0.6)
                    .collect();
                // A live cache is a strided view of a larger allocation.
                let k =
                    Array::from_slice(&data, &[1, 4, len + 16, 256]).index((.., .., 0..len, ..));
                let v = k.multiply(Array::from_f32(-0.7))?;
                let apply = |q: &Array, k: &Array, v: &Array, causal| {
                    lisa_metal::jit::sdpa_vector(
                        q.t.device(),
                        &q.t,
                        &k.t,
                        &v.t,
                        1. / 16.,
                        causal,
                        None,
                    )
                    .map(Array::new)
                };
                let wide = apply(&q, &k, &v, true)?;
                for i in 0..rows {
                    let single = apply(
                        &q.index((.., .., i..i + 1, ..)).contiguous()?,
                        &k.index((.., .., 0..prefix + i + 1, ..)),
                        &v.index((.., .., 0..prefix + i + 1, ..)),
                        false,
                    )?;
                    assert_eq!(
                        single.reshape(&[-1])?.to_vec1::<f32>()?,
                        wide.index((.., .., i..i + 1, ..))
                            .contiguous()?
                            .reshape(&[-1])?
                            .to_vec1::<f32>()?,
                        "prefix={prefix}, rows={rows}, query={i}"
                    );
                }
            }
        }
        Ok(())
    }
    #[test]
    fn f32_cache_keeps_precision_across_growth_and_restore() -> Result<()> {
        let values: Vec<f32> = (0..260 * 256)
            .map(|i| 1.0001 + (i % 19) as f32 * 0.0001)
            .collect();
        let all = Array::from_slice(&values, &[1, 1, 260, 256]);
        let mut cache = FullAttentionCache::with_dtype(
            crate::metal_context::execution_context(),
            1,
            256,
            Dtype::Float32,
        );
        cache.update(
            &all.index((.., .., 0..1, ..)),
            &all.index((.., .., 0..1, ..)),
        )?;
        let layer = LayerCache::Full(cache);
        let saved = layer.snapshot_prefix();
        let mut restored = LayerCache::Full(FullAttentionCache::with_dtype(
            crate::metal_context::execution_context(),
            1,
            256,
            Dtype::Float32,
        ));
        restored.restore_prefix(&saved);
        let LayerCache::Full(mut cache) = restored else {
            unreachable!()
        };
        let (k, v) = cache.update(
            &all.index((.., .., 1..260, ..)),
            &all.index((.., .., 1..260, ..)),
        )?;
        for array in [k, v] {
            assert_eq!(array.dtype(), Dtype::Float32);
            array.eval()?;
            assert_eq!(array.reshape(&[-1])?.to_vec1::<f32>()?, values);
        }
        Ok(())
    }
}

#[cfg(test)]
mod checkpoint_tests {}
