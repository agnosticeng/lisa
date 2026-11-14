//! Request bindings for the model-owned bounded media-attention graph.
use crate::core::graph::PlanCache;
use anyhow::ensure;
use lisa_backend::ScalarType;
use lisa_metal::{Array, Dtype, device::MetalDevice};
use lisa_models::media_attention::{self, Geometry};
crate::metal_context_local! {
    static PLANS:std::cell::RefCell<PlanCache<MetalDevice,Geometry>>=std::cell::RefCell::new(PlanCache::new(crate::metal_context::device(),8));
}
pub fn attention(
    q: &Array,
    k: &Array,
    v: &Array,
    scale: f32,
    mask: Option<&Array>,
) -> anyhow::Result<Array> {
    let shape = |x: &Array| -> anyhow::Result<[i32; 4]> {
        x.shape()
            .try_into()
            .map_err(|_| anyhow::anyhow!("media attention requires rank four"))
    };
    let dtype = if [q.dtype(), k.dtype(), v.dtype()].contains(&Dtype::Float32) {
        Dtype::Float32
    } else {
        q.dtype()
    };
    let output_dtype = match dtype {
        Dtype::Float32 => ScalarType::F32,
        Dtype::Float16 => ScalarType::F16,
        Dtype::Bfloat16 => ScalarType::BF16,
        _ => anyhow::bail!("unsupported media attention dtype"),
    };
    let geometry = Geometry {
        q: shape(q)?,
        k: shape(k)?,
        v: shape(v)?,
        output_dtype,
        scale: scale.to_bits(),
        mask: mask.is_some(),
    };
    geometry.validate()?;
    if let Some(mask) = mask {
        ensure!(
            mask.shape() == [1, 1, geometry.q[2], geometry.k[2]],
            "invalid media attention mask"
        );
    }
    let mut inputs = vec![q, k, v];
    if let Some(mask) = mask {
        inputs.push(mask);
    }
    PLANS.with(|cache| {
        cache
            .borrow_mut()
            .try_get_or_prepare(geometry.clone(), || media_attention::graph(&geometry))?
            .run(&inputs, &mut [])?
            .into_one()
            .map_err(anyhow::Error::msg)
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    use lisa_metal::{ops, ops::indexing::IndexOp};
    fn original_attention(
        q: &Array,
        k: &Array,
        v: &Array,
        scale: f32,
        mask: Option<&Array>,
    ) -> anyhow::Result<Array> {
        ensure!(
            q.ndim() == 4
                && k.ndim() == 4
                && v.ndim() == 4
                && q.dim(0) == 1
                && k.dim(0) == 1
                && v.dim(0) == 1
                && k.dim(1) > 0
                && q.dim(1) > 0
                && q.dim(2) > 0
                && k.dim(2) > 0
                && q.dim(3) == k.dim(3)
                && q.dim(1) % k.dim(1) == 0
                && k.dim(1) == v.dim(1)
                && k.dim(2) == v.dim(2),
            "invalid media attention geometry"
        );
        if let Some(mask) = mask {
            ensure!(
                mask.shape() == [1, 1, q.dim(2), k.dim(2)],
                "invalid media attention mask"
            );
        }
        let dtype = if [q.dtype(), k.dtype(), v.dtype()].contains(&Dtype::Float32) {
            Dtype::Float32
        } else {
            q.dtype()
        };
        let repeat = (q.dim(1) / k.dim(1)) as usize;
        let k = k
            .as_dtype(Dtype::Float32)?
            .repeat_axis(repeat, 1)?
            .transpose_axes(&[0, 1, 3, 2])?
            .contiguous()?
            .reshape(&[q.dim(1), k.dim(3), k.dim(2)])?;
        let v = v
            .as_dtype(Dtype::Float32)?
            .repeat_axis(repeat, 1)?
            .reshape(&[q.dim(1), v.dim(2), v.dim(3)])?;
        let mut chunks = Vec::new();
        for start in (0..q.dim(2)).step_by(256) {
            let end = (start + 256).min(q.dim(2));
            let q = q
                .index((.., .., start..end, ..))
                .as_dtype(Dtype::Float32)?
                .multiply(&Array::from_f32(scale))?;
            let q = q.reshape(&[q.dim(1), end - start, q.dim(3)])?;
            let mut scores = q.matmul(&k)?;
            if let Some(mask) = mask {
                scores = scores.add(&mask.index((.., .., start..end, ..)).reshape(&[
                    1,
                    end - start,
                    k.dim(2),
                ])?)?;
            }
            let out = scores.softmax_axis(-1)?.matmul(&v)?.as_dtype(dtype)?;
            out.eval_enqueue()?;
            chunks.push(out.expand_dims(0)?);
        }
        Ok(ops::concatenate(&chunks.iter().collect::<Vec<_>>(), 2)?)
    }

    #[test]
    fn grouped_attention_mask_and_chunk_tail_match_scalar_reference() -> anyhow::Result<()> {
        let (heads, kv_heads, nq, nk, d) = (4usize, 2usize, 259usize, 7usize, 8usize);
        let values = |n: usize, a: usize| {
            (0..n)
                .map(|i| ((i * a % 43) as f32 - 21.) / 32.)
                .collect::<Vec<_>>()
        };
        let qv = values(heads * nq * d, 7);
        let kv = values(kv_heads * nk * d, 11);
        let vv = values(kv_heads * nk * d, 13);
        let q = Array::from_slice(&qv, &[1, heads as i32, nq as i32, d as i32]);
        let k = Array::from_slice(&kv, &[1, kv_heads as i32, nk as i32, d as i32]);
        let v = Array::from_slice(&vv, &[1, kv_heads as i32, nk as i32, d as i32]);
        let mask: Vec<f32> = (0..nq * nk)
            .map(|i| {
                if i % nk == nk - 1 {
                    f32::NEG_INFINITY
                } else {
                    0.
                }
            })
            .collect();
        let mask = Array::from_slice(&mask, &[1, 1, nq as i32, nk as i32]);
        let actual = attention(&q, &k, &v, 0.25, Some(&mask))?.contiguous()?;
        actual.eval()?;
        let actual = actual.as_slice::<f32>();
        for head in 0..heads {
            for row in 0..nq {
                let kh = head / (heads / kv_heads);
                let scores: Vec<f32> = (0..nk - 1)
                    .map(|j| {
                        (0..d)
                            .map(|c| {
                                qv[(head * nq + row) * d + c] * 0.25 * kv[(kh * nk + j) * d + c]
                            })
                            .sum::<f32>()
                            .exp()
                    })
                    .collect();
                let denominator: f32 = scores.iter().sum();
                for c in 0..d {
                    let expected: f32 = scores
                        .iter()
                        .enumerate()
                        .map(|(j, s)| s / denominator * vv[(kh * nk + j) * d + c])
                        .sum();
                    let got = actual[(head * nq + row) * d + c];
                    assert!(
                        (got - expected).abs() < 2e-6,
                        "head={head} row={row} c={c}: {got} vs {expected}"
                    );
                }
            }
        }
        Ok(())
    }
    #[test]
    fn media_plan_preserves_original_bits_and_chunk_boundaries() -> anyhow::Result<()> {
        for dtype in [Dtype::Bfloat16, Dtype::Float16, Dtype::Float32] {
            for n in [1, 259] {
                let make = |shape: &[i32], seed: i32| -> anyhow::Result<Array> {
                    let data = (0..shape.iter().product())
                        .map(|i| ((i * seed % 43) as f32 - 21.) / 32.)
                        .collect::<Vec<_>>();
                    Ok(Array::from_slice(&data, shape).as_dtype(dtype)?)
                };
                let q = make(&[1, 4, n, 8], 7)?;
                let k = make(&[1, 2, 7, 8], 11)?;
                let v = make(&[1, 2, 7, 12], 13)?;
                let mask = Array::from_slice(
                    &(0..n * 7)
                        .map(|i| if i % 7 == 6 { f32::NEG_INFINITY } else { 0. })
                        .collect::<Vec<_>>(),
                    &[1, 1, n, 7],
                );
                for mask in [None, Some(&mask)] {
                    let expected =
                        original_attention(&q, &k, &v, 0.25, mask)?.as_dtype(Dtype::Float32)?;
                    let expected = expected
                        .as_slice::<f32>()
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>();
                    for _ in 0..2 {
                        let actual = attention(&q, &k, &v, 0.25, mask)?.as_dtype(Dtype::Float32)?;
                        assert_eq!(
                            actual
                                .as_slice::<f32>()
                                .iter()
                                .map(|x| x.to_bits())
                                .collect::<Vec<_>>(),
                            expected,
                            "dtype={dtype:?} rows={n}"
                        );
                    }
                }
            }
        }
        Ok(())
    }
}
