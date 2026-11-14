#![cfg(test)]
use super::*;
use crate::core::quant::QuantizedLinearExecution;
use lisa_metal::{Dtype, ops};
fn projection(n: i32, k: i32, bits: i32, group_size: i32) -> QuantizedLinear {
    QuantizedLinear {
        weight: Array::from_slice(
            &(0..n * k * bits / 32)
                .map(|i| (i as u32).wrapping_mul(0x1234567).wrapping_add(0x89abcdef))
                .collect::<Vec<_>>(),
            &[n, k * bits / 32],
        ),
        scales: Array::from_slice(
            &vec![half::bf16::from_f32(0.0078125); (n * k / group_size) as usize],
            &[n, k / group_size],
        ),
        biases: Array::from_slice(
            &vec![half::bf16::from_f32(-0.0625); (n * k / group_size) as usize],
            &[n, k / group_size],
        ),
        group_size,
        bits,
    }
}
fn same(a: &Array, b: &Array) {
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.dtype(), b.dtype());
    if a.dtype() == Dtype::Uint32 {
        assert_eq!(a.to_vec1::<u32>().unwrap(), b.to_vec1::<u32>().unwrap());
    } else {
        let bits = |a: &Array| {
            a.as_dtype(Dtype::Float32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(a), bits(b));
    }
}
#[test]
fn draft_selection_matches_original_shortlist_and_full_head() -> anyhow::Result<()> {
    let hidden = Array::from_slice(
        &(0..256)
            .map(|i| half::bf16::from_f32(((i * 37 % 127) - 63) as f32 / 128.))
            .collect::<Vec<_>>(),
        &[1, 256],
    );
    let coarse = projection(1025, 256, 2, 64);
    for bits in [2, 4, 8] {
        for use_coarse in [false, true] {
            for confidence in [false, true] {
                let target = projection(1025, 256, bits, 64);
                let got = select(&hidden, &target, use_coarse.then_some(&coarse), confidence)?;
                let (id, exact, candidates, logp) = if use_coarse {
                    let scores = coarse.forward(&hidden)?;
                    let ids = ops::top32_indices(&scores)?.reshape(&[32])?;
                    let w = target.weight.take_axis(&ids, 0)?;
                    let s = target.scales.take_axis(&ids, 0)?;
                    let b = target.biases.take_axis(&ids, 0)?;
                    let exact = ops::quantized_matmul(
                        &hidden,
                        &w,
                        &s,
                        Some(&b),
                        true,
                        target.group_size,
                        target.bits,
                    )?;
                    let local = ops::indexing::argmax_axis(&exact, -1, None)?.reshape(&[1])?;
                    let id = ids.take_axis(&local, 0)?.reshape(&[1, 1])?;
                    let logp = exact.softmax_axis(-1)?.take_axis(&local, -1)?.log()?;
                    (id, exact, Some(ids), confidence.then_some(logp))
                } else {
                    let scores = target.forward(&hidden)?;
                    let id = ops::indexing::argmax_axis(&scores, -1, None)?.reshape(&[1, 1])?;
                    let logp = if confidence {
                        Some(scores.softmax_axis(-1)?.take_axis(&id, -1)?.log()?)
                    } else {
                        None
                    };
                    (id, scores, None, logp)
                };
                same(&got.token, &id);
                same(&got.scores, &exact);
                assert_eq!(got.candidates.is_some(), candidates.is_some());
                assert_eq!(got.confidence.is_some(), logp.is_some());
                if let Some(expected) = &candidates {
                    same(got.candidates.as_ref().unwrap(), expected);
                }
                if let Some(expected) = &logp {
                    same(got.confidence.as_ref().unwrap(), expected);
                }
            }
        }
    }
    Ok(())
}
#[test]
fn explicit_requantization_preserves_original_packed_bytes() -> anyhow::Result<()> {
    for bits in [2, 4, 8] {
        for group in [32, 64, 128] {
            let source = projection(33, 256, bits, group);
            let full =
                ops::dequantize(&source.weight, &source.scales, &source.biases, group, bits)?;
            full.eval()?;
            let expected = ops::quantize(&full, 64, 2)?;
            let actual = crate::bindings::affine::requantize(
                &source.weight,
                &source.scales,
                &source.biases,
                (group, bits),
                (64, 2),
            )?;
            same(&actual.0, &expected.0);
            same(&actual.1, &expected.1);
            same(&actual.2, &expected.2);
        }
    }
    Ok(())
}
