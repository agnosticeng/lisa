#![cfg(test)]
use super::*;
use lisa_metal::ops;
#[path = "glm53_dense_mlp_oracle.rs"]
mod oracle;
fn values(shape: &[i32], seed: usize) -> Array {
    Array::from_slice(
        &(0..shape.iter().product::<i32>() as usize)
            .map(|i| ((i * 17 + seed) % 127) as f32 / 151. - 0.3)
            .collect::<Vec<_>>(),
        shape,
    )
}
fn linear(shape: &[i32], seed: usize, dtype: Dtype) -> anyhow::Result<QuantizedLinear> {
    let (w, s, b) = ops::quantize(&values(shape, seed).as_dtype(dtype)?, 64, 8)?;
    Ok(QuantizedLinear {
        weight: w,
        scales: s,
        biases: b,
        group_size: 64,
        bits: 8,
    })
}
#[test]
fn dense_expert_plans_preserve_paired_separate_native_and_capture_bits() -> anyhow::Result<()> {
    for dtype in [Dtype::Float32, Dtype::Bfloat16] {
        let gate = linear(&[2048, 4096], 1, dtype)?;
        let up = linear(&[2048, 4096], 7, dtype)?;
        let down = linear(&[4096, 2048], 11, dtype)?;
        for paired in [false, true] {
            let old = oracle::DenseMlp {
                gate_up: oracle::GateUpProjection::from_parts(gate.clone(), up.clone(), paired)?,
                down: down.clone(),
                limit: 10.,
            };
            let new = DenseMlp {
                context: crate::metal_context::execution_context(),
                gate_up: GateUpProjection::from_parts(
                    &crate::metal_context::execution_context(),
                    gate.clone(),
                    up.clone(),
                    paired,
                )?,
                down: down.clone(),
                limit: 10.,
            };
            for native in [false, true] {
                let _mode =
                    crate::bindings::glm53_affine_moe::Glm53Q8GateUpNativeScope::new(native);
                for (batch, tokens) in [(1, 1), (1, 8), (1, 16), (1, 17), (2, 1), (2, 3)] {
                    let x = values(&[batch, tokens, 4096], 23);
                    for capture in [false, true] {
                        let ca = Capture::default();
                        let cb = Capture::default();
                        let a = old.forward_with_capture(&x, capture.then_some(&ca), 3)?;
                        let b = new.forward_with_capture(&x, capture.then_some(&cb), 3)?;
                        assert_eq!(a.shape(), b.shape());
                        assert_eq!(
                            a.to_vec1::<u32>()?,
                            b.to_vec1::<u32>()?,
                            "B{batch} T{tokens} native{native} paired{paired} metadata{dtype:?} capture{capture}"
                        );
                        let (a, b) = (ca.entries.lock().unwrap(), cb.entries.lock().unwrap());
                        assert_eq!(a.len(), b.len());
                        for (k, v) in a.iter() {
                            assert_eq!(
                                v.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                                b[k].iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                                "capture {k}"
                            );
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

use crate::core::capture::Capture;
