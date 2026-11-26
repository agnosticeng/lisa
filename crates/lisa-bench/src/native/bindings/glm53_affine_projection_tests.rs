#![cfg(test)]
use super::*;
use lisa_metal::{Dtype, ops};
#[path = "glm53_affine_projection_oracle.rs"]
mod oracle;
fn values(shape: &[i32], seed: usize) -> Array {
    Array::from_slice(
        &(0..shape.iter().product::<i32>() as usize)
            .map(|i| ((i * 17 + seed) % 127) as f32 / 151. - 0.3)
            .collect::<Vec<_>>(),
        shape,
    )
}
#[test]
fn affine_plans_preserve_native_fallback_and_cohort_bits() -> anyhow::Result<()> {
    for (input, output) in [(4096, 1536), (192, 64)] {
        for dtype in [Dtype::Float32, Dtype::Bfloat16] {
            let source = values(&[output, input], 11).as_dtype(dtype)?;
            let (w, s, b) = ops::quantize(&source, 64, 8)?;
            let w = QuantizedLinear {
                weight: w,
                scales: s,
                biases: b,
                group_size: 64,
                bits: 8,
            };
            for native in [true, false] {
                for shape in [
                    vec![1, 1, input],
                    vec![1, 8, input],
                    vec![1, 16, input],
                    vec![1, 17, input],
                    vec![1, 24, input],
                    vec![2, 1, input],
                    vec![2, 3, input],
                    vec![1, input],
                ] {
                    let x = values(&shape, 23);
                    let a = oracle::original(&w, &x, native)?;
                    let b = forward(&w, &x, native)?;
                    assert_eq!(a.shape(), b.shape());
                    assert_eq!(
                        a.to_vec1::<u32>()?,
                        b.to_vec1::<u32>()?,
                        "{shape:?} metadata{dtype:?} native{native}"
                    );
                }
            }
        }
    }
    Ok(())
}
