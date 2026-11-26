#![cfg(test)]
use super::*;
use crate::bindings::deepseek41_moe::tests::tensor;
#[path = "glm53_kda_projection_oracle.rs"]
mod oracle;
fn values(shape: &[i32], seed: usize) -> Array {
    Array::from_slice(
        &(0..shape.iter().map(|&n| n as usize).product::<usize>())
            .map(|i| ((i * 17 + seed) % 127) as f32 / 151. - 0.3)
            .collect::<Vec<_>>(),
        shape,
    )
}
fn parts(x: (Array, Array, Array, Array, Array, Array)) -> [Array; 6] {
    [x.0, x.1, x.2, x.3, x.4, x.5]
}
fn equal(a: &Array, b: &Array) -> anyhow::Result<()> {
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.to_vec1::<u32>()?, b.to_vec1::<u32>()?);
    Ok(())
}
#[test]
fn kda_projection_plans_preserve_fused_separate_and_packed_bits() -> anyhow::Result<()> {
    for dtype in [Dtype::Bfloat16, Dtype::Float32] {
        let weight = |shape: &[i32], seed| -> anyhow::Result<Projection> {
            Ok(values(shape, seed).as_dtype(dtype)?.into())
        };
        let source = oracle::Source {
            q_w: weight(&[8192, 4096], 11)?,
            k_w: weight(&[8192, 4096], 23)?,
            v_w: weight(&[8192, 4096], 37)?,
            f_a_w: weight(&[128, 4096], 47)?,
            g_a_w: weight(&[128, 4096], 59)?,
            b_w: weight(&[64, 4096], 71)?,
            f_b_w: weight(&[8192, 128], 83)?,
            g_b_w: weight(&[8192, 128], 97)?,
        };
        for (batch, rows) in [(1, 1), (8, 1), (1, 8), (1, 9), (2, 3)] {
            let x = values(&[batch, rows, 4096], 103);
            let old = parts(source.input_projections(&x)?);
            let new = parts(inputs(
                [
                    &source.q_w,
                    &source.k_w,
                    &source.v_w,
                    &source.f_a_w,
                    &source.g_a_w,
                    &source.b_w,
                ],
                &x,
                None,
            )?);
            for (a, b) in old.iter().zip(&new) {
                equal(a, b)?;
            }
            let a = source.gate_projections(&old[3], &old[4])?;
            let b = gates([&source.f_b_w, &source.g_b_w], &new[3], &new[4])?;
            equal(&a.0, &b.0)?;
            equal(&a.1, &b.1)?;
        }
    }
    let weight = |shape: &[i32], seed| -> anyhow::Result<Projection> {
        Ok(Projection::Packed(
            tensor(lisa_gguf::TensorType::Q8_0, shape, seed)?.into_prepared()?,
        ))
    };
    let source = oracle::Source {
        q_w: weight(&[8192, 4096], 11)?,
        k_w: weight(&[8192, 4096], 23)?,
        v_w: weight(&[8192, 4096], 37)?,
        f_a_w: weight(&[128, 4096], 47)?,
        g_a_w: weight(&[128, 4096], 59)?,
        b_w: weight(&[64, 4096], 71)?,
        f_b_w: weight(&[8192, 128], 83)?,
        g_b_w: weight(&[8192, 128], 97)?,
    };
    for rows in [1, 3, 8] {
        let x = values(&[1, rows, 4096], 103);
        let a = parts(source.input_projections(&x)?);
        let b = parts(inputs(
            [
                &source.q_w,
                &source.k_w,
                &source.v_w,
                &source.f_a_w,
                &source.g_a_w,
                &source.b_w,
            ],
            &x,
            None,
        )?);
        for (a, b) in a.iter().zip(&b) {
            equal(a, b)?;
        }
        let a = source.gate_projections(&a[3], &a[4])?;
        let b = gates([&source.f_b_w, &source.g_b_w], &b[3], &b[4])?;
        equal(&a.0, &b.0)?;
        equal(&a.1, &b.1)?;
    }
    Ok(())
}
