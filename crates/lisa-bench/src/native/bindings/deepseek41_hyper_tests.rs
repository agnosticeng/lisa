#![cfg(test)]
//! The original residual call sequence is the oracle, including separate casts.
use super::*;
use crate::bindings::deepseek41_moe::tests::{projection_check, tensor};
use lisa_gguf::TensorType;
use lisa_metal::kernels::ggml as native;
fn values(shape: &[i32], seed: usize) -> Array {
    let n = shape.iter().map(|&x| x as usize).product();
    Array::from_slice(
        &(0..n)
            .map(|i| ((i * 17 + seed) % 113) as f32 / 71. - 0.7)
            .collect::<Vec<_>>(),
        shape,
    )
}
#[test]
fn residual_plans_preserve_original_four_stream_bits_and_dispatches() -> Result<()> {
    let stream = Stream::gpu();
    let mixer = tensor(TensorType::F32, &[24, 20480], 17)?;
    let weight = mixer.plain_view()?;
    let scale = Array::from_slice(&[0.03f32, 0.11, 0.07], &[3]);
    let base = values(&[24], 5);
    let norm = values(&[5120], 13);
    let model = Hc::bound(
        super::super::deepseek41_projection::matrix(&mixer)?,
        vec![weight.clone(), scale.clone(), base.clone(), norm.clone()],
        &stream,
    )?;
    let plans = ResidualPlans::new(&stream)?;
    for seed in [19, 103] {
        let residual = values(&[1, 4, 5120], seed);
        let previous = Array::from_slice(&[0.21f32, 0.33, 0.19, 0.27], &[4]);
        let flat = native::rms_norm(&residual.reshape(&[1, 20480])?, None, 1e-20, false, &stream)?;
        let mix = native::plain_matvec(&weight, &flat, &stream)?;
        let old = native::hc_input(
            &mix, &residual, &previous, &scale, &base, &norm, 1e-6, 1e-20, &stream,
        )?;
        let new = model.forward(&residual, &previous)?;
        for (a, b) in [
            (&old.split, &new.split),
            (&old.collapsed, &new.collapsed),
            (&old.norm, &new.norm),
        ] {
            assert_eq!(a.to_vec1::<u32>()?, b.to_vec1::<u32>()?);
        }
        projection_check(
            &stream,
            || {
                let hidden = native::hc_collapse(&residual, &previous, &stream)?;
                Ok(native::rms_norm(
                    &hidden,
                    Some(&norm),
                    1e-20,
                    true,
                    &stream,
                )?)
            },
            || plans.head(&residual, &previous, &norm),
        )?;
    }
    for rows in [1, 3, 8] {
        let residual = values(&[rows, 4, 5120], 11);
        let block = values(&[rows, 5120], 37);
        let add = values(&[rows, 5120], 57);
        let split = values(&[rows, 24], 7);
        projection_check(
            &stream,
            || {
                Ok(native::hc_expand(
                    &block, None, &residual, &split, false, &stream,
                )?)
            },
            || plans.attention(&block, &residual, &split),
        )?;
        projection_check(
            &stream,
            || {
                Ok(native::hc_expand(
                    &block,
                    Some(&add),
                    &residual,
                    &split,
                    true,
                    &stream,
                )?)
            },
            || plans.ffn(&block, &add, &residual, &split),
        )?;
    }
    Ok(())
}
