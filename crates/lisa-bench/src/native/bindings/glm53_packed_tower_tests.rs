#![cfg(test)]
use super::*;
use crate::bindings::deepseek41_moe::tests::{counts, difference, tensor};
use lisa_gguf::TensorType;
use lisa_metal::kernels::sparse::kernels as gpu;
fn values(shape: &[i32], seed: usize) -> Array {
    Array::from_slice(
        &(0..shape.iter().product::<i32>() as usize)
            .map(|i| ((i * 17 + seed) % 127) as f32 / 151. - 0.3)
            .collect::<Vec<_>>(),
        shape,
    )
}
fn rms_original(x: &Array, w: &Array) -> Result<Array> {
    gpu::glm53_ds4_rms_norm(x, w, 1e-5, &Stream::thread_local_or_default()).context("RMS declined")
}
fn mean_original(x: &Array) -> Result<Array> {
    let w = Array::from_slice(&[0.25f32; 4], &[1, 1, 4]);
    gpu::hc_weighted_sum(
        &x.reshape(&[1, 1, 4, 4096])?,
        &w,
        &Stream::thread_local_or_default(),
    )
    .context("mean declined")
}
fn check<const N: usize>(
    old: impl FnOnce() -> Result<[Array; N]>,
    new: impl FnOnce() -> Result<[Array; N]>,
) -> Result<()> {
    let before = counts();
    let a = old()?;
    let between = counts();
    let b = new()?;
    let after = counts();
    for (a, b) in a.iter().zip(&b) {
        assert_eq!(a.shape(), b.shape());
        assert_eq!(a.to_vec1::<u32>()?, b.to_vec1::<u32>()?);
    }
    if std::env::var_os("LISA_TRACE").is_some() {
        assert_eq!(difference(&before, &between), difference(&between, &after));
    }
    Ok(())
}
#[test]
fn packed_tower_plans_preserve_embedding_norm_and_draft_bits() -> Result<()> {
    let x = values(&[1, 1, 4096], 1);
    let w = values(&[4096], 7);
    let state = values(&[1, 1, 4, 4096], 11);
    check(
        || Ok([Array::concatenate(&[&x, &x, &x, &x], 2)?.reshape(&[1, 1, 4, 4096])?]),
        || run(Request::Expand, &[&x]),
    )?;
    check(
        || {
            let m = state.reshape(&[1, 1, 16384])?;
            Ok([rms_original(&mean_original(&m)?, &w)?, m])
        },
        || run(Request::Final, &[&state, &w]),
    )?;
    check(
        || Ok([rms_original(&mean_original(&state)?, &w)?]),
        || run(Request::MeanNorm, &[&state, &w]),
    )?;
    let a = values(&[1, 1, 4096], 23);
    let b = values(&[1, 1, 4096], 29);
    check(
        || {
            let (r, n) =
                gpu::glm53_mtp_add_rms_norm(&x, &a, &w, 1e-5, &Stream::thread_local_or_default())
                    .context("add norm declined")?;
            Ok([r, n])
        },
        || run(Request::AddNorm, &[&x, &a, &w]),
    )?;
    check(
        || {
            Ok([
                gpu::glm53_mtp_add3(&x, &a, &b, &Stream::thread_local_or_default())
                    .context("add three declined")?,
            ])
        },
        || run(Request::AddThree, &[&x, &a, &b]),
    )?;
    for kind in [TensorType::Q8_0, TensorType::Q4_K] {
        let projection = Projection::Packed(tensor(kind, &[4096, 8192], 37)?);
        check(
            || Ok([projection.forward(&Array::concatenate(&[&x, &a], 2)?)?]),
            || {
                run(
                    Request::JoinProject(projection.descriptor()?),
                    &[&x, &a, &projection],
                )
            },
        )?;
    }
    for n in [1, 3, 8] {
        let xs = (0..n).map(|i| values(&[1, 1, 4096], i)).collect::<Vec<_>>();
        let refs = xs.iter().collect::<Vec<_>>();
        check(
            || Ok([Array::concatenate(&refs, 1)?]),
            || run(Request::Join { count: n }, &refs),
        )?;
    }
    Ok(())
}
