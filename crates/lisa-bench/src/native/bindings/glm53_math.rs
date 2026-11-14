//! Native specialization and conformance fixtures for portable GLM bindings.
use anyhow::{Context, Result, ensure};
use lisa_metal::{Array, Dtype, Stream, device::MetalDevice};
pub fn linear_projection(x: &Array, w: &Array) -> Result<Array> {
    lisa_engine::bindings::glm53_math::linear_projection(
        &crate::metal_context::execution_context(),
        x,
        w,
    )
}
pub fn l2norm_last(x: &Array, eps: f32) -> lisa_metal::error::Result<Array> {
    lisa_engine::bindings::glm53_math::l2norm_last(
        &crate::metal_context::execution_context(),
        x,
        eps,
    )
    .map_err(|e| lisa_metal::error::Error::msg(e.to_string()))
}
pub fn gated_rms_norm(x: &Array, w: &Array, g: &Array, eps: f32) -> Result<Array> {
    lisa_engine::bindings::glm53_math::gated_rms_norm(
        &crate::metal_context::execution_context(),
        x,
        w,
        g,
        eps,
    )
}
pub fn rms_norm_last(x: &Array, w: Option<&Array>, eps: f32) -> Result<Array> {
    lisa_engine::bindings::glm53_math::rms_norm_last(
        &crate::metal_context::execution_context(),
        x,
        w,
        eps,
    )
}
#[cfg(test)]
#[path = "glm53_math_oracle.rs"]
mod oracle;
#[cfg(test)]
mod tests {
    use super::*;
    use crate::bindings::deepseek41_moe::tests::{counts, difference};

    fn values(shape: &[i32], seed: usize) -> Array {
        Array::from_slice(
            &(0..shape.iter().map(|&d| d as usize).product::<usize>())
                .map(|i| ((i * 17 + seed) % 127) as f32 / 83. - 0.6)
                .collect::<Vec<_>>(),
            shape,
        )
    }
    fn check(
        old: impl FnOnce() -> anyhow::Result<Array>,
        new: impl FnOnce() -> anyhow::Result<Array>,
    ) -> anyhow::Result<()> {
        let a = counts();
        let old = old()?;
        let b = counts();
        let new = new()?;
        let c = counts();
        assert_eq!(old.shape(), new.shape());
        assert_eq!(old.to_vec1::<u32>()?, new.to_vec1::<u32>()?);
        if std::env::var_os("LISA_TRACE").is_some() {
            assert_eq!(difference(&a, &b), difference(&b, &c));
        }
        Ok(())
    }
    #[test]
    fn glm_normalization_and_projection_plans_preserve_exact_operation_chains() -> anyhow::Result<()>
    {
        for shape in [
            vec![1, 1, 64, 128],
            vec![1, 8, 64, 128],
            vec![2, 3, 64, 128],
        ] {
            let x = values(&shape, 11);
            let w = values(&[128], 37);
            let gate = values(&shape, 47);
            check(
                || Ok(oracle::l2norm_last(&x, 1e-6)?),
                || Ok(l2norm_last(&x, 1e-6)?),
            )?;
            for weighted in [false, true] {
                let w = weighted.then_some(&w);
                check(
                    || oracle::rms_norm_last(&x, w, 1e-5),
                    || rms_norm_last(&x, w, 1e-5),
                )?;
            }
            check(
                || oracle::gated_rms_norm(&x, &w, &gate, 1e-5),
                || gated_rms_norm(&x, &w, &gate, 1e-5),
            )?;
        }
        for (batch, rows) in [(1, 1), (1, 8), (1, 9), (2, 3)] {
            let x = values(&[batch, rows, 128], 73);
            for dtype in [
                lisa_metal::Dtype::Float32,
                lisa_metal::Dtype::Bfloat16,
                lisa_metal::Dtype::Float16,
            ] {
                let w = values(&[256, 128], 91).as_dtype(dtype)?;
                check(
                    || oracle::linear_projection(&x, &w),
                    || linear_projection(&x, &w),
                )?;
            }
        }
        Ok(())
    }
}
