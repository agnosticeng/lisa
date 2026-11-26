//! Native conformance adapters for portable image bindings.
#[cfg(test)]
use lisa_metal::Array;
#[cfg(test)]
use lisa_metal::Dtype;
#[cfg(test)]
use lisa_metal::device::MetalDevice;
#[cfg(test)]
use std::collections::HashMap;
#[cfg(test)]
pub type Weights = lisa_engine::bindings::flux2::weights::Weights<MetalDevice>;
#[cfg(test)]
pub(crate) fn info(x: &Array) -> anyhow::Result<lisa_models::flux::symbolic::TensorInfo> {
    lisa_engine::bindings::flux2::weights::info(&crate::metal_context::execution_context(), x)
}
#[cfg(test)]
pub(crate) fn run(graph: lisa_models::graph::Graph, inputs: &[&Array]) -> anyhow::Result<Array> {
    lisa_engine::bindings::flux2::weights::run(
        &crate::metal_context::execution_context(),
        graph,
        inputs,
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dense_layout_cache_preserves_values_and_leading_dimensions() -> anyhow::Result<()> {
        for dtype in [Dtype::Bfloat16, Dtype::Float32] {
            let matrix = Array::from_slice(
                &(0..20).map(|v| v as f32 / 16.0 - 0.5).collect::<Vec<_>>(),
                &[5, 4],
            )
            .as_dtype(dtype)?;
            let weights = Weights::from_map(
                crate::metal_context::execution_context(),
                HashMap::from([("projection.weight".into(), matrix.clone())]),
            );
            let x = Array::from_slice(
                &(0..24).map(|v| v as f32 / 32.0).collect::<Vec<_>>(),
                &[1, 2, 3, 4],
            )
            .as_dtype(dtype)?;
            let reference = x
                .reshape(&[6, 4])?
                .matmul(&matrix.transpose_axes(&[1, 0])?)?
                .reshape(&[1, 2, 3, 5])?
                .as_dtype(Dtype::Float32)?;
            reference.eval()?;
            for _ in 0..2 {
                let actual = weights.lin("projection", &x)?.as_dtype(Dtype::Float32)?;
                actual.eval()?;
                assert_eq!(actual.shape(), reference.shape());
                assert_eq!(actual.as_slice::<f32>(), reference.as_slice::<f32>());
                assert_eq!(weights.1.borrow().len(), 1);
            }
        }
        Ok(())
    }
}
