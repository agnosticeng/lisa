//! Native specialization and conformance fixtures for portable GLM bindings.
use anyhow::{Context, Result, ensure};
use lisa_metal::{Array, Dtype, Stream, device::MetalDevice};
use lisa_models::glm53::kda::{Spec, Stage};
pub(crate) fn run<const N: usize>(
    spec: Spec,
    stage: Stage,
    inputs: &[&Array],
) -> Result<[Array; N]> {
    lisa_engine::bindings::glm53_kda::run(
        &crate::metal_context::execution_context(),
        spec,
        stage,
        inputs,
    )
}
pub(crate) fn run_many(spec: Spec, stage: Stage, inputs: &[&Array]) -> Result<Vec<Array>> {
    lisa_engine::bindings::glm53_kda::run_many(
        &crate::metal_context::execution_context(),
        spec,
        stage,
        inputs,
    )
}
pub(crate) fn forward(
    key: lisa_models::glm53::kda::Forward,
    inputs: &[&Array],
) -> Result<([Array; 3], Vec<(String, Array)>)> {
    lisa_engine::bindings::glm53_kda::forward(
        &crate::metal_context::execution_context(),
        key,
        inputs,
    )
}
