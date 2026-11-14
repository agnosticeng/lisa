//! Native specialization and conformance fixtures for portable GLM bindings.
use super::glm53_projection::{Projection, ProjectionExecution};
use anyhow::{Context, Result, ensure};
use lisa_metal::{Array, Dtype, Stream, device::MetalDevice};
pub fn linear(p: &Projection, x: &Array) -> Result<Array> {
    lisa_engine::bindings::glm53_kda_projection::linear(
        &crate::metal_context::execution_context(),
        p,
        x,
    )
}
pub fn inputs(
    p: [&Projection; 6],
    x: &Array,
    groups: Option<usize>,
) -> Result<(Array, Array, Array, Array, Array, Array)> {
    lisa_engine::bindings::glm53_kda_projection::inputs(
        &crate::metal_context::execution_context(),
        p,
        x,
        groups,
    )
}
pub fn gates(p: [&Projection; 2], a: &Array, b: &Array) -> Result<(Array, Array)> {
    lisa_engine::bindings::glm53_kda_projection::gates(
        &crate::metal_context::execution_context(),
        p,
        a,
        b,
    )
}
#[cfg(test)]
#[path = "glm53_kda_projection_tests.rs"]
mod tests;
