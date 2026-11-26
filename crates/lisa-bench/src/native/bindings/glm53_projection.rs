//! Native specialization and conformance fixtures for portable GLM bindings.
use anyhow::Result;
use lisa_metal::{Array, device::MetalDevice};
pub type Projection = lisa_engine::bindings::glm53::packed::projection::Projection<MetalDevice>;
pub fn plain(w: &crate::core::gguf_resident::ResidentWeights, name: &str) -> Result<Array> {
    lisa_engine::bindings::glm53::packed::projection::plain(&**w, name)
}
pub trait ProjectionExecution {
    fn descriptor(&self) -> Result<lisa_models::glm53::packed::Projection>;
    fn forward(&self, x: &Array) -> Result<Array>;
}
impl ProjectionExecution for Projection {
    fn descriptor(&self) -> Result<lisa_models::glm53::packed::Projection> {
        self.descriptor_on(&crate::metal_context::device())
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        self.forward_on(&crate::metal_context::execution_context(), x)
    }
}
