//! Concrete construction of portable model plan bindings.
use crate::core::gguf_resident::ResidentWeights;
use anyhow::Result;
use lisa_metal::{Array, Stream, device::MetalDevice};
use lisa_models::deepseek41::moe as model;
pub type Routing = lisa_engine::bindings::deepseek41_moe::Routing<Array>;
pub type Output = lisa_engine::bindings::deepseek41_moe::Output<Array>;
pub struct Moe(lisa_engine::bindings::deepseek41_moe::Moe<MetalDevice>);
impl std::ops::Deref for Moe {
    type Target = lisa_engine::bindings::deepseek41_moe::Moe<MetalDevice>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Moe {
    pub fn load(
        weights: &ResidentWeights,
        layer: usize,
        experts: usize,
        stream: &Stream,
    ) -> Result<Self> {
        Ok(Self(lisa_engine::bindings::deepseek41_moe::Moe::load(
            &**weights,
            layer,
            experts,
            &MetalDevice::on_stream(stream),
        )?))
    }
    #[cfg(test)]
    pub fn bound(weights: Vec<Array>, spec: model::Spec, stream: &Stream) -> Result<Self> {
        Ok(Self(lisa_engine::bindings::deepseek41_moe::Moe::bound(
            weights,
            spec,
            &MetalDevice::on_stream(stream),
        )?))
    }
}
#[cfg(test)]
#[path = "deepseek41_moe_tests.rs"]
pub(crate) mod tests;

#[cfg(test)]
use lisa_models::packed::Projection;
