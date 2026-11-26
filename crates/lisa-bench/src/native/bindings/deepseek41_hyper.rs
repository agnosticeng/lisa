//! Concrete construction of portable model plan bindings.
use crate::core::gguf_resident::ResidentWeights;
use anyhow::Result;
use lisa_metal::{Array, Stream, device::MetalDevice};
pub type Input = lisa_engine::bindings::deepseek41::hyper::Input<Array>;
pub struct Hc(lisa_engine::bindings::deepseek41::hyper::Hc<MetalDevice>);
impl std::ops::Deref for Hc {
    type Target = lisa_engine::bindings::deepseek41::hyper::Hc<MetalDevice>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Hc {
    pub fn load(
        weights: &ResidentWeights,
        layer: usize,
        kind: &str,
        stream: &Stream,
    ) -> Result<Self> {
        Ok(Self(lisa_engine::bindings::deepseek41::hyper::Hc::load(
            &**weights,
            layer,
            kind,
            &MetalDevice::on_stream(stream),
        )?))
    }
    #[cfg(test)]
    pub fn bound(
        matrix: lisa_models::deepseek41::projection::Matrix,
        weights: Vec<Array>,
        stream: &Stream,
    ) -> Result<Self> {
        Ok(Self(lisa_engine::bindings::deepseek41::hyper::Hc::bound(
            matrix,
            weights,
            &MetalDevice::on_stream(stream),
        )?))
    }
}
pub struct ResidualPlans(lisa_engine::bindings::deepseek41::hyper::ResidualPlans<MetalDevice>);
impl std::ops::Deref for ResidualPlans {
    type Target = lisa_engine::bindings::deepseek41::hyper::ResidualPlans<MetalDevice>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl ResidualPlans {
    pub fn new(stream: &Stream) -> Result<Self> {
        Ok(Self(
            lisa_engine::bindings::deepseek41::hyper::ResidualPlans::new(&MetalDevice::on_stream(
                stream,
            ))?,
        ))
    }
}
#[cfg(test)]
#[path = "deepseek41_hyper_tests.rs"]
mod tests;
