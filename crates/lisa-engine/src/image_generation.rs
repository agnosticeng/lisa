//! Portable image pipeline selection and execution.
use crate::{
    bindings::{BindingDevice, flux2, flux3},
    execution::DeviceContext,
};
pub use flux2::Options;
use lisa_backend::ManagedDevice;
use std::path::Path;

pub struct Pipeline<D: BindingDevice + ManagedDevice> {
    kind: Kind<D>,
}
enum Kind<D: BindingDevice + ManagedDevice> {
    Flux2(flux2::Pipeline<D>),
    Flux3(flux3::Pipeline<D>),
}
impl<D: BindingDevice + ManagedDevice> Pipeline<D> {
    pub fn default_steps(&self) -> u32 {
        match &self.kind {
            Kind::Flux2(_) => flux2::DEFAULT_STEPS,
            Kind::Flux3(_) => flux3::DEFAULT_STEPS,
        }
    }
    pub fn load(context: DeviceContext<D>, dir: &Path) -> anyhow::Result<Self> {
        let kind = if crate::weights::flux3::is_checkpoint(dir) {
            Kind::Flux3(crate::weights::flux3::load(context, dir)?)
        } else {
            Kind::Flux2(crate::weights::flux2::load(context, dir)?)
        };
        Ok(Self { kind })
    }

    pub fn generate(
        &self,
        prompt: &str,
        options: &Options,
        progress: &mut dyn FnMut(&str, u32, u32) -> anyhow::Result<()>,
    ) -> anyhow::Result<Vec<u8>> {
        match &self.kind {
            Kind::Flux2(p) => p.generate(prompt, options, progress),
            Kind::Flux3(p) => p.generate(prompt, options, progress),
        }
    }
}
pub fn is_model_type(model_type: &str) -> bool {
    matches!(model_type, "flux2-klein" | "flux3-klein")
}
