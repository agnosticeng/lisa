//! Shared image-generation dispatch for the CLI, HTTP server and native app.
use super::{flux2, flux3};
pub use flux2::Options;
use std::path::Path;

pub enum Pipeline {
    Flux2(flux2::Pipeline),
    Flux3(flux3::Pipeline),
}
impl Pipeline {
    pub fn default_steps(&self) -> u32 {
        match self {
            Self::Flux2(_) => flux2::DEFAULT_STEPS,
            Self::Flux3(_) => flux3::DEFAULT_STEPS,
        }
    }
    pub fn load(dir: &Path) -> anyhow::Result<Self> {
        if flux3::is_checkpoint(dir) {
            Ok(Self::Flux3(flux3::Pipeline::load(dir)?))
        } else {
            Ok(Self::Flux2(flux2::Pipeline::load(dir)?))
        }
    }
    pub fn generate(
        &self,
        prompt: &str,
        options: &Options,
        progress: &mut dyn FnMut(&str, u32, u32) -> anyhow::Result<()>,
    ) -> anyhow::Result<Vec<u8>> {
        match self {
            Self::Flux2(p) => p.generate(prompt, options, progress),
            Self::Flux3(p) => p.generate(prompt, options, progress),
        }
    }
}
pub fn is_model_type(model_type: &str) -> bool {
    matches!(model_type, "flux2-klein" | "flux3-klein")
}
