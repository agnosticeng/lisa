//! Native FLUX.2 Klein 4B text-to-image pipeline on LISA's Metal runtime.
//! Architecture and operation order: mlx-serve 6c430614 (MIT), flux2 plugin.
//! Native conformance adapters for portable image bindings.
use lisa_metal::device::MetalDevice;
pub type Pipeline = lisa_engine::bindings::flux2::Pipeline<MetalDevice>;
pub use lisa_engine::bindings::flux2::DEFAULT_STEPS;
pub use lisa_engine::bindings::flux2::{DEFAULT_MODEL, Options};
pub use lisa_engine::weights::flux2::is_checkpoint;
#[cfg(test)]
mod plan_oracle;
#[cfg(test)]
mod plan_tests;
#[cfg(test)]
mod reference_tests;
pub(crate) mod vae;
pub(crate) mod weights;
