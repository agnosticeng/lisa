//! Native FLUX.2 Klein 4B text-to-image pipeline on LISA's Metal runtime.
//! Architecture and operation order: mlx-serve 6c430614 (MIT), flux2 plugin.
//! Native conformance adapters for portable image bindings.
use lisa_metal::{Array, Dtype, device::MetalDevice};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};
pub type Pipeline = lisa_engine::bindings::flux2::Pipeline<MetalDevice>;
pub use lisa_engine::bindings::flux2::{DEFAULT_STEPS, is_checkpoint};
mod dit;
mod text;
pub use lisa_engine::bindings::flux2::{DEFAULT_MODEL, Options};
#[cfg(test)]
mod plan_oracle;
#[cfg(test)]
mod plan_tests;
#[cfg(test)]
mod reference_tests;
pub(crate) mod vae;
pub(crate) mod weights;
