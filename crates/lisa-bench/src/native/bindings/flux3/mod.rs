//! Native FLUX 3 Image Klein text-to-image inference.
//! Adapted from BFL's Apache-2.0 reference bfc8889a4497f846416f253a3972a1941ac698b9.
//! Copyright 2026 Black Forest Labs, Inc. See LICENSE for the full notice.
//! Native conformance adapters for portable image bindings.
use lisa_metal::{Array, Dtype, device::MetalDevice};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};
pub type Pipeline = lisa_engine::bindings::flux3::Pipeline<MetalDevice>;
pub use lisa_engine::bindings::flux3::{DEFAULT_STEPS, is_checkpoint};
mod dit;
mod text;
use crate::bindings::flux2::{Options, vae, weights::Weights};
pub use lisa_engine::bindings::flux3::REPO;
use lisa_models::flux::pipeline::schedule3 as schedule;
fn decoder_weights(dir: &Path) -> anyhow::Result<Weights> {
    lisa_engine::bindings::flux3::decoder_weights(&crate::metal_context::execution_context(), dir)
}
#[cfg(test)]
pub(crate) mod tests;
