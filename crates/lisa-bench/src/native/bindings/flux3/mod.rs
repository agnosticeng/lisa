//! Native FLUX 3 Image Klein text-to-image inference.
//! Adapted from BFL's Apache-2.0 reference bfc8889a4497f846416f253a3972a1941ac698b9.
//! Copyright 2026 Black Forest Labs, Inc. See LICENSE for the full notice.
//! Native conformance adapters for portable image bindings.
use lisa_metal::device::MetalDevice;
pub type Pipeline = lisa_engine::bindings::flux3::Pipeline<MetalDevice>;
pub use lisa_engine::bindings::flux3::DEFAULT_STEPS;
pub use lisa_engine::weights::flux3::REPO;
pub use lisa_engine::weights::flux3::is_checkpoint;
use lisa_models::flux::pipeline::schedule3 as schedule;
#[cfg(test)]
pub(crate) mod tests;
