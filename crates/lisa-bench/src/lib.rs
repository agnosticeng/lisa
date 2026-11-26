//! Compiled conformance oracles and diagnostic entry points.
//! Native diagnostics and independent numerical conformance oracles.
//! Applications bind portable engine plans through their own selected provider.
#[cfg(test)]
mod conformance;
mod native;
pub use native::*;

pub(crate) use native::metal_storage;
