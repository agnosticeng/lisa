//! Native conformance adapters for portable image bindings.
use lisa_metal::{Array, Dtype, device::MetalDevice};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};
pub fn encode(dir: &Path, prompt: &str) -> anyhow::Result<Array> {
    lisa_engine::bindings::flux2::text::encode(
        &crate::metal_context::execution_context(),
        dir,
        prompt,
    )
}
