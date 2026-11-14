//! Native conformance adapters for portable image bindings.
use lisa_metal::{Array, Dtype, device::MetalDevice};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};
pub type Dit = lisa_engine::bindings::flux2::dit::Dit<MetalDevice>;
