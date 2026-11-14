//! Native conformance adapters for portable image bindings.
use lisa_metal::{Array, Dtype, device::MetalDevice};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};
pub type Encoder = lisa_engine::bindings::flux3::text::Encoder<MetalDevice>;
