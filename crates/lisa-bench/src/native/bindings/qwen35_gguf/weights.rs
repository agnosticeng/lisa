//! Provider specialization retained by native conformance fixtures.
pub use lisa_engine::bindings::qwen35_gguf::weights::{Config, validate};
pub type Bound =
    lisa_engine::bindings::qwen35_gguf::weights::Bound<lisa_metal::device::MetalDevice>;
