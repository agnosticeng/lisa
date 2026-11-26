//! Provider specialization retained by native conformance fixtures.
#[cfg(test)]
pub use lisa_engine::weights::qwen35::gguf::Config;
#[cfg(test)]
pub type Bound =
    lisa_engine::bindings::qwen35::gguf::weights::Bound<lisa_metal::device::MetalDevice>;
