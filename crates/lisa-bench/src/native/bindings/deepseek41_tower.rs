//! Native application type for the portable DeepSeek model binding.
pub type Tower = lisa_engine::bindings::deepseek41_tower::Tower<
    lisa_metal::device::MetalDevice,
    crate::metal_language::MetalLanguageRuntime,
>;
