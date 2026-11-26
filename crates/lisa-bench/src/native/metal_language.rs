//! Application construction of the portable language execution adapter.
pub type MetalLanguageRuntime = lisa_engine::language::runtime::Runtime<
    lisa_metal::device::MetalDevice,
    crate::core::sampler::Sampler,
>;
pub fn runtime() -> MetalLanguageRuntime {
    MetalLanguageRuntime::new(
        crate::metal_context::execution_context(),
        lisa_engine::tensor_ops::TensorLaw::Native,
    )
}
