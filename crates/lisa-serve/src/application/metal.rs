//! Metal provider construction at the application boundary.
pub type Device = lisa_metal::device::MetalDevice;
pub type LanguageRuntime = lisa_engine::language_runtime::Runtime<Device, Sampler>;
pub type Language = lisa_engine::model_binding::Language<Device>;
pub type Sampler = lisa_engine::sampler::Sampler<Device>;
pub type Session = lisa_engine::conversation::Conversation<LanguageRuntime>;
pub type Batch = lisa_engine::batching::Batch<LanguageRuntime>;
pub type ContinuousBatch = lisa_engine::batching::ContinuousBatch<LanguageRuntime>;
pub type Request = lisa_engine::scheduling::Request<Sampler>;
pub type LayerCache = lisa_engine::cache::LayerCache<Device>;
pub type PrefixCache = lisa_engine::conversation_cache::PrefixCache<Device>;
pub type ImagePipeline = lisa_engine::image_generation::Pipeline<Device>;
/// A fresh model execution owns its prepared bindings and diagnostics.
pub fn context() -> anyhow::Result<lisa_engine::execution::DeviceContext<Device>> {
    lisa_metal::runtime::check_platform_support()?;
    let device = Device::system_default();
    Ok(lisa_engine::execution::DeviceContext::with_observer(
        device,
        lisa_engine::execution::Observer {
            enabled: lisa_metal::trace::enabled_for,
            span: |name, detail| {
                lisa_metal::trace::enabled_for(name).then(|| {
                    Box::new(match detail {
                        Some(d) => lisa_metal::trace::span_detail(name, d),
                        None => lisa_metal::trace::span(name),
                    }) as Box<dyn std::any::Any>
                })
            },
        },
    ))
}
pub fn runtime(context: lisa_engine::execution::DeviceContext<Device>) -> LanguageRuntime {
    LanguageRuntime::new(context, lisa_engine::tensor_ops::TensorLaw::Native)
}
pub fn load_qwen4(
    path: &std::path::Path,
    config: lisa_models::qwen4::config::ModelConfig,
) -> anyhow::Result<lisa_engine::bindings::qwen4_tower::Tower<Device>> {
    let context = context()?;
    lisa_metal::warm_kernels();
    lisa_engine::bindings::qwen4_tower::Tower::load_on(runtime(context), path, config)
}
pub fn prefix_spill() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .map(|home| home.join(".lisa/prefix_spill"))
}
pub fn memory_line(tag: &str) {
    let mb = |r: lisa_metal::error::Result<usize>| r.map(|b| b >> 20).unwrap_or(0);
    eprintln!(
        "[mlx-mem {tag}] active {} MiB | private {} MiB | budget {} MiB",
        mb(lisa_metal::memory::active_memory()),
        mb(lisa_metal::memory::private_allocated_memory()),
        lisa_engine::memory::ram_budget() >> 20
    );
}
