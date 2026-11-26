//! Metal provider construction at the application boundary.
pub type Device = lisa_metal::device::MetalDevice;
pub type LanguageRuntime = lisa_engine::language::runtime::Runtime<Device, Sampler>;
pub type Language = lisa_engine::model_binding::Language<Device>;
pub type Sampler = lisa_engine::sampling::sampler::Sampler<Device>;
pub type Session = lisa_engine::conversation::Conversation<LanguageRuntime>;
pub type Batch = lisa_engine::batching::Batch<LanguageRuntime>;
pub type ContinuousBatch = lisa_engine::batching::ContinuousBatch<LanguageRuntime>;
pub type Request = lisa_engine::scheduling::Request<Sampler>;
pub type LayerCache = lisa_engine::cache::LayerCache<Device>;
pub type PrefixCache = lisa_engine::conversation::cache::PrefixCache<Device>;
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
) -> anyhow::Result<lisa_engine::bindings::qwen4::tower::Tower<Device>> {
    let context = context()?;
    lisa_metal::warm_kernels();
    lisa_engine::weights::qwen4::load_on(runtime(context), path, config)
}
/// Numerical identity of the device kernels, for models that declare none.
pub fn kernel_law() -> u64 {
    lisa_metal::runtime::kernel_law()
}
/// Token ids as a `[1, S]` device tensor.
pub fn token_row(ids: &[u32]) -> lisa_metal::Array {
    lisa_metal::Array::from_slice(
        &ids.iter().map(|&t| t as i32).collect::<Vec<_>>(),
        &[1i32, ids.len() as i32],
    )
}
/// Mean of `[1, S, H]` hidden states over the sequence, read back as f32.
pub fn mean_pool(hidden: &lisa_metal::Array) -> anyhow::Result<Vec<f32>> {
    let pooled = hidden
        .mean_axis(1, false)?
        .as_dtype(lisa_metal::Dtype::Float32)?;
    pooled.eval()?;
    Ok(pooled.as_slice::<f32>().to_vec())
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

/// Sparse HTTP milestones, independent of GPU execution. The diagnostic log
/// survives an externally stopped server; trace JSON also receives the event.
pub(crate) fn trace_http(name: &'static str) {
    if !lisa_metal::trace::enabled_for(name) {
        return;
    }
    static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let elapsed = START.get_or_init(std::time::Instant::now).elapsed();
    lisa_metal::trace::event(name);
    eprintln!(
        "[http.trace] {}us {:?} {name}",
        elapsed.as_micros(),
        std::thread::current().id(),
    );
}
