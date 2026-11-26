//! Portable checkpoint binding. Applications supply an owned device context.
use crate::{
    bindings::{glm53::affine::moe::ExpertDevice, qwen35::tower::Qwen35Tower},
    checkpoint::{self, ModelKind, config::ReadConfig},
    language::{LanguageModel, runtime::Runtime},
    sampling::sampler::Sampler,
};
use lisa_backend::ReadbackDevice;
use std::path::Path;
pub type Language<D> = Box<dyn LanguageModel<Runtime<D, Sampler<D>>>>;
pub fn language<D: ExpertDevice + ReadbackDevice>(
    runtime: Runtime<D, Sampler<D>>,
    path: &Path,
    kind: ModelKind,
) -> anyhow::Result<Language<D>>
where
    D::Error: From<anyhow::Error>,
{
    let context = runtime.context.clone();
    anyhow::ensure!(
        crate::memory::kv_format() != crate::memory::KvFormat::Int8 || kind == ModelKind::Inkling2,
        "--kv-format int8 is supported only by Inkling 2; {kind:?} keeps its native key/value format"
    );
    let mut model: Language<D> = match kind {
        ModelKind::Inkling2 => Box::new(crate::weights::inkling2::gguf::load(runtime, path)?),
        ModelKind::Olala => Box::new(crate::weights::olala::load(runtime, path)?),
        ModelKind::Lfm3 => Box::new(crate::weights::lfm3::load(runtime, path)?),
        ModelKind::Qwen35 => {
            let cfg =
                lisa_models::qwen3_5::config::Qwen35Config::from_json(&path.join("config.json"))?;
            Box::new(Qwen35Tower::from_execution(
                runtime,
                crate::weights::qwen35::mlx::load(context, path, cfg)?,
            ))
        }
        ModelKind::Qwen35Gguf => Box::new(Qwen35Tower::from_execution(
            runtime,
            crate::weights::qwen35::gguf::load(context, path)?,
        )),
        ModelKind::Qwen4 => {
            let cfg =
                lisa_models::qwen4::config::ModelConfig::from_json(&path.join("config.json"))?;
            Box::new(crate::weights::qwen4::load_on(runtime, path, cfg)?)
        }
        ModelKind::Glm53 => {
            let cfg = checkpoint::glm53::ModelConfig::from_json(&path.join("config.json"))?;
            Box::new(crate::weights::glm53::affine::load(runtime, path, cfg)?)
        }
        ModelKind::Glm53Gguf => Box::new(crate::weights::glm53::packed::load(runtime, path)?),
        ModelKind::DeepSeek41 => {
            Box::new(crate::weights::deepseek41::load(context, runtime, path)?)
        }
        _ => anyhow::bail!("{kind:?} is not a language model"),
    };
    if let Some(capacity) = checkpoint::default_context_capacity(path)?
        && model.max_position_embeddings() != capacity
    {
        model.set_context_capacity(capacity)?;
    }
    Ok(model)
}
