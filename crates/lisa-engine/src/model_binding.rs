//! Portable checkpoint binding. Applications supply an owned device context.
use crate::{
    bindings::{
        glm53_affine_moe::ExpertDevice, qwen35_gguf::execution::GgufExecution,
        qwen35_mlx::MlxExecution, qwen35_tower::Qwen35Tower,
    },
    checkpoint::{self, ModelKind},
    core::config::ReadConfig,
    language::LanguageModel,
    language_runtime::Runtime,
    sampler::Sampler,
};
use lisa_backend::ReadbackDevice;
use std::path::Path;
pub type Language<D> = Box<dyn LanguageModel<Runtime<D, Sampler<D>>>>;
pub fn language<D: ExpertDevice + ReadbackDevice>(
    runtime: Runtime<D, Sampler<D>>,
    path: &Path,
    kind: ModelKind,
    draft: Option<&Path>,
) -> anyhow::Result<Language<D>>
where
    D::Error: From<anyhow::Error>,
{
    let context = runtime.context.clone();
    anyhow::ensure!(
        draft.is_none() || kind == ModelKind::Qwen35,
        "DSpine requires a compatible MLX Qwen 3.8 27B checkpoint"
    );
    let mut model: Language<D> = match kind {
        ModelKind::Qwen35 => {
            let cfg =
                lisa_models::qwen3_5::config::Qwen35Config::from_json(&path.join("config.json"))?;
            Box::new(Qwen35Tower::from_execution(
                runtime,
                MlxExecution::load_with_dspine(context, path, cfg, draft)?,
            ))
        }
        ModelKind::Qwen35Gguf => Box::new(Qwen35Tower::from_execution(
            runtime,
            GgufExecution::load(context, path)?,
        )),
        ModelKind::Qwen4 => {
            let cfg =
                lisa_models::qwen4::config::ModelConfig::from_json(&path.join("config.json"))?;
            Box::new(crate::bindings::qwen4_tower::Tower::load_on(
                runtime, path, cfg,
            )?)
        }
        ModelKind::Glm53 => {
            let cfg = checkpoint::glm53::ModelConfig::from_json(&path.join("config.json"))?;
            Box::new(crate::bindings::glm53_affine_tower::Tower::load_on(
                runtime, path, cfg,
            )?)
        }
        ModelKind::Glm53Gguf => Box::new(crate::bindings::glm53_packed_tower::Tower::load_on(
            runtime, path,
        )?),
        ModelKind::DeepSeek41 => Box::new(crate::bindings::deepseek41_tower::Tower::load(
            context, runtime, path,
        )?),
        _ => anyhow::bail!("{kind:?} is not a language model"),
    };
    if let Some(capacity) = checkpoint::default_context_capacity(path)? {
        if model.max_position_embeddings() != capacity {
            model.set_context_capacity(capacity)?;
        }
    }
    Ok(model)
}
