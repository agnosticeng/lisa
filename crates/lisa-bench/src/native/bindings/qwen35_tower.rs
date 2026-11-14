//! Application construction of the generic Qwen tower.
use crate::bindings::qwen35_mlx::MlxExecution;
use anyhow::Result;
use lisa_models::qwen3_5::config::Qwen35Config;
use std::path::Path;
pub type Qwen35Tower =
    lisa_engine::bindings::qwen35_tower::Qwen35Tower<lisa_metal::device::MetalDevice>;
pub trait Qwen35Load: Sized {
    fn load(dir: &Path, config: Qwen35Config) -> Result<Self>;
    fn load_with_dspine(dir: &Path, config: Qwen35Config, draft: &Path) -> Result<Self>;
    fn from_weights(w: crate::core::loader::Weights, config: Qwen35Config) -> Result<Self>;
    fn from_weights_cached(
        w: crate::core::loader::Weights,
        config: Qwen35Config,
        dir: Option<&Path>,
    ) -> Result<Self>;
    fn load_gguf(path: &Path) -> Result<Self>;
}
impl Qwen35Load for Qwen35Tower {
    fn load(dir: &Path, config: Qwen35Config) -> Result<Self> {
        Ok(Self::from_execution(
            crate::metal_language::runtime(),
            MlxExecution::load(crate::metal_context::execution_context(), dir, config)?,
        ))
    }
    fn load_with_dspine(dir: &Path, config: Qwen35Config, draft: &Path) -> Result<Self> {
        Ok(Self::from_execution(
            crate::metal_language::runtime(),
            MlxExecution::load_with_dspine(
                crate::metal_context::execution_context(),
                dir,
                config,
                Some(draft),
            )?,
        ))
    }
    fn from_weights(w: crate::core::loader::Weights, config: Qwen35Config) -> Result<Self> {
        Ok(Self::from_execution(
            crate::metal_language::runtime(),
            MlxExecution::from_weights(crate::metal_context::execution_context(), w, config)?,
        ))
    }
    fn from_weights_cached(
        w: crate::core::loader::Weights,
        config: Qwen35Config,
        dir: Option<&Path>,
    ) -> Result<Self> {
        Ok(Self::from_execution(
            crate::metal_language::runtime(),
            MlxExecution::from_weights_cached(
                crate::metal_context::execution_context(),
                w,
                config,
                dir,
            )?,
        ))
    }
    fn load_gguf(path: &Path) -> Result<Self> {
        Ok(Self::from_execution(
            crate::metal_language::runtime(),
            crate::bindings::qwen35_gguf::execution::GgufExecution::load(
                crate::metal_context::execution_context(),
                path,
            )?,
        ))
    }
}
