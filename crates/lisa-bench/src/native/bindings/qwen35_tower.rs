//! Application construction of the generic Qwen tower.
use anyhow::Result;
use lisa_engine::weights::qwen35::mlx as qwen35_mlx;
use lisa_models::qwen3_5::config::Qwen35Config;
use std::path::Path;
pub type Qwen35Tower =
    lisa_engine::bindings::qwen35::tower::Qwen35Tower<lisa_metal::device::MetalDevice>;
pub trait Qwen35Load: Sized {
    fn load(dir: &Path, config: Qwen35Config) -> Result<Self>;
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
            qwen35_mlx::load(crate::metal_context::execution_context(), dir, config)?,
        ))
    }
    fn from_weights(w: crate::core::loader::Weights, config: Qwen35Config) -> Result<Self> {
        Ok(Self::from_execution(
            crate::metal_language::runtime(),
            qwen35_mlx::from_weights(crate::metal_context::execution_context(), w, config)?,
        ))
    }
    fn from_weights_cached(
        w: crate::core::loader::Weights,
        config: Qwen35Config,
        dir: Option<&Path>,
    ) -> Result<Self> {
        Ok(Self::from_execution(
            crate::metal_language::runtime(),
            qwen35_mlx::from_weights_cached(
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
            lisa_engine::weights::qwen35::gguf::load(
                crate::metal_context::execution_context(),
                path,
            )?,
        ))
    }
}
