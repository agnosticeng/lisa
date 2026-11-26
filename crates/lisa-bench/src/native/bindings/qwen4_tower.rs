//! Application construction for the portable Flash-Next bindings.
use lisa_metal::device::MetalDevice;
pub type Tower = lisa_engine::bindings::qwen4::tower::Tower<MetalDevice>;
pub use super::qwen4_decoder::{Block, DecoderLayer};
pub trait TowerLoad {
    fn load(
        dir: &std::path::Path,
        config: lisa_models::qwen4::config::ModelConfig,
    ) -> anyhow::Result<Self>
    where
        Self: Sized;
}
impl TowerLoad for Tower {
    fn load(
        dir: &std::path::Path,
        config: lisa_models::qwen4::config::ModelConfig,
    ) -> anyhow::Result<Self> {
        lisa_metal::warm_kernels();
        lisa_engine::weights::qwen4::load_on(crate::metal_language::runtime(), dir, config)
    }
}
