#[cfg(test)]
use super::{
    glm53_affine_moe::DenseMlp,
    glm53_affine_tower_ops::{self as plans, one as plan_one},
    glm53_hyper::Hc,
};
use crate::checkpoint::glm53::ModelConfig;
#[cfg(test)]
use crate::core::cache::IndexerTape;
#[cfg(test)]
use crate::core::cache::LayerCache;
#[cfg(test)]
use crate::core::capture::Capture;
#[cfg(test)]
use crate::core::loader::Weights;
#[cfg(test)]
use crate::core::quant::QuantizedEmbedding;
#[cfg(test)]
use crate::core::quant::QuantizedLinear;
pub use lisa_engine::bindings::glm53::policy::INC_STEP;
#[cfg(test)]
use lisa_metal::{Array, Dtype};
#[cfg(test)]
use lisa_models::glm53::affine_tower::{self as model, Request as R};
use std::path::Path;
#[cfg(test)]
const DS4_GLM53_NATIVE_BATCH_ROW_LIMIT: usize = model::NATIVE_BATCH_ROW_LIMIT;
#[cfg(test)]
const DS4_GLM53_PREFILL_CHUNK_TOKENS: usize = model::PREFILL_CHUNK;
#[cfg(test)]
fn ds4_native_session_batch_position_supported(r: usize, p: usize, l: usize) -> bool {
    model::batch_position_supported(r, p, l)
}
pub type Tower =
    lisa_engine::bindings::glm53::affine::tower::Tower<lisa_metal::device::MetalDevice>;
pub type Layer =
    lisa_engine::bindings::glm53::affine::tower::Layer<lisa_metal::device::MetalDevice>;
pub type Attn = lisa_engine::bindings::glm53::affine::tower::Attn<lisa_metal::device::MetalDevice>;
pub type Mlp = lisa_engine::bindings::glm53::affine::tower::Mlp<lisa_metal::device::MetalDevice>;
pub type MtpBlock =
    lisa_engine::bindings::glm53::affine::tower::MtpBlock<lisa_metal::device::MetalDevice>;
pub trait TowerLoad {
    fn load(dir: &Path, config: ModelConfig) -> anyhow::Result<Self>
    where
        Self: Sized;
}
impl TowerLoad for Tower {
    fn load(dir: &Path, config: ModelConfig) -> anyhow::Result<Self> {
        lisa_engine::weights::glm53::affine::load(crate::metal_language::runtime(), dir, config)
    }
}
#[cfg(test)]
fn rms(x: &Array, w: &Array, eps: f32) -> anyhow::Result<Array> {
    lisa_engine::bindings::glm53::affine::tower::rms(
        &crate::metal_context::execution_context(),
        x,
        w,
        eps,
    )
}
#[cfg(test)]
fn final_rms(x: &Array, w: &Array, eps: f32) -> anyhow::Result<Array> {
    lisa_engine::bindings::glm53::affine::tower::final_rms(
        &crate::metal_context::execution_context(),
        x,
        w,
        eps,
    )
}
#[cfg(test)]
fn final_hc_mean(x: &Array) -> anyhow::Result<Array> {
    lisa_engine::bindings::glm53::affine::tower::final_hc_mean(
        &crate::metal_context::execution_context(),
        x,
    )
}
#[cfg(test)]
fn mtp_rms(x: &Array, w: &Array, eps: f32) -> anyhow::Result<Array> {
    lisa_engine::bindings::glm53::affine::tower::mtp_rms(
        &crate::metal_context::execution_context(),
        x,
        w,
        eps,
    )
}
#[cfg(test)]
fn narrow(x: &Array, axis: usize, start: usize, len: usize) -> lisa_metal::error::Result<Array> {
    Ok(lisa_engine::bindings::glm53::affine::tower::narrow(
        &crate::metal_context::execution_context(),
        x,
        axis,
        start,
        len,
    )?)
}

#[cfg(test)]
#[path = "glm53_affine_tower_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "glm53_affine_tower_plan_tests.rs"]
mod plan_tests;
