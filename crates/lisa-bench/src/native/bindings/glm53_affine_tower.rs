use super::{
    glm53_affine_moe::{DenseMlp, MoE},
    glm53_affine_tower_ops::{self as plans, one as plan_one},
    glm53_attention::Dsa,
    glm53_hyper::Hc,
    glm53_recurrent::Kda,
};
use crate::{
    checkpoint::glm53::ModelConfig,
    core::{
        cache::{FullAttentionCache, GdnCache, IndexerTape, LayerCache},
        capture::Capture,
        loader::{self, LazyTensorStore, Weights},
        quant::{QuantizedEmbedding, QuantizedLinear},
    },
};
pub use lisa_engine::bindings::glm53_policy::INC_STEP;
use lisa_metal::{Array, Dtype};
use lisa_models::glm53::affine_tower::{self as model, Request as R};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};
const DS4_GLM53_NATIVE_BATCH_ROW_LIMIT: usize = model::NATIVE_BATCH_ROW_LIMIT;
const DS4_GLM53_PREFILL_CHUNK_TOKENS: usize = model::PREFILL_CHUNK;
const DS4_GLM53_DEFAULT_CONTEXT: usize = model::DEFAULT_CONTEXT;
fn ds4_native_session_batch_position_supported(r: usize, p: usize, l: usize) -> bool {
    model::batch_position_supported(r, p, l)
}
pub type Tower = lisa_engine::bindings::glm53_affine_tower::Tower<lisa_metal::device::MetalDevice>;
pub type Layer = lisa_engine::bindings::glm53_affine_tower::Layer<lisa_metal::device::MetalDevice>;
pub type Attn = lisa_engine::bindings::glm53_affine_tower::Attn<lisa_metal::device::MetalDevice>;
pub type Mlp = lisa_engine::bindings::glm53_affine_tower::Mlp<lisa_metal::device::MetalDevice>;
pub type MtpBlock =
    lisa_engine::bindings::glm53_affine_tower::MtpBlock<lisa_metal::device::MetalDevice>;
pub trait TowerLoad {
    fn load(dir: &Path, config: ModelConfig) -> anyhow::Result<Self>
    where
        Self: Sized;
}
impl TowerLoad for Tower {
    fn load(dir: &Path, config: ModelConfig) -> anyhow::Result<Self> {
        Self::load_on(crate::metal_language::runtime(), dir, config)
    }
}
fn rms(x: &Array, w: &Array, eps: f32) -> anyhow::Result<Array> {
    Ok(lisa_engine::bindings::glm53_affine_tower::rms(
        &crate::metal_context::execution_context(),
        x,
        w,
        eps,
    )?)
}
fn final_rms(x: &Array, w: &Array, eps: f32) -> anyhow::Result<Array> {
    Ok(lisa_engine::bindings::glm53_affine_tower::final_rms(
        &crate::metal_context::execution_context(),
        x,
        w,
        eps,
    )?)
}
fn final_hc_mean(x: &Array) -> anyhow::Result<Array> {
    Ok(lisa_engine::bindings::glm53_affine_tower::final_hc_mean(
        &crate::metal_context::execution_context(),
        x,
    )?)
}
fn mtp_rms(x: &Array, w: &Array, eps: f32) -> anyhow::Result<Array> {
    Ok(lisa_engine::bindings::glm53_affine_tower::mtp_rms(
        &crate::metal_context::execution_context(),
        x,
        w,
        eps,
    )?)
}
fn reshape(x: &Array, shape: &[i32]) -> lisa_metal::error::Result<Array> {
    Ok(lisa_engine::bindings::glm53_affine_tower::reshape(
        &crate::metal_context::execution_context(),
        x,
        shape,
    )?)
}
fn promote(x: &Array) -> lisa_metal::error::Result<Array> {
    Ok(lisa_engine::bindings::glm53_affine_tower::promote(
        &crate::metal_context::execution_context(),
        x,
    )?)
}
fn narrow(x: &Array, axis: usize, start: usize, len: usize) -> lisa_metal::error::Result<Array> {
    Ok(lisa_engine::bindings::glm53_affine_tower::narrow(
        &crate::metal_context::execution_context(),
        x,
        axis,
        start,
        len,
    )?)
}
fn join(arrays: &[&Array], axis: i32) -> lisa_metal::error::Result<Array> {
    Ok(lisa_engine::bindings::glm53_affine_tower::join(
        &crate::metal_context::execution_context(),
        arrays,
        axis,
    )?)
}
fn i32_arr(_rt_of: &Array, tokens: &[u32]) -> anyhow::Result<Array> {
    Ok(lisa_engine::bindings::glm53_affine_tower::i32_arr(
        &crate::metal_context::execution_context(),
        _rt_of,
        tokens,
    )?)
}

#[cfg(test)]
#[path = "glm53_affine_tower_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "glm53_affine_tower_capture.rs"]
mod migration_capture;
#[cfg(test)]
use migration_capture::*;

#[cfg(test)]
#[path = "glm53_affine_tower_plan_tests.rs"]
mod plan_tests;

use crate::core::quant::QuantizedLinearExecution;
