pub use super::glm53_dense_mlp::{
    DenseMlp, GateUpProjection, clamped_swiglu, fuse_gate_up, split_gate_up,
};
#[cfg(test)]
use super::glm53_oracle_helpers::narrow;
use crate::core::{
    capture::Capture,
    loader::{LazyTensorStore, TensorSource, Weights},
    quant::QuantizedLinear,
};
use lisa_backend::ResourceDevice;
pub use lisa_engine::bindings::glm53_policy::{Glm53Q8GateUpNativeScope, native_gate_up_enabled};
use lisa_metal::{Array, Dtype, ops};
use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc, Mutex, Weak,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};
pub type MoE = lisa_engine::bindings::glm53_affine_moe::MoE<lisa_metal::device::MetalDevice>;
pub type ExpertParameters =
    lisa_engine::bindings::glm53_affine_moe::ExpertParameters<lisa_metal::device::MetalDevice>;
pub type CachedExpert =
    lisa_engine::bindings::glm53_affine_moe::CachedExpert<lisa_metal::device::MetalDevice>;
pub type ExpertCacheState =
    lisa_engine::bindings::glm53_affine_moe::ExpertCacheState<lisa_metal::device::MetalDevice>;
pub type ExpertPools =
    lisa_engine::bindings::glm53_affine_moe::ExpertPools<lisa_metal::device::MetalDevice>;
pub type ExpertStageMessage =
    lisa_engine::bindings::glm53_affine_moe::ExpertStageMessage<lisa_metal::device::MetalDevice>;
pub type GroupedMoeWorkspace =
    lisa_engine::bindings::glm53_affine_moe::GroupedMoeWorkspace<lisa_metal::device::MetalDevice>;
pub type GroupedMoeBuffers =
    lisa_engine::bindings::glm53_affine_moe::GroupedMoeBuffers<lisa_metal::device::MetalDevice>;
pub type GroupedMoeViews =
    lisa_engine::bindings::glm53_affine_moe::GroupedMoeViews<lisa_metal::device::MetalDevice>;
fn selected_prefetch_bytes(expert_ids: &[usize], expert_cache_bytes: &[usize]) -> Option<usize> {
    expert_ids.iter().try_fold(0usize, |total, &expert_id| {
        total.checked_add(*expert_cache_bytes.get(expert_id)?)
    })
}

#[cfg(test)]
fn encode_f32(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[cfg(all(test, feature = "glm53-fixtures"))]
fn encode_i32(values: &[i32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[cfg(all(test, feature = "glm53-fixtures"))]
fn encode_u32(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

#[cfg(test)]
#[path = "glm53_affine_moe_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "glm53_moe_grouped_workspace_tests.rs"]
mod grouped_workspace_tests;

#[cfg(test)]
#[path = "glm53_moe_grouped_workspace_bounds_tests.rs"]
mod grouped_workspace_bounds_tests;

#[cfg(test)]
#[path = "glm53_moe_address_oracle.rs"]
mod address_oracle;

#[cfg(test)]
#[path = "glm53_moe_address_tests.rs"]
pub(crate) mod address_tests;
