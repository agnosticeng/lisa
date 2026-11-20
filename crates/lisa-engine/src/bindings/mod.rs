//! Portable weight, state and plan bindings. Device selection belongs to applications.
pub mod deepseek41_engram;
pub mod deepseek41_hyper;
pub mod deepseek41_moe;
pub mod deepseek41_projection;
pub mod deepseek41_state;
pub mod deepseek41_weights;
pub mod packed;
pub trait BindingDevice:
    crate::indexer_cache::CacheDevice<Tensor: Send + Sync>
    + lisa_backend::HostAccessDevice
    + Send
    + Sync
{
}
impl<D> BindingDevice for D
where
    D: crate::indexer_cache::CacheDevice + lisa_backend::HostAccessDevice + Send + Sync,
    D::Tensor: Send + Sync,
{
}
pub mod deepseek41_attention;
pub mod deepseek41_graph;
pub mod deepseek41_tower;

pub mod normalization;

pub mod qwen35_execution;
pub mod qwen35_tower;

pub mod dense;
pub mod gated_delta;

pub mod qwen35_attention;

pub mod qwen35_decoder;

pub mod qwen35_mtp;

pub mod dspine_bridge;
pub mod dspine_layers;

pub mod dspine;

pub mod affine;
pub mod dspine_retrieval;
pub mod dspine_retrieval_plans;
pub mod qwen35_readout;

pub mod dspine_policy;
pub mod qwen35_entry;

pub mod qwen35_mlx;

pub mod qwen4_hyper;

pub mod qwen4_moe;

pub mod qwen4_indexer;

pub mod qwen4_attention;

pub mod qwen4_ple;

pub mod qwen4_decoder;

pub mod qwen4_mtp;

pub mod qwen4_tower;

pub mod qwen35_gguf;

pub mod glm53_policy;

pub mod glm53_math;

pub mod glm53_indexer;

pub mod glm53_hyper;

pub mod glm53_projection;

pub mod glm53_kda;

pub mod glm53_kda_projection;

pub mod glm53_moe;

pub mod glm53_recurrent;

pub mod glm53_packed_attention;

pub mod glm53_packed_tower;

pub mod flux2;
pub mod flux3;

pub mod glm53_attention_mode;

pub mod glm53_affine_projection;

pub mod glm53_moe_ops;

pub mod glm53_attention_ops;

pub mod glm53_absorbed_attention;

pub mod glm53_dense_mlp;

pub mod tensor_io;

pub mod glm53_affine_tower_ops;
pub mod glm53_attention;

#[cfg(feature = "glm53-fixtures")]
pub mod glm53_test_fixtures;

pub mod glm53_affine_moe;

pub mod glm53_affine_tower;
pub mod glm53_law;

pub mod lfm3;

pub mod olala;
