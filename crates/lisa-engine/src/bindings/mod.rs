//! Portable weight, state and plan bindings. Device selection belongs to applications.
pub mod deepseek41;
pub mod flux2;
pub mod flux3;
pub mod glm53;
pub mod inkling2;
pub mod lfm3;
pub mod olala;
pub mod qwen35;
pub mod qwen4;
pub mod shared;

pub trait BindingDevice:
    crate::cache::indexer::CacheDevice<Tensor: Send + Sync>
    + lisa_backend::HostAccessDevice
    + Send
    + Sync
{
}
impl<D> BindingDevice for D
where
    D: crate::cache::indexer::CacheDevice + lisa_backend::HostAccessDevice + Send + Sync,
    D::Tensor: Send + Sync,
{
}

// Public aliases for callers using the flat model binding paths.
pub use deepseek41::attention as deepseek41_attention;
pub use deepseek41::engram as deepseek41_engram;
pub use deepseek41::graph as deepseek41_graph;
pub use deepseek41::hyper as deepseek41_hyper;
pub use deepseek41::moe as deepseek41_moe;
pub use deepseek41::projection as deepseek41_projection;
pub use deepseek41::state as deepseek41_state;
pub use deepseek41::tower as deepseek41_tower;
pub use glm53::affine::dense_mlp as glm53_dense_mlp;
pub use glm53::affine::moe as glm53_affine_moe;
pub use glm53::affine::moe::ops as glm53_moe_ops;
pub use glm53::affine::projection as glm53_affine_projection;
pub use glm53::affine::tower as glm53_affine_tower;
pub use glm53::affine::tower::ops as glm53_affine_tower_ops;
pub use glm53::attention as glm53_attention;
pub use glm53::attention::absorbed as glm53_absorbed_attention;
pub use glm53::attention::indexer as glm53_indexer;
pub use glm53::attention::mode as glm53_attention_mode;
pub use glm53::attention::ops as glm53_attention_ops;
pub use glm53::hyper as glm53_hyper;
pub use glm53::kda as glm53_kda;
pub use glm53::kda::projection as glm53_kda_projection;
pub use glm53::kda::recurrent as glm53_recurrent;
pub use glm53::law as glm53_law;
pub use glm53::math as glm53_math;
pub use glm53::packed::attention as glm53_packed_attention;
pub use glm53::packed::moe as glm53_moe;
pub use glm53::packed::projection as glm53_projection;
pub use glm53::packed::tower as glm53_packed_tower;
pub use glm53::policy as glm53_policy;
pub use qwen4::attention as qwen4_attention;
pub use qwen4::decoder as qwen4_decoder;
pub use qwen4::hyper as qwen4_hyper;
pub use qwen4::indexer as qwen4_indexer;
pub use qwen4::moe as qwen4_moe;
pub use qwen4::mtp as qwen4_mtp;
pub use qwen4::ple as qwen4_ple;
pub use qwen4::tower as qwen4_tower;
pub use qwen35::attention as qwen35_attention;
pub use qwen35::decoder as qwen35_decoder;
pub use qwen35::entry as qwen35_entry;
pub use qwen35::execution as qwen35_execution;
pub use qwen35::gguf as qwen35_gguf;
pub use qwen35::mlx as qwen35_mlx;
pub use qwen35::mtp as qwen35_mtp;
pub use qwen35::readout as qwen35_readout;
pub use qwen35::tower as qwen35_tower;
pub use shared::{affine, dense, gated_delta, normalization, tensor_io};
