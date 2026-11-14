//! Native conformance bindings and arithmetic oracles for generic engine plans.
pub(crate) mod affine;
pub(crate) mod dense;
pub mod gated_delta;
pub mod qwen35_attention;
pub(crate) mod qwen35_decoder;
pub(crate) mod qwen35_entry;
pub(crate) mod qwen35_execution;
pub(crate) mod qwen35_mlx;
pub mod qwen35_mtp;
pub(crate) mod qwen35_readout;
pub mod qwen35_tower;

pub mod dspine;
pub(crate) mod dspine_bridge;
pub(crate) mod dspine_layers;
pub mod dspine_policy;
pub(crate) mod dspine_retrieval;
pub(crate) mod dspine_retrieval_plans;

pub mod flux2;
pub mod flux3;

pub mod qwen35_gguf;

pub mod qwen35;

pub mod deepseek41_moe;
pub mod packed;

pub mod deepseek41_projection;

pub mod glm53_projection;

pub mod glm53_moe;

pub mod deepseek41_engram;

pub mod deepseek41_weights;

pub mod deepseek41_hyper;

pub mod glm53_hyper;
pub mod qwen4;
pub mod qwen4_attention;
pub mod qwen4_decoder;
pub mod qwen4_hyper;
pub mod qwen4_indexer;
pub mod qwen4_moe;
pub mod qwen4_mtp;
pub mod qwen4_ple;
pub mod qwen4_tower;

pub mod deepseek41_attention;

pub mod deepseek41_graph;

pub mod deepseek41_tower;

pub mod deepseek41_state;

pub mod deepseek41;

pub mod glm53_math;

pub mod glm53_kda_projection;

pub(crate) mod glm53_kda;

pub mod glm53_recurrent;

pub mod glm53_indexer;

pub mod glm53_packed_attention;

pub mod glm53_packed_tower;

pub mod glm53_affine_projection;

pub mod glm53_dense_mlp;

pub mod glm53_absorbed_attention;
pub mod glm53_attention_mode;

pub(crate) mod glm53_attention_ops;

pub mod glm53_attention;

pub(crate) mod glm53_moe_ops;

pub mod glm53_affine_moe;

pub(crate) mod glm53_affine_tower_ops;

pub mod glm53_affine_tower;
#[cfg(test)]
pub(crate) mod glm53_oracle_helpers;
#[cfg(test)]
pub mod glm53_smoke;
#[cfg(test)]
pub(crate) mod glm53_test_fixtures;
