//! Native conformance bindings and arithmetic oracles for generic engine plans.
#[cfg(test)]
pub(crate) mod affine;
#[cfg(test)]
pub(crate) mod dense;
#[cfg(test)]
pub mod gated_delta;
#[cfg(test)]
pub mod qwen35_attention;
#[cfg(test)]
pub(crate) mod qwen35_entry;
#[cfg(test)]
pub mod qwen35_mtp;
#[cfg(test)]
pub(crate) mod qwen35_readout;
pub mod qwen35_tower;

pub mod flux2;
#[cfg(test)]
pub mod flux3;

pub mod qwen35_gguf;

#[cfg(test)]
pub mod qwen35;

pub mod deepseek41_moe;
#[cfg(test)]
pub mod packed;

pub mod deepseek41_projection;
#[cfg(test)]
mod deepseek41_projection_oracle;

pub mod glm53_projection;

#[cfg(test)]
pub mod glm53_moe;

pub mod deepseek41_engram;

pub mod deepseek41_weights;

#[cfg(test)]
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
#[cfg(test)]
mod glm53_math_oracle;

#[cfg(test)]
pub mod glm53_kda_projection;

#[cfg(test)]
pub mod glm53_recurrent;

pub mod glm53_indexer;

#[cfg(test)]
pub mod glm53_packed_attention;

#[cfg(test)]
pub mod glm53_packed_tower;

pub mod glm53_affine_projection;

pub mod glm53_dense_mlp;

#[cfg(test)]
pub mod glm53_absorbed_attention;
pub mod glm53_attention_mode;

#[cfg(test)]
pub(crate) mod glm53_attention_ops;

#[cfg(test)]
pub mod glm53_attention;

pub mod glm53_affine_moe;

pub(crate) mod glm53_affine_tower_ops;

pub mod glm53_affine_tower;
#[cfg(test)]
pub(crate) mod glm53_oracle_helpers;
