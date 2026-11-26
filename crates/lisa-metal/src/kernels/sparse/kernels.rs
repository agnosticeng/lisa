//! GLM-5.3 KDA Metal kernels (phase 3 port, the reference `glm53_kda.metal` reference).
//!
//! The KDA prepare, recurrence, and output kernels port the reference's F32 execution
//! order. MLX-loaded weights and projections feed the same operation graph.

use crate::ffi::{KernelInput, MetalKernel, OutputArg, TemplateArg};
use crate::kernels::EXACT_HEADER;
use crate::{Array, Dtype, Stream};
use objc2_metal::MTLDevice;
#[path = "attention.rs"]
mod attention;
#[path = "hyperconnection.rs"]
mod hyperconnection;
#[path = "indexing.rs"]
mod indexing;
#[path = "linear.rs"]
mod linear;
#[path = "moe.rs"]
mod moe;
#[path = "normalization.rs"]
mod normalization;
#[path = "recurrent.rs"]
mod recurrent;
#[path = "routing.rs"]
mod routing;
pub use attention::dsa_decode_scores;
pub use attention::dsa_decode_weighted_v;
pub use attention::dsa_indexed_decode_fused;
pub use attention::dsa_indexed_lora_wide_native;
pub use attention::dsa_indexer_topk;
pub use attention::dsa_indexer_topk_shuffle;
pub use attention::dsa_o_project;
pub use attention::dsa_qk_lowrank_project;
pub use attention::dsa_qkv_project_pair;
pub use attention::dsa_qkv_rms_norm_rows;
pub use attention::dsa_value_project_rows;
pub use hyperconnection::hc_expand4;
pub use hyperconnection::hc_mix_rms;
pub use hyperconnection::hc_mix_rms_bf16_gemv;
pub use hyperconnection::hc_split_sinkhorn;
pub use hyperconnection::hc_split_weighted_sum_norm4;
pub use hyperconnection::hc_weighted_sum;
pub use hyperconnection::shared_q8_swiglu_ds4;
pub use indexing::expand_pool_selection;
pub use indexing::expand_pool_selection_for_attention;
pub use indexing::indexer_pool_fused;
pub use indexing::indexer_scores;
pub use indexing::indexer_scores_batch;
pub use indexing::indexer_scores_rows4;
pub use linear::affine_q8_mv_ds4;
pub use linear::bf16_gemm;
pub use linear::bf16_gemv;
pub use linear::bf16_kda_project;
pub use linear::bf16_kda_project_with_nsg;
pub use linear::bf16_pair_gemv;
pub use linear::bf16_qkv;
pub use moe::moe_expert_route_map;
pub use moe::moe_gather_qmv;
pub use moe::moe_gather_qmv_addr;
pub use moe::moe_gather_qmv_addr_ds4;
pub use moe::moe_gather_qmv_sum8_addr;
pub use moe::moe_gather_qmv_sum8_addr_add_shared;
pub use moe::moe_gather_qmv_sum8_addr_add_shared_down;
pub use moe::moe_gather_qmv_sum8_addr_ds4;
pub use moe::moe_gather_qmv_sum8_mapped;
pub use moe::moe_mapped_resource_table;
pub use moe::moe_pair_swiglu;
pub use moe::moe_pair_swiglu_addr;
pub use moe::moe_pair_swiglu_mapped;
pub use moe::moe_sum8;
pub use normalization::ds4_rms_norm;
pub use normalization::mtp_add_rms_norm;
pub use normalization::mtp_add3;
pub use normalization::mtp_kv_lora_rms_norm;
pub use normalization::mtp_rms_norm;
pub use normalization::qkv_norm_compact;
pub use normalization::qkv_norm_compact_into;
pub use normalization::rms_row_fused;
pub use normalization::swiglu_fused;
pub use normalization::weighted_clamped_swiglu;
pub use recurrent::kda_decode;
pub use recurrent::kda_output_norm;
pub use recurrent::kda_prepare;
pub use recurrent::kda_recurrence;
pub use routing::router_matvec_f32;
pub use routing::router_select_fused;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

#[cfg(test)]
#[path = "tests_q2_decode.rs"]
mod tests_q2_decode;
pub use attention::dsa_indexed_lora_short;
