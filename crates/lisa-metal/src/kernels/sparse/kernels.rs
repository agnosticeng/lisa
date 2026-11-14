//! GLM-5.3 KDA Metal kernels (phase 3 port, ds4 `glm53_kda.metal` reference).
//!
//! The KDA prepare, recurrence, and output kernels port ds4's F32 execution
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
#[allow(unused_imports)]
use attention::DSA_INDEXED_DECODE_FUSED_SOURCE;
#[allow(unused_imports)]
use attention::DSA_INDEXED_DECODE_NATIVE_SOURCE;
#[allow(unused_imports)]
use attention::DSA_INDEXED_LORA_NATIVE_SOURCE;
#[allow(unused_imports)]
use attention::DSA_INDEXED_LORA_SOURCE;
#[allow(unused_imports)]
use attention::DSA_O_PROJECT_SOURCE;
#[allow(unused_imports)]
use attention::DSA_QK_LOWRANK_NATIVE_SOURCE;
#[allow(unused_imports)]
use attention::DSA_QK_LOWRANK_SOURCE;
#[allow(unused_imports)]
use attention::DSA_QKV_NORM_ROWS_SOURCE;
#[allow(unused_imports)]
use attention::DSA_QKV_PROJECT_PAIR_SOURCE;
#[allow(unused_imports)]
use attention::DSA_SCORES_SOURCE;
#[allow(unused_imports)]
use attention::DSA_VALUE_PROJECT_NATIVE_MMA_SOURCE;
#[allow(unused_imports)]
use attention::DSA_VALUE_PROJECT_NATIVE_SOURCE;
#[allow(unused_imports)]
use attention::DSA_VALUE_PROJECT_SOURCE;
#[allow(unused_imports)]
use attention::DSA_WV_SOURCE;
#[allow(unused_imports)]
use attention::Ds4IndexedLoraArgs;
pub use attention::dsa_decode_scores;
pub use attention::dsa_decode_weighted_v;
#[allow(unused_imports)]
pub use attention::dsa_indexed_decode_fused;
#[allow(unused_imports)]
use attention::dsa_indexed_decode_fused_capture_kernel;
#[allow(unused_imports)]
use attention::dsa_indexed_decode_fused_kernel;
#[allow(unused_imports)]
use attention::dsa_indexed_decode_native_kernel;
#[allow(unused_imports)]
use attention::dsa_indexed_lora_kernel;
#[allow(unused_imports)]
pub use attention::dsa_indexed_lora_wide_native;
pub use attention::dsa_indexer_topk;
pub use attention::dsa_indexer_topk_shuffle;
#[allow(unused_imports)]
use attention::dsa_indexer_topk_with_shuffle;
pub use attention::dsa_o_project;
#[allow(unused_imports)]
use attention::dsa_o_project_kernel;
#[allow(unused_imports)]
use attention::dsa_qk_lowrank_kernel;
#[allow(unused_imports)]
use attention::dsa_qk_lowrank_native_kernel;
pub use attention::dsa_qk_lowrank_project;
#[allow(unused_imports)]
use attention::dsa_qkv_norm_rows_kernel;
#[allow(unused_imports)]
use attention::dsa_qkv_project_pair_kernel;
#[allow(unused_imports)]
use attention::dsa_scores_kernel;
#[allow(unused_imports)]
use attention::dsa_value_project_kernel;
#[allow(unused_imports)]
use attention::dsa_value_project_native_kernel;
pub use attention::dsa_value_project_rows;
#[allow(unused_imports)]
use attention::dsa_wv_kernel;
pub use attention::glm53_dsa_qkv_project_pair;
pub use attention::glm53_dsa_qkv_rms_norm_rows;
#[allow(unused_imports)]
use hyperconnection::HC_EXPAND_SOURCE;
#[allow(unused_imports)]
use hyperconnection::HC_MIX_RMS_BF16_GEMV_SOURCE;
#[allow(unused_imports)]
use hyperconnection::HC_MIX_RMS_SOURCE;
#[allow(unused_imports)]
use hyperconnection::HC_SPLIT_NORM4_SOURCE;
#[allow(unused_imports)]
use hyperconnection::HC_SPLIT_SOURCE;
#[allow(unused_imports)]
use hyperconnection::HC_WSUM_SOURCE;
#[allow(unused_imports)]
use hyperconnection::SHARED_Q8_SWIGLU_SOURCE;
pub use hyperconnection::glm53_hc_mix_rms;
pub use hyperconnection::glm53_hc_mix_rms_bf16_gemv;
#[allow(unused_imports)]
use hyperconnection::hc_expand_kernel;
pub use hyperconnection::hc_expand4;
#[allow(unused_imports)]
use hyperconnection::hc_mix_rms_bf16_gemv_kernel;
#[allow(unused_imports)]
use hyperconnection::hc_mix_rms_kernel;
#[allow(unused_imports)]
use hyperconnection::hc_split_kernel;
#[allow(unused_imports)]
use hyperconnection::hc_split_norm4_kernel;
pub use hyperconnection::hc_split_sinkhorn;
pub use hyperconnection::hc_split_weighted_sum_norm4;
pub use hyperconnection::hc_weighted_sum;
#[allow(unused_imports)]
use hyperconnection::hc_wsum_kernel;
pub use hyperconnection::shared_q8_swiglu_ds4;
#[allow(unused_imports)]
use hyperconnection::shared_q8_swiglu_ds4_kernel;
#[allow(unused_imports)]
use indexing::INDEXER_EXPAND_SELECTION_SOURCE;
#[allow(unused_imports)]
use indexing::INDEXER_POOL_SOURCE;
#[allow(unused_imports)]
use indexing::INDEXER_SCORES_BATCH_SOURCE;
#[allow(unused_imports)]
use indexing::INDEXER_SCORES_DECODE_ROWS4_SOURCE;
#[allow(unused_imports)]
use indexing::INDEXER_SCORES_DECODE_SOURCE;
#[allow(unused_imports)]
use indexing::INDEXER_SCORES_TILED_SOURCE;
#[allow(unused_imports)]
use indexing::INDEXER_TOPK_MERGE_SOURCE;
#[allow(unused_imports)]
use indexing::INDEXER_TOPK_SORT_SOURCE;
pub use indexing::glm53_expand_pool_selection;
pub use indexing::glm53_expand_pool_selection_for_attention;
#[allow(unused_imports)]
use indexing::glm53_expand_pool_selection_mode;
pub use indexing::glm53_indexer_scores;
pub use indexing::glm53_indexer_scores_batch;
pub use indexing::glm53_indexer_scores_rows4;
#[allow(unused_imports)]
use indexing::glm53_indexer_scores_with_decode_rows4;
#[allow(unused_imports)]
use indexing::indexer_expand_selection_kernel;
pub use indexing::indexer_pool_fused;
#[allow(unused_imports)]
use indexing::indexer_pool_kernel;
#[allow(unused_imports)]
use indexing::indexer_scores_batch_kernel;
#[allow(unused_imports)]
use indexing::indexer_scores_decode_kernel;
#[allow(unused_imports)]
use indexing::indexer_scores_decode_rows4_kernel;
#[allow(unused_imports)]
use indexing::indexer_scores_tiled_kernel;
#[allow(unused_imports)]
use indexing::indexer_topk_merge_kernel;
#[allow(unused_imports)]
use indexing::indexer_topk_sort_kernel;
#[allow(unused_imports)]
use linear::AFFINE_Q8_MV_DS4_SOURCE;
#[allow(unused_imports)]
use linear::BF16_GEMM_SOURCE;
#[allow(unused_imports)]
use linear::BF16_GEMV_ROWS2_SOURCE;
#[allow(unused_imports)]
use linear::BF16_GEMV_SOURCE;
#[allow(unused_imports)]
use linear::BF16_KDA_PROJECT_SOURCE;
#[allow(unused_imports)]
use linear::BF16_PAIR_GEMV_SOURCE;
#[allow(unused_imports)]
use linear::BF16_QKV_SOURCE;
#[allow(unused_imports)]
use linear::affine_q8_mv_ds4_kernel;
#[allow(unused_imports)]
use linear::bf16_gemm_kernel;
#[allow(unused_imports)]
use linear::bf16_gemv_kernel;
#[allow(unused_imports)]
use linear::bf16_gemv_rows2_kernel;
#[allow(unused_imports)]
use linear::bf16_kda_project_kernel;
#[allow(unused_imports)]
use linear::bf16_pair_gemv_kernel;
#[allow(unused_imports)]
use linear::bf16_projection_nsg;
#[allow(unused_imports)]
use linear::bf16_qkv_kernel;
pub use linear::glm53_affine_q8_mv_ds4;
pub use linear::glm53_bf16_gemm;
pub use linear::glm53_bf16_gemv;
pub use linear::glm53_bf16_kda_project;
pub use linear::glm53_bf16_kda_project_with_nsg;
pub use linear::glm53_bf16_pair_gemv;
pub use linear::glm53_bf16_qkv;
#[allow(unused_imports)]
use linear::glm53_bf16_qkv_inner;
#[allow(unused_imports)]
use moe::MOE_EXPERT_ROUTE_MAP_SOURCE;
#[allow(unused_imports)]
use moe::MOE_GATHER_ADDR_DS4_SOURCE;
#[allow(unused_imports)]
use moe::MOE_GATHER_ADDR_SOURCE;
#[allow(unused_imports)]
use moe::MOE_GATHER_SOURCE;
#[allow(unused_imports)]
use moe::MOE_GATHER_SUM8_ADDR_DS4_SOURCE;
#[allow(unused_imports)]
use moe::MOE_GATHER_SUM8_ADDR_SOURCE;
#[allow(unused_imports)]
use moe::MOE_GATHER_SUM8_MAPPED_BODY;
#[allow(unused_imports)]
use moe::MOE_MAPPED_RESOURCE_HELPER;
#[allow(unused_imports)]
use moe::MOE_MAPPED_RESOURCE_INPUTS;
#[allow(unused_imports)]
use moe::MOE_PAIR_SWIGLU_ADDR_SOURCE;
#[allow(unused_imports)]
use moe::MOE_PAIR_SWIGLU_MAPPED_BODY;
#[allow(unused_imports)]
use moe::MOE_PAIR_SWIGLU_SOURCE;
#[allow(unused_imports)]
use moe::MOE_SUM8_SOURCE;
pub use moe::moe_expert_route_map;
#[allow(unused_imports)]
use moe::moe_expert_route_map_kernel;
#[allow(unused_imports)]
use moe::moe_gather_addr_ds4_kernel;
#[allow(unused_imports)]
use moe::moe_gather_addr_kernel;
#[allow(unused_imports)]
use moe::moe_gather_header;
#[allow(unused_imports)]
use moe::moe_gather_kernel;
pub use moe::moe_gather_qmv;
pub use moe::moe_gather_qmv_addr;
pub use moe::moe_gather_qmv_addr_ds4;
pub use moe::moe_gather_qmv_sum8_addr;
pub use moe::moe_gather_qmv_sum8_addr_add_shared;
pub use moe::moe_gather_qmv_sum8_addr_add_shared_down;
pub use moe::moe_gather_qmv_sum8_addr_ds4;
#[allow(unused_imports)]
use moe::moe_gather_qmv_sum8_addr_impl;
pub use moe::moe_gather_qmv_sum8_mapped;
#[allow(unused_imports)]
use moe::moe_gather_sum8_addr_ds4_kernel;
#[allow(unused_imports)]
use moe::moe_gather_sum8_addr_kernel;
#[allow(unused_imports)]
use moe::moe_gather_sum8_mapped_kernel;
pub use moe::moe_mapped_resource_table;
pub use moe::moe_pair_swiglu;
pub use moe::moe_pair_swiglu_addr;
#[allow(unused_imports)]
use moe::moe_pair_swiglu_addr_kernel;
#[allow(unused_imports)]
use moe::moe_pair_swiglu_kernel;
pub use moe::moe_pair_swiglu_mapped;
#[allow(unused_imports)]
use moe::moe_pair_swiglu_mapped_kernel;
pub use moe::moe_sum8;
#[allow(unused_imports)]
use moe::moe_sum8_kernel;
#[allow(unused_imports)]
use moe::padded_moe_resource_inputs;
#[allow(unused_imports)]
use normalization::MTP_ADD_RMS_NORM_SOURCE;
#[allow(unused_imports)]
use normalization::MTP_ADD3_SOURCE;
#[allow(unused_imports)]
use normalization::MTP_KV_LORA_RMS_NORM_SOURCE;
#[allow(unused_imports)]
use normalization::MTP_RMS_NORM_SOURCE;
#[allow(unused_imports)]
use normalization::QKV_NORM_COMPACT_SOURCE;
#[allow(unused_imports)]
use normalization::RMS_ROW_SOURCE;
#[allow(unused_imports)]
use normalization::SWIGLU_SOURCE;
#[allow(unused_imports)]
use normalization::ds4_rms_norm_kernel;
pub use normalization::glm53_ds4_rms_norm;
pub use normalization::glm53_mtp_add_rms_norm;
pub use normalization::glm53_mtp_add3;
pub use normalization::glm53_mtp_kv_lora_rms_norm;
pub use normalization::glm53_mtp_rms_norm;
pub use normalization::glm53_qkv_norm_compact;
pub use normalization::glm53_qkv_norm_compact_into;
#[allow(unused_imports)]
use normalization::mtp_add_rms_norm_kernel;
#[allow(unused_imports)]
use normalization::mtp_add3_kernel;
#[allow(unused_imports)]
use normalization::mtp_kv_lora_rms_norm_kernel;
#[allow(unused_imports)]
use normalization::qkv_norm_compact_kernel;
pub use normalization::rms_row_fused;
#[allow(unused_imports)]
use normalization::rms_row_kernel;
pub use normalization::swiglu_fused;
#[allow(unused_imports)]
use normalization::swiglu_kernel;
#[allow(unused_imports)]
use recurrent::KDA_DECODE_SOURCE;
#[allow(unused_imports)]
use recurrent::KDA_OUTPUT_SOURCE;
#[allow(unused_imports)]
use recurrent::KDA_PREPARE_SOURCE;
#[allow(unused_imports)]
use recurrent::KDA_RECURRENCE_SOURCE;
pub use recurrent::kda_decode;
#[allow(unused_imports)]
use recurrent::kda_decode_kernel;
#[allow(unused_imports)]
use recurrent::kda_kernel;
#[allow(unused_imports)]
use recurrent::kda_output_kernel;
pub use recurrent::kda_output_norm;
pub use recurrent::kda_prepare;
#[allow(unused_imports)]
use recurrent::kda_prepare_kernel;
pub use recurrent::kda_recurrence;
#[allow(unused_imports)]
use routing::ROUTER_MV_SMALL_SOURCE;
#[allow(unused_imports)]
use routing::ROUTER_MV_SOURCE;
#[allow(unused_imports)]
use routing::ROUTER_SELECT_SOURCE;
pub use routing::router_matvec_f32;
#[allow(unused_imports)]
use routing::router_matvec_f32_small;
#[allow(unused_imports)]
use routing::router_mv_kernel;
#[allow(unused_imports)]
use routing::router_mv_small_kernel;
pub use routing::router_select_fused;
#[allow(unused_imports)]
use routing::router_select_kernel;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;

#[cfg(test)]
#[path = "tests_q2_decode.rs"]
mod tests_q2_decode;
pub use attention::dsa_indexed_lora_short;
