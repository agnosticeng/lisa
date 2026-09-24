//! Qwen 3.8 Flash-Next Metal kernels.
//!
//! The host wrappers and the helper libraries live in these modules; the Metal
//! sources are co-located under `kernels/` (and `kernels/qsa/`). The generic
//! kernels stay shared in `backend::metal::kernels/{common,nax}`.

pub mod kernels;
pub mod moe_decode;
pub mod moe_helpers;
pub mod nax_helpers;
pub mod prefill_indirect;
pub mod qsa;
