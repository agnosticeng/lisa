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

/// Force `get_or_init` on every Metal-kernel static in every qwen4 module.
///
/// Called from `Tower::load` so all kernel handles exist before the first
/// forward pass. Note: a `MetalKernel` handle is created without compiling;
/// the actual Metal pipeline is compiled lazily on first `apply`, keyed by
/// the call site's template args + dtypes (`Tower::warmup` then exercises
/// the prefill-2048 and S=1 shapes so those pipelines are built at load too).
pub fn warm_kernels() {
    let stream = lisa_mlx::Stream::gpu();
    kernels::warm(&stream);
    moe_decode::warm(&stream);
    prefill_indirect::warm();
    qsa::warm();
    // `moe_helpers` / `nax_helpers` only carry `const` shader source strings
    // (include_str!) — no kernel statics, nothing to warm.
}
