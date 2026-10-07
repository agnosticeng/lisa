use std::sync::Arc;

use objc2_metal::MTLSize;

use crate::array::{Array, Dtype};
use crate::error::{Error, Result};
use crate::runtime::{Buffer, ComputePipeline, ConstVal, MetalRuntime};

use super::compile::{builtin_template_def, compile_builtin, get_2d_grid_dims, type_string};
use super::reduce::reduce_axis0_sum;
use super::{Device, Tensor};
use super::{
    MLX_GEMM_PREAMBLE, MLX_QUANTIZED_PREAMBLE, MLX_QUANTIZED_UTILS_PREAMBLE, MLX_UTILS_PREAMBLE,
};
use super::compile::compile_builtin_math;

/// Mirror of MLX `qmv_fast_k_alignment` (`quantized.cpp:147`): the K step in
/// `qmv_fast_impl` must divide K for the fast path to be valid.
fn qmv_fast_k_alignment(bits: i32) -> i32 {
    let pack_factor = 32 / bits.max(1);
    pack_factor * (if bits == 2 { 1 } else { 2 }) * 32
}

/// The K alignment (and n % 8) condition under which `affine_qmv_fast` is the
/// M=1 reference kernel — shared by the dispatch gate of
/// [`affine_verify_qmm`] so the verify-width kernel is only taken exactly
/// where it is bit-identical to the per-row qmv_fast chain.
pub(crate) fn qmv_fast_eligible(k: usize, n: usize, bits: i32) -> bool {
    n % 8 == 0 && (k as i32) % qmv_fast_k_alignment(bits) == 0
}

/// Precompile the verify-width specializations M = 2..=7 (bf16) so the MTP
/// verify never pays a runtime JIT stall (the kernels are shape-independent
/// — K/N ride as runtime constants — so one compile per M covers every
/// projection). Mirrors `warm_kernels` for the fast ops.
pub fn warm_verify_qmm(device: &Device, group_size: i32, bits: i32) {
    let type_str = match type_string(crate::array::Dtype::Bfloat16) {
        Ok(t) => t,
        Err(_) => return,
    };
    for m in 2..=7usize {
        let kname = format!("affine_verify_qmm_{type_str}_gs_{group_size}_b_{bits}_m_{m}");
        let template_def = builtin_template_def(
            &kname,
            "affine_verify_qmm",
            &[
                type_str.to_string(),
                group_size.to_string(),
                bits.to_string(),
                m.to_string(),
            ],
        );
        let source = format!(
            "{MLX_UTILS_PREAMBLE}{MLX_GEMM_PREAMBLE}{MLX_QUANTIZED_UTILS_PREAMBLE}{MLX_QUANTIZED_PREAMBLE}{VERIFY_QMM_SOURCE}{template_def}"
        );
        let _ = compile_builtin(device, &source, &kname);
        // The split-K lane: one specialization per (M, K_PARTS); BN is derived
        // from M. Both K_PARTS variants warmed — the tower
        // mixes N >= 4096 and N < 4096 projections.
        for k_parts in [2usize, 4] {
            let bn = 2; // measured BN for every M (see verify_qmm_splitk_lane)
            let kname = format!(
                "affine_verify_qmm_splitk_{type_str}_gs_{group_size}_b_{bits}_m_{m}_bn_{bn}_kp_{k_parts}_ppt_1"
            );
            let template_def = builtin_template_def(
                &kname,
                "affine_verify_qmm_splitk",
                &[
                    type_str.to_string(),
                    group_size.to_string(),
                    bits.to_string(),
                    m.to_string(),
                    bn.to_string(),
                    k_parts.to_string(),
                    "1".to_string(),
                    "0".to_string(),
                ],
            );
            let source = format!(
                "{MLX_UTILS_PREAMBLE}{MLX_GEMM_PREAMBLE}{MLX_QUANTIZED_UTILS_PREAMBLE}{MLX_QUANTIZED_PREAMBLE}{VERIFY_QMM_SOURCE}{template_def}"
            );
            let _ = compile_builtin(device, &source, &kname);
        }
        // The msg lane (huge-N, lm_head class): one specialization per
        // (M, BN); BN is derived from M.
        if let Some(bn) = verify_qmm_msg_lane(m, 5120, 151_936, bits) {
            let kname =
                format!("affine_verify_qmm_msg_{type_str}_gs_{group_size}_b_{bits}_m_{m}_bn_{bn}");
            let template_def = builtin_template_def(
                &kname,
                "affine_verify_qmm_msg",
                &[
                    type_str.to_string(),
                    group_size.to_string(),
                    bits.to_string(),
                    m.to_string(),
                    bn.to_string(),
                    "8".to_string(),
                ],
            );
            let source = format!(
                "{MLX_UTILS_PREAMBLE}{MLX_GEMM_PREAMBLE}{MLX_QUANTIZED_UTILS_PREAMBLE}{MLX_QUANTIZED_PREAMBLE}{VERIFY_QMM_SOURCE}{template_def}"
            );
            let _ = compile_builtin(device, &source, &kname);
        }
    }
}

/// Affine `quantized_matmul` for a single-batch (B = 1) vector input, matching
/// MLX's `qmv`/`qmv_fast` path (`quantized.cpp:461`). `w` is the packed uint32
/// weight `[N, K*bits/32]`; `scales`/`biases` are `[N, K/group_size]`.
///
/// Returns `(kernel_name, generated_source, func)` for the chosen variant.
pub fn affine_qmv_source(
    x: &Tensor,
    w: &Tensor,
    group_size: i32,
    bits: i32,
) -> Result<(String, String)> {
    let (_, k) = x.dims2();
    let (n, _) = w.dims2();
    let type_str = type_string(x.dtype())?;
    let fast = n % 8 == 0 && (k as i32) % qmv_fast_k_alignment(bits) == 0;
    let func = if fast {
        "affine_qmv_fast"
    } else {
        "affine_qmv"
    };
    let mut kname = String::from("affine_");
    kname.push_str(if fast { "qmv_fast_" } else { "qmv_" });
    kname.push_str(type_str);
    kname.push_str(&format!("_gs_{group_size}_b_{bits}_batch_0"));
    // template args: type, group_size, bits, batched=false, has_global_scale=false,
    // results_per_simdgroup=4 (bool prints as 0/1, as in C++ ostream).
    let template_def = builtin_template_def(
        &kname,
        func,
        &[
            type_str.to_string(),
            group_size.to_string(),
            bits.to_string(),
            "0".to_string(),
            "0".to_string(),
            "4".to_string(),
        ],
    );
    let source = format!(
        "{MLX_UTILS_PREAMBLE}{MLX_GEMM_PREAMBLE}{MLX_QUANTIZED_UTILS_PREAMBLE}{MLX_QUANTIZED_PREAMBLE}{template_def}"
    );
    Ok((kname, source))
}

/// JIT-compile the affine qmv variant for these shapes through the runtime.
pub fn affine_qmv_pipeline_jit(
    device: &Device,
    x: &Tensor,
    w: &Tensor,
    group_size: i32,
    bits: i32,
) -> Result<ComputePipeline> {
    let (kname, source) = affine_qmv_source(x, w, group_size, bits)?;
    compile_builtin(device, &source, &kname)
}

/// Affine `quantized_matmul` using a JIT-compiled pipeline (see
/// [`affine_qmv_pipeline_jit`]).
pub fn affine_qmv_fast(
    device: &Device,
    x: &Tensor,
    w: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: i32,
    bits: i32,
) -> Result<Tensor> {
    let pipeline = affine_qmv_pipeline_jit(device, x, w, group_size, bits)?;
    affine_qmv_dispatch(device, &pipeline, x, w, scales, biases)
}

/// Dispatch a prebuilt affine qmv pipeline over these inputs.
pub fn affine_qmv_dispatch(
    device: &Device,
    pipeline: &ComputePipeline,
    x: &Tensor,
    w: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
) -> Result<Tensor> {
    let mdev = device;
    let (m, k) = x.dims2();
    let (n, _) = w.dims2();
    let out_dtype = x.dtype();

    let ycount = m * n;
    let ybuf = mdev.buffer((ycount) as usize * (out_dtype).size_of(), "qmv_out")?;
    let y = Array::from_parts(mdev, ybuf.clone(), &vec![m, n], out_dtype);

    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(pipeline);

    let bind = |index: usize, t: &Tensor| -> Result<()> {
        let (ms, layout) = t.buffer_and_layout();
        if !layout.is_contiguous() {
            crate::bail!("affine_qmv: non-contiguous input");
        }
        enc.set_input(index, Some(ms), layout.offset * t.dtype().size_of());
        Ok(())
    };
    bind(0, w)?;
    bind(1, scales)?;
    bind(2, biases)?;
    bind(3, x)?;
    enc.set_output(4, Some(&ybuf), 0);
    let kk = k as i32;
    enc.set_bytes(5, &kk);
    let nn = n as i32;
    enc.set_bytes(6, &nn);

    let bn = 8usize;
    enc.dispatch_groups_size(
        MTLSize {
            width: m,
            height: n.div_ceil(bn),
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 2,
            depth: 1,
        },
    );
    Ok(y)
}

/// Norm-in-qmv mega-kernel (specs/16 phase 1): ONE dispatch computes
/// `(bf16(x + r), rms_norm(bf16(x + r)), qmv(rms_norm))` — the fused
/// residual-add + RMSNorm prologue inside `affine_qmv_fast`. Bit-exactness
/// argument lives in the shader header; pinned word-for-word by the
/// `qmv_addnorm_bitexact_vs_fused_chain` test. Decode GEMV shapes only
/// (`qmv_fast_eligible`, axis_size > 4096 so the looped-rms association
/// applies, group_size 64 / bits 4, bf16).
///
/// Returns `(y, sum)` plus `normed` when `write_norm` is set (the GDN
/// in_proj site still consumes the normed row downstream).
#[allow(clippy::too_many_arguments)]
pub fn affine_qmv_fast_addnorm(
    device: &Device,
    x: &Tensor,
    r: &Tensor,
    norm_weight: &Tensor,
    w: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    eps: f32,
    group_size: i32,
    bits: i32,
    write_norm: bool,
) -> Result<(Tensor, Tensor, Option<Tensor>)> {
    if group_size != 64 || bits != 4 {
        crate::bail!("addnorm qmv: only gs64/b4 is pinned");
    }
    let (m, k) = x.dims2();
    let (n, _) = w.dims2();
    if m != 1 {
        crate::bail!("addnorm qmv: decode GEMV only (M=1)");
    }
    if !qmv_fast_eligible(k, n, bits) || k <= 4096 {
        crate::bail!("addnorm qmv: shape ineligible (k={k}, n={n})");
    }
    let axis_size = k as u32;
    let emu = super::norm::looped_rms_tgs();
    if emu == 0 || emu > 1024 || emu % 32 != 0 {
        crate::bail!("addnorm qmv: bad emulated threadgroup size {emu}");
    }
    let type_str = type_string(x.dtype())?;
    if type_str != "bfloat16_t" {
        crate::bail!("addnorm qmv: only bfloat16 pinned");
    }
    let wn = if write_norm { 1 } else { 0 };
    let kname =
        format!("affine_qmv_fast_addnorm_{type_str}_gs_{group_size}_b_{bits}_wn_{wn}");
    let template_def = builtin_template_def(
        &kname,
        "affine_qmv_fast_addnorm",
        &[
            type_str.to_string(),
            group_size.to_string(),
            bits.to_string(),
            wn.to_string(),
        ],
    );
    let source = format!(
        "{MLX_UTILS_PREAMBLE}{MLX_GEMM_PREAMBLE}{MLX_QUANTIZED_UTILS_PREAMBLE}{MLX_QUANTIZED_PREAMBLE}{}{template_def}",
        crate::jit::MLX_NORM_QMV_SOURCE
    );
    let pipeline = compile_builtin(device, &source, &kname)?;

    let out_dtype = x.dtype();
    let ybuf = device.buffer((m * n) as usize * out_dtype.size_of(), "qmv_out")?;
    let y = Array::from_parts(device, ybuf.clone(), &vec![m, n], out_dtype);
    let sbuf = device.buffer(k as usize * out_dtype.size_of(), "fused_add_rms_sum")?;
    let sum = Array::from_parts(device, sbuf.clone(), &vec![m, k], out_dtype);
    // ALWAYS allocated and bound: the kernel stages the normed row to this
    // scratch unconditionally (the stock qmv_fast_impl re-reads it); nil
    // buffer bindings are UB on the write + read-back path.
    let nb = device.buffer(k as usize * out_dtype.size_of(), "fused_add_rms_norm")?;
    let normed = if write_norm {
        Some(Array::from_parts(device, nb.clone(), &vec![m, k], out_dtype))
    } else {
        None
    };

    let bind = |enc: &crate::runtime::ComputeEncoder, index: usize, t: &Tensor| -> Result<()> {
        let (ms, layout) = t.buffer_and_layout();
        if !layout.is_contiguous() {
            crate::bail!("addnorm qmv: non-contiguous input");
        }
        enc.set_input(index, Some(ms), layout.offset * t.dtype().size_of());
        Ok(())
    };

    let guard = device.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    bind(&enc, 0, w)?;
    bind(&enc, 1, scales)?;
    bind(&enc, 2, biases)?;
    bind(&enc, 3, x)?;
    bind(&enc, 4, r)?;
    {
        let (ms, wl) = norm_weight.buffer_and_layout();
        if !wl.is_contiguous() {
            crate::bail!("addnorm qmv: non-contiguous norm weight");
        }
        enc.set_input(5, Some(ms), wl.start_offset());
        let stride: u32 = if norm_weight.rank() == 1 {
            wl.stride()[0] as u32
        } else {
            0
        };
        enc.set_bytes(13, &stride);
    }
    enc.set_output(6, Some(&sbuf), 0);
    enc.set_output(7, Some(&nb), 0);
    enc.set_output(8, Some(&ybuf), 0);
    enc.set_bytes(9, &eps);
    let kk = k as i32;
    enc.set_bytes(10, &kk);
    let nn = n as i32;
    enc.set_bytes(11, &nn);
    enc.set_bytes(12, &axis_size);
    enc.set_bytes(14, &emu);

    let bn = 8usize;
    enc.dispatch_groups_size(
        MTLSize {
            width: m,
            height: n.div_ceil(bn),
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 2,
            depth: 1,
        },
    );
    Ok((y, sum, normed))
}

/// Tail-ULP variant of [`affine_qmv_fast_addnorm`] (specs/16 §7, campaign
/// close): same signature and contract as the bit-exact mega-kernel except
/// the norm numerics relax to the §9.7 tail-ULP rule (<= 1 bf16 ULP + argmax
/// equality vs `fast::fused_add_rms_norm` + `quantized_matmul`, pinned by
/// `qmv_addnorm_tailulp_vs_fused_chain`). The prologue is a plain 64-thread
/// strided reduction with the s row resident in THREADGROUP memory — no
/// emulated-lane replay, no device scratch, no device barrier (the two costs
/// that made phase 1b regress in situ, §5.2). Deterministic: every
/// threadgroup of a row derives bit-identical values; only tid.y==0 writes
/// the device outputs.
#[allow(clippy::too_many_arguments)]
pub fn affine_qmv_fast_addnorm_tu(
    device: &Device,
    x: &Tensor,
    r: &Tensor,
    norm_weight: &Tensor,
    w: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    eps: f32,
    group_size: i32,
    bits: i32,
    write_norm: bool,
) -> Result<(Tensor, Tensor, Option<Tensor>)> {
    if group_size != 64 || bits != 4 {
        crate::bail!("addnorm-tu qmv: only gs64/b4 is pinned");
    }
    let (m, k) = x.dims2();
    let (n, _) = w.dims2();
    if m != 1 {
        crate::bail!("addnorm-tu qmv: decode GEMV only (M=1)");
    }
    if k > 5120 {
        crate::bail!("addnorm-tu qmv: threadgroup row buffer sized for k<=5120 (k={k})");
    }
    if !qmv_fast_eligible(k, n, bits) || k <= 4096 {
        crate::bail!("addnorm-tu qmv: shape ineligible (k={k}, n={n})");
    }
    let axis_size = k as u32;
    let type_str = type_string(x.dtype())?;
    if type_str != "bfloat16_t" {
        crate::bail!("addnorm-tu qmv: only bfloat16 pinned");
    }
    let wn = if write_norm { 1 } else { 0 };
    let kname = format!(
        "affine_qmv_fast_addnorm_tu_{type_str}_gs_{group_size}_b_{bits}_wn_{wn}"
    );
    let template_def = builtin_template_def(
        &kname,
        "affine_qmv_fast_addnorm_tu",
        &[
            type_str.to_string(),
            group_size.to_string(),
            bits.to_string(),
            wn.to_string(),
        ],
    );
    let source = format!(
        "{MLX_UTILS_PREAMBLE}{MLX_GEMM_PREAMBLE}{MLX_QUANTIZED_UTILS_PREAMBLE}{MLX_QUANTIZED_PREAMBLE}{}{template_def}",
        crate::jit::MLX_NORM_QMV_SOURCE
    );
    let pipeline = compile_builtin(device, &source, &kname)?;

    let out_dtype = x.dtype();
    let ybuf = device.buffer((m * n) as usize * out_dtype.size_of(), "qmv_out")?;
    let y = Array::from_parts(device, ybuf.clone(), &vec![m, n], out_dtype);
    let sbuf = device.buffer(k as usize * out_dtype.size_of(), "fused_add_rms_sum")?;
    let sum = Array::from_parts(device, sbuf.clone(), &vec![m, k], out_dtype);
    let nb = device.buffer(k as usize * out_dtype.size_of(), "fused_add_rms_norm")?;
    let normed = if write_norm {
        Some(Array::from_parts(device, nb.clone(), &vec![m, k], out_dtype))
    } else {
        None
    };

    let bind = |enc: &crate::runtime::ComputeEncoder, index: usize, t: &Tensor| -> Result<()> {
        let (ms, layout) = t.buffer_and_layout();
        if !layout.is_contiguous() {
            crate::bail!("addnorm-tu qmv: non-contiguous input");
        }
        enc.set_input(index, Some(ms), layout.offset * t.dtype().size_of());
        Ok(())
    };

    let guard = device.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    bind(&enc, 0, w)?;
    bind(&enc, 1, scales)?;
    bind(&enc, 2, biases)?;
    bind(&enc, 3, x)?;
    bind(&enc, 4, r)?;
    {
        let (ms, wl) = norm_weight.buffer_and_layout();
        if !wl.is_contiguous() {
            crate::bail!("addnorm-tu qmv: non-contiguous norm weight");
        }
        enc.set_input(5, Some(ms), wl.start_offset());
        let stride: u32 = if norm_weight.rank() == 1 {
            wl.stride()[0] as u32
        } else {
            0
        };
        enc.set_bytes(13, &stride);
    }
    enc.set_output(6, Some(&sbuf), 0);
    enc.set_output(7, Some(&nb), 0);
    enc.set_output(8, Some(&ybuf), 0);
    enc.set_bytes(9, &eps);
    let kk = k as i32;
    enc.set_bytes(10, &kk);
    let nn = n as i32;
    enc.set_bytes(11, &nn);
    enc.set_bytes(12, &axis_size);

    let bn = 8usize;
    enc.dispatch_groups_size(
        MTLSize {
            width: m,
            height: n.div_ceil(bn),
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 2,
            depth: 1,
        },
    );
    Ok((y, sum, normed))
}

/// Verify-width quantized matmul (the `verifyQmm` lane,
/// split-K family):
/// M = 2..8 input rows served by ONE weight stream. Tiling collapses the
/// qmv_fast input-row grid axis; the M rows ride an unrolled inner loop
/// reusing the same weight words. Numerics are bit-identical to the per-row
/// `affine_qmv_fast` chain by construction (same qdot/load_vector expression
/// trees, same pack assignment, same simd_sum reduction) — pinned by the
/// `verify_qmm_bitexact_vs_qmv` test.
///
/// Requires `qmv_fast_eligible(k, n, bits)` (n % 8 == 0, K aligned) and
/// 2 <= m; the caller (`ops::quantized_matmul`) enforces both.
pub fn affine_verify_qmm(
    device: &Device,
    x: &Tensor,
    w: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: i32,
    bits: i32,
) -> Result<Tensor> {
    let (m, k) = x.dims2();
    let (n, _) = w.dims2();
    debug_assert!(m >= 2 && m <= 16);
    debug_assert!(qmv_fast_eligible(k, n, bits));
    let type_str = type_string(x.dtype())?;
    // One specialization per M (distinct host_name — two specializations
    // sharing a name bind the wrong binary).
    let kname = format!("affine_verify_qmm_{type_str}_gs_{group_size}_b_{bits}_m_{m}");
    let template_def = builtin_template_def(
        &kname,
        "affine_verify_qmm",
        &[
            type_str.to_string(),
            group_size.to_string(),
            bits.to_string(),
            m.to_string(),
        ],
    );
    let source = format!(
        "{MLX_UTILS_PREAMBLE}{MLX_GEMM_PREAMBLE}{MLX_QUANTIZED_UTILS_PREAMBLE}{MLX_QUANTIZED_PREAMBLE}{VERIFY_QMM_SOURCE}{template_def}"
    );
    let pipeline = compile_builtin(device, &source, &kname)?;

    let out_dtype = x.dtype();
    let ycount = m * n;
    let ybuf = device.buffer(ycount as usize * out_dtype.size_of(), "verify_qmm_out")?;
    let y = Array::from_parts(device, ybuf.clone(), &vec![m, n], out_dtype);

    let guard = device.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    let bind = |index: usize, t: &Tensor| -> Result<()> {
        let (ms, layout) = t.buffer_and_layout();
        if !layout.is_contiguous() {
            crate::bail!("affine_verify_qmm: non-contiguous input");
        }
        enc.set_input(index, Some(ms), layout.offset * t.dtype().size_of());
        Ok(())
    };
    bind(0, w)?;
    bind(1, scales)?;
    bind(2, biases)?;
    bind(3, x)?;
    enc.set_output(4, Some(&ybuf), 0);
    let kk = k as i32;
    enc.set_bytes(5, &kk);
    let nn = n as i32;
    enc.set_bytes(6, &nn);

    // The tile: one threadgroup column (all M rows), 8 output columns per
    // threadgroup, 2 simdgroups x 32 threads — qmv_fast's threadgroup shape.
    enc.dispatch_groups_size(
        MTLSize {
            width: 1,
            height: n.div_ceil(8),
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 2,
            depth: 1,
        },
    );
    Ok(y)
}

/// The split-K lane gate, restricted to the plain-SIMD splitk lane lisa
/// ports: M 2..=7, N >= 512, N < 100000 (huge-N is the msg lane — NOT ported,
/// the lm_head class stays on qmv_wide), K % 64 == 0, N % BN == 0, K % 8 == 0
/// (pack geometry), (K/8) % K_PARTS == 0. BN = 2 (MEASURED deviation — see
/// below); K_PARTS (2 for N >= 4096, else 4). Returns the (bn, k_parts) tile
/// when the lane applies. 4-bit only (the shipping trunk class; mixed-plain
/// stays off here).
/// Shipped packs-per-thread for the split-K verify tile (specs/15 §6): the
/// PPT extension over the 1-pack body. 1 = the exact shipped loop
/// shape — FALSIFIED in situ (5 interleaved base/new round-cost pairs,
/// 27B): the isolated-bench win (~8% at M=5) does NOT convert — S7
/// median 98.06 vs 94.54 ms base (+3.7%), S5 and 16k wash. Kernel param
/// and the tile dispatch stay as opt-in infra.
pub const SPLITK_PPT: usize = 1;

pub fn verify_qmm_splitk_lane(m: usize, k: usize, n: usize, bits: i32) -> Option<(usize, usize)> {
    if bits != 4 {
        return None;
    }
    if !(2..=8).contains(&m) {
        return None;
    }
    // M=8 joined the split-K verify tile (specs/08 m16 campaign): the only
    // wider MROWS that does not spill (M>=9 measures 1500-4500 us/dis
    // isolated vs 180 at M=8) — measured, isolated bench_verify_qmm_m_sweep.
    if n < 512 || n >= 100_000 {
        return None;
    }
    if k % 64 != 0 || k % 8 != 0 {
        return None;
    }
    if n % 2 != 0 {
        return None;
    }
    let k_parts = if n >= 4096 { 2 } else { 4 };
    if (k / 8) % k_parts != 0 {
        return None;
    }
    // BN = 2 for every M — a MEASURED deviation (the obvious 4 columns through
    // M=6 does not hold). A BN=4 tile stack-spills on our Metal
    // compiler at M=5 (2.85 ms vs 0.65 ms at M=5, isolated single-shot);
    // the SIMD lanes differ — measure, don't inherit.
    // In-situ A/B seam (specs/09c sweep): `LISA_VERIFY_QMM_TILE=bn:kp`
    // overrides the resolved tile for the eligibility-unchanged lane. Host
    // side only; unset means the measured constants above.
    if let Some((bn, kp)) = tile_env_override() {
        if n % bn != 0 || (k / 8) % kp != 0 {
            return None;
        }
        return Some((bn, kp));
    }
    Some((2, k_parts))
}

/// The `LISA_VERIFY_QMM_TILE=bn:kp` A/B override, parsed once per process.
/// Malformed or out-of-range values resolve to None (stock tile); the
/// eligibility gates above still apply to the overridden tile.
fn tile_env_override() -> Option<(usize, usize)> {
    use std::sync::OnceLock;
    static TILE: OnceLock<Option<(usize, usize)>> = OnceLock::new();
    *TILE.get_or_init(|| {
        std::env::var("LISA_VERIFY_QMM_TILE")
            .ok()
            .and_then(|raw| parse_tile_override(&raw))
    })
}

/// Pure parser for the tile override (test seam; `bn:kp`, bn 1..=8, kp 1..=32).
pub fn parse_tile_override(raw: &str) -> Option<(usize, usize)> {
    let mut it = raw.split(':');
    let bn: usize = it.next()?.parse().ok()?;
    let kp: usize = it.next()?.parse().ok()?;
    if !(1..=8).contains(&bn) || !(1..=32).contains(&kp) || it.next().is_some() {
        return None;
    }
    Some((bn, kp))
}

/// Raw split-K verify tile dispatch with EXPLICIT tile parameters (bn,
/// k_parts, ppt) — the sweep harness entry point; the production path is
/// `affine_verify_qmm_splitk`, which resolves the tile via
/// `verify_qmm_splitk_lane`. ppt = packs per thread per K-iteration (our
/// PPT extension; bit-identical output for any ppt, see the .metal note).
pub fn affine_verify_qmm_splitk_tile(
    device: &Device,
    x: &Tensor,
    w: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: i32,
    bits: i32,
    bn: usize,
    k_parts: usize,
    ppt: usize,
) -> Result<Tensor> {
    affine_verify_qmm_splitk_tile_ex(
        device, x, w, scales, biases, group_size, bits, bn, k_parts, ppt, false, false,
    )
}

/// Variant axis of the split-K verify tile under the tail-ULP contract:
/// - `vload`: 4 consecutive packs per thread per iteration loaded with ONE
///   16-byte uint4 per column (VLOAD template) — requires per_part % 4 == 0.
/// - `fast_math`: compile the specialization with `Math::Fast` (Metal fast
///   math) instead of the stock `SafeNoLang`.
/// Both change fp32 codegen ONLY (no numerics pin yet for fast-math — the
/// tail-ULP tests cover each combination they exercise).
pub fn affine_verify_qmm_splitk_tile_ex(
    device: &Device,
    x: &Tensor,
    w: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: i32,
    bits: i32,
    bn: usize,
    k_parts: usize,
    ppt: usize,
    vload: bool,
    fast_math: bool,
) -> Result<Tensor> {
    let (m, k) = x.dims2();
    let (n, _) = w.dims2();
    if bn == 0 || n % bn != 0 || (k / 8) % k_parts != 0 || ppt == 0 || ppt > 4 {
        crate::bail!(
            "affine_verify_qmm_splitk_tile: bad tile (m={m} k={k} n={n} bn={bn} kp={k_parts} ppt={ppt})"
        );
    }
    if vload && ((k / 8 / k_parts) % 4 != 0) {
        crate::bail!(
            "affine_verify_qmm_splitk_tile: vload needs per_part % 4 == 0 (k={k} kp={k_parts})"
        );
    }
    let type_str = type_string(x.dtype())?;
    // One specialization per (M, BN, K_PARTS, PPT, VLOAD, FM) with a distinct
    // host_name (two specializations sharing a name
    // bind the wrong binary). The stock combination keeps the historical
    // name (warm_kernels / disk-cache compatibility).
    let variant = if vload || fast_math {
        format!("_vl{}_fm{}", vload as u8, fast_math as u8)
    } else {
        String::new()
    };
    let kname = format!(
        "affine_verify_qmm_splitk{variant}_{type_str}_gs_{group_size}_b_{bits}_m_{m}_bn_{bn}_kp_{k_parts}_ppt_{ppt}"
    );
    let template_def = builtin_template_def(
        &kname,
        "affine_verify_qmm_splitk",
        &[
            type_str.to_string(),
            group_size.to_string(),
            bits.to_string(),
            m.to_string(),
            bn.to_string(),
            k_parts.to_string(),
            ppt.to_string(),
            (vload as u8).to_string(),
        ],
    );

    let source = format!(
        "{MLX_UTILS_PREAMBLE}{MLX_GEMM_PREAMBLE}{MLX_QUANTIZED_UTILS_PREAMBLE}{MLX_QUANTIZED_PREAMBLE}{VERIFY_QMM_SOURCE}{template_def}"
    );
    let pipeline = if fast_math {
        compile_builtin_math(device, &source, &kname, crate::runtime::Math::Fast)?
    } else {
        compile_builtin(device, &source, &kname)?
    };

    let out_dtype = x.dtype();
    let ycount = m * n;
    let ybuf = device.buffer(ycount as usize * out_dtype.size_of(), "verify_qmm_splitk_out")?;
    let y = Array::from_parts(device, ybuf.clone(), &vec![m, n], out_dtype);

    let guard = device.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    let bind = |index: usize, t: &Tensor| -> Result<()> {
        let (ms, layout) = t.buffer_and_layout();
        if !layout.is_contiguous() {
            crate::bail!("affine_verify_qmm_splitk: non-contiguous input");
        }
        enc.set_input(index, Some(ms), layout.offset * t.dtype().size_of());
        Ok(())
    };
    bind(0, w)?;
    bind(1, scales)?;
    bind(2, biases)?;
    bind(3, x)?;
    enc.set_output(4, Some(&ybuf), 0);
    let kk = k as i32;
    enc.set_bytes(5, &kk);
    let nn = n as i32;
    enc.set_bytes(6, &nn);

    // Their exact geometry: threadgroups of (32*K_PARTS,1,1) threads, one
    // threadgroup per BN-column tile (grid = (1, N/BN)).
    enc.dispatch_groups_size(
        MTLSize {
            width: 1,
            height: n / bn,
            depth: 1,
        },
        MTLSize {
            width: 32 * k_parts,
            height: 1,
            depth: 1,
        },
    );
    Ok(y)
}

/// Dispatch the split-K verify tile (`affine_verify_qmm_splitk`): the
/// splitk lane, tail-ULP class by construction (fp32 sum order differs from
/// qmv_wide). Gated by `verify_qmm_splitk_lane`; the caller scopes engagement
/// to the MTP verify forward only (ops::quantized_matmul's verify_splitk
/// thread-local).
pub fn affine_verify_qmm_splitk(
    device: &Device,
    x: &Tensor,
    w: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: i32,
    bits: i32,
) -> Result<Tensor> {
    let (m, k) = x.dims2();
    let (n, _) = w.dims2();
    let Some((bn, k_parts)) = verify_qmm_splitk_lane(m, k, n, bits) else {
        crate::bail!(
            "affine_verify_qmm_splitk: shape ineligible (m={m} k={k} n={n} bits={bits})"
        );
    };
    affine_verify_qmm_splitk_tile(
        device, x, w, scales, biases, group_size, bits, bn, k_parts, SPLITK_PPT,
    )
}

/// The serial (M=1) split-K qmv lane: the reference's split-K accumulation
/// structure ported to the decode qmv path. The verify tile
/// (`affine_verify_qmm_splitk`) is their comptime emission ported
/// line-for-line; the serial lane takes the SAME kernel body at MROWS = 1 —
/// K partitions of whole quantization groups across K_PARTS simdgroups of
/// one threadgroup, fp32 partition accumulators, part-ordered combine,
/// single bf16 write (their exact in-kernel partial reduction; no separate
/// reduce dispatch needed at M=1 since one threadgroup owns all K parts of
/// its BN columns).
///
/// Partition count mirrors their decode-class lane: K_PARTS = 2 for
/// N >= 4096, else 4; BN = 2 (the measured constant). 4-bit only. Gates:
/// N >= 512, N < 100000 (huge-N stays on qmv_fast), K % 64 == 0,
/// K % 8 == 0 (pack geometry), N % 2 == 0, (K/8) % K_PARTS == 0.
/// Portage-1:1 note: the fp32 sum order differs from `affine_qmv_fast`
/// (per-part chains + part-ordered reduction) — legal reorder, goldens
/// re-captured.
pub fn qmv_splitk_lane(k: usize, n: usize, bits: i32) -> Option<usize> {
    if bits != 4 || n < 512 || n >= 100_000 {
        return None;
    }
    if k % 64 != 0 || k % 8 != 0 || n % 2 != 0 {
        return None;
    }
    let k_parts = if n >= 4096 { 2 } else { 4 };
    if (k / 8) % k_parts != 0 {
        return None;
    }
    Some(k_parts)
}

/// Dispatch the serial split-K qmv tile (`affine_verify_qmm_splitk` at
/// MROWS = 1): the decode GEMV path. Gated by [`qmv_splitk_lane`].
/// Kernel-level opt-in — NOT wired into `quantized_matmul` (FALSIFIED in
/// situ, specs/07 §7): goldens pass 310/310 with the lane wired, but the
/// serial 27B step is a wash (paired interleaved audits, base/new/new/base:
/// 36.66/37.21 vs 36.75/36.20 ms median) and the kp=4 sweep on the wide-N
/// projections regresses hard (85-88 ms vs 36). Kept as opt-in infra for a
/// future per-shape lane or an in-kernel geometry revision.
pub fn affine_qmv_splitk(
    device: &Device,
    x: &Tensor,
    w: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: i32,
    bits: i32,
) -> Result<Tensor> {
    let (m, k) = x.dims2();
    let (n, _) = w.dims2();
    if m != 1 {
        crate::bail!("affine_qmv_splitk: serial decode GEMV only (M=1)");
    }
    let Some(k_parts) = qmv_splitk_lane(k, n, bits) else {
        crate::bail!("affine_qmv_splitk: shape ineligible (k={k} n={n} bits={bits})");
    };
    let bn = 2usize;
    let type_str = type_string(x.dtype())?;
    let kname = format!(
        "affine_qmv_splitk_{type_str}_gs_{group_size}_b_{bits}_bn_{bn}_kp_{k_parts}_ppt_1"
    );
    let template_def = builtin_template_def(
        &kname,
        "affine_verify_qmm_splitk",
        &[
            type_str.to_string(),
            group_size.to_string(),
            bits.to_string(),
            "1".to_string(),
            bn.to_string(),
            k_parts.to_string(),
            "1".to_string(),
            "0".to_string(),
        ],
    );
    let source = format!(
        "{MLX_UTILS_PREAMBLE}{MLX_GEMM_PREAMBLE}{MLX_QUANTIZED_UTILS_PREAMBLE}{MLX_QUANTIZED_PREAMBLE}{VERIFY_QMM_SOURCE}{template_def}"
    );
    let pipeline = compile_builtin(device, &source, &kname)?;

    let out_dtype = x.dtype();
    let ybuf = device.buffer(n * out_dtype.size_of(), "qmv_splitk_out")?;
    let y = Array::from_parts(device, ybuf.clone(), &vec![1, n], out_dtype);

    let guard = device.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    let bind = |index: usize, t: &Tensor| -> Result<()> {
        let (ms, layout) = t.buffer_and_layout();
        if !layout.is_contiguous() {
            crate::bail!("affine_qmv_splitk: non-contiguous input");
        }
        enc.set_input(index, Some(ms), layout.offset * t.dtype().size_of());
        Ok(())
    };
    bind(0, w)?;
    bind(1, scales)?;
    bind(2, biases)?;
    bind(3, x)?;
    enc.set_output(4, Some(&ybuf), 0);
    let kk = k as i32;
    enc.set_bytes(5, &kk);
    let nn = n as i32;
    enc.set_bytes(6, &nn);

    // Their exact geometry: threadgroups of (32*K_PARTS, 1, 1) threads, one
    // threadgroup per BN-column tile (grid = (1, N/BN)).
    enc.dispatch_groups_size(
        MTLSize {
            width: 1,
            height: n / bn,
            depth: 1,
        },
        MTLSize {
            width: 32 * k_parts,
            height: 1,
            depth: 1,
        },
    );
    Ok(y)
}

/// Compile `source` to a content-addressed `.metallib` under the temp dir and
/// return its path. Compiling one instantiation takes ~0.2 s.
/// `mlx_gemm.metal` / `gemv.metal` (Apple MLX steel kernels vendored
/// through the kernels crate; see NOTICE). The general `matmul` path uses
/// these so its accumulation is MLX-bit-exact.
const DENSE_GEMM: &str = include_str!("../shaders/common/matmul/dense_gemm.metal");
const DENSE_GEMV: &str = include_str!("../shaders/common/matmul/dense_gemv.metal");

/// The huge-N (lm_head class) lane gate, the msg arm: M 2..=7, N >= 100000
/// (the msg threshold — our qwen3_5 lm_head N = 151936 qualifies natively),
/// K % 64 == 0, N % BN == 0, K % 8 == 0 (pack geometry). BN: 4 through M=6,
/// 2 at M=7. 4-bit only (the shipping
/// trunk class). Returns the BN tile when the lane applies.
pub fn verify_qmm_msg_lane(m: usize, k: usize, n: usize, bits: i32) -> Option<usize> {
    if bits != 4 || !(2..=7).contains(&m) {
        return None;
    }
    if n < 100_000 || k % 64 != 0 || k % 8 != 0 {
        return None;
    }
    let bn = if m <= 6 { 4 } else { 2 };
    if n % bn != 0 {
        return None;
    }
    Some(bn)
}

/// Dispatch the msg verify tile (`affine_verify_qmm_msg`): the
/// huge-N lane. NOT wired into
/// `quantized_matmul` — FALSIFIED in situ (specs/19 ledger): the tile wins
/// isolated single-shot (1.48 vs 1.95 ms at m=7 k=5120 n=151936) but loses
/// the production e2e decode wall consistently, order-independently
/// (128-tok d6: 2.56 s base vs 3.7-5.4 s msg) — the full-K strided
/// per-thread chain starves the mixed-pipeline drain exactly like the
/// bit-exact full-K tile does (the reason splitk exists). Kept as
/// kernel-level opt-in (warm + tail-ULP pin) for a future
/// 2-packs-per-thread variant or an M4 packing change.
pub fn affine_verify_qmm_msg(
    device: &Device,
    x: &Tensor,
    w: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: i32,
    bits: i32,
) -> Result<Tensor> {
    const NSG: usize = 8; // VQMM_MSG_NSG — the sweep winner
    let (m, k) = x.dims2();
    let (n, _) = w.dims2();
    let Some(bn) = verify_qmm_msg_lane(m, k, n, bits) else {
        crate::bail!("affine_verify_qmm_msg: shape ineligible (m={m} k={k} n={n} bits={bits})");
    };
    let type_str = type_string(x.dtype())?;
    // One specialization per (M, BN) with a distinct host_name (two
    // specializations sharing a name bind the wrong
    // binary). NSG is a fixed design constant.
    let kname =
        format!("affine_verify_qmm_msg_{type_str}_gs_{group_size}_b_{bits}_m_{m}_bn_{bn}");
    let template_def = builtin_template_def(
        &kname,
        "affine_verify_qmm_msg",
        &[
            type_str.to_string(),
            group_size.to_string(),
            bits.to_string(),
            m.to_string(),
            bn.to_string(),
            NSG.to_string(),
        ],
    );
    let source = format!(
        "{MLX_UTILS_PREAMBLE}{MLX_GEMM_PREAMBLE}{MLX_QUANTIZED_UTILS_PREAMBLE}{MLX_QUANTIZED_PREAMBLE}{VERIFY_QMM_SOURCE}{template_def}"
    );
    let pipeline = compile_builtin(device, &source, &kname)?;

    let out_dtype = x.dtype();
    let ycount = m * n;
    let ybuf = device.buffer(ycount as usize * out_dtype.size_of(), "verify_qmm_msg_out")?;
    let y = Array::from_parts(device, ybuf.clone(), &vec![m, n], out_dtype);

    let guard = device.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    let bind = |index: usize, t: &Tensor| -> Result<()> {
        let (ms, layout) = t.buffer_and_layout();
        if !layout.is_contiguous() {
            crate::bail!("affine_verify_qmm_msg: non-contiguous input");
        }
        enc.set_input(index, Some(ms), layout.offset * t.dtype().size_of());
        Ok(())
    };
    bind(0, w)?;
    bind(1, scales)?;
    bind(2, biases)?;
    bind(3, x)?;
    enc.set_output(4, Some(&ybuf), 0);
    let kk = k as i32;
    enc.set_bytes(5, &kk);
    let nn = n as i32;
    enc.set_bytes(6, &nn);

    // Their exact geometry: (32*NSG, 1, 1)-thread threadgroups, one
    // threadgroup column per NSG*BN output columns; the in-kernel n0 guard
    // covers the ceil-div tail.
    let cols = bn * NSG;
    let tg_count = (n + cols - 1) / cols;
    enc.dispatch_groups_size(
        MTLSize {
            width: 1,
            height: tg_count,
            depth: 1,
        },
        MTLSize {
            width: 32 * NSG,
            height: 1,
            depth: 1,
        },
    );
    Ok(y)
}

/// The verify-width S>1 quantized matmul kernel (`affine_verify_qmm`): the
/// qmv_fast numerics with the input-row grid axis collapsed into the
/// threadgroup — one weight stream for the whole verify width. See the
/// kernel header for the split-K provenance.
pub(crate) const VERIFY_QMM_SOURCE: &str = include_str!("../shaders/common/matmul/verify_qmm.metal");

/// The NAX m16 verify tile (`affine_verify_qmm_nax_m16`): the dedicated
/// M 8..=16 lane for M5-class matrix units. See the kernel header for the
/// verbatim-port provenance and the numerics class.
pub(crate) const VERIFY_QMM_NAX_M16_SOURCE: &str =
    include_str!("../shaders/common/matmul/verify_qmm_nax_m16.metal");

/// The m16 lane gate, mirroring the reference's lane table: the NAX m16 tile
/// takes M 8..=16 by default (past the plain-SIMD region, below the tiled
/// 32x32 lane), 4-bit, N % 32 == 0 (one output column per lane per
/// threadgroup), K % 128 == 0 (16-deep BK loop over K/8 per-simdgroup
/// chunks), N < 100000 (huge-N stays off the custom lanes). The M5-class
/// hardware gate is `device.nax()`.
pub fn verify_qmm_nax_m16_lane(m: usize, k: usize, n: usize, bits: i32, group_size: i32) -> bool {
    bits == 4
        && (8..=16).contains(&m)
        && n % 32 == 0
        && k % 128 == 0
        && group_size % 16 == 0
        && k % group_size.max(1) as usize == 0
        && n < 100_000
}

/// Dispatch the NAX m16 verify tile — kernel-level opt-in (the msg tile
/// precedent), NOT wired into `quantized_matmul`. The MPP probe diagnosis
/// (specs/08 item 2): the kernel never needed additive-binary linkage — it
/// needed (a) the NAX compile class (`compile_nax_jit`, Metal 4 language —
/// MSL 3.2 rejects the MPP+bfloat16 source outright) and (b) the template
/// type spelled `bfloat16_t` (the bare `bfloat16` name is ambiguous under
/// Metal 4) and (c) the grid on the X axis (MPP kernels silently skip
/// threadgroups launched with grid.y > 0). With all three the kernel
/// compiles and executes through lisa's ordinary `newLibraryWithSource`
/// path, but it measured SLOWER than stock at every M (specs/08 step 3) —
/// unwired. `x` must be the PADDED [16, K] tile; returns [16, N].
pub fn affine_verify_qmm_nax_m16(
    device: &Device,
    x: &Tensor,
    w: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: i32,
    bits: i32,
) -> Result<Tensor> {
    let (_, k) = x.dims2();
    let (n, _) = w.dims2();
    if !verify_qmm_nax_m16_lane(16, k, n, bits, group_size) {
        crate::bail!(
            "affine_verify_qmm_nax_m16: shape ineligible (k={k} n={n} bits={bits} gs={group_size})"
        );
    }
    if !device.nax() {
        crate::bail!("affine_verify_qmm_nax_m16: requires a NAX (M5-class) device");
    }
    // The template type must be `bfloat16_t` — the MLX preambles' typedef —
    // NOT `bfloat16`: under the NAX language class the Metal stdlib defines
    // `metal::bfloat16` (metal_extended_vector), which makes the bare name
    // ambiguous and the library fails to compile. Same convention the working
    // NAX kernels use (`qmm_nax`, `gather_qmm_rhs_nax`).
    let type_str = match x.dtype() {
        crate::jit::DType::BF16 => "bfloat16_t".to_string(),
        other => type_string(other)?.to_string(),
    };
    let kname = format!(
        "affine_verify_qmm_nax_m16_{type_str}_gs_{group_size}_b_{bits}_k_{k}"
    );
    let template_def = builtin_template_def(
        &kname,
        "affine_verify_qmm_nax_m16",
        &[
            type_str.clone(),
            group_size.to_string(),
            bits.to_string(),
            k.to_string(),
        ],
    );
    // The MLX preambles define `bfloat16_t` (the template type name the
    // specialization instantiates); MPP supplies the tensor ops.
    let source = format!(
        "{MLX_UTILS_PREAMBLE}{MLX_GEMM_PREAMBLE}{MLX_QUANTIZED_UTILS_PREAMBLE}{MLX_QUANTIZED_PREAMBLE}{VERIFY_QMM_NAX_M16_SOURCE}{template_def}"
    );
    // The NAX compile class (Math::Safe -> Metal 4 language version): the MPP
    // tensor ops are Metal-4 language features. The old `compile_builtin`
    // (Math::SafeNoLang -> MSL 3.2) could not compile this source at all.
    let pipeline = super::nax::compile_nax_jit(device, &source, &kname, &[])?;

    let ybuf = device.buffer(16 * n as usize * x.dtype().size_of(), "verify_qmm_nax_m16_out")?;
    let guard = device.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    let bind = |index: usize, t: &Tensor| -> Result<()> {
        let (ms, layout) = t.buffer_and_layout();
        if !layout.is_contiguous() {
            crate::bail!("affine_verify_qmm_nax_m16: non-contiguous input");
        }
        enc.set_input(index, Some(ms), layout.offset * t.dtype().size_of());
        Ok(())
    };
    bind(0, x)?;
    // w is the packed u32 word tensor; the kernel reads it as bytes.
    bind(1, w)?;
    bind(2, scales)?;
    bind(3, biases)?;
    enc.set_output(4, Some(&ybuf), 0);
    let nn = n as i32;
    enc.set_bytes(5, &nn);
    // threadgroup = 256 threads (8 simdgroups); one threadgroup per 32
    // output columns. The grid rides the X axis: the MPP probe (specs/08
    // item 2) showed MPP tensor-op kernels silently SKIP every threadgroup
    // launched with grid.y > 0 (no GPU error, no output) — the reference
    // dispatches (256, N/32, 1), keeping the varying axis on X.
    enc.dispatch_groups_size(
        MTLSize { width: n / 32, height: 1, depth: 1 },
        MTLSize { width: 256, height: 1, depth: 1 },
    );
    Ok(Array::from_parts(device, ybuf, &vec![16, n as usize], x.dtype()))
}

/// `GemmParams` (`kernels/mlx_gemm.rs`).
#[repr(C)]
struct GemmParams {
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldd: i32,
    tiles_n: i32,
    tiles_m: i32,
    batch_stride_a: isize,
    batch_stride_b: isize,
    batch_stride_d: isize,
    swizzle_log: i32,
    gemm_k_iterations_aligned: i32,
    batch_ndim: i32,
}

/// The Metal device class tile selection keys on: the last character
/// of the architecture name (`applegpu_*`).
fn device_type(rt: &Arc<crate::runtime::MetalRuntime>) -> char {
    rt.architecture_name().chars().last().unwrap_or('m')
}

/// MLX `affine_dequantize` (quantized.cpp): JIT-compile the dequantize kernel
/// for these template args and dispatch it. Replaces the host dequantize in
/// `ops::dequantize` (a full GPU->CPU->GPU round trip per call; the embedding
/// forward runs it per token, which showed up as ~130 `cast_host` allocs/token).
pub fn affine_dequantize(
    device: &Device,
    w: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: i32,
    bits: i32,
) -> Result<Tensor> {
    let mdev = device;
    // w is [.., kq] u32 (packed); scales/biases [.., groups]; out [.., k].
    let kq = *w.dims().last().unwrap();
    let per_word = 32 / bits as usize;
    let k = kq * per_word;
    let count: usize = w.size() / kq * k;
    let out_dims: Vec<usize> = {
        let mut d2 = w.dims().to_vec();
        *d2.last_mut().unwrap() = k;
        d2
    };
    let obuf = mdev.buffer(count * scales.dtype().size_of(), "deq_out")?;
    let out = Array::from_parts(mdev, obuf.clone(), &out_dims, scales.dtype());

    let ty = type_string(scales.dtype())?;
    // MLX instantiates the *out* dtype as T; w is read as bytes.
    let kname = format!("affine_dequantize_{ty}_gs_{group_size}_b_{bits}");
    let template_def = builtin_template_def(
        &kname,
        "affine_dequantize",
        &[
            ty.to_string(),
            group_size.to_string(),
            bits.to_string(),
            "0".to_string(),
        ],
    );
    let source = format!(
        "{MLX_UTILS_PREAMBLE}{MLX_GEMM_PREAMBLE}{MLX_QUANTIZED_UTILS_PREAMBLE}{MLX_QUANTIZED_PREAMBLE}{template_def}"
    );
    let pipeline = compile_builtin(device, &source, &kname)?;

    // MLX dims: one thread per pack (8 values for 4-bit); 2-D grid.
    let packs_per_int: usize = if bits == 3 || bits == 5 {
        8
    } else {
        8 / bits as usize
    };
    let nthreads = count / packs_per_int.max(1);
    let mut grid_shape = w.dims().to_vec();
    *grid_shape.last_mut().unwrap() *= 4; // uint8 per uint32
    let (gw, gh) = {
        // get_2d_grid_dims over the byte-expanded shape
        let strides: Vec<i64> = {
            let mut sv = vec![1i64; grid_shape.len()];
            for i in (0..grid_shape.len().saturating_sub(1)).rev() {
                sv[i] = sv[i + 1] * grid_shape[i + 1] as i64;
            }
            sv
        };
        get_2d_grid_dims(&grid_shape, &strides)
    };
    let tg = pipeline
        .max_total_threads_per_threadgroup()
        .min(nthreads)
        .max(1);

    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    enc.set_input(
        0,
        Some(w.buffer_and_layout().0),
        w.buffer_and_layout().1.offset * 4,
    );
    enc.set_input(
        1,
        Some(scales.buffer_and_layout().0),
        scales.buffer_and_layout().1.offset * scales.dtype().size_of(),
    );
    enc.set_input(
        2,
        Some(biases.buffer_and_layout().0),
        biases.buffer_and_layout().1.offset * biases.dtype().size_of(),
    );
    enc.set_output(3, Some(out.buffer_and_layout().0), 0);
    enc.dispatch_threads_3d((gw.max(1), gh.max(1), 1), (tg, 1, 1));
    Ok(out)
}

/// `call_mlx_gemv` (kernels/mlx_gemm.rs:270): the M=1 / N=1 path of the
/// reference `matmul`, so its numerics match bit-for-bit.
#[allow(clippy::too_many_arguments)]
fn dense_gemv(
    rt: &Arc<MetalRuntime>,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_stride: &[usize],
    lhs_offset: usize,
    lhs_buffer: &Arc<Buffer>,
    rhs_stride: &[usize],
    rhs_offset: usize,
    rhs_buffer: &Arc<Buffer>,
    out: &crate::array::Array,
    dt: Dtype,
) -> Result<()> {
    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];

    let (lda, a_trans) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k as i32, false)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m as i32, true)
    } else {
        return Err(Error::Msg(format!(
            "mlx gemv: non-contiguous lhs {lhs_stride:?} mnk {m},{n},{k}"
        )));
    };
    let (ldb, b_trans) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        (n as i32, false)
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        (k as i32, true)
    } else {
        return Err(Error::Msg(format!(
            "mlx gemv: non-contiguous rhs {rhs_stride:?} mnk {m},{n},{k}"
        )));
    };

    let is_b_matrix = n != 1;
    let transpose_mat = if is_b_matrix { !b_trans } else { a_trans };
    let mat_ld = if is_b_matrix {
        ldb as usize
    } else {
        lda as usize
    };
    let in_vec_size = k;
    let out_vec_size = if is_b_matrix { n } else { m };

    let (mat_buffer, mat_offset, vec_buffer, vec_offset) = if is_b_matrix {
        (rhs_buffer, rhs_offset, lhs_buffer, lhs_offset)
    } else {
        (lhs_buffer, lhs_offset, rhs_buffer, rhs_offset)
    };

    let vec_batch_stride: i64 = if is_b_matrix {
        if lhs_stride.len() > 2 {
            lhs_stride[lhs_stride.len() - 3] as i64
        } else {
            k as i64
        }
    } else {
        if rhs_stride.len() > 2 {
            rhs_stride[rhs_stride.len() - 3] as i64
        } else {
            k as i64
        }
    };
    let mat_batch_stride: i64 = if is_b_matrix {
        if rhs_stride.len() > 2 {
            rhs_stride[rhs_stride.len() - 3] as i64
        } else {
            0
        }
    } else {
        if lhs_stride.len() > 2 {
            lhs_stride[lhs_stride.len() - 3] as i64
        } else {
            0
        }
    };

    let (bm, bn, sm, sn, tm, tn) = if transpose_mat {
        let (sm, sn) = if in_vec_size >= 8192 && out_vec_size >= 2048 {
            (4usize, 8usize)
        } else {
            (8, 4)
        };
        let bn = if out_vec_size >= 2048 {
            16usize
        } else if out_vec_size >= 512 {
            4
        } else {
            2
        };
        let tn: usize = if out_vec_size < 4 { 1 } else { 4 };
        (1usize, bn, sm, sn, 4usize, tn)
    } else {
        let (bm, bn, sm, sn): (usize, usize, usize, usize) = if in_vec_size <= 64 {
            (1, 1, 8, 4)
        } else if in_vec_size >= 16 * out_vec_size {
            (1, 8, 1, 32)
        } else if out_vec_size >= 4096 {
            (8, 1, 1, 32)
        } else {
            (4, 1, 1, 32)
        };
        let tm: usize = if out_vec_size < 4 { 1 } else { 4 };
        (bm, bn, sm, sn, tm, 4usize)
    };

    let dtype_str = match dt {
        Dtype::Float32 => "float32",
        Dtype::Float16 => "float16",
        Dtype::Bfloat16 => "bfloat16",
        other => return Err(Error::Msg(format!("mlx gemv dtype {other:?}"))),
    };
    let kernel_prefix = if transpose_mat { "gemv_t" } else { "gemv" };
    let name = format!(
        "{}_{}_bm{}_bn{}_sm{}_sn{}_tm{}_tn{}_nc0_axpby0",
        kernel_prefix, dtype_str, bm, bn, sm, sn, tm, tn
    );
    let source = if transpose_mat {
        DENSE_GEMV
    } else {
        DENSE_GEMV
    };
    let pipeline =
        rt.compile_with_constants(source, &name, crate::runtime::Math::SafeNoLang, &[])?;

    let _esz = dt.size_of();
    let guard = rt.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    enc.set_input(0, Some(mat_buffer), mat_offset);
    enc.set_input(1, Some(vec_buffer), vec_offset);
    enc.set_output(3, Some(out.buffer_and_layout().0), 0);
    enc.set_bytes(4, &(in_vec_size as i32));
    enc.set_bytes(5, &(out_vec_size as i32));
    enc.set_bytes(6, &(mat_ld as i32));
    enc.set_bytes(7, &1.0f32); // alpha
    enc.set_bytes(8, &0.0f32); // beta
    enc.set_bytes(9, &1i32); // batch_ndim
    let batch_shape = [b as i32];
    let vec_strides = [vec_batch_stride];
    let mat_strides = [mat_batch_stride];
    let bias_strides = [0i64];
    enc.set_bytes_directly(10, 4, batch_shape.as_ptr().cast());
    enc.set_bytes_directly(11, 8, vec_strides.as_ptr().cast());
    enc.set_bytes_directly(12, 8, mat_strides.as_ptr().cast());
    enc.set_bytes_directly(13, 8, bias_strides.as_ptr().cast());
    enc.set_bytes(14, &1i32); // bias_stride

    let n_out_per_tgp = if transpose_mat {
        bn * sn * tn
    } else {
        bm * sm * tm
    };
    let n_tgp = out_vec_size.div_ceil(n_out_per_tgp);
    enc.dispatch_groups_3d((n_tgp, 1, b), (32, bn, bm));
    Ok(())
}

/// `call_mlx_gemm` (kernels/mlx_gemm.rs:453): the MLX `matmul`
/// kernel, tile selection, batch collapse and parameters, so the general
/// matmul path is bit-identical to MLX's.
#[allow(clippy::too_many_arguments)]
pub fn dense_gemm(
    rt: &Arc<MetalRuntime>,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_stride: &[usize],
    lhs_offset: usize,
    lhs_buffer: &Arc<Buffer>,
    rhs_stride: &[usize],
    rhs_offset: usize,
    rhs_buffer: &Arc<Buffer>,
    out: &crate::array::Array,
    dt: Dtype,
) -> Result<()> {
    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];

    let (lda, a_trans) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k as i32, false)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m as i32, true)
    } else {
        return Err(Error::Msg(format!(
            "mlx gemm: non-contiguous lhs {lhs_stride:?} mnk {m},{n},{k}"
        )));
    };
    let (ldb, b_trans) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        (n as i32, false)
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        (k as i32, true)
    } else {
        return Err(Error::Msg(format!(
            "mlx gemm: non-contiguous rhs {rhs_stride:?} mnk {m},{n},{k}"
        )));
    };

    if m == 1 || n == 1 {
        return dense_gemv(
            rt,
            (b, m, n, k),
            lhs_stride,
            lhs_offset,
            lhs_buffer,
            rhs_stride,
            rhs_offset,
            rhs_buffer,
            out,
            dt,
        );
    }

    // Batch collapse: A contiguous in batch + B broadcast (2-D rhs) -> [b*m, k].
    let (effective_batch, effective_m, batch_collapsed) = {
        let mut eb = b;
        let mut em = m;
        let mut collapsed = false;
        if b > 1 && !a_trans {
            let a_batch = if lhs_stride.len() > 2 {
                lhs_stride[lhs_stride.len() - 3]
            } else {
                m * k
            };
            let b_batch = if rhs_stride.len() > 2 {
                rhs_stride[rhs_stride.len() - 3]
            } else {
                0
            };
            if a_batch == m * k && b_batch == 0 {
                eb = 1;
                em = b * m;
                collapsed = true;
            }
        }
        (eb, em, collapsed)
    };
    let m = effective_m;
    let b = effective_batch;

    // Tile selection (MLX GEMM_TPARAM_MACRO; medium device = arch ending 's').
    let total_output = b * m * n;
    let is_large = total_output >= (1 << 20);
    let devc = device_type(rt);
    let (bm, bn, bk, wm, wn) = if m < 16 {
        (32, 32, 16, 2, 2)
    } else if devc == 's' || devc == 'm' {
        if dt == Dtype::Float32 {
            if !is_large {
                if !a_trans && b_trans {
                    (32, 64, 16, 1, 2)
                } else {
                    (64, 32, 32, 2, 2)
                }
            } else {
                (64, 64, 16, 2, 2)
            }
        } else if is_large {
            if 2 * m.max(n) > k {
                (64, 64, 16, 1, 2)
            } else if !a_trans && b_trans {
                (64, 32, 32, 2, 2)
            } else {
                (32, 64, 16, 1, 2)
            }
        } else if !a_trans && b_trans {
            (64, 32, 32, 2, 2)
        } else {
            (64, 64, 16, 1, 2)
        }
    } else if devc == 'd' {
        if is_large {
            if dt != Dtype::Float32 {
                if 2 * m.max(n) > k {
                    (64, 64, 16, 1, 2)
                } else if !a_trans && b_trans {
                    (64, 32, 32, 2, 2)
                } else {
                    (32, 64, 16, 1, 2)
                }
            } else {
                (64, 64, 16, 2, 2)
            }
        } else if dt != Dtype::Float32 {
            if !a_trans && b_trans {
                (64, 32, 32, 2, 2)
            } else {
                (64, 64, 16, 1, 2)
            }
        } else if !a_trans && b_trans {
            (32, 64, 16, 1, 2)
        } else {
            (64, 32, 32, 2, 2)
        }
    } else {
        if !a_trans && b_trans {
            (64, 32, 32, 2, 2)
        } else if dt != Dtype::Float32 {
            (64, 64, 16, 1, 2)
        } else {
            (64, 64, 16, 2, 2)
        }
    };

    let has_batch = b > 1;
    let swizzle_log = 0;
    let tile_swizzle = 1usize << swizzle_log;
    let tn = n.div_ceil(bn);
    let tm = m.div_ceil(bm);
    let tn = tn * tile_swizzle;
    let tm = tm.div_ceil(tile_swizzle);

    let (batch_stride_a, batch_stride_b) = if batch_collapsed {
        (0isize, 0isize)
    } else {
        let a_stride = if lhs_stride.len() > 2 {
            lhs_stride[lhs_stride.len() - 3] as isize
        } else {
            (m * k) as isize
        };
        let b_stride = if rhs_stride.len() > 2 {
            rhs_stride[rhs_stride.len() - 3] as isize
        } else {
            (n * k) as isize
        };
        (a_stride, b_stride)
    };

    let params = GemmParams {
        m: m as i32,
        n: n as i32,
        k: k as i32,
        lda: if batch_collapsed { k as i32 } else { lda },
        ldb,
        ldd: n as i32,
        tiles_n: tn as i32,
        tiles_m: tm as i32,
        swizzle_log,
        batch_stride_a,
        batch_stride_b,
        batch_stride_d: (m * n) as isize,
        batch_ndim: 1,
        gemm_k_iterations_aligned: (k / bk) as i32,
    };

    let dtype_str = match dt {
        Dtype::Float32 => "f32",
        Dtype::Float16 => "f16",
        Dtype::Bfloat16 => "bf16",
        other => return Err(Error::Msg(format!("mlx gemm dtype {other:?}"))),
    };
    let trans_str = match (a_trans, b_trans) {
        (false, false) => "nn",
        (true, false) => "tn",
        (false, true) => "nt",
        (true, true) => "tt",
    };
    let name = format!(
        "gemm_{}_{}_{}_{}_{}_{}_{}_{}",
        trans_str, dtype_str, dtype_str, bm, bn, bk, wm, wn
    );
    let consts: Vec<(usize, ConstVal)> = vec![
        (10, ConstVal::Bool(has_batch)),
        (100, ConstVal::Bool(false)),
        (110, ConstVal::Bool(false)),
        (200, ConstVal::Bool(m % bm == 0)),
        (201, ConstVal::Bool(n % bn == 0)),
        (202, ConstVal::Bool(k % bk == 0)),
        (300, ConstVal::Bool(false)),
    ];
    let pipeline =
        rt.compile_with_constants(DENSE_GEMM, &name, crate::runtime::Math::SafeNoLang, &consts)?;

    let guard = rt.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    enc.set_input(0, Some(lhs_buffer), lhs_offset);
    enc.set_input(1, Some(rhs_buffer), rhs_offset);
    enc.set_output(3, Some(out.buffer_and_layout().0), 0);
    enc.set_bytes(4, &params);
    enc.set_bytes(6, &(b as i32));
    let batch_strides = [batch_stride_a, batch_stride_b];
    enc.set_bytes_directly(7, 16, batch_strides.as_ptr().cast());
    enc.dispatch_groups_3d((tn, tm, b), (32, wn, wm));
    Ok(())
}

/// MLX `qmv_wide` (`quantized.cpp:544`): the affine qmv that reuses each weight
/// group across up to 5 input vectors. MLX takes this for `2 <= M <
/// vector_limit` (`use_qmv_wide` is true for affine on gen>=15).
pub fn qmv_wide(
    device: &Device,
    x: &Tensor,
    w: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: i32,
    bits: i32,
) -> Result<Tensor> {
    let mdev = device;
    let (m, k) = x.dims2();
    let (n, _) = w.dims2();
    let n_tiles = m.div_ceil(5);
    let vecs_per_tg = m.div_ceil(n_tiles);
    let k_lanes = 8usize;
    let rows_per_tg = (32 / k_lanes) * 2;
    let type_str = type_string(x.dtype())?;
    let kname = format!(
        "affine_qmv_wide_{type_str}_gs_{group_size}_b_{bits}_nv_{vecs_per_tg}_kl_{k_lanes}_batch_0"
    );
    let template_def = builtin_template_def(
        &kname,
        "affine_qmv_wide",
        &[
            type_str.to_string(),
            group_size.to_string(),
            bits.to_string(),
            vecs_per_tg.to_string(),
            k_lanes.to_string(),
            "0".to_string(),
        ],
    );
    let source = format!(
        "{MLX_UTILS_PREAMBLE}{MLX_GEMM_PREAMBLE}{MLX_QUANTIZED_UTILS_PREAMBLE}{MLX_QUANTIZED_PREAMBLE}{template_def}"
    );
    let pipeline = compile_builtin(device, &source, &kname)?;

    let count = m * n;
    let obuf = mdev.buffer((count) as usize * (x.dtype()).size_of(), "qmv_wide_out")?;
    let out = Array::from_parts(mdev, obuf.clone(), &vec![m, n], x.dtype());
    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    let bind = |index: usize, t: &Tensor| -> Result<()> {
        let (ms, layout) = t.buffer_and_layout();
        enc.set_input(index, Some(ms), layout.offset * t.dtype().size_of());
        Ok(())
    };
    bind(0, w)?;
    bind(1, scales)?;
    bind(2, biases)?;
    bind(3, x)?;
    enc.set_output(4, Some(&obuf), 0);
    for (i, v) in [k as i32, n as i32, m as i32].into_iter().enumerate() {
        enc.set_bytes(5 + i, &v);
    }
    enc.dispatch_groups_size(
        MTLSize {
            width: m.div_ceil(vecs_per_tg),
            height: n.div_ceil(rows_per_tg),
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 2,
            depth: 1,
        },
    );
    Ok(out)
}

/// MLX `get_qmv_batch_limit` (`quantized.cpp:85`): the M threshold below which
/// quantized matmul takes the qmv path instead of a GEMM.
pub fn qmv_batch_limit(k: usize, n: usize) -> usize {
    if k <= 2048 && n <= 2048 {
        33
    } else if k <= 4096 && n <= 4096 {
        25
    } else {
        13
    }
}

/// MLX affine `qmm_splitk` (`quantized.cpp:1120`) — the prefill dense path
/// (transpose=true, B=1, M >= vector_limit). Split-K over K with a sum reduce.
pub fn affine_qmm_splitk(
    device: &Device,
    x: &Tensor,
    w: &Tensor,
    scales: &Tensor,
    biases: &Tensor,
    group_size: i32,
    bits: i32,
) -> Result<Tensor> {
    let mdev = device;
    let (m, k) = x.dims2();
    let (n, _) = w.dims2();
    let (bm, bn) = (32usize, 32usize);
    let n_tiles = n.div_ceil(bn);
    let m_tiles = m.div_ceil(bm);
    let current_tgs = n_tiles * m_tiles;
    let k_align = (group_size.max(32)) as usize;
    let mut split_k = (512 / current_tgs).max(1);
    split_k = split_k.min(k / k_align);
    while split_k > 1 && k % (split_k * k_align) != 0 {
        split_k -= 1;
    }
    if split_k <= 1 {
        crate::bail!("affine_qmm_splitk: split_k <= 1 (use qmm path)");
    }
    let k_partition_size = (k / split_k) as i32;

    let type_str = type_string(x.dtype())?;
    let aligned = n % 32 == 0;
    let kname = format!(
        "affine_qmm_t_splitk_{type_str}_gs_{group_size}_b_{bits}_{}",
        if aligned { "_alN_true" } else { "_alN_false" }
    );
    let def = builtin_template_def(
        &kname,
        "affine_qmm_t_splitk",
        &[
            type_str.to_string(),
            group_size.to_string(),
            bits.to_string(),
            (if aligned { "true" } else { "false" }).to_string(),
        ],
    );
    let source = format!(
        "{MLX_UTILS_PREAMBLE}{MLX_GEMM_PREAMBLE}{MLX_QUANTIZED_UTILS_PREAMBLE}{MLX_QUANTIZED_PREAMBLE}{def}"
    );
    let pipeline = compile_builtin(device, &source, &kname)?;

    // intermediate [split_k, m, n]
    let icount = split_k * m * n;
    let ibuf = mdev.buffer(
        (icount) as usize * (x.dtype()).size_of(),
        "qmm_splitk_inter",
    )?;
    let inter = Array::from_parts(mdev, ibuf.clone(), &vec![split_k, m, n], x.dtype());

    let guard = mdev.commands.encoder()?;
    let enc = guard.encoder();
    enc.set_pipeline(&pipeline);
    let bind = |index: usize, t: &Tensor| -> Result<()> {
        let (ms, layout) = t.buffer_and_layout();
        enc.set_input(index, Some(ms), layout.offset * t.dtype().size_of());
        Ok(())
    };
    bind(0, w)?;
    bind(1, scales)?;
    bind(2, biases)?;
    bind(3, x)?;
    enc.set_output(4, Some(&ibuf), 0);
    let kk = k as i32;
    enc.set_bytes(5, &kk);
    let nn = n as i32;
    enc.set_bytes(6, &nn);
    let mm = m as i32;
    enc.set_bytes(7, &mm);
    enc.set_bytes(8, &k_partition_size);
    let skps = (m * n) as i32;
    enc.set_bytes(9, &skps);
    enc.dispatch_groups_size(
        MTLSize {
            width: n_tiles,
            height: m_tiles,
            depth: split_k,
        },
        MTLSize {
            width: 32,
            height: 2,
            depth: 2,
        },
    );
    drop(guard);

    // Sum the split-K partials (axis 0) with MLX's strided reduce.
    reduce_axis0_sum(device, &inter)
}
