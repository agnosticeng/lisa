// The MPP probe (specs/08 item 2 diagnosis): does an MPP `matmul2d`
// (simdgroup execution) kernel compile AND execute through lisa's
// `newLibraryWithSource` path? Every stage prints a timestamp and the tests
// are `#[ignore]` — run them explicitly:
//   cargo test -p lisa-mlx --lib mpp_probe -- --ignored --nocapture
// A watchdog exits the process after 180 s stuck on one stage, so a hang
// reports WHICH stage instead of blocking the session.
#![cfg(test)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use crate::jit::Device;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLDevice, MTLResourceOptions, MTLSize,
};

static STAGE: AtomicUsize = AtomicUsize::new(0);
static T0: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

fn mark(stage: &str) {
    let t0 = *T0.get_or_init(Instant::now);
    STAGE.fetch_add(1, Ordering::SeqCst);
    eprintln!("[mpp-probe +{:>8.2?}] {stage}", t0.elapsed());
}

fn arm_watchdog() {
    T0.get_or_init(Instant::now);
    std::thread::spawn(|| {
        for _ in 0..180 {
            std::thread::sleep(Duration::from_secs(1));
        }
        let t0 = *T0.get_or_init(Instant::now);
        eprintln!(
            "[mpp-probe +{:>8.2?}] WATCHDOG: stuck >180 s at stage {} — exiting 99",
            t0.elapsed(),
            STAGE.load(Ordering::SeqCst)
        );
        std::process::exit(99);
    });
}

// A minimal MPP matmul2d kernel: A [16,16] device float, B [32,16] staged in
// threadgroup memory (one row per lane), C = A x B^T staged threadgroup, lane
// writes its column of y [16,32]. The descriptor and cooperative-tensor
// pattern mirror the m16 tile exactly (float instead of bf16 to keep the
// probe free of the type preamble).
const MPP_PROBE_SOURCE: &str = r#"
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

[[kernel]] void mpp_probe(
    const device float* a [[buffer(0)]],
    const device float* b [[buffer(1)]],
    device float* y [[buffer(2)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_position_in_threadgroup]],
    uint tgp [[threadgroup_position_in_grid]]) {

    threadgroup float b_tile[32 * 16];
    threadgroup float partial[32 * 16];

    constexpr auto desc = matmul2d_descriptor(
        16, 32, 16, false, false, false,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, metal::execution_simdgroup> op;

    tensor<device float, dextents<int, 2>, tensor_inline> A(
        (device float*)a, dextents<int, 2>{16, 16}, array<int, 2>{1, 16});
    tensor<threadgroup float, dextents<int, 2>, tensor_inline> B(
        b_tile, dextents<int, 2>{32, 16}, array<int, 2>{1, 32});
    tensor<threadgroup float, dextents<int, 2>, tensor_inline> C(
        partial, dextents<int, 2>{32, 16}, array<int, 2>{1, 32});

    auto ct_c = op.template get_destination_cooperative_tensor<
        tensor<device float, extents<int, 16, 16>, tensor_inline>,
        tensor<threadgroup float, extents<int, 32, 16>, tensor_inline>,
        float>();
    _Pragma("unroll")
    for (uint16_t i = 0; i < ct_c.get_capacity(); ++i) {
        ct_c[i] = 0.0f;
    }
    for (int k = 0; k < 16; ++k) {
        b_tile[int(lane) + 32 * k] = b[int(lane) * 16 + k];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    auto tA = A.template slice<16, 16>(0, 0);
    auto tB = B.template slice<32, 16>(0, 0);
    op.run(tA, tB, ct_c);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    auto tC = C.template slice<32, 16>(0, 0);
    ct_c.store(tC);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32) {
        for (int r = 0; r < 16; ++r) {
            y[int(tgp) * 512 + r * 32 + int(tid)] = partial[int(tid) + 32 * r];
        }
    }
}
"#;

fn rt() -> Device {
    std::sync::Arc::new(crate::runtime::MetalRuntime::new(4).expect("metal runtime"))
}

// Compile the probe under the NAX language class (Math::Safe) — the same
// options the working `qmm_nax`/`gather_qmm_rhs_nax` kernels use.
#[test]
#[ignore]
fn mpp_probe_nax_lang_compile_and_run() {
    arm_watchdog();
    let rt = rt();
    mark("runtime up");
    eprintln!("[mpp-probe] device nax = {}", rt.nax());
    mark("compile start");
    let pipeline = rt
        .compile_with(MPP_PROBE_SOURCE, "mpp_probe", crate::runtime::Math::Safe)
        .expect("compile");
    mark("compile done");
    assert!(rt.nax(), "probe requires a NAX device");

    // Raw command buffer — bypasses lisa's Commands batching/fence machinery
    // entirely, so the probe isolates Metal itself.
    let dev = rt.metal_device();
    let queue = dev.newCommandQueue().unwrap();
    let len = 16 * 16 * 4;
    let mk = |l: usize| {
        dev.newBufferWithLength_options(l, MTLResourceOptions::StorageModeShared)
            .unwrap_or_else(|| panic!("buffer"))
    };
    let abuf = mk(len);
    let bbuf = mk(32 * 16 * 4);
    let ybuf = mk(4 * 16 * 32 * 4);
    // A: a[i][k] = i + k. B: b[j][k] = j * 16 + k (identity-ish values).
    let a: Vec<f32> = (0..16)
        .flat_map(|i| (0..16).map(move |k| (i + k) as f32))
        .collect();
    let b: Vec<f32> = (0..32)
        .flat_map(|j| (0..16).map(move |k| (j * 16 + k) as f32))
        .collect();
    unsafe {
        std::ptr::copy_nonoverlapping(a.as_ptr(), abuf.contents().as_ptr().cast(), len / 4);
        std::ptr::copy_nonoverlapping(b.as_ptr(), bbuf.contents().as_ptr().cast(), b.len());
        std::ptr::write_bytes(ybuf.contents().as_ptr().cast::<u8>(), 0xEE, 4 * 16 * 32 * 4);
    }
    let cb = queue.commandBuffer().unwrap();
    let enc = cb
        .computeCommandEncoderWithDispatchType(objc2_metal::MTLDispatchType::Serial)
        .unwrap();
    enc.setComputePipelineState(pipeline.raw());
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&abuf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&bbuf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&ybuf), 0, 2);
    }
    mark("dispatch");
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: 4,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    mark("commit");
    cb.commit();
    mark("wait");
    cb.waitUntilCompleted();
    mark("wait done");
    assert_eq!(
        cb.status(),
        objc2_metal::MTLCommandBufferStatus::Completed,
        "command buffer status"
    );
    let y =
        unsafe { std::slice::from_raw_parts(ybuf.contents().as_ptr().cast::<f32>(), 4 * 16 * 32) };
    // Every threadgroup computes the same tile into its own slot
    // (y[tgp*512 ..]); a dead threadgroup leaves the 0xEE sentinel untouched.
    let mut bad = 0;
    for g in 0..4 {
        for r in 0..16 {
            for c in 0..32 {
                let want: f32 = (0..16).map(|k| (r + k) as f32 * (c * 16 + k) as f32).sum();
                let got = y[g * 512 + r * 32 + c];
                if got.to_bits() == 0xEEEE_EEEE {
                    eprintln!("[mpp-probe] tg{g} y[{r}][{c}] NEVER WRITTEN");
                    bad += 1;
                } else if (got - want).abs() > 1e-3 * want.abs().max(1.0) {
                    bad += 1;
                    if bad < 4 {
                        eprintln!("[mpp-probe] tg{g} MISMATCH y[{r}][{c}] = {got} want {want}");
                    }
                }
            }
        }
    }
    eprintln!("[mpp-probe] mismatches: {bad}/2048 (4 threadgroups x 512)");
    assert_eq!(bad, 0, "MPP matmul2d probe produced wrong values");
}

// The m16 kernel: compile under the NAX language class (MSL 3.2 cannot
// compile the MPP+bfloat16 source at all — see the ledger), then a full
// dispatch on synthetic 4-bit weights against a CPU reference.
#[test]
#[ignore]
fn mpp_probe_m16_compile_and_run() {
    arm_watchdog();
    let rt = rt();
    eprintln!("[mpp-probe] device nax = {}", rt.nax());
    assert!(rt.nax());
    mark(
        "m16 compile (Safe/NAX class — MSL 3.2 cannot compile MPP+bfloat16, see the 3.2 error note in the ledger)",
    );
    let source = format!(
        "{}{}{}{}{}{}",
        crate::jit::MLX_UTILS_PREAMBLE,
        crate::jit::MLX_GEMM_PREAMBLE,
        crate::jit::MLX_QUANTIZED_UTILS_PREAMBLE,
        crate::jit::MLX_QUANTIZED_PREAMBLE,
        crate::jit::VERIFY_QMM_NAX_M16_SOURCE,
        crate::jit::builtin_template_def(
            "affine_verify_qmm_nax_m16_bfloat16_t_gs_64_b_4_k_128",
            "affine_verify_qmm_nax_m16",
            &[
                "bfloat16_t".to_string(),
                "64".to_string(),
                "4".to_string(),
                "128".to_string()
            ],
        ),
    );
    let name = "affine_verify_qmm_nax_m16_bfloat16_t_gs_64_b_4_k_128";
    let t = Instant::now();
    let pipeline = rt
        .compile_with(&source, name, crate::runtime::Math::Safe)
        .expect("m16 compile (Safe)");
    mark("m16 compile done (Safe/NAX)");
    eprintln!(
        "[mpp-probe] m16 compile took {:.2?}, max threads/tg = {}",
        t.elapsed(),
        pipeline.max_total_threads_per_threadgroup()
    );

    // Functional check: x [16,128] bf16 (rows 8..15 = garbage-padded), 4-bit
    // weights N=64 (two threadgroups), K=128, gs=64 -> two groups. Compare
    // rows 0..7 against a CPU reference; rows 8..15 are padding (not checked).
    use half::bf16;
    let (k, n) = (128usize, 128usize);
    let mut xh: Vec<bf16> = Vec::with_capacity(16 * k);
    for r in 0..16 {
        for c in 0..k {
            xh.push(bf16::from_f32(
                ((r as f32) * 0.25 + (c as f32) * 0.03125) - 1.0,
            ));
        }
    }
    // 4-bit values 0..15 per byte-pair; N rows of K bytes packed 2/byte.
    let mut wq = vec![0u8; n * k / 2];
    for j in 0..n {
        for c in 0..k / 2 {
            wq[j * k / 2 + c] = (((j + c) % 16) | (((j * 3 + c * 7) % 16) << 4)) as u8;
        }
    }
    let ngroups = k / 64;
    let mut scales = Vec::with_capacity(n * ngroups);
    let mut biases = Vec::with_capacity(n * ngroups);
    for j in 0..n {
        for g in 0..ngroups {
            scales.push(bf16::from_f32(0.0078125 + 0.001 * g as f32));
            biases.push(bf16::from_f32(-0.5 + 0.01 * j as f32));
        }
    }
    let deq = |q: u32, j: usize, g: usize| -> f32 {
        q as f32 * scales[j * ngroups + g].to_f32() + biases[j * ngroups + g].to_f32()
    };
    // CPU reference y[r][j] = sum_c x[r][c] * deq(q(w, j, c)) with deq value
    // ROUNDED TO bf16 first (the kernel stores T into B_tile).
    let mut want = vec![0f32; 16 * n];
    for r in 0..8 {
        for j in 0..n {
            let mut acc = 0f32;
            for c in 0..k {
                let byte = wq[j * k / 2 + c / 2];
                let q = if c % 2 == 0 {
                    (byte & 0xF) as u32
                } else {
                    (byte >> 4) as u32
                };
                let g = c / 64;
                let bv = bf16::from_f32(deq(q, j, g)).to_f32();
                acc += xh[r * k + c].to_f32() * bv;
            }
            want[r * n + j] = acc;
        }
    }
    let mk = |l: usize, name: &str| {
        rt.device()
            .newBufferWithLength_options(l, MTLResourceOptions::StorageModeShared)
            .unwrap_or_else(|| panic!("buffer {name}"))
    };
    let xbuf = mk(16 * k * 2, "x");
    let wbuf = mk(wq.len(), "w");
    let sbuf = mk(scales.len() * 2, "s");
    let bbuf = mk(biases.len() * 2, "b");
    let ybuf = mk(16 * n * 2, "y");
    unsafe {
        std::ptr::copy_nonoverlapping(
            xh.as_ptr() as *const u8,
            xbuf.contents().as_ptr().cast::<u8>(),
            16 * k * 2,
        );
        std::ptr::copy_nonoverlapping(wq.as_ptr(), wbuf.contents().as_ptr().cast::<u8>(), wq.len());
        std::ptr::copy_nonoverlapping(
            scales.as_ptr() as *const u8,
            sbuf.contents().as_ptr().cast::<u8>(),
            scales.len() * 2,
        );
        std::ptr::copy_nonoverlapping(
            biases.as_ptr() as *const u8,
            bbuf.contents().as_ptr().cast::<u8>(),
            biases.len() * 2,
        );
        std::ptr::write_bytes(ybuf.contents().as_ptr().cast::<u8>(), 0xEE, 16 * n * 2);
    }
    // Raw command buffer again (isolation).
    let queue = rt.device().newCommandQueue().unwrap();
    let cb = queue.commandBuffer().unwrap();
    let enc = cb
        .computeCommandEncoderWithDispatchType(objc2_metal::MTLDispatchType::Serial)
        .unwrap();
    enc.setComputePipelineState(pipeline.raw());
    unsafe {
        enc.setBuffer_offset_atIndex(Some(&xbuf), 0, 0);
        enc.setBuffer_offset_atIndex(Some(&wbuf), 0, 1);
        enc.setBuffer_offset_atIndex(Some(&sbuf), 0, 2);
        enc.setBuffer_offset_atIndex(Some(&bbuf), 0, 3);
        enc.setBuffer_offset_atIndex(Some(&ybuf), 0, 4);
    }
    let nn = n as i32;
    unsafe {
        enc.setBytes_length_atIndex(
            std::ptr::NonNull::new(&nn as *const i32 as *mut core::ffi::c_void).unwrap(),
            4,
            5,
        );
    }
    mark("m16 dispatch");
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: n / 32,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    enc.endEncoding();
    cb.commit();
    mark("m16 wait");
    cb.waitUntilCompleted();
    mark("m16 wait done");
    eprintln!(
        "[mpp-probe] cb error = {:?}",
        cb.error().map(|e| e.localizedDescription().to_string())
    );
    assert_eq!(
        cb.status(),
        objc2_metal::MTLCommandBufferStatus::Completed,
        "cb status"
    );
    let yh = unsafe { std::slice::from_raw_parts(ybuf.contents().as_ptr().cast::<u16>(), 16 * n) };
    for (r, j) in [
        (0usize, 31usize),
        (0, 32),
        (0, 33),
        (1, 32),
        (7, 63),
        (1, 0),
    ] {
        eprintln!("[mpp-probe] y[{r}][{j}] bits = 0x{:04x}", yh[r * n + j]);
    }
    let mut worst = 0f32;
    let mut bigbad = 0usize;
    let mut worst2 = (0f32, 0f32, (0usize, 0usize));
    let mut bad = 0usize;
    for r in 0..8 {
        for j in 0..n {
            let got = bf16::from_bits(yh[r * n + j]).to_f32();
            let w = want[r * n + j];
            let d = (got - w).abs();
            worst = worst.max(d);
            if d > worst2.1 {
                worst2 = (got, d, (r, j));
            }
            if d.abs() > 100.0 {
                bigbad += 1;
            }
            // The output is bf16: allow one bf16 ulp of the output plus a
            // small fp32-reorder margin (the kernel's partial tree differs
            // from the sequential reference).
            let ulp = 2f32.powf(w.abs().max(1.0f32).log2().floor()) / 128.0;
            if d > ulp * 2.0 {
                bad += 1;
                if bad < 40 {
                    eprintln!("[mpp-probe] m16 MISMATCH y[{r}][{j}] = {got} want {w}");
                }
            }
        }
    }
    eprintln!("[mpp-probe] m16 worst abs diff = {worst}, bad = {bad}/512, wild(>100) = {bigbad}");
    eprintln!(
        "[mpp-probe] m16 worst2: y[{:?}] got {} want {} abs {} rel {}",
        worst2.2,
        worst2.0,
        want[worst2.2.0 * n + worst2.2.1],
        worst2.1,
        worst2.1 / want[worst2.2.0 * n + worst2.2.1].abs().max(1e-6)
    );
    assert_eq!(bad, 0, "m16 kernel produced wrong values");
}
