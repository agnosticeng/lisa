//! GPU-slot micro-bench: the step-decode kernels, dispatched back-to-back
//! through the REAL per-op path (per_buffer=16), measured per dispatch via
//! the command-buffer GPU-time counters (`LISA_TRACE` must be on so per-label
//! dispatch counts exist).
//!
//! `LISA_TRACE=1 cargo test -p lisa-mlx --lib bench_kernel_gpu_slot -- --ignored --nocapture`

use crate::ops::{array_ops, fast, zeros, Array, Dtype, Stream};

const KERNEL: &str = r#"
#include <metal_stdlib>
using namespace metal;
[[kernel]] void double_it(
    const device float* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    out[i] = x[i] * 2.0f;
}
"#;

/// Repeat `f` n times (one dispatch per call through the real per-op path),
/// flush, and print the per-dispatch GPU-slot µs attributed to `label`.
fn gpu_slot_us(gpu_label: &str, label: &str, n: usize, mut f: impl FnMut()) {
    let label_gpu = |find: &str| {
        crate::ops::runtime_label_gpu_ms()
            .into_iter()
            .find(|(l, _)| l == find)
            .map(|(_, ns)| ns)
            .unwrap_or(0)
    };
    let before = label_gpu(gpu_label);
    let t = std::time::Instant::now();
    for _ in 0..n {
        f();
    }
    Stream::thread_local_or_default()
        .runtime()
        .commands
        .flush_and_wait()
        .unwrap();
    let wall_us = t.elapsed().as_micros() as f64 / n as f64;
    let after = label_gpu(gpu_label);
    println!(
        "{label:<40} gpu-slot {:>8.1} us/dis   wall {:>8.1} us/dis   (n={n})",
        (after - before) as f64 / n as f64 / 1e3,
        wall_us,
    );
}

/// Cold-stream qmv: `reps` DISTINCT weight sets (~50 MB each at 5120->17408)
/// cycled `rounds` times, so no read is cache-resident. This is the honest
/// instrument for the serial qmv bandwidth ledger (bench_gpu_slot's single
/// set is cache-resident and reads an impossible 770 GB/s).
#[test]
#[ignore]
fn bench_qmv_cold_stream() {
    let stream = Stream::thread_local_or_default();
    let _ = &stream;
    let k = 5120i32;
    let n = 17408i32;
    let reps = 24usize;
    let rounds = 16usize;
    let mut state: u64 = 0x9e3779b97f4a7c15;
    let mut rnd = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut sets = Vec::new();
    for _ in 0..reps {
        let wv: Vec<u32> = (0..(n as usize) * (k as usize / 8)).map(|_| rnd() as u32).collect();
        let wq = Array::from_slice(&wv, &[n, k / 8]);
        let sv: Vec<f32> = (0..(n as usize) * (k as usize / 64)).map(|_| (rnd() % 1000) as f32 / 1e6).collect();
        let sc = Array::from_slice(&sv, &[n, k / 64]);
        let bi = f32arr(&[n, k / 64], 0.0);
        let sc16 = Array::new(sc.t.to_dtype(Dtype::Bfloat16).unwrap());
        let bi16 = Array::new(bi.t.to_dtype(Dtype::Bfloat16).unwrap());
        sets.push((wq, sc16, bi16));
    }
    sets.iter().for_each(|(w, s, b)| {
        let _ = (w.t.eval(), s.t.eval(), b.t.eval());
    });
    let x = bf16(&[1, k], 0.01);
    let bytes_per = (n as usize * k as usize / 2)
        + 2 * (n as usize * k as usize / 64) * 2;
    // Warm the pipeline JIT once.
    let _ = crate::ops::quantized_matmul(&x, &sets[0].0, &sets[0].1, Some(&sets[0].2), true, 64, 4)
        .unwrap()
        .t
        .eval();
    let total = reps * rounds;
    let pre_gpu: u64 = crate::ops::runtime_label_gpu_ms()
        .into_iter()
        .find(|(l, _)| l.contains("affine_qmv"))
        .map(|(_, ns)| ns)
        .unwrap_or(0);
    let t = std::time::Instant::now();
    let mut first: Option<f32> = None;
    for r in 0..total {
        let (w, s, b) = &sets[r % reps];
        let y = crate::ops::quantized_matmul(&x, w, s, Some(b), true, 64, 4).unwrap();
        // Eval each output so the GPU pipeline is never host-starved, and the
        // read of every byte is real.
        if r == 0 {
            y.t.eval().unwrap();
            let f = y.t.to_dtype(Dtype::Float32).unwrap();
            f.eval().unwrap();
            first = Some(f.item::<f32>());
        } else {
            y.t.eval().unwrap();
        }
    }
    let dt = t.elapsed().as_secs_f64();
    let gpu_ms = crate::ops::runtime_label_gpu_ms()
        .into_iter()
        .find(|(l, _)| l.contains("affine_qmv"))
        .map(|(_, ns)| ns)
        .unwrap_or(0);
    // Per-round banding: round r reads each set once (reps dispatches).
    println!(
        "qmv_cold_stream: {total} dispatches, {} MiB read in {:.1} ms = {:.0} GB/s (first {first:?})",
        total * bytes_per >> 20,
        dt * 1e3,
        (total * bytes_per) as f64 / dt / 1e9,
    );
    let _ = pre_gpu;
    println!(
        "qmv_cold_stream: kernel gpu {:.1} ms total = {:.0} GB/s of pure kernel stream",
        (gpu_ms - pre_gpu) as f64 / 1e6,
        (total * bytes_per) as f64 / ((gpu_ms - pre_gpu) as f64 / 1e9),
    );
}

fn bf16(shape: &[i32], fill: f32) -> Array {
    let dims: Vec<i32> = shape.to_vec();
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let v = vec![fill; n];
    let a = Array::from_slice(&v, &dims);
    Array::new(a.t.to_dtype(Dtype::Bfloat16).unwrap())
}

fn f32arr(shape: &[i32], fill: f32) -> Array {
    let dims: Vec<i32> = shape.to_vec();
    let n: usize = shape.iter().map(|&d| d as usize).product();
    Array::from_slice(&vec![fill; n], &dims)
}

#[test]
#[ignore]
fn bench_kernel_gpu_slot() {
    let stream = Stream::thread_local_or_default();
    let rt = stream.runtime().clone();

    // Minimal kernel (the `double_it` harness) as the floor reference.
    let pipe = rt.compile(KERNEL, "double_it").unwrap();
    let xb = rt.buffer(1024, "x").unwrap();
    let ob = rt.buffer(1024, "out").unwrap();
    gpu_slot_us("double_it", "minimal_double_it", 256, || {
        let g = rt.commands.encoder().unwrap();
        let e = g.encoder();
        e.set_pipeline(&pipe);
        e.set_input(0, Some(&xb), 0);
        e.set_output(1, Some(&ob), 0);
        e.dispatch_threads((256, 1, 1), (64, 1, 1));
    });

    // qmv_fast 5120 -> 17408 (gate/up of the 27B MLP, 4-bit gs64): ~45 MB read.
    let k = 5120i32;
    let n = 17408i32;
    let x = bf16(&[1, k], 0.01);
    let wq = zeros::<u32>(&[n, k / 8]).unwrap();
    let sc = f32arr(&[n, k / 64], 1.0);
    let bi = f32arr(&[n, k / 64], 0.0);
    gpu_slot_us("affine_qmv_serial", "affine_qmv_serial 5120->17408", 256, || {
        array_ops::quantized_matmul(&x, &wq, &sc, Some(&bi), true, 64, 4).unwrap();
    });

    // qmv_fast 5120 -> 6144 (attn projection scale): ~16 MB read.
    let n2 = 6144i32;
    let wq2 = zeros::<u32>(&[n2, k / 8]).unwrap();
    let sc2 = f32arr(&[n2, k / 64], 1.0);
    let bi2 = f32arr(&[n2, k / 64], 0.0);
    gpu_slot_us("affine_qmv_serial", "affine_qmv_serial 5120->6144", 256, || {
        array_ops::quantized_matmul(&x, &wq2, &sc2, Some(&bi2), true, 64, 4).unwrap();
    });

    // swiglu2 [1,17408] (the MLP activation, decode shape).
    let g = bf16(&[1, 17408], 0.5);
    let u = bf16(&[1, 17408], 0.5);
    gpu_slot_us("custom_kernel_track_swiglu2__bfloat16_t_1_17408_bfloat16_t_bfloat16_t_bfloat16_t", "swiglu2 [1,17408]", 256, || {
        crate::kernels::swiglu2(&g, &u, &stream);
    });

    // fused add+norm [1,5120] (the B1 kernel) and plain rms.
    let a = bf16(&[1, 5120], 0.5);
    let b = bf16(&[1, 5120], 0.5);
    let wt = bf16(&[5120], 1.0);
    gpu_slot_us("fused_add_rms_looped_bfloat16", "fused_add_rms [1,5120]", 256, || {
        fast::fused_add_rms_norm(&a, &b, &wt, 1e-6).unwrap();
    });
    gpu_slot_us("rms_looped_bfloat16", "rms [1,5120]", 256, || {
        fast::rms_norm(&a, Some(&wt), 1e-6).unwrap();
    });

    // The real MLP decode sequence: qmv(gate) qmv(up) swiglu2 qmv(down), to
    // see whether mixing kernels changes the per-dispatch slot price.
    let wq3 = zeros::<u32>(&[k, n / 8]).unwrap();
    let sc3 = f32arr(&[k, n / 64], 1.0);
    let bi3 = f32arr(&[k, n / 64], 0.0);
    gpu_slot_us("(mixed)", "mlp_trio (qmv x2 + swiglu2 + qmv)", 64, || {
        let gout = array_ops::quantized_matmul(&x, &wq, &sc, Some(&bi), true, 64, 4).unwrap();
        let uout = array_ops::quantized_matmul(&x, &wq, &sc, Some(&bi), true, 64, 4).unwrap();
        let act = match crate::kernels::swiglu2(&gout, &uout, &stream) {
            Some(act) => act,
            None => gout.multiply(&uout).unwrap(),
        };
        let _d = array_ops::quantized_matmul(&act, &wq3, &sc3, Some(&bi3), true, 64, 4).unwrap();
    });

    // The VERIFY-width phase shapes (M = 3 / 7 rows): the same phase kernels
    // dispatched at the S-row verify sizes, to split the in-situ verify
    // forward's phase cost into kernel cost vs launch-context cost. The
    // gpu-slot counter pipelines back-to-back INDEPENDENT launches here, so
    // the gpu-slot column is a floor (real exec per launch when pipelined);
    // wall shows the enqueue path. Compare against the in-situ per-launch
    // prices from `verify-audit` before attributing anything.
    for rows in [3usize, 7usize] {
        let a = bf16(&[rows as i32, 5120], 0.5);
        let b = bf16(&[rows as i32, 5120], 0.5);
        let wt = bf16(&[5120], 1.0);
        gpu_slot_us(
            "fused_add_rms_looped_bfloat16",
            &format!("fused_add_rms [{rows},5120]"),
            256,
            || { let _ = fast::fused_add_rms_norm(&a, &b, &wt, 1e-6).unwrap(); },
        );
        let g = bf16(&[rows as i32, 17408], 0.5);
        let u = bf16(&[rows as i32, 17408], 0.5);
        gpu_slot_us(
            &format!("custom_kernel_track_swiglu2__bfloat16_t_{rows}_17408_bfloat16_t_bfloat16_t_bfloat16_t"),
            &format!("swiglu2 [{rows},17408]"),
            256,
            || {
                crate::kernels::swiglu2(&g, &u, &stream);
            },
        );
        let xr = bf16(&[rows as i32, 5120], 0.5);
        gpu_slot_us("rms_looped_bfloat16", &format!("rms [{rows},5120]"), 256, || {
            fast::rms_norm(&xr, Some(&wt), 1e-6).unwrap();
        });
    }

    // The M 8..=16 verify-qmm sweep (specs/08 m16 lane campaign): the stock
    // path takes qmv_wide (per-row chain) at M 8..12 and the tiled
    // affine_qmm_splitk at M >= 13, and the wide rounds pay a 30-50 % per-
    // token penalty. Per-dispatch stock vs the split-K VERIFY tile (bn=2,
    // kp=2 — the shipped constants; the raw tile entry has no MROWS gate,
    // the production lane does) at the trunk shapes, to split where the
    // penalty lives: qmm kernel cost vs launch/occupancy vs the tiled lane.
    // Spill shows up as a huge gpu-slot/dis (the BN=4 M=5 precedent).
    for rows in [8usize, 9, 10, 11, 12, 13, 16] {
        let xr = bf16(&[rows as i32, 5120], 0.01);
        let wq = zeros::<u32>(&[17408, 5120 / 8]).unwrap();
        let sc = f32arr(&[17408, 5120 / 64], 1.0);
        let bi = f32arr(&[17408, 5120 / 64], 0.0);
        gpu_slot_us(
            // M 8..12 stock = qmv_wide; M >= 13 stock = qmm_nax (compile name
            // read from the jit log on first run).
            if rows < 13 { "affine_qmv_wide_float_gs_64_b_4_nv_4_kl_8_batch_0" } else { "qmm_nax" },
            &format!("STOCK qmm [{rows},5120]->17408"),
            64,
            || {
                let _ = array_ops::quantized_matmul(&xr, &wq, &sc, Some(&bi), true, 64, 4)
                    .unwrap();
            },
        );
        // The NAX-tiled lane (M >= 13 stock): same entry as stock above for
        // m < 13 — for m >= 13 stock already routes qmm_nax (affine_qmm_splitk
        // bails at split_k<=1 for this grid).
        let wq2 = zeros::<u32>(&[17408, 5120 / 8]).unwrap();
        gpu_slot_us(
            "qmm_nax",
            &format!("TILED qmm_nax [{rows},5120]->17408"),
            64,
            || {
                let _ = crate::jit::qmm_nax(
                    &xr.t.device(),
                    &xr.t,
                    &wq2.t,
                    &sc.t,
                    &bi.t,
                    64,
                    4,
                )
                .unwrap();
            },
        );
        // Candidate m16 lane: the shipped split-K verify tile at MROWS>7.
        let wq3 = zeros::<u32>(&[17408, 5120 / 8]).unwrap();
        gpu_slot_us(
            &format!(
                "affine_verify_qmm_splitk_float_gs_64_b_4_m_{rows}_bn_2_kp_2_ppt_2"
            ),
            &format!("VERIFY-TILE bn2kp2 [{rows},5120]->17408"),
            64,
            || {
                let _ = crate::jit::affine_verify_qmm_splitk_tile(
                    &xr.t.device(),
                    &xr.t,
                    &wq3.t,
                    &sc.t,
                    &bi.t,
                    64,
                    4,
                    2,
                    2,
                    2,
                )
                .unwrap();
            },
        );
    }
}

/// The m16 verify-qmm attribution sweep (specs/08 item 2): per-dispatch
/// eval+sync wall (warmup first — JIT compiles excluded) for the stock path
/// vs the split-K verify tile at MROWS > 7 across M 8..=16, at the trunk
/// verify shapes. n=32 back-to-back dispatches with ONE flush amortises the
/// per-launch slot the way the real verify forward enqueues them; the
/// isolated single-shot is the floor. A stack-spilled MROWS shows as a huge
/// per-dis time (the BN=4 M=5 precedent: 2.85 ms vs 0.65 ms).
///
/// `LISA_DEVICE=metal cargo test -p lisa-mlx --release --lib
///  bench_verify_qmm_m_sweep -- --ignored --nocapture`
#[test]
#[ignore]
fn bench_verify_qmm_m_sweep() {
    fn time_us(n: usize, mut f: impl FnMut()) -> f64 {
        let stream = Stream::thread_local_or_default();
        let rt = stream.runtime().clone();
        f(); // warmup (JIT compile)
        rt.commands.flush_and_wait().unwrap();
        let t = std::time::Instant::now();
        for _ in 0..n {
            f();
        }
        rt.commands.flush_and_wait().unwrap();
        t.elapsed().as_micros() as f64 / n as f64
    }

    for rows in [8usize, 9, 10, 11, 12, 13, 16] {
        let xr = bf16(&[rows as i32, 5120], 0.01);
        let wq = zeros::<u32>(&[17408, 5120 / 8]).unwrap();
        let sc = f32arr(&[17408, 5120 / 64], 1.0);
        let bi = f32arr(&[17408, 5120 / 64], 0.0);
        // The NAX m16 tile: M-independent cost (the fixed [16, K] tile always
        // streams), so one arm per rows is redundant — measured at the first
        // rows only; the print repeats it for comparison.
        let m16 = if rows == 8 {
            let xr16 = bf16(&[16, 5120], 0.01);
            time_us(32, || {
                let _ = crate::jit::affine_verify_qmm_nax_m16(
                    &xr16.t.device(),
                    &xr16.t,
                    &wq.t,
                    &sc.t,
                    &bi.t,
                    64,
                    4,
                )
                .unwrap();
            })
        } else {
            f64::NAN
        };
        let stock = time_us(32, || {
            let _ = array_ops::quantized_matmul(&xr, &wq, &sc, Some(&bi), true, 64, 4).unwrap();
        });
        let nax = time_us(32, || {
            let _ = crate::jit::qmm_nax(&xr.t.device(), &xr.t, &wq.t, &sc.t, &bi.t, 64, 4)
                .unwrap();
        });
        let tile = |bn: usize, kp: usize| {
            time_us(32, || {
                let _ = crate::jit::affine_verify_qmm_splitk_tile(
                    &xr.t.device(),
                    &xr.t,
                    &wq.t,
                    &sc.t,
                    &bi.t,
                    64,
                    4,
                    bn,
                    kp,
                    crate::jit::SPLITK_PPT,
                )
                .unwrap();
            })
        };
        // Two-tile pass: MROWS=8 on rows 0..8 + MROWS=(rows-8) on the tail —
        // the same fp32 chain per output element as the shipped splitk tile
        // (MROWS does not enter the reduction), at the price of one extra
        // weight stream. Measured at the tail x as a plain host copy.
        let twotile = if rows > 8 {
            let tail = rows - 8;
            let xt = bf16(&[tail as i32, 5120], 0.01);
            time_us(32, || {
                let _ = crate::jit::affine_verify_qmm_splitk_tile(
                    &xr.t.device(), &xr.t, &wq.t, &sc.t, &bi.t, 64, 4, 2, 2,
                    crate::jit::SPLITK_PPT,
                )
                .unwrap();
                let _ = crate::jit::affine_verify_qmm_splitk_tile(
                    &xr.t.device(), &xt.t, &wq.t, &sc.t, &bi.t, 64, 4, 2, 2,
                    crate::jit::SPLITK_PPT,
                )
                .unwrap();
            })
        } else {
            0.0
        };
        println!(
            "M={rows:>2}  stock {stock:>8.1}  qmm_nax {nax:>8.1}  m16 {m16:>8.1}  tile bn2kp2 {kp2:>9.1}  bn2kp1 {kp1:>9.1}  bn2kp4 {kp4:>9.1}  twotile {twotile:>9.1}",
            kp2 = tile(2, 2),
            kp1 = tile(2, 1),
            kp4 = tile(2, 4),
        );
    }
}

/// Peak streaming-READ bandwidth of this box: the number `specs/07`'s
/// that the tree never actually measured. If the real peak is well above
/// 819 GB/s then 723 is a kernel-efficiency problem, not a floor.
///
/// Grid-stride read of a 1 GiB bf16 buffer, reduced per threadgroup and
/// flushed with one atomic per warp, so the traffic is `n * 2` bytes read and
/// a negligible write. `dispatch_threads` + `flush_and_wait` (eval+sync — the
/// harness rule: enqueue-only timings fake wins).
///
/// `LISA_DEVICE=metal cargo test -p lisa-mlx --release --lib bench_peak_stream
///  -- --ignored --nocapture`
const STREAM_READ: &str = r#"
#include <metal_stdlib>
using namespace metal;

[[kernel]] void stream_read(
    const device half4* x [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint tgsize [[threads_per_threadgroup]],
    uint tgnum [[threadgroups_per_grid]])
{
    const uint stride = tgnum * tgsize;
    float4 a0 = float4(0.0f), a1 = float4(0.0f), a2 = float4(0.0f), a3 = float4(0.0f);
    uint i = gid * tgsize + lid;
    const uint step = stride * 4u;
    // 4 independent 8-byte loads in flight (32 B/iteration/thread).
    for (; i + 3u * stride < NBOUND4; i += step) {
        a0 += float4(x[i]);
        a1 += float4(x[i + stride]);
        a2 += float4(x[i + 2u * stride]);
        a3 += float4(x[i + 3u * stride]);
    }
    for (; i < NBOUND4; i += stride) {
        a0 += float4(x[i]);
    }
    float s = (a0.x + a0.y + a0.z + a0.w) + (a1.x + a1.y + a1.z + a1.w)
            + (a2.x + a2.y + a2.z + a2.w) + (a3.x + a3.y + a3.z + a3.w);
    s = simd_sum(s);
    if ((lid & 31u) == 0u) {
        atomic_fetch_add_explicit((device atomic_float*)out, s, memory_order_relaxed);
    }
}
"#;

#[test]
#[ignore]
fn bench_peak_stream() {
    let stream = Stream::thread_local_or_default();
    let rt = stream.runtime().clone();
    // 4 GiB working set (2G halves): far past any cache, so it is DRAM traffic.
    const N: usize = 2 * 1024 * 1024 * 1024;
    const N4: usize = N / 4;
    let src = STREAM_READ.replace("NBOUND4", &N4.to_string());
    let pipe = rt.compile(&src, "stream_read").unwrap();
    let xb = rt.buffer(N * 2, "stream_x").unwrap();
    let ob = rt.buffer(64 * 1024, "stream_out").unwrap();
    let iters = 10usize;
    let bytes = (N * 2) as f64;
    let gib = bytes / (1024.0 * 1024.0 * 1024.0);
    let mut best = 0.0f64;
    for threads in [
        256usize * 4096,
        256 * 16384,
        256 * 65536,
        256 * 262144,
    ] {
        let before = std::time::Instant::now();
        for _ in 0..iters {
            let g = rt.commands.encoder().unwrap();
            let e = g.encoder();
            e.set_pipeline(&pipe);
            e.set_input(0, Some(&xb), 0);
            e.set_output(1, Some(&ob), 0);
            e.dispatch_threads((threads, 1, 1), (256, 1, 1));
        }
        rt.commands.flush_and_wait().unwrap();
        let secs = before.elapsed().as_secs_f64() / iters as f64;
        let gbs = bytes / secs / 1e9;
        if gbs > best {
            best = gbs;
        }
        println!(
            "peak stream read: threads={threads:>9}  {:>8.2} ms/pass  {:>7.1} GB/s  ({gib:.2} GiB/pass)",
            secs * 1e3,
            gbs
        );
    }
    println!(">>> PEAK STREAMING READ = {best:.1} GB/s (specs/07 assumed 819)");
}
