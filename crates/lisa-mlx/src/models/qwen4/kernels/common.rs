use crate::ffi::{MetalKernel, OutputArg};
use lisa_mlx::ops::indexing::IndexOp;
use lisa_mlx::{Array, Dtype, Stream, ops};

/// Shared helpers for the fused elementwise kernels, ported verbatim from the
/// engine's `TrackFastKernels.exactHeader`.
pub const EXACT_HEADER: &str = include_str!("../../../shaders/common/exact_header.metal");
/// `track_copy_rows_into`: append `src` `[R, s, D]` into `dst` `[R, cap, D]` at
/// row offset `off`, touching only the `s` appended rows. Replaces the
/// `slice_assign` (O(cap)) in the KV cache and the indexer tape.
const COPY_ROWS_SOURCE: &str = include_str!("../../../shaders/common/data/copy_rows.metal");

static COPY_ROWS: std::sync::OnceLock<Option<MetalKernel>> = std::sync::OnceLock::new();

fn copy_rows_kernel() -> &'static Option<MetalKernel> {
    COPY_ROWS.get_or_init(|| {
        MetalKernel::new(
            "track_copy_rows_into",
            &["src", "dst", "off"],
            &["out"],
            COPY_ROWS_SOURCE,
            "",
            true,
            false,
        )
        .ok()
    })
}

/// Append `src` `[.., s, D]` into the caller's `dst` `[.., cap, D]` buffer at
/// row offset `off`. `dst` is bound as both a shape-carrying input and the
/// preallocated output so the encoder sees the dependency on the cache buffer.
pub fn copy_rows_into(src: &Array, dst: &Array, off: i32, stream: &Stream) -> Option<Array> {
    let kernel = copy_rows_kernel().as_ref()?;
    let off_a = Array::from_int(off);
    let inputs: [&Array; 3] = [src, dst, &off_a];
    let outputs = [OutputArg {
        shape: dst.shape().to_vec(),
        dtype: Dtype::Bfloat16,
    }];
    let grid = (src.size() as i32, 1, 1);
    let out = kernel
        .apply_into(&inputs, &[], grid, (256, 1, 1), &outputs, &[dst], stream)
        .ok()?;
    out.into_iter().next()
}
/// Diagnostic: time the NAX indirect expert GEMMs against a dense quantized
/// matmul of the same FLOPs (`lisa indirect-bench`).
pub fn indirect_bench() {
    use std::time::Instant;
    let m = 10240i32;
    let k = 2560i32;
    let n = 640i32;
    let e = 512i32;
    let mut seed = 7u64;
    let mut rnd = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 40) as f32 / 8388608.0) - 1.0
    };
    let wv: Vec<f32> = (0..(e * n * k) as usize).map(|_| rnd() * 0.1).collect();
    let w = Array::from_slice(&wv, &[e, n, k])
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let (gw, gs, gb) = ops::quantize(&w, 32, 4).unwrap();
    let wdv: Vec<f32> = (0..(e * k * n) as usize).map(|_| rnd() * 0.1).collect();
    let wd = Array::from_slice(&wdv, &[e, k, n])
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let (dw, ds, db) = ops::quantize(&wd, 32, 4).unwrap();
    let xv: Vec<f32> = (0..(1024 * k) as usize).map(|_| rnd()).collect();
    let x = Array::from_slice(&xv, &[1, 1024, k])
        .as_dtype(Dtype::Bfloat16)
        .unwrap();
    let _ = (
        gw.eval(),
        gs.eval(),
        gb.eval(),
        dw.eval(),
        ds.eval(),
        db.eval(),
        x.eval(),
    );

    let idx: Vec<u32> = (0..m).map(|i| (i / 20) as u32).collect();
    let sorted_idx = Array::from_slice(&idx, &[m]);
    let tok: Vec<u32> = (0..m).map(|i| (i / 10) as u32).collect();
    let token_rows = Array::from_slice(&tok, &[m]);
    let stream = lisa_mlx::Stream::thread_local_or_default();
    let tiles = crate::prefill_indirect::tile_table(&sorted_idx, m, e, &stream).unwrap();
    let max_t = crate::prefill_indirect::max_tiles(m, e);
    let _ = tiles.eval();

    // warmup
    let act = crate::prefill_indirect::gate_up(
        &x,
        &gw,
        &gs,
        &gb,
        &gw,
        &gs,
        &gb,
        &sorted_idx,
        &token_rows,
        &tiles,
        max_t,
        n,
        k,
        m,
        &stream,
    )
    .unwrap();
    let _ = act.eval();
    let dn = crate::prefill_indirect::down(
        &act,
        &dw,
        &ds,
        &db,
        &sorted_idx,
        &tiles,
        max_t,
        k,
        n,
        m,
        &stream,
    )
    .unwrap();
    let _ = dn.eval();

    for i in 0..6 {
        let t = Instant::now();
        let a = crate::prefill_indirect::gate_up(
            &x,
            &gw,
            &gs,
            &gb,
            &gw,
            &gs,
            &gb,
            &sorted_idx,
            &token_rows,
            &tiles,
            max_t,
            n,
            k,
            m,
            &stream,
        )
        .unwrap();
        let _ = a.eval();
        println!(
            "  gate_up iter {i}: {:.2} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
    }

    let t = Instant::now();
    for _ in 0..5 {
        let a = crate::prefill_indirect::down(
            &act,
            &dw,
            &ds,
            &db,
            &sorted_idx,
            &tiles,
            max_t,
            k,
            n,
            m,
            &stream,
        )
        .unwrap();
        let _ = a.eval();
    }
    println!(
        "down indirect:    {:.2} ms/iter",
        t.elapsed().as_secs_f64() * 200.0
    );

    // dense reference: one expert's weight over all M rows (same FLOPs)
    let xd = x.reshape(&[1024, k]).unwrap();
    let (w2, s2, b2) = ops::quantize(&w.index((0, .., ..)), 32, 4).unwrap();
    let _ = (w2.eval(), s2.eval(), b2.eval());
    let f = || ops::quantized_matmul(&xd, &w2, &s2, Some(&b2), true, 32, 4);
    let _ = f().unwrap().eval();
    let t = Instant::now();
    for _ in 0..5 {
        let _ = f().unwrap().eval();
    }
    println!(
        "dense qmm M=1024: {:.2} ms/iter (x10 to compare M=10240)",
        t.elapsed().as_secs_f64() * 200.0
    );
}

/// Force `get_or_init` on every kernel static in this module (see
/// `crate::models::qwen4::warm_kernels`).
pub fn warm() {
    let _ = copy_rows_kernel();
}

#[cfg(test)]
mod copy_rows_tests {
    use crate::{Array, Stream};

    /// The KV cache hands `copy_rows_into` a transposed v view; the kernel must
    /// read it through its declared strides and produce EXACTLY the bytes the
    /// contiguous src would have written (the slice_assign fallback it replaces
    /// is the full-KV-buffer copy this wave removed).
    #[test]
    fn copy_rows_into_strided_src_matches_contiguous() {
        let stream = Stream::thread_local_or_default();
        let (b, hk, s, d, cap, off) = (1usize, 4usize, 3usize, 256usize, 64usize, 5usize);
        let mut src_v: Vec<half::bf16> = Vec::with_capacity(b * hk * s * d);
        let mut x = 0u32;
        for _ in 0..b * hk * s * d {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            src_v.push(half::bf16::from_f32(((x >> 16) as f32 / 65536.0) * 2.0 - 1.0));
        }
        // Contiguous src [b, hk, s, d].
        let dense = Array::from_slice(&src_v, &[b as _, hk as _, s as _, d as _]);
        // A logically identical src held as a strided view: transpose out and
        // back (the layout the KV cache actually receives for v).
        let strided = dense
            .transpose_axes(&[0, 2, 1, 3])
            .expect("t1")
            .transpose_axes(&[0, 2, 1, 3])
            .expect("t2");

        let mk_dst = || {
            Array::from_slice(
                &vec![half::bf16::from_f32(f32::NAN); b * hk * cap * d],
                &[b as _, hk as _, cap as _, d as _],
            )
        };
        let mut dst_a = mk_dst();
        let mut dst_b = mk_dst();
        let ra = super::copy_rows_into(&dense, &mut dst_a, off as i32, &stream);
        let rb = super::copy_rows_into(&strided, &mut dst_b, off as i32, &stream);
        assert!(ra.is_some(), "dense src declined");
        assert!(rb.is_some(), "strided src declined (strides not engaged)");
        dst_a.eval().expect("eval a");
        dst_b.eval().expect("eval b");
        let va = dst_a.to_vec1::<half::bf16>().expect("read a");
        let vb = dst_b.to_vec1::<half::bf16>().expect("read b");
        let mism = va.iter().zip(vb.iter()).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
        assert_eq!(mism, 0, "strided-src copy differs from contiguous-src copy");
        // And the rows actually landed at `off` (dst [b, hk, cap, d]).
        let base = (0 * cap * d + off * d) as usize;
        assert!(va[base].to_f32() != f32::NAN);
    }
}
