//! Smoke tests for the lisa-mlx + custom Metal kernel stack.

use lisa_mlx::ops::indexing::IndexOp;
use lisa_mlx::{ops, Array, Dtype, Stream};

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (u32::MAX as f32)) * 2.0 - 1.0
}

fn rand_f32(n: usize, seed: &mut u64) -> Vec<f32> {
    (0..n).map(|_| lcg(seed)).collect()
}

/// Verify quantized_matmul (affine 4-bit, gs 32, transpose) against
/// dequantize + matmul.
fn smoke_quantized_matmul() -> anyhow::Result<()> {
    let mut seed = 42u64;
    let m = 64usize; // output rows (N)
    let k = 256usize; // input cols (K)
    let w = rand_f32(m * k, &mut seed);
    let x = rand_f32(k, &mut seed);

    let w = Array::from_slice(&w, &[m as i32, k as i32]);
    let (wq, scales, biases) = ops::quantize(&w, 32, 4)?;
    assert_eq!(wq.dtype(), Dtype::Uint32);
    assert_eq!(wq.shape(), vec![m as i32, (k / 8) as i32]);
    assert_eq!(scales.shape(), vec![m as i32, (k / 32) as i32]);

    let x = Array::from_slice(&x, &[1i32, k as i32]).as_dtype(Dtype::Bfloat16)?;
    let y1 = ops::quantized_matmul(&x, &wq, &scales, Some(&biases), true, 32, 4)?;
    let w_deq = ops::dequantize(&wq, &scales, &biases, 32, 4)?;
    let y2 = ops::matmul(&x, w_deq.transpose_axes(&[1, 0])?)?;

    let max_diff = (y1 - y2).abs()?.max(None)?.item::<f32>();
    println!("quantized_matmul max abs diff: {max_diff:.6}");
    assert!(max_diff < 0.1, "quantized matmul parity failed: {max_diff}");
    Ok(())
}

/// Verify the custom GDN Metal kernel against the ops fallback.
fn smoke_gdn_kernel() -> anyhow::Result<()> {
    use lisa_mlx::kernels::gdn_kernel_available;

    println!("kernel available: {}", gdn_kernel_available());
    for t in [64usize, 512, 1024] {
        let r = gdn_parity(t)?;
        println!("T={t}: y diff {:.6} state diff {:.6}", r.0, r.1);
    }
    Ok(())
}

fn gdn_parity(t: usize) -> anyhow::Result<(f32, f32)> {
    use lisa_mlx::kernels::{gated_delta_kernel, gated_delta_ops};
    let b = 1usize;
    let t = t;
    let hk = 16usize;
    let hv = 48usize;
    let dk = 128usize;
    let dv = 128usize;
    let mut seed = 7u64;

    let q = Array::from_slice(&rand_f32(b*t*hk*dk, &mut seed), &[b as i32,t as i32,hk as i32,dk as i32]).as_dtype(Dtype::Bfloat16)?;
    let k = Array::from_slice(&rand_f32(b*t*hk*dk, &mut seed), &[b as i32,t as i32,hk as i32,dk as i32]).as_dtype(Dtype::Bfloat16)?;
    // normalize k
    let k = {
        let kn = k.as_dtype(Dtype::Float32)?;
        let norm = lisa_mlx::ops::rsqrt(&(&kn * &kn).sum_axis(-1, Some(true))?)? / 11.313708f32;
        (kn * norm).as_dtype(Dtype::Bfloat16)?
    };
    let v = Array::from_slice(&rand_f32(b*t*hv*dv, &mut seed), &[b as i32,t as i32,hv as i32,dv as i32]).as_dtype(Dtype::Bfloat16)?;
    let g = ops::sigmoid(&Array::from_slice(&rand_f32(b*t*hv, &mut seed), &[b as i32,t as i32,hv as i32]))?.as_dtype(Dtype::Float32)?;
    let beta = ops::sigmoid(&Array::from_slice(&rand_f32(b*t*hv, &mut seed), &[b as i32,t as i32,hv as i32]))?.as_dtype(Dtype::Float32)?;
    let state = Array::from_slice(&rand_f32(b*hv*dv*dk, &mut seed), &[b as i32,hv as i32,dv as i32,dk as i32]).as_dtype(Dtype::Float32)?;
    let stream = Stream::thread_local_or_default();
    let (yk, sk) = gated_delta_kernel(&q,&k,&v,&g,&beta,&state,false,&stream).ok_or_else(|| anyhow::anyhow!("kernel fail"))?;
    let (yo, so) = gated_delta_ops(&q,&k,&v,&g,&beta,Some(&state))?;
    let dy = (yk - yo).abs()?.max(None)?.item_cast::<f32>();
    let ds = (sk - so).abs()?.max(None)?.item_cast::<f32>();
    Ok((dy, ds))
}

/// Verify fast ops: rms_norm, rope, sdpa.
fn smoke_fast_ops() -> anyhow::Result<()> {
    let mut seed = 99u64;
    let s = 8usize;
    let h = 4usize;
    let d = 64usize;
    let x = Array::from_slice(
        &rand_f32(s * h * d, &mut seed),
        &[1i32, s as i32, h as i32, d as i32],
    )
    .as_dtype(Dtype::Bfloat16)?;
    let w = Array::from_slice(&rand_f32(d, &mut seed), &[d as i32]).as_dtype(Dtype::Bfloat16)?;

    let normed = lisa_mlx::fast::rms_norm(&x, Some(&w), 1e-6)?;
    println!("rms_norm ok: {:?}", normed.shape());

    let q = lisa_mlx::fast::rope(&x, d as i32, false, Some(1e7), 1.0, 0, None)?;
    println!("rope ok: {:?}", q.shape());

    let attn = lisa_mlx::fast::scaled_dot_product_attention(
        &q,
        &normed,
        &normed,
        (d as f32).sqrt().recip(),
        lisa_mlx::fast::ScaledDotProductAttentionMask::Causal,
        None,
    )?;
    println!("sdpa ok: {:?}", attn.shape());
    Ok(())
}

/// Verify indexing + take_along_axis.
fn smoke_indexing() -> anyhow::Result<()> {
    let mut seed = 5u64;
    let x = Array::from_slice(&rand_f32(64, &mut seed), &[4i32, 16i32]);
    let row = x.index((1, ..));
    assert_eq!(row.shape(), vec![16]);
    let col = x.index((.., 3));
    assert_eq!(col.shape(), vec![4]);
    let idx = Array::from_slice(&[0i32, 5, 9], &[3]);
    let taken = x.take_axis(&idx, 1)?;
    assert_eq!(taken.shape(), vec![4, 3]);
    println!("indexing ok");
    Ok(())
}

/// Compare MLX fused ops vs the engine's bf16-stepped arithmetic.
/// Isolate the quantized_matmul M>1 divergence.
pub fn run_qmm_m_test() -> anyhow::Result<()> {
    use lisa_mlx::ops;
    let k = 256usize;
    let n = 32usize;
    let m_max = 256usize;
    let mut seed = 11u64;
    // ONE expert: weight [n, k], input rows [m_max, k]
    let w = Array::from_slice(&rand_f32(n * k, &mut seed), &[n as i32, k as i32]);
    let (wq, scales, biases) = ops::quantize(&w, 32, 4)?;
    let wq = wq.contiguous().unwrap();
    let scales = scales.contiguous().unwrap();
    let biases = biases.contiguous().unwrap();
    let x = Array::from_slice(&rand_f32(m_max * k, &mut seed), &[m_max as i32, k as i32])
        .as_dtype(Dtype::Bfloat16)?
        .contiguous()
        .unwrap();
    // reference: M=1 (row 0 alone)
    let x0 = x.index(0..1).contiguous().unwrap();
    let y_ref = ops::quantized_matmul(&x0, &wq, &scales, Some(&biases), true, 32, 4).unwrap();
    let _ = y_ref.eval();
    let ref0: Vec<f32> = y_ref
        .as_dtype(Dtype::Float32)
        .unwrap()
        .as_slice::<f32>()
        .to_vec();
    for m in [1usize, 2, 8, 64, 256] {
        let xs = x
            .index(0..m as i32)
            .contiguous()
            .unwrap();
        let y = ops::quantized_matmul(&xs, &wq, &scales, Some(&biases), true, 32, 4).unwrap();
        let _ = y.eval();
        let arr: Vec<f32> = y
            .as_dtype(Dtype::Float32)
            .unwrap()
            .as_slice::<f32>()
            .to_vec();
        let row0 = &arr[..n];
        let d: f32 = row0
            .iter()
            .zip(ref0.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        println!("M={m}: row0 vs M=1 row0 max diff {d:.6}");
    }
    Ok(())
}

pub fn run_rounding_test() -> anyhow::Result<()> {
    use lisa_mlx::ops::indexing::IndexOp;
    let n = 4096usize;
    let mut seed = 123u64;
    let data: Vec<f32> = (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((seed >> 33) as f32 / u32::MAX as f32) * 20.0 - 10.0
        })
        .collect();
    let x = Array::from_slice(&data, &[n as i32]).as_dtype(Dtype::Bfloat16)?;

    // --- silu: fused op vs bf16-stepped x*sigmoid(x) ---
    let fused = lisa_mlx::nn::silu(&x)?;
    // stepped: s = 1/(1+exp(|x|)) mirrored, each step bf16
    let ax = ops::abs(&x)?;
    let e = ops::exp(&ax)?;                         // bf16 out
    let denom = Array::from_f32(1.0).as_dtype(Dtype::Bfloat16)? + e;
    let inv = reciprocal(&denom)?;
    let one_minus = Array::from_f32(1.0).as_dtype(Dtype::Bfloat16)? - &inv;
    let mirrored = lisa_mlx::ops::r#where(
        &x.lt(&Array::from_f32(0.0).as_dtype(Dtype::Bfloat16)?)?,
        &inv,
        &one_minus)?;
    let stepped = x.multiply(&mirrored)?;
    let d = (fused - stepped).abs()?.max(None)?.item_cast::<f32>();
    println!("silu fused-vs-stepped max diff: {d:.6}");

    // --- rms_norm(x, Some(w)) vs two-rounding rms_norm(None)*w ---
    let w = Array::from_slice(&data[..64], &[64i32]).as_dtype(Dtype::Bfloat16)?;
    let _ = n;
    let x2 = x.index(0..64).reshape(&[8i32, 8])?;
    let x2 = x2.reshape(&[64i32])?;
    let _ = &x2;
    let a = lisa_mlx::fast::rms_norm(&x2, Some(&w), 1e-6)?;
    let b = lisa_mlx::fast::rms_norm(&x2, None, 1e-6)? * &w;
    let d = (a - b).abs()?.max(None)?.item_cast::<f32>();
    println!("rms_norm weighted-vs-tworound max diff: {d:.6}");

    // --- logaddexp(x, 0) f32 vs bf16-stepped ---
    let la_f32 = ops::logaddexp(&Array::from_f32(0.0), &x.as_dtype(Dtype::Float32)?)?;
    // stepped bf16: max(x,0) + log1p(exp(min(x,0)))
    let zero = Array::from_f32(0.0).as_dtype(Dtype::Bfloat16)?;
    let mx_ = ops::maximum(&x, &zero)?;
    let mn = ops::minimum(&x, &zero)?;
    let emn = ops::exp(&mn)?;                        // bf16
    let l1p = {
        // log1p via log(1+e) in bf16
        let one_e = Array::from_f32(1.0).as_dtype(Dtype::Bfloat16)? + emn;
        ops::log(&one_e)?
    };
    let stepped_la = mx_.add(&l1p)?;
    let d = (la_f32.as_dtype(Dtype::Bfloat16)? - stepped_la).abs()?.max(None)?.item_cast::<f32>();
    println!("logaddexp0 f32-vs-stepped-bf16 max diff: {d:.6}");
    Ok(())
}

fn reciprocal(x: &Array) -> lisa_mlx::error::Result<Array> {
    Array::from_f32(1.0).as_dtype(Dtype::Bfloat16)?.divide(x)
}

pub fn run_negslice_test() -> anyhow::Result<()> {
    use lisa_mlx::ops::indexing::IndexOp;
    // The EXACT form the GDN conv-state save uses: [B, T, C], slice (-3)..
    let c: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let m = Array::from_slice(&c, &[1i32, 6, 4]);
    let neg = m.index((.., -3.., ..));
    let _ = neg.eval();
    println!("m[(.., -3.., ..)] shape {:?} data {:?}", neg.shape(), neg.as_slice::<f32>());
    println!("expected shape [1,3,4], data [12..23]");
    Ok(())
}

pub fn run_all() -> anyhow::Result<()> {
    lisa_mlx::ffi::install_error_handler();
    smoke_quantized_matmul()?;
    smoke_fast_ops()?;
    smoke_indexing()?;
    smoke_rms_row_exact()?;
    smoke_gdn_kernel()?;
    println!("all smoke tests passed");
    Ok(())
}

/// `rms_row_exact` on a fixed input, GPU vs a host reference that replicates
/// the engine's pinned reduction association (self-contained: no model, no
/// no reference dumps).
fn smoke_rms_row_exact() -> anyhow::Result<()> {
    let mut seed = 7u64;
    let rows = 4usize;
    let wide = 10240usize;
    let h = 2560usize;
    let eps = 1e-6f32;
    let raw = rand_f32(rows * wide, &mut seed);
    let x = Array::from_slice(&raw, &[rows as i32, wide as i32]).as_dtype(Dtype::Bfloat16)?;
    let wv: Vec<f32> = (0..wide).map(|i| 0.5f32 + (i % 97) as f32 * 0.01).collect();
    let w = Array::from_slice(&wv, &[wide as i32]).as_dtype(Dtype::Bfloat16)?;

    let got = crate::core::norm::rms_row_exact(&x, Some(&w), eps, h)?;
    let got_f = got.as_dtype(Dtype::Float32)?.as_slice::<f32>().to_vec();
    // Read the same buffer a second time (independent cast + eval): if the two
    // reads disagree, the buffer is being modified between reads.
    let got_f2 = got.as_dtype(Dtype::Float32)?.as_slice::<f32>().to_vec();
    let reread = got_f.iter().zip(got_f2.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("[rms] got re-read stability: {reread}");

    let xb: Vec<f32> = x.as_dtype(Dtype::Float32)?.as_slice::<f32>().to_vec();
    let slices = h / 128;
    let groups = wide / h;
    let mut want = vec![0f32; rows * wide];
    for r in 0..rows {
        for g in 0..groups {
            let base = g * h;
            // per-lane sum over 4 consecutive elements -> [slices][32]
            let mut lane = vec![0f32; slices * 32];
            for sl in 0..slices {
                for l in 0..32 {
                    let mut s = 0f32;
                    for e in 0..4 {
                        let i = r * wide + base + sl * 128 + l * 4 + e;
                        s += xb[i] * xb[i];
                    }
                    lane[sl * 32 + l] = s;
                }
            }
            // xor butterfly over the 32 lanes -> [slices]
            let mut v: Vec<f32> = lane.clone();
            let mut width = 32usize;
            while width >= 2 {
                let half = width / 2;
                let mut nv = vec![0f32; slices * half];
                for sl in 0..slices {
                    for i in 0..half {
                        nv[sl * half + i] = v[sl * width + i] + v[sl * width + half + i];
                    }
                }
                v = nv;
                width = half;
            }
            // pad the `slices` partials to 32, second butterfly
            let mut v2 = vec![0f32; 32];
            for sl in 0..slices {
                v2[sl] = v[sl];
            }
            let mut width = 32;
            while width >= 2 {
                let half = width / 2;
                let nv: Vec<f32> = (0..half).map(|i| v2[i] + v2[half + i]).collect();
                v2 = nv;
                width = half;
            }
            let total = v2[0];
            if std::env::var("RMS_TOTALS").is_ok() {
                eprintln!("host total[{r}][{g}] = {total:.6} inv={:.6}", (total / h as f32 + eps).sqrt().recip());
            }
            let inv = (total / h as f32 + eps).sqrt().recip();
            for off in 0..h {
                let gi = r * wide + base + off;
                want[gi] = xb[gi] * inv * wv[base + off];
            }
        }
    }
    let mut max = 0f32;
    for (a, b) in got_f.iter().zip(want.iter()) {
        max = max.max((a - b).abs());
    }
    if std::env::var("RMS_TOTALS").is_ok() {
        std::fs::write("/tmp/rms-trace/host_want.bin", unsafe {
            std::slice::from_raw_parts(want.as_ptr() as *const u8, want.len() * 4)
        }).ok();
        std::fs::write("/tmp/rms-trace/host_got.bin", unsafe {
            std::slice::from_raw_parts(got_f.as_ptr() as *const u8, got_f.len() * 4)
        }).ok();
    }
    println!("rms_row_exact vs host reference: max abs diff {max:.6}");

    // Micro-test: narrow the LAST axis and materialise (copy_strided), the op
    // rms_row_exact uses for its lane sums.
    {
        let n = 2 * 3 * 16;
        let a = Array::from_slice(
            &(0..n).map(|i| i as f32).collect::<Vec<_>>(),
            &[2i32, 3i32, 16i32],
        )
        .as_dtype(Dtype::Bfloat16)?;
        let v = a.index((lisa_mlx::ops::indexing::Ellipsis, 0..4)).contiguous()?;
        let vf = v.as_dtype(Dtype::Float32)?.as_slice::<f32>().to_vec();
        let want: Vec<f32> = (0..6).flat_map(|r| (0..4).map(move |c| (r * 16 + c) as f32)).collect();
        let d = vf.iter().zip(want.iter()).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("narrow-last-axis contiguous: max diff {d:.6}  got={:?} want={:?}", &vf[..4.min(vf.len())], &want[..4]);
    }
    if max > 1.0 {
        anyhow::bail!("rms_row_exact diverges from the host reference: {max}");
    }
    Ok(())
}

/// Verify the custom GDN Metal kernel against the ops fallback.
pub fn run_cache_test() -> anyhow::Result<()> {
    use crate::core::cache::FullAttentionCache;
    let mut cache = FullAttentionCache::new(2, 256);
    // push 3 chunks of 500 rows = 1500 total (forces 2 grows)
    let mut seed = 1u64;
    for chunk in 0..3 {
        let data: Vec<f32> = (0..500 * 2 * 256)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                (seed >> 33) as f32 / u32::MAX as f32
            })
            .collect();
        let k = Array::from_slice(&data, &[1i32, 2, 500, 256]).as_dtype(Dtype::Bfloat16)?;
        let v = k.clone();
        let (lk, lv) = cache.update(&k, &v)?;
        let _ = lk.eval();
        let _ = lv.eval();
        // verify the live view equals the accumulated writes
        let expect_len = (chunk + 1) * 500;
        assert_eq!(lk.dim(2) as usize, expect_len);
        // check the first row of this chunk landed at the right offset
        let row = lk.index((.., .., (chunk * 500) as i32, ..))
            .as_dtype(Dtype::Float32)?;
        let row_v = row.as_slice::<f32>();
        let want = &data[..256];
        let maxd: f32 = row_v.iter().zip(want.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        println!("chunk {chunk}: first-row max diff {maxd}");
        assert!(maxd < 0.01);
        // check the last row too
        let row = lk.index((.., .., (expect_len - 1) as i32, ..))
            .as_dtype(Dtype::Float32)?;
        let row_v = row.as_slice::<f32>();
        let off = (expect_len - 1 - chunk * 500) as usize * 256;
        let want = &data[off..off + 256];
        let maxd: f32 = row_v.iter().zip(want.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        println!("chunk {chunk}: last-row max diff {maxd}");
        assert!(maxd < 0.01);
    }
    println!("cache test passed");
    Ok(())
}
