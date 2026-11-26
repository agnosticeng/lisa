#![cfg(test)]
use super::*;

#[test]
fn glm53_mhc_split_normed_matches_separate_path() {
    if !lisa_metal::device::metal_available() {
        eprintln!("[glm53-mhc-normed] skipping: no Metal device available");
        return;
    }
    let (b, t, c, h) = (1usize, 2usize, 4usize, 4096usize);
    let fn_values: Vec<f32> = (0..24 * c * h)
        .map(|i| ((i * 17 % 101) as f32 - 50.0) * 0.0001)
        .collect();
    let fn_w = Array::from_slice(&fn_values, &[24, (c * h) as i32])
        .as_dtype(Dtype::Bfloat16)
        .expect("cast mHC projection to BF16");
    let base_values: Vec<f32> = (0..24).map(|i| ((i as f32) - 11.5) * 0.009).collect();
    let state_values: Vec<f32> = (0..b * t * c * h)
        .map(|i| ((i * 7 % 43) as f32 - 21.0) * 0.002)
        .collect();
    let norm_values: Vec<f32> = (0..h).map(|i| 0.8 + (i % 23) as f32 * 0.007).collect();
    let hc = Hc {
        context: crate::metal_context::execution_context(),
        fn_w: fn_w.clone(),
        base: Array::from_slice(&base_values, &[24]),
        scale: Array::from_slice(&[0.7f32, 0.5, 0.3], &[3]),
        plans: plans(),
        hc_mult: c,
        hidden: h,
        eps: 1e-6,
        sinkhorn_iters: 4,
        rms_eps: 1e-6,
    };
    let state = Array::from_slice(&state_values, &[b as i32, t as i32, c as i32, h as i32]);
    let norm_weight = Array::from_slice(&norm_values, &[h as i32]);

    // Compare the reference BF16 row-kernel path with the same artifact weights
    // promoted to F32 and multiplied through the generic MLX path.
    let flat = state
        .as_dtype(Dtype::Float32)
        .expect("promote mHC state")
        .reshape(&[(b * t) as i32, (c * h) as i32])
        .expect("flatten mHC state");
    let ms = flat
        .multiply(&flat)
        .expect("square mHC state")
        .mean_axis(1, true)
        .expect("mHC RMS mean");
    let normed = flat
        .multiply(
            ms.add(Array::from_f32_as(1e-6, Dtype::Float32))
                .expect("add mHC epsilon")
                .rsqrt()
                .expect("mHC RMS inverse"),
        )
        .expect("normalize mHC state");
    let generic_mix = normed
        .matmul(
            fn_w.as_dtype(Dtype::Float32)
                .expect("promote mHC weights")
                .transpose_axes(&[1, 0])
                .expect("transpose mHC weights"),
        )
        .expect("generic mHC projection")
        .to_vec1::<f32>()
        .expect("read generic mHC projection");
    let (_, _, _, _, row_mix) = hc.mix_rows(&state).expect("BF16 mHC row projection");
    let row_mix = row_mix.to_vec1::<f32>().expect("read BF16 mHC projection");
    for (i, (&got, &want)) in row_mix.iter().zip(&generic_mix).enumerate() {
        assert!((got - want).abs() < 2e-3, "mHC mix {i}: {got} vs {want}");
    }

    // Cross the reference's eight-row dispatch boundary to cover the tiled GEMM
    // route used by larger prompt-prefill chunks in mHC as well.
    let gemm_rows = 9usize;
    let gemm_state_values: Vec<f32> = (0..gemm_rows * c * h)
        .map(|i| ((i * 13 % 59) as f32 - 29.0) * 0.0015)
        .collect();
    let gemm_state = Array::from_slice(
        &gemm_state_values,
        &[gemm_rows as i32, 1, c as i32, h as i32],
    );
    let flat = gemm_state
        .as_dtype(Dtype::Float32)
        .expect("promote mHC GEMM state")
        .reshape(&[gemm_rows as i32, (c * h) as i32])
        .expect("flatten mHC GEMM state");
    let ms = flat
        .multiply(&flat)
        .expect("square mHC GEMM state")
        .mean_axis(1, true)
        .expect("mHC GEMM RMS mean");
    let normed = flat
        .multiply(
            ms.add(Array::from_f32_as(1e-6, Dtype::Float32))
                .expect("add mHC GEMM epsilon")
                .rsqrt()
                .expect("mHC GEMM RMS inverse"),
        )
        .expect("normalize mHC GEMM state");
    let generic_mix = normed
        .matmul(
            fn_w.as_dtype(Dtype::Float32)
                .expect("promote mHC GEMM weights")
                .transpose_axes(&[1, 0])
                .expect("transpose mHC GEMM weights"),
        )
        .expect("generic mHC GEMM projection")
        .to_vec1::<f32>()
        .expect("read generic mHC GEMM projection");
    let (_, _, _, _, gemm_mix) = hc.mix_rows(&gemm_state).expect("BF16 mHC tiled GEMM");
    let gemm_mix = gemm_mix.to_vec1::<f32>().expect("read mHC tiled GEMM");
    for (i, (&got, &want)) in gemm_mix.iter().zip(&generic_mix).enumerate() {
        assert!(
            (got - want).abs() < 2e-3,
            "mHC GEMM mix {i}: {got} vs {want}"
        );
    }

    let (want_xc, want_post, want_comb) = hc.split(&state).expect("reference split");
    let ms = want_xc
        .multiply(&want_xc)
        .expect("square")
        .contiguous()
        .expect("contiguous")
        .mean_axis(2, true)
        .expect("mean");
    let want_normed = want_xc
        .multiply(
            ms.add(Array::from_f32_as(1e-6, Dtype::Float32))
                .expect("epsilon")
                .rsqrt()
                .expect("rsqrt"),
        )
        .expect("normalize")
        .multiply(&norm_weight)
        .expect("weight");
    let (got_xc, got_normed, got_post, got_comb) = hc
        .split_normed(&state, &norm_weight, 1e-6)
        .expect("fused mHC path")
        .expect("specialized HC=4 kernel should dispatch");
    assert_eq!(got_xc.shape(), [b as i32, t as i32, h as i32]);
    assert_eq!(got_normed.shape(), [b as i32, t as i32, h as i32]);
    assert_eq!(got_post.shape(), [b as i32, t as i32, c as i32]);
    assert_eq!(got_comb.shape(), [b as i32, t as i32, c as i32, c as i32]);

    for (label, got, want) in [
        ("collapsed", got_xc, want_xc),
        ("normalized", got_normed, want_normed),
        ("post", got_post, want_post),
        ("comb", got_comb, want_comb),
    ] {
        let got = got.to_vec1::<f32>().expect("read fused output");
        let want = want.to_vec1::<f32>().expect("read reference output");
        assert_eq!(got.len(), want.len(), "{label} length");
        for (i, (&a, &b)) in got.iter().zip(&want).enumerate() {
            let tol = if label == "normalized" { 3e-5 } else { 4e-6 };
            assert!((a - b).abs() <= tol, "{label}[{i}]: {a} vs {b}");
        }
    }
}

#[path = "glm53_hyper_oracle.rs"]
mod original;
#[test]
fn model_hyper_plans_preserve_original_bits_and_dispatch_boundaries() -> Result<()> {
    use crate::bindings::deepseek41_moe::tests::{counts, difference, projection_check};
    let stream = lisa_metal::Stream::gpu();
    let values = |shape: &[i32], seed: usize| {
        let n = shape.iter().map(|&n| n as usize).product();
        Array::from_slice(
            &(0..n)
                .map(|i| ((i * 17 + seed) % 103) as f32 / 113. - 0.4)
                .collect::<Vec<_>>(),
            shape,
        )
    };
    for h in [256usize, 4096] {
        for dtype in [Dtype::Bfloat16, Dtype::Float32, Dtype::Float16] {
            let matrix = values(&[24, 4 * h as i32], 11).as_dtype(dtype)?;
            let base = values(&[24], 37);
            let scale = Array::from_slice(&[0.7f32, 0.5, 0.3], &[3]);
            let old = original::bound(matrix.clone(), base.clone(), scale.clone(), h);
            let new = Hc {
                context: crate::metal_context::execution_context(),
                fn_w: matrix,
                base,
                scale,
                hc_mult: 4,
                hidden: h,
                eps: 1e-6,
                sinkhorn_iters: 4,
                rms_eps: 1e-6,
                plans: plans(),
            };
            for (b, t) in [(1, 1), (1, 8), (1, 9), (2, 2)] {
                let state = values(&[b, t, 4, h as i32], 53);
                stream.runtime().commands.flush_and_wait()?;
                let before = counts();
                let a = old.split(&state)?;
                stream.runtime().commands.flush_and_wait()?;
                let middle = counts();
                let z = new.split(&state)?;
                stream.runtime().commands.flush_and_wait()?;
                let after = counts();
                for (a, z) in [(&a.0, &z.0), (&a.1, &z.1), (&a.2, &z.2)] {
                    assert_eq!(a.shape(), z.shape());
                    assert_eq!(
                        a.to_vec1::<u32>()?,
                        z.to_vec1::<u32>()?,
                        "split h={h} b={b} t={t} dtype={dtype:?}"
                    );
                }
                if !after.is_empty() {
                    assert_eq!(difference(&before, &middle), difference(&middle, &after));
                }
                projection_check(
                    &stream,
                    || old.expand(&a.0, &state, &a.1, &a.2),
                    || new.expand(&z.0, &state, &z.1, &z.2),
                )?;
                if h == 4096 {
                    let norm = values(&[h as i32], 23);
                    let before = counts();
                    let a = old.split_normed_with_mixes(&state, &norm, 1e-5)?.unwrap();
                    stream.runtime().commands.flush_and_wait()?;
                    let middle = counts();
                    let z = new.split_normed_with_mixes(&state, &norm, 1e-5)?.unwrap();
                    stream.runtime().commands.flush_and_wait()?;
                    let after = counts();
                    for (a, z) in [
                        (&a.0, &z.0),
                        (&a.1, &z.1),
                        (&a.2, &z.2),
                        (&a.3, &z.3),
                        (&a.4, &z.4),
                        (&a.5, &z.5),
                    ] {
                        assert_eq!(a.shape(), z.shape());
                        assert_eq!(a.to_vec1::<u32>()?, z.to_vec1::<u32>()?);
                    }
                    if !after.is_empty() {
                        assert_eq!(difference(&before, &middle), difference(&middle, &after));
                    }
                } else {
                    assert!(
                        new.split_normed(&state, &values(&[h as i32], 23), 1e-5)?
                            .is_none()
                    );
                }
            }
        }
    }
    Ok(())
}
