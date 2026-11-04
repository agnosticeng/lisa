use super::*;
#[cfg(feature = "glm53-fixtures")]
use lisa_mlx::ops::indexing::IndexOp;

#[test]
fn glm53_mhc_split_normed_matches_separate_path() {
    if !lisa_mlx::device::metal_available() {
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
        fn_w: fn_w.clone(),
        base: Array::from_slice(&base_values, &[24]),
        scale: Array::from_slice(&[0.7f32, 0.5, 0.3], &[3]),
        scale_vals: [0.7, 0.5, 0.3],
        hc_mult: c,
        hidden: h,
        eps: 1e-6,
        sinkhorn_iters: 4,
        rms_eps: 1e-6,
    };
    let state = Array::from_slice(&state_values, &[b as i32, t as i32, c as i32, h as i32]);
    let norm_weight = Array::from_slice(&norm_values, &[h as i32]);

    // Compare the ds4 BF16 row-kernel path with the same artifact weights
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
            &ms.add(&Array::from_f32_as(1e-6, Dtype::Float32))
                .expect("add mHC epsilon")
                .rsqrt()
                .expect("mHC RMS inverse"),
        )
        .expect("normalize mHC state");
    let generic_mix = normed
        .matmul(
            &fn_w
                .as_dtype(Dtype::Float32)
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

    // Cross ds4's eight-row dispatch boundary to cover the tiled GEMM
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
            &ms.add(&Array::from_f32_as(1e-6, Dtype::Float32))
                .expect("add mHC GEMM epsilon")
                .rsqrt()
                .expect("mHC GEMM RMS inverse"),
        )
        .expect("normalize mHC GEMM state");
    let generic_mix = normed
        .matmul(
            &fn_w
                .as_dtype(Dtype::Float32)
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
            &ms.add(&Array::from_f32_as(1e-6, Dtype::Float32))
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

#[cfg(feature = "glm53-fixtures")]
#[test]
fn glm53_mhc_mix_rows_layer21_capture_matches_ds4_fixture() {
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-mhc-mix-rows-ds4] skipping: no Metal device available");
        return;
    }
    const ROWS: usize = 4;
    const STREAMS: usize = 4;
    const HIDDEN: usize = 4096;
    const WIDTH: usize = STREAMS * HIDDEN;
    const STATE_VALUES: usize = ROWS * WIDTH;
    const WEIGHT_VALUES: usize = 24 * WIDTH;
    const WEIGHT_BYTES: usize = WEIGHT_VALUES * std::mem::size_of::<u16>();
    const MIX_VALUES: usize = ROWS * 24;
    const MIX_BYTES: usize = MIX_VALUES * std::mem::size_of::<f32>();

    let norm_fixture =
        &crate::models::glm53::test_fixtures::read("mhc_mix_norm_ds4_layer21_ids1234.f32");
    let projection_fixture =
        &crate::models::glm53::test_fixtures::read("mhc_mix_projection_ds4_layer21_ids1234.bin");
    assert_eq!(norm_fixture.len(), 4 * STATE_VALUES * 4);
    assert_eq!(projection_fixture.len(), 2 * (WEIGHT_BYTES + MIX_BYTES));
    let state_values = norm_fixture
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte float")))
        .collect::<Vec<_>>();
    let mut projection_offset = 0;
    for (half, state_offset) in [("attention", 0), ("FFN", 2 * STATE_VALUES)] {
        let mut weight_values = Vec::with_capacity(WEIGHT_VALUES);
        for bytes in
            projection_fixture[projection_offset..projection_offset + WEIGHT_BYTES].chunks_exact(2)
        {
            let bf16 = u16::from_le_bytes(bytes.try_into().expect("two-byte BF16"));
            weight_values.push(f32::from_bits(u32::from(bf16) << 16));
        }
        projection_offset += WEIGHT_BYTES;
        let fn_w = Array::from_slice(&weight_values, &[24, WIDTH as i32])
            .as_dtype(Dtype::Bfloat16)
            .expect("cast captured mHC projection to BF16");
        let expected = projection_fixture[projection_offset..projection_offset + MIX_BYTES]
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte float")))
            .collect::<Vec<_>>();
        projection_offset += MIX_BYTES;

        let hc = Hc {
            fn_w,
            base: Array::from_slice(&vec![0.0f32; 24], &[24]),
            scale: Array::from_slice(&[1.0f32; 3], &[3]),
            scale_vals: [1.0; 3],
            hc_mult: STREAMS,
            hidden: HIDDEN,
            eps: 1e-6,
            sinkhorn_iters: 20,
            rms_eps: 1e-5,
        };
        let state = Array::from_slice(
            &state_values[state_offset..state_offset + STATE_VALUES],
            &[1, ROWS as i32, STREAMS as i32, HIDDEN as i32],
        );
        let (b, t, c, h, actual) = hc.mix_rows(&state).expect("production mHC mix_rows");
        assert_eq!((b, t, c, h), (1, ROWS, STREAMS, HIDDEN));
        let actual = actual.to_vec1::<f32>().expect("read production mixer rows");
        for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "layer-21 {half} Hc::mix_rows[{index}]: {got} vs DS4 {want}"
            );
        }

        let first_state = state.index((.., 0..1, .., ..));
        let (_, _, _, _, first_actual) = hc
            .mix_rows(&first_state)
            .expect("single-row fused production mHC mix_rows");
        let first_actual = first_actual
            .to_vec1::<f32>()
            .expect("read single-row fused mixer");
        for (index, (&got, &want)) in first_actual.iter().zip(&expected[..24]).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "layer-21 single-row {half} Hc::mix_rows[{index}]: {got} vs DS4 {want}"
            );
        }
    }
}

#[cfg(feature = "glm53-fixtures")]
#[test]
fn glm53_mhc_mix_rows_prefill_layer21_rows9_matches_ds4_fixture() {
    if !lisa_mlx::device::metal_available() {
        eprintln!("[glm53-mhc-mix-rows9-ds4] skipping: no Metal device available");
        return;
    }
    const ROWS: usize = 9;
    const STREAMS: usize = 4;
    const HIDDEN: usize = 4096;
    const WIDTH: usize = STREAMS * HIDDEN;
    const STATE_VALUES: usize = ROWS * WIDTH;
    const STATE_BYTES: usize = STATE_VALUES * std::mem::size_of::<f32>();
    const WEIGHT_VALUES: usize = 24 * WIDTH;
    const WEIGHT_BYTES: usize = WEIGHT_VALUES * std::mem::size_of::<u16>();
    const MIX_VALUES: usize = ROWS * 24;
    const MIX_BYTES: usize = MIX_VALUES * std::mem::size_of::<f32>();

    let projection_fixture =
        &crate::models::glm53::test_fixtures::read("mhc_mix_projection_ds4_layer21_ids1234.bin");
    let fixture = &crate::models::glm53::test_fixtures::read("mhc_mix_rows9_ds4_layer21.bin");
    assert_eq!(projection_fixture.len(), 2 * (WEIGHT_BYTES + 4 * 24 * 4));
    assert_eq!(fixture.len(), STATE_BYTES + MIX_BYTES);
    let mut weight_values = Vec::with_capacity(WEIGHT_VALUES);
    for bytes in projection_fixture[..WEIGHT_BYTES].chunks_exact(2) {
        let bf16 = u16::from_le_bytes(bytes.try_into().expect("two-byte BF16"));
        weight_values.push(f32::from_bits(u32::from(bf16) << 16));
    }
    let fn_w = Array::from_slice(&weight_values, &[24, WIDTH as i32])
        .as_dtype(Dtype::Bfloat16)
        .expect("cast captured mHC projection to BF16");
    let state_values = fixture[..STATE_BYTES]
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte float")))
        .collect::<Vec<_>>();
    let expected = fixture[STATE_BYTES..]
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte float")))
        .collect::<Vec<_>>();
    let state = Array::from_slice(
        &state_values,
        &[1, ROWS as i32, STREAMS as i32, HIDDEN as i32],
    );
    let hc = Hc {
        fn_w,
        base: Array::from_slice(&vec![0.0f32; 24], &[24]),
        scale: Array::from_slice(&[1.0f32; 3], &[3]),
        scale_vals: [1.0; 3],
        hc_mult: STREAMS,
        hidden: HIDDEN,
        eps: 1e-6,
        sinkhorn_iters: 20,
        rms_eps: 1e-5,
    };
    let (b, t, c, h, actual) = hc
        .mix_rows(&state)
        .expect("production mHC prefill mix_rows");
    assert_eq!((b, t, c, h), (1, ROWS, STREAMS, HIDDEN));
    let actual = actual
        .to_vec1::<f32>()
        .expect("read production mHC prefill mixes");
    for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "layer-21 nine-row mHC Hc::mix_rows[{index}]: {got} vs DS4 {want}"
        );
    }
}
