use super::linear::*;
use super::*;
use crate::ops::indexing::IndexOp;
use crate::{Array, Stream};

#[test]
fn moe_expert_route_map_matches_token_major_oracle() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-moe-route-map] skipping: no Metal device available");
        return;
    }

    const N_TOKENS: usize = 96;
    const TOP_K: usize = 8;
    const N_EXPERTS: usize = 288;
    const TILE: usize = 32;
    let stream = Stream::thread_local_or_default();
    let selected_values = (0..N_TOKENS)
        .flat_map(|token| (0..TOP_K).map(move |rank| ((token * 17 + rank * 31) % N_EXPERTS) as i32))
        .collect::<Vec<_>>();
    let selected = Array::from_slice(&selected_values, &[(N_TOKENS * TOP_K) as i32]);

    let (route_ids, counts, work_count, work_items) =
        moe_expert_route_map(&selected, N_TOKENS, TOP_K, N_EXPERTS, TILE, &stream)
            .expect("route-map kernel should compile and dispatch");
    route_ids.eval().expect("route IDs should evaluate");
    counts.eval().expect("route counts should evaluate");
    work_count.eval().expect("work count should evaluate");
    work_items.eval().expect("work items should evaluate");

    let actual_ids = route_ids.t.to_vec::<i32>().expect("read route IDs");
    let actual_counts = counts.t.to_vec::<i32>().expect("read route counts");
    let actual_work_count = work_count.t.to_vec::<i32>().expect("read work count")[0];
    let actual_work_items = work_items.t.to_vec::<i32>().expect("read work items");
    let mut expected_work_items = Vec::new();

    for expert in 0..N_EXPERTS {
        let expected_routes = selected_values
            .iter()
            .enumerate()
            .filter_map(|(slot, &selected_expert)| {
                (selected_expert == expert as i32).then_some(slot as i32)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            actual_counts[expert],
            expected_routes.len() as i32,
            "expert {expert} count; actual counts prefix: {:?}; route prefix: {:?}",
            &actual_counts[..actual_counts.len().min(16)],
            &actual_ids[..actual_ids.len().min(16)]
        );
        let route_start = expert * N_TOKENS;
        assert_eq!(
            &actual_ids[route_start..route_start + expected_routes.len()],
            expected_routes.as_slice(),
            "expert {expert} route list differs"
        );
        for row_start in (0..expected_routes.len()).step_by(TILE) {
            expected_work_items.push(expert as i32);
            expected_work_items.push(row_start as i32);
        }
    }

    assert_eq!(actual_work_count as usize, expected_work_items.len() / 2);
    assert_eq!(
        &actual_work_items[..expected_work_items.len()],
        expected_work_items.as_slice()
    );
    let sorted_routes = actual_ids
        .as_chunks::<N_TOKENS>()
        .0
        .iter()
        .zip(actual_counts.iter())
        .flat_map(|(routes, &count)| routes[..count as usize].iter().copied())
        .collect::<Vec<_>>();
    let mut expected_all = (0..selected_values.len() as i32).collect::<Vec<_>>();
    let mut actual_all = sorted_routes;
    expected_all.sort_unstable();
    actual_all.sort_unstable();
    assert_eq!(
        actual_all, expected_all,
        "route map must include every slot once"
    );
}

#[test]
fn affine_q2_mpp_matches_half_weight_oracle() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-q2-mpp] skipping: no Metal device available");
        return;
    }

    const M: usize = 32;
    const K: usize = 512;
    const N: usize = 64;
    let mut packed = vec![0u8; N * K / 4];
    for n in 0..N {
        for k in 0..K {
            let code = ((n * 3 + k * 7 + k / 128) % 4) as u8;
            packed[n * (K / 4) + k / 4] |= code << ((k % 4) * 2);
        }
    }
    let scales = (0..N * (K / 128))
        .map(|i| 0.0007 + (i % 13) as f32 * 0.00003)
        .collect::<Vec<_>>();
    let biases = (0..N * (K / 128))
        .map(|i| 0.0001 + (i % 7) as f32 * 0.00002)
        .collect::<Vec<_>>();
    let x_values = (0..M * K)
        .map(|i| 0.0009 + (i * 17 % 251) as f32 * 0.0000007)
        .collect::<Vec<_>>();
    let x = Array::from_slice(&x_values, &[M as i32, K as i32]);
    let w = Array::from_slice(&packed, &[N as i32, (K / 4) as i32]);
    let scales_array = Array::from_slice(&scales, &[N as i32, (K / 128) as i32]);
    let biases_array = Array::from_slice(&biases, &[N as i32, (K / 128) as i32]);
    let actual = crate::jit::affine_q2_mpp_prefill(
        x.t.device(),
        &x.t,
        &w.t,
        &scales_array.t,
        &biases_array.t,
    )
    .expect("Q2 MPP tile should compile and dispatch")
    .to_vec1::<f32>()
    .expect("read Q2 MPP output");

    let mut expected = vec![0.0f32; M * N];
    for m in 0..M {
        for n in 0..N {
            let mut acc = 0.0f32;
            for k in 0..K {
                let code = (packed[n * (K / 4) + k / 4] >> ((k % 4) * 2)) & 0x03;
                let group = n * (K / 128) + k / 128;
                let weight = f32::from(half::f16::from_f32(
                    f32::from(code) * scales[group] + biases[group],
                ));
                acc += x_values[m * K + k] * weight;
            }
            expected[m * N + n] = acc;
        }
    }

    let (max_abs, rel_l2) = actual.iter().zip(&expected).fold(
        (0.0f32, (0.0f64, 0.0f64)),
        |(max_abs, (error_sq, reference_sq)), (&got, &want)| {
            let delta = got - want;
            (
                max_abs.max(delta.abs()),
                (
                    error_sq + f64::from(delta) * f64::from(delta),
                    reference_sq + f64::from(want) * f64::from(want),
                ),
            )
        },
    );
    let rel_l2 = (rel_l2.0 / rel_l2.1.max(f64::MIN_POSITIVE)).sqrt();
    eprintln!("[glm53-q2-mpp] max_abs={max_abs:e} rel_l2={rel_l2:e}");
    assert!(max_abs < 5.0e-7, "Q2 MPP max abs error {max_abs:e}");
    assert!(rel_l2 < 5.0e-4, "Q2 MPP relative L2 error {rel_l2:e}");
}

#[test]
fn router_mv_small_matches_matmul() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-router-small] skipping: no Metal device available");
        return;
    }
    let (out_dim, in_dim) = (19usize, 4096usize);
    let stream = Stream::thread_local_or_default();
    for tokens in 2..=8 {
        let weights: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.001)
            .collect();
        let x: Vec<f32> = (0..tokens * in_dim)
            .map(|i| ((i * 29 % 97) as f32 - 48.0) * 0.002)
            .collect();
        let weights = Array::from_slice(&weights, &[out_dim as i32, in_dim as i32]);
        let x = Array::from_slice(&x, &[tokens as i32, in_dim as i32]);
        let actual = router_matvec_f32(&weights, &x, &stream)
            .expect("DS4 small-batch router matvec should dispatch")
            .to_vec1::<f32>()
            .expect("router result readback");
        let transposed = weights.transpose_axes(&[1, 0]).expect("transpose weights");
        let expected = x
            .matmul(&transposed)
            .expect("reference router matmul")
            .to_vec1::<f32>()
            .expect("reference readback");
        let max_abs = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[glm53-router-small] tokens={tokens} max_abs={max_abs}");
        assert!(max_abs < 1e-4, "tokens={tokens} router max error {max_abs}");
    }
}

#[test]
fn router_mv_plain_matches_matmul() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-router-plain] skipping: no Metal device available");
        return;
    }
    let (out_dim, in_dim) = (19usize, 4096usize);
    let weights: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| ((i * 19 % 127) as f32 - 63.0) * 0.0008)
        .collect();
    let weights = Array::from_slice(&weights, &[out_dim as i32, in_dim as i32]);
    let stream = Stream::thread_local_or_default();
    for tokens in [1usize, 9] {
        let x: Vec<f32> = (0..tokens * in_dim)
            .map(|i| ((i * 31 % 113) as f32 - 56.0) * 0.0015)
            .collect();
        let x = Array::from_slice(&x, &[tokens as i32, in_dim as i32]);
        let actual = router_matvec_f32(&weights, &x, &stream)
            .expect("DS4 plain router matvec should dispatch")
            .to_vec1::<f32>()
            .expect("router result readback");
        let transposed = weights.transpose_axes(&[1, 0]).expect("transpose weights");
        let expected = x
            .matmul(&transposed)
            .expect("reference router matmul")
            .to_vec1::<f32>()
            .expect("reference readback");
        let max_abs = actual
            .iter()
            .zip(&expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[glm53-router-plain] tokens={tokens} max_abs={max_abs}");
        assert!(max_abs < 1e-4, "tokens={tokens} router max error {max_abs}");
    }
}

fn bf16_row_reference(weights: &[f32], x: &[f32], in_dim: usize) -> Vec<f32> {
    let out_dim = weights.len() / in_dim;
    let mut out = vec![0.0f32; out_dim];
    for row in 0..out_dim {
        let w = &weights[row * in_dim..(row + 1) * in_dim];
        let mut lanes = [0.0f32; 32];
        for lane in 0..32 {
            let mut k = lane;
            while k + 224 < in_dim {
                for offset in (0..256).step_by(32) {
                    lanes[lane] = w[k + offset].mul_add(x[k + offset], lanes[lane]);
                }
                k += 256;
            }
            while k < in_dim {
                lanes[lane] = w[k].mul_add(x[k], lanes[lane]);
                k += 32;
            }
        }
        out[row] = lanes.into_iter().sum();
    }
    out
}

fn assert_bf16_row_close(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label} output length");
    for (i, (&a, &b)) in got.iter().zip(want).enumerate() {
        assert!((a - b).abs() <= 2e-4, "{label}[{i}]: {a} vs {b}");
    }
}

fn bf16_gemv_rows2_weights(len: usize) -> Vec<half::bf16> {
    (0..len)
        .map(|i| {
            let bits = (i as u32)
                .wrapping_mul(0x9e37_79b9)
                .wrapping_add(0x85eb_ca6b);
            let bits = bits ^ (bits >> 16);
            let sign = ((bits >> 16) & 0x8000) as u16;
            let exponent = (119 + ((bits >> 8) & 7)) as u16;
            half::bf16::from_bits(sign | (exponent << 7) | (bits & 127) as u16)
        })
        .collect()
}

fn bf16_gemv_rows2_inputs(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let bits = (i as u32)
                .wrapping_mul(0x7feb_352d)
                .wrapping_add(0x846c_a68b);
            let bits = bits ^ (bits >> 15);
            let exponent = 119 + ((bits >> 23) & 7);
            f32::from_bits((bits & 0x807f_ffff) | (exponent << 23))
        })
        .collect()
}

fn assert_bf16_rows2_matches_single_rows(
    label: &str,
    weights: &Array,
    x: &Array,
    reference_weights: &Array,
    reference_x: &[f32],
    stream: &Stream,
) {
    let (out_dim, in_dim) = (weights.dim(0) as usize, weights.dim(1) as usize);
    assert_eq!(x.shape(), &[2, in_dim as i32]);
    assert_eq!(reference_x.len(), 2 * in_dim);
    let actual = bf16_gemv(weights, x, stream).expect("two-row BF16 GEMV should dispatch");
    assert_eq!(actual.shape(), &[2, out_dim as i32]);
    let actual = actual.to_vec1::<f32>().expect("read two-row BF16 GEMV");
    for row in 0..2 {
        // A one-row call retains the original reference GEMV kernel, so the
        // reference never enters the new shared-weight two-row dispatch.
        let reference_row = Array::from_slice(
            &reference_x[row * in_dim..(row + 1) * in_dim],
            &[1, in_dim as i32],
        );
        let expected = bf16_gemv(reference_weights, &reference_row, stream)
            .expect("original one-row BF16 GEMV should dispatch")
            .to_vec1::<f32>()
            .expect("read original one-row BF16 GEMV");
        assert_eq!(expected.len(), out_dim);
        for (channel, (&got, &want)) in actual[row * out_dim..(row + 1) * out_dim]
            .iter()
            .zip(&expected)
            .enumerate()
        {
            assert!(
                got.is_finite() && want.is_finite(),
                "{label}: nonfinite output"
            );
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{label}: row={row}, channel={channel}, two-row={got}, one-row={want}"
            );
        }
    }
}

#[test]
fn bf16_rows2_gemv_matches_single_rows_bitwise_at_tails() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-bf16-rows2-tails] skipping: no Metal device available");
        return;
    }
    let stream = Stream::thread_local_or_default();
    // Include partial SIMD groups, the eight-FMA unroll boundary, and
    // partial output threadgroups on both supported NSG schedules.
    for (out_dim, in_dim) in [
        (64, 128),
        (128, 64),
        (1, 1),
        (7, 31),
        (9, 32),
        (15, 33),
        (17, 224),
        (31, 225),
        (33, 255),
        (37, 256),
        (65, 257),
        (67, 323),
        (1023, 257),
        (1024, 257),
        (1037, 323),
    ] {
        let weights = Array::from_slice(
            &bf16_gemv_rows2_weights(out_dim * in_dim),
            &[out_dim as i32, in_dim as i32],
        );
        let input = bf16_gemv_rows2_inputs(2 * in_dim);
        let x = Array::from_slice(&input, &[2, in_dim as i32]);
        assert_bf16_rows2_matches_single_rows(
            &format!("N={out_dim}, K={in_dim}"),
            &weights,
            &x,
            &weights,
            &input,
            &stream,
        );
    }
}

#[test]
fn bf16_rows2_gemv_matches_single_rows_bitwise_at_kda_shapes() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-bf16-rows2-kda] skipping: no Metal device available");
        return;
    }
    let stream = Stream::thread_local_or_default();
    for (out_dim, in_dim) in [(8192, 4096), (4096, 8192)] {
        // Import BF16 directly: each matrix is 64 MiB, with no full F32
        // staging/cast allocation. Finish readback before the next matrix.
        let weights = Array::from_slice(
            &bf16_gemv_rows2_weights(out_dim * in_dim),
            &[out_dim as i32, in_dim as i32],
        );
        let input = bf16_gemv_rows2_inputs(2 * in_dim);
        let x = Array::from_slice(&input, &[2, in_dim as i32]);
        assert_bf16_rows2_matches_single_rows(
            &format!("KDA N={out_dim}, K={in_dim}"),
            &weights,
            &x,
            &weights,
            &input,
            &stream,
        );
    }
}

#[test]
fn bf16_rows2_gemv_matches_single_rows_bitwise_with_views() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-bf16-rows2-views] skipping: no Metal device available");
        return;
    }
    let stream = Stream::thread_local_or_default();
    let (out_dim, in_dim) = (1037usize, 323usize);
    for padding in [0usize, 3] {
        let stride = in_dim + padding;
        let column = usize::from(padding != 0);
        let weight_values = bf16_gemv_rows2_weights((out_dim + 2) * stride);
        let input_values = bf16_gemv_rows2_inputs(4 * stride);
        let weight_storage =
            Array::from_slice(&weight_values, &[(out_dim + 2) as i32, stride as i32]);
        let input_storage = Array::from_slice(&input_values, &[4, stride as i32]);
        let weights = Array::new(
            weight_storage
                .t
                .narrow(0, 1, out_dim)
                .unwrap()
                .narrow(1, column, in_dim)
                .unwrap(),
        );
        let x = Array::new(
            input_storage
                .t
                .narrow(0, 1, 2)
                .unwrap()
                .narrow(1, column, in_dim)
                .unwrap(),
        );
        for view in [&weights, &x] {
            assert!(
                view.t.layout().offset > 0,
                "the view must exercise a binding offset"
            );
            assert_eq!(view.t.layout().is_contiguous(), padding == 0);
        }
        // Pack the reference from host values independently of the view's
        // strides and offset; the wrapper must materialize padded rows.
        let packed_weights: Vec<_> = (1..=out_dim)
            .flat_map(|row| {
                weight_values[row * stride + column..row * stride + column + in_dim]
                    .iter()
                    .copied()
            })
            .collect();
        let packed_input: Vec<_> = (1..=2)
            .flat_map(|row| {
                input_values[row * stride + column..row * stride + column + in_dim]
                    .iter()
                    .copied()
            })
            .collect();
        let reference_weights =
            Array::from_slice(&packed_weights, &[out_dim as i32, in_dim as i32]);
        assert_bf16_rows2_matches_single_rows(
            &format!("offset view, row padding={padding}"),
            &weights,
            &x,
            &reference_weights,
            &packed_input,
            &stream,
        );
    }
}

#[test]
fn bf16_projection_dispatch_covers_all_rows() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-bf16-projection] skipping: no Metal device available");
        return;
    }
    let (in_dim, out_dim, rows) = (323usize, 9usize, 8usize);
    let stream = Stream::thread_local_or_default();
    let make_weights = |salt: usize| {
        let values: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| (((i * 19 + salt * 23) % 127) as f32 - 63.0) * 0.001)
            .collect();
        let weights = Array::from_slice(&values, &[out_dim as i32, in_dim as i32])
            .as_dtype(Dtype::Bfloat16)
            .expect("cast weights to BF16");
        let rounded = weights
            .as_dtype(Dtype::Float32)
            .expect("promote BF16 weights")
            .to_vec1::<f32>()
            .expect("read rounded weights");
        (weights, rounded)
    };
    let x_values: Vec<f32> = (0..rows * in_dim)
        .map(|i| (((i * 11 + 3) % 97) as f32 - 48.0) * 0.002)
        .collect();
    let x_batch = Array::from_slice(&x_values, &[rows as i32, in_dim as i32]);
    let (gemv_w, gemv_w_rounded) = make_weights(0);
    let gemv =
        bf16_gemv(&gemv_w, &x_batch, &stream).expect("BF16 GEMV should dispatch for every row");
    let gemv_out = gemv.to_vec1::<f32>().expect("read GEMV output");
    let mut gemv_want = Vec::with_capacity(rows * out_dim);
    for row in 0..rows {
        gemv_want.extend(bf16_row_reference(
            &gemv_w_rounded,
            &x_values[row * in_dim..(row + 1) * in_dim],
            in_dim,
        ));
    }
    assert_bf16_row_close("GEMV", &gemv_out, &gemv_want);

    let x_decode = Array::from_slice(&x_values[..in_dim], &[1, in_dim as i32]);
    let (q_w, q_ref) = make_weights(1);
    let (k_w, k_ref) = make_weights(2);
    let (v_w, v_ref) = make_weights(3);
    if bf16_projection_nsg(&x_decode) == 8 {
        assert!(bf16_qkv(&q_w, &k_w, &v_w, &x_decode, &stream).is_none());
    }
    let (q, k, v) = bf16_qkv_inner(&q_w, &k_w, &v_w, &x_decode, &stream)
        .expect("fused BF16 qkv should dispatch for every output row");
    let x_row = &x_values[..in_dim];
    for (name, actual, weight) in [("Q", q, q_ref), ("K", k, k_ref), ("V", v, v_ref)] {
        let actual = actual.to_vec1::<f32>().expect("read qkv output");
        let expected = bf16_row_reference(&weight, x_row, in_dim);
        assert_bf16_row_close(name, &actual, &expected);
    }
}

#[test]
fn bf16_fused_kda_inputs_match_six_gemvs_bitwise() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-bf16-kda-project] skipping: no Metal device available");
        return;
    }
    // Exercise multiple 512-wide chunks in the fused KDA GEMV schedule.
    let (in_dim, qkv_dim, low_dim, heads) = (1024usize, 64usize, 32usize, 16usize);
    let stream = Stream::thread_local_or_default();
    let make_weights = |rows: usize, salt: usize| {
        let values: Vec<f32> = (0..rows * in_dim)
            .map(|i| (((i * 29 + salt * 17) % 127) as f32 - 63.0) * 0.003)
            .collect();
        Array::from_slice(&values, &[rows as i32, in_dim as i32])
            .as_dtype(Dtype::Bfloat16)
            .expect("cast KDA weights to BF16")
    };
    let q = make_weights(qkv_dim, 1);
    let k = make_weights(qkv_dim, 2);
    let v = make_weights(qkv_dim, 3);
    let fa = make_weights(low_dim, 4);
    let ga = make_weights(low_dim, 5);
    let beta = make_weights(heads, 6);
    for rows in [1usize, 2, 8] {
        let x_values: Vec<f32> = (0..rows * in_dim)
            .map(|i| (((i * 37 + 11) % 101) as f32 - 50.0) * 0.002)
            .collect();
        let x = Array::from_slice(&x_values, &[rows as i32, in_dim as i32]);
        let mut fused_schedules = Vec::new();
        for nsg_override in [None, Some(4), Some(16)] {
            let fused_start = crate::runtime_dispatch_count();
            let (packed, fused_fa, fused_ga) =
                bf16_kda_project_with_nsg(&q, &k, &v, &fa, &ga, &beta, &x, nsg_override, &stream)
                    .expect("fused KDA projection should dispatch");
            assert_eq!(
                crate::runtime_dispatch_count() - fused_start,
                1,
                "rows={rows}, nsg={nsg_override:?}: fused KDA inputs should use one dispatch"
            );
            fused_schedules.push((
                nsg_override,
                packed.to_vec1::<f32>().expect("read packed projections"),
                fused_fa.to_vec1::<f32>().expect("read fused f_a"),
                fused_ga.to_vec1::<f32>().expect("read fused g_a"),
            ));
        }
        let staged_start = crate::runtime_dispatch_count();
        let references = [&q, &k, &v, &beta].map(|weights| {
            bf16_gemv(weights, &x, &stream)
                .expect("reference BF16 GEMV should dispatch")
                .to_vec1::<f32>()
                .expect("read reference GEMV")
        });
        let mut expected = Vec::with_capacity(rows * (3 * qkv_dim + heads));
        // Q, K, and V have qkv_dim rows; beta has its own smaller width.
        for row in 0..rows {
            for projection in &references[..3] {
                expected.extend_from_slice(&projection[row * qkv_dim..(row + 1) * qkv_dim]);
            }
            expected.extend_from_slice(&references[3][row * heads..(row + 1) * heads]);
        }
        let expected_fa = bf16_gemv(&fa, &x, &stream)
            .expect("reference f_a GEMV should dispatch")
            .to_vec1::<f32>()
            .expect("read reference f_a");
        let expected_ga = bf16_gemv(&ga, &x, &stream)
            .expect("reference g_a GEMV should dispatch")
            .to_vec1::<f32>()
            .expect("read reference g_a");
        assert_eq!(
            crate::runtime_dispatch_count() - staged_start,
            6,
            "rows={rows}: staged KDA inputs should use six dispatches"
        );
        for (nsg_override, packed, fused_fa, fused_ga) in fused_schedules {
            assert_eq!(
                packed.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "rows={rows}, nsg={nsg_override:?}: fused q/k/v/beta must preserve separate DS4 GEMV bits"
            );
            assert_eq!(
                fused_fa.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected_fa.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "rows={rows}, nsg={nsg_override:?}: fused f_a must preserve separate DS4 GEMV bits"
            );
            assert_eq!(
                fused_ga.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected_ga.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "rows={rows}, nsg={nsg_override:?}: fused g_a must preserve separate DS4 GEMV bits"
            );
        }
    }
}

#[test]
fn bf16_pair_gemv_matches_separate_ds4_rows_bitwise() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-bf16-pair-gemv] skipping: no Metal device available");
        return;
    }
    let (in_dim_a, out_dim_a, in_dim_b, out_dim_b) = (323usize, 37usize, 197usize, 19usize);
    let stream = Stream::thread_local_or_default();
    let make_weights = |salt: usize, out_dim: usize, in_dim: usize| {
        let values: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| (((i * 19 + salt * 23) % 127) as f32 - 63.0) * 0.001)
            .collect();
        Array::from_slice(&values, &[out_dim as i32, in_dim as i32])
            .as_dtype(Dtype::Bfloat16)
            .expect("cast paired GEMV weights to BF16")
    };
    let weights_a = make_weights(3, out_dim_a, in_dim_a);
    let weights_b = make_weights(7, out_dim_b, in_dim_b);
    for rows in [1usize, 2, 8] {
        let values_a: Vec<f32> = (0..rows * in_dim_a)
            .map(|i| (((i * 11 + 5) % 97) as f32 - 48.0) * 0.002)
            .collect();
        let values_b: Vec<f32> = (0..rows * in_dim_b)
            .map(|i| (((i * 13 + 9) % 103) as f32 - 51.0) * 0.0015)
            .collect();
        let x_a = Array::from_slice(&values_a, &[rows as i32, in_dim_a as i32]);
        let x_b = Array::from_slice(&values_b, &[rows as i32, in_dim_b as i32]);
        let before = crate::runtime_dispatch_count();
        let (paired_a, paired_b) = bf16_pair_gemv(&weights_a, &x_a, &weights_b, &x_b, &stream)
            .expect("paired BF16 GEMV should dispatch");
        assert_eq!(
            crate::runtime_dispatch_count() - before,
            1,
            "rows={rows}: paired projections should use one dispatch"
        );
        let separate_a = bf16_gemv(&weights_a, &x_a, &stream)
            .expect("reference A GEMV should dispatch")
            .to_vec1::<f32>()
            .expect("read reference A GEMV");
        let separate_b = bf16_gemv(&weights_b, &x_b, &stream)
            .expect("reference B GEMV should dispatch")
            .to_vec1::<f32>()
            .expect("read reference B GEMV");
        let paired_a = paired_a.to_vec1::<f32>().expect("read paired A GEMV");
        let paired_b = paired_b.to_vec1::<f32>().expect("read paired B GEMV");
        assert_eq!(
            paired_a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            separate_a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "rows={rows}: paired f_b projection changed DS4 GEMV bits"
        );
        assert_eq!(
            paired_b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            separate_b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "rows={rows}: paired g_b projection changed DS4 GEMV bits"
        );
    }
}

#[test]
fn bf16_tiled_gemm_matches_ds4_operand_staging() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-bf16-gemm] skipping: no Metal device available");
        return;
    }
    // Cross both the 64-feature and 32-row tile edges: the first tile
    // takes the reference's direct store path and the remaining tiles exercise its
    // masked output copy.
    let (in_dim, out_dim, rows) = (323usize, 65usize, 33usize);
    let weight_values: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| (((i * 19 + 7) % 127) as f32 - 63.0) * 0.001)
        .collect();
    let weights = Array::from_slice(&weight_values, &[out_dim as i32, in_dim as i32])
        .as_dtype(Dtype::Bfloat16)
        .expect("cast GEMM weights to BF16");
    let activation_values: Vec<f32> = (0..rows * in_dim)
        .map(|i| (((i * 11 + 3) % 97) as f32 - 48.0) * 0.002)
        .collect();
    let x = Array::from_slice(&activation_values, &[rows as i32, in_dim as i32]);
    let stream = Stream::thread_local_or_default();
    let result = bf16_gemm(&weights, &x, &stream)
        .expect("tiled BF16 GEMM should dispatch for more than eight rows");
    let actual = result.to_vec1::<f32>().expect("read GEMM output");
    let rounded_weights = weights
        .as_dtype(Dtype::Float32)
        .expect("promote rounded BF16 weights")
        .to_vec1::<f32>()
        .expect("read rounded BF16 weights");

    // The reference stages both BF16 weights and F32 activations through F16 before
    // the F32-accumulating simdgroup matrix multiply.
    for row in 0..rows {
        for col in 0..out_dim {
            let mut expected = 0.0f32;
            for k in 0..in_dim {
                let w = half::f16::from_f32(rounded_weights[col * in_dim + k]).to_f32();
                let a = half::f16::from_f32(activation_values[row * in_dim + k]).to_f32();
                expected = w.mul_add(a, expected);
            }
            let got = actual[row * out_dim + col];
            assert!(
                (got - expected).abs() <= 2e-5,
                "GEMM[{row},{col}]: {got} vs {expected}"
            );
        }
    }
}

#[test]
fn bf16_tiled_gemm_reuses_long_k_tiles_safely() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-bf16-gemm] skipping: no Metal device available");
        return;
    }
    // Exercise 128 shared-memory tile reuses, the schedule that exposed a
    // missing threadgroup barrier in the real 4,096-wide KDA projections.
    let (in_dim, out_dim, rows) = (4096usize, 65usize, 33usize);
    let weight_values: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| (((i * 19 + 7) % 127) as f32 - 63.0) * 0.001)
        .collect();
    let weights = Array::from_slice(&weight_values, &[out_dim as i32, in_dim as i32])
        .as_dtype(Dtype::Bfloat16)
        .expect("cast long-K GEMM weights to BF16");
    let activation_values: Vec<f32> = (0..rows * in_dim)
        .map(|i| (((i * 11 + 3) % 97) as f32 - 48.0) * 0.002)
        .collect();
    let x = Array::from_slice(&activation_values, &[rows as i32, in_dim as i32]);
    let stream = Stream::thread_local_or_default();
    let result = bf16_gemm(&weights, &x, &stream).expect("long-K tiled BF16 GEMM should dispatch");
    let actual = result.to_vec1::<f32>().expect("read long-K GEMM output");
    let rounded_weights = weights
        .as_dtype(Dtype::Float32)
        .expect("promote rounded long-K BF16 weights")
        .to_vec1::<f32>()
        .expect("read rounded long-K BF16 weights");

    for row in 0..rows {
        for col in 0..out_dim {
            let mut expected = 0.0f32;
            for k in 0..in_dim {
                let w = half::f16::from_f32(rounded_weights[col * in_dim + k]).to_f32();
                let a = half::f16::from_f32(activation_values[row * in_dim + k]).to_f32();
                expected = w.mul_add(a, expected);
            }
            let got = actual[row * out_dim + col];
            assert!(
                (got - expected).abs() <= 2e-5,
                "long-K GEMM[{row},{col}]: {got} vs {expected}"
            );
        }
    }
}

#[test]
fn router_selection_matches_reference_and_compiles() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-router] skipping: no Metal device available");
        return;
    }
    let (rows, experts, top_k) = (2usize, 7usize, 3usize);
    let logits_values = vec![
        -1.2f32, 0.4, 2.1, -0.3, 1.6, 0.1, -2.4, 0.8, -0.7, 0.2, 1.4, -1.1, 0.5, 2.3,
    ];
    let bias_values = vec![0.01f32, -0.03, 0.02, 0.05, -0.04, 0.0, 0.025];
    let logits = Array::from_slice(&logits_values, &[rows as i32, experts as i32]);
    let bias = Array::from_slice(&bias_values, &[experts as i32]);
    let stream = Stream::thread_local_or_default();
    let (selected, weights) = router_select_fused(&logits, &bias, top_k, 2.5, &stream)
        .expect("GLM router kernel should compile and dispatch");
    let selected = selected.to_vec1::<i32>().expect("read selected experts");
    let weights = weights.to_vec1::<f32>().expect("read router weights");

    for row in 0..rows {
        let probs: Vec<f32> = logits_values[row * experts..(row + 1) * experts]
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        let mut order: Vec<usize> = (0..experts).collect();
        order.sort_by(|&a, &b| {
            (probs[b] + bias_values[b])
                .total_cmp(&(probs[a] + bias_values[a]))
                .then_with(|| a.cmp(&b))
        });
        let expected = &order[..top_k];
        for rank in 0..top_k {
            assert_eq!(selected[row * top_k + rank], expected[rank] as i32);
        }
        let denom = expected
            .iter()
            .map(|&expert| probs[expert])
            .sum::<f32>()
            .max(6.103515625e-5);
        for rank in 0..top_k {
            let want = probs[expected[rank]] / denom * 2.5;
            let got = weights[row * top_k + rank];
            assert!(
                (got - want).abs() <= 2e-5,
                "row {row} rank {rank}: {got} vs {want}"
            );
        }
    }
}

#[test]
fn moe_sum8_matches_ds4_slot_order() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-moe-sum8] skipping: no Metal device available");
        return;
    }
    let (tokens, width) = (3usize, 513usize);
    let values: Vec<f32> = (0..tokens * 8 * width)
        .map(|i| (((i * 37 + 11) % 251) as f32 - 125.0) * 0.0031)
        .collect();
    let rows = Array::from_slice(&values, &[(tokens * 8) as i32, width as i32]);
    let stream = Stream::thread_local_or_default();
    let actual = moe_sum8(&rows, tokens, width, &stream)
        .expect("ordered eight-expert reduction should dispatch")
        .to_vec1::<f32>()
        .expect("read ordered eight-expert reduction");
    for token in 0..tokens {
        for column in 0..width {
            let base = token * 8 * width + column;
            let mut expected = values[base];
            for slot in 1..8 {
                expected += values[base + slot * width];
            }
            let index = token * width + column;
            assert_eq!(
                actual[index].to_bits(),
                expected.to_bits(),
                "token {token}, column {column}: {} vs {expected}",
                actual[index]
            );
        }
    }
}

#[test]
fn moe_sum8_shared_q8_down_inline_scalars_matches_stock_qmv() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-moe-shared-down] skipping: no Metal device available");
        return;
    }

    let (experts, out_rows, routed_in, shared_in) = (2usize, 16usize, 512usize, 256usize);
    let down_w = Array::from_slice(
        &vec![0u32; experts * out_rows * (routed_in / 16)],
        &[experts as i32, out_rows as i32, (routed_in / 16) as i32],
    );
    let down_scales = Array::from_slice(
        &vec![0.0f32; experts * out_rows * (routed_in / 128)],
        &[experts as i32, out_rows as i32, (routed_in / 128) as i32],
    );
    let down_biases = down_scales.clone();
    let mut addresses = vec![0i64; experts * 9];
    for expert in 0..experts {
        for (field, tensor) in [
            (6, down_w.index((expert as i32, .., ..))),
            (7, down_scales.index((expert as i32, .., ..))),
            (8, down_biases.index((expert as i32, .., ..))),
        ] {
            addresses[expert * 9 + field] =
                tensor.gpu_address().expect("expert down address") as i64;
        }
    }
    let address_table = Array::from_slice(&addresses, &[(experts * 9) as i32]);
    let selected = Array::from_slice(&[0i32; 8], &[8]);
    let routed = Array::from_slice(&vec![0.0f32; 8 * routed_in], &[8, routed_in as i32]);

    let shared_values: Vec<f32> = (0..out_rows * shared_in)
        .map(|i| (((i * 31 + 7) % 127) as f32 - 63.0) * 0.002)
        .collect();
    let shared_dense = Array::from_slice(&shared_values, &[out_rows as i32, shared_in as i32]);
    let (shared_w, shared_scales, shared_biases) =
        crate::ops::quantize(&shared_dense, 64, 8).expect("quantize shared Q8 down weight");
    assert_eq!(shared_w.shape(), [out_rows as i32, (shared_in / 4) as i32]);
    let shared_x_values: Vec<f32> = (0..shared_in)
        .map(|i| (((i * 13 + 5) % 101) as f32 - 50.0) * 0.003)
        .collect();
    let shared_x = Array::from_slice(&shared_x_values, &[1, shared_in as i32]);
    let stream = Stream::thread_local_or_default();
    let expected = crate::ops::quantized_matmul(
        &shared_x,
        &shared_w,
        &shared_scales,
        Some(&shared_biases),
        true,
        64,
        8,
    )
    .expect("stock shared Q8 down projection")
    .to_vec1::<f32>()
    .expect("read stock shared Q8 output");
    let keepalive = [&down_w, &down_scales, &down_biases];
    let actual = moe_gather_qmv_sum8_addr_add_shared_down(
        &address_table,
        &routed,
        &selected,
        &keepalive,
        &shared_x,
        &shared_w,
        &shared_scales,
        &shared_biases,
        out_rows,
        Dtype::Float32,
        None,
        &stream,
    )
    .expect("inline-scalar shared Q8 down/sum8 dispatch")
    .to_vec1::<f32>()
    .expect("read fused shared Q8 output");
    assert_eq!(actual.len(), expected.len());
    for (row, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() <= 1e-5,
            "shared down row {row}: LISA {got}, stock QMV {want}"
        );
    }
}

#[test]
fn hc_weighted_sum_matches_ds4_decode_order() {
    let h = 4096usize;
    let values: Vec<f32> = (0..4 * h)
        .map(|i| (((i * 17 + 9) % 211) as i32 - 105) as f32 * 0.125)
        .collect();
    let x = Array::from_slice(&values, &[1, 1, 4, h as i32]);
    let w = Array::from_slice(&[0.25f32; 4], &[1, 1, 4]);
    let stream = Stream::thread_local_or_default();
    let actual = hc_weighted_sum(&x, &w, &stream)
        .expect("HC weighted-sum dispatch")
        .to_vec1::<f32>()
        .expect("read HC output");
    assert_eq!(actual.len(), h);
    for d in 0..h {
        let mut expected = 0.0f32;
        for c in 0..4 {
            expected += values[c * h + d] * 0.25;
        }
        assert_eq!(actual[d].to_bits(), expected.to_bits(), "channel {d}");
    }
}

#[test]
fn hc_mix_rms_matches_reference() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-hc-rms] skipping: no Metal device available");
        return;
    }
    let (rows, width) = (2usize, 16_384usize);
    let values: Vec<f32> = (0..rows * width)
        .map(|i| (((i * 29 + 17) % 4093) as f32 - 2046.0) * 0.0007)
        .collect();
    let input = Array::from_slice(&values, &[rows as i32, width as i32]);
    let stream = Stream::thread_local_or_default();
    let actual = hc_mix_rms(&input, 1e-6, &stream)
        .expect("ds4-shaped plain RMSNorm should dispatch")
        .to_vec1::<f32>()
        .expect("read normalized mHC input");
    let ones = Array::from_slice(&vec![1.0f32; width], &[width as i32]);
    let weighted = mtp_rms_norm(&input, &ones, 1e-6, &stream)
        .expect("ds4-shaped weighted RMSNorm should dispatch")
        .to_vec1::<f32>()
        .expect("read weighted RMSNorm output");
    for (index, (&plain, &weighted)) in actual.iter().zip(&weighted).enumerate() {
        assert_eq!(
            plain.to_bits(),
            weighted.to_bits(),
            "plain/weighted ds4 RMSNorm mismatch at {index}"
        );
    }

    for row in 0..rows {
        let xs = &values[row * width..(row + 1) * width];
        let mut square_sum = 0.0f32;
        for x in xs {
            square_sum += x * x;
        }
        let scale = 1.0f32 / (square_sum / width as f32 + 1e-6).sqrt();
        for (column, &x) in xs.iter().enumerate() {
            let want = x * scale;
            let got = actual[row * width + column];
            assert!(
                (got - want).abs() <= 2e-6,
                "row={row} column={column}: {got} vs {want}"
            );
        }
    }
}

#[test]
fn mtp_add3_matches_left_associated_f32_reference() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-mtp-add3] skipping: no Metal device available");
        return;
    }
    let n = 4096usize;
    let a: Vec<f32> = (0..n)
        .map(|i| (((i * 29 + 7) % 4093) as f32 - 2046.0) * 0.00031)
        .collect();
    let b: Vec<f32> = (0..n)
        .map(|i| (((i * 17 + 3) % 2053) as f32 - 1026.0) * 0.00019)
        .collect();
    let c: Vec<f32> = (0..n)
        .map(|i| (((i * 13 + 11) % 1021) as f32 - 510.0) * 0.00027)
        .collect();
    let a_array = Array::from_slice(&a, &[1, 1, n as i32]);
    let b_array = Array::from_slice(&b, &[1, 1, n as i32]);
    let c_array = Array::from_slice(&c, &[1, 1, n as i32]);
    let stream = Stream::thread_local_or_default();
    let got = mtp_add3(&a_array, &b_array, &c_array, &stream)
        .expect("DS4-ordered MTP add3 should dispatch")
        .to_vec1::<f32>()
        .expect("read MTP add3 result");
    for i in 0..n {
        let want = a[i] + b[i] + c[i];
        assert_eq!(got[i].to_bits(), want.to_bits(), "MTP add3[{i}]");
    }
}

#[test]
fn moe_gather_kernels_compile_on_packed_q2_layout() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-moe-gather] skipping: no Metal device available");
        return;
    }
    let stream = Stream::thread_local_or_default();
    let (experts, hidden, mid) = (2usize, 512usize, 8usize);
    let x = Array::from_slice(&vec![0.25f32; hidden], &[1, hidden as i32]);
    let selected = Array::from_slice(&[1i32], &[1]);

    let down_w = Array::from_slice(
        &vec![0u32; experts * hidden * (hidden / 16)],
        &[experts as i32, hidden as i32, (hidden / 16) as i32],
    );
    let down_s = Array::from_slice(
        &vec![0.0f32; experts * hidden * (hidden / 128)],
        &[experts as i32, hidden as i32, (hidden / 128) as i32],
    );
    let down_b = Array::from_slice(
        &vec![0.0f32; experts * hidden * (hidden / 128)],
        &[experts as i32, hidden as i32, (hidden / 128) as i32],
    );
    let gathered = moe_gather_qmv(
        &down_w,
        &down_s,
        &down_b,
        &x,
        &selected,
        hidden,
        false,
        Dtype::Float32,
        &stream,
    )
    .expect("gathered down-projection kernel should compile and dispatch");
    assert!(
        gathered
            .to_vec1::<f32>()
            .expect("read gathered projection")
            .iter()
            .all(|&v| v == 0.0)
    );

    let pair_w = Array::from_slice(
        &vec![0u32; experts * 2 * mid * (hidden / 16)],
        &[experts as i32, (2 * mid) as i32, (hidden / 16) as i32],
    );
    let pair_s = Array::from_slice(
        &vec![0.0f32; experts * 2 * mid * (hidden / 128)],
        &[experts as i32, (2 * mid) as i32, (hidden / 128) as i32],
    );
    let pair_b = Array::from_slice(
        &vec![0.0f32; experts * 2 * mid * (hidden / 128)],
        &[experts as i32, (2 * mid) as i32, (hidden / 128) as i32],
    );
    let route_weights = Array::from_slice(&[1.25f32], &[1]);
    let pair = moe_pair_swiglu(
        &pair_w,
        &pair_s,
        &pair_b,
        &x,
        &selected,
        &route_weights,
        mid,
        1,
        10.0,
        &stream,
    )
    .expect("gathered pair-SwiGLU kernel should compile and dispatch");
    assert!(
        pair.to_vec1::<f32>()
            .expect("read gathered pair output")
            .iter()
            .all(|&v| v == 0.0)
    );
}

#[test]
fn moe_nonzero_q2_gathers_match_affine_qmv() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-moe-q2] skipping: no Metal device available");
        return;
    }

    let stream = Stream::thread_local_or_default();
    let (experts, hidden, mid) = (2usize, 512usize, 8usize);
    let x_values: Vec<f32> = (0..hidden)
        .map(|i| ((i * 17 % 83) as f32 - 41.0) * 0.0009)
        .collect();
    let x = Array::from_slice(&x_values, &[1, hidden as i32]);
    let selected = Array::from_slice(&[1i32], &[1]);

    let down_values: Vec<f32> = (0..experts * 16 * hidden)
        .map(|i| ((i * 23 % 109) as f32 - 54.0) * 0.0011)
        .collect();
    let down_dense = Array::from_slice(&down_values, &[experts as i32, 16, hidden as i32]);
    let (down_w, down_s, down_b) =
        crate::ops::quantize(&down_dense, 128, 2).expect("quantize synthetic down weights");
    let down = moe_gather_qmv(
        &down_w,
        &down_s,
        &down_b,
        &x,
        &selected,
        16,
        false,
        Dtype::Float32,
        &stream,
    )
    .expect("gathered down projection should dispatch");
    let reference_down = crate::ops::quantized_matmul(
        &x,
        &down_w
            .index((1, .., ..))
            .contiguous()
            .expect("slice down weights"),
        &down_s
            .index((1, .., ..))
            .contiguous()
            .expect("slice down scales"),
        Some(
            &down_b
                .index((1, .., ..))
                .contiguous()
                .expect("slice down biases"),
        ),
        true,
        128,
        2,
    )
    .expect("reference down projection");
    let got = down.to_vec1::<f32>().expect("read gathered down output");
    let want = reference_down
        .to_vec1::<f32>()
        .expect("read reference down output");
    for (i, (&a, &b)) in got.iter().zip(&want).enumerate() {
        assert!((a - b).abs() <= 3e-4, "down[{i}]: {a} vs {b}");
    }

    // The reference-shaped down kernel loops over the eight selected experts
    // inside one output-row tile. It must preserve the current per-expert
    // QMV and left-associated sum8 result bit-for-bit.
    let routed_x_values: Vec<f32> = (0..2 * 8 * hidden)
        .map(|i| ((i * 19 % 127) as f32 - 63.0) * 0.0007)
        .collect();
    let routed_x = Array::from_slice(&routed_x_values, &[16, hidden as i32]);
    let selected8_ids = [1i32, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0];
    let selected8 = Array::from_slice(&selected8_ids, &[16]);
    let mut down_addresses = vec![0i64; experts * 9];
    for expert in 0..experts {
        for (field, tensor) in [
            (6, down_w.index((expert as i32, .., ..))),
            (7, down_s.index((expert as i32, .., ..))),
            (8, down_b.index((expert as i32, .., ..))),
        ] {
            down_addresses[expert * 9 + field] =
                tensor.gpu_address().expect("down tensor address") as i64;
        }
    }
    let down_address_table = Array::from_slice(&down_addresses, &[(experts * 9) as i32]);
    let down_keepalive = [&down_w, &down_s, &down_b];
    let fused_sum = moe_gather_qmv_sum8_addr(
        &down_address_table,
        &routed_x,
        &selected8,
        &down_keepalive,
        16,
        Dtype::Float32,
        None,
        &stream,
    )
    .expect("selected-expert down sum8 should dispatch")
    .to_vec1::<f32>()
    .expect("read selected-expert down sum8");
    let ds4_fused_sum = moe_gather_qmv_sum8_addr_ds4(
        &down_address_table,
        &routed_x,
        &selected8,
        &down_keepalive,
        None,
        16,
        Dtype::Float32,
        None,
        &stream,
    )
    .expect("DS4-exact down sum8 should dispatch")
    .to_vec1::<f32>()
    .expect("read DS4-exact down sum8");
    assert_eq!(fused_sum.len(), ds4_fused_sum.len());
    for (i, (&actual, &expected)) in fused_sum.iter().zip(&ds4_fused_sum).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "generic and DS4-exact down sum8[{i}] differ"
        );
    }

    let routed_x_f16 = routed_x
        .as_dtype(Dtype::Float16)
        .expect("round routed activation to DS4 F16");
    let per_expert_f16 = moe_gather_qmv_addr(
        &down_address_table,
        &routed_x_f16,
        &selected8,
        &down_keepalive,
        16,
        true,
        Dtype::Float32,
        None,
        &stream,
    )
    .expect("rounded-F16 routed down projections should dispatch");
    let separate_sum_f16 = moe_sum8(&per_expert_f16, 2, 16, &stream)
        .expect("rounded-F16 separate down sum should dispatch")
        .to_vec1::<f32>()
        .expect("read rounded-F16 separate down sum");
    let fused_sum_f16 = moe_gather_qmv_sum8_addr(
        &down_address_table,
        &routed_x_f16,
        &selected8,
        &down_keepalive,
        16,
        Dtype::Float32,
        None,
        &stream,
    )
    .expect("F16-input selected-expert down sum8 should dispatch")
    .to_vec1::<f32>()
    .expect("read F16-input selected-expert down sum8");
    for (i, (&actual, &expected)) in fused_sum_f16.iter().zip(&separate_sum_f16).enumerate() {
        assert!(
            (actual - expected).abs() <= 3e-4,
            "F16 down sum8[{i}] fused={actual} separate={expected}"
        );
    }

    let shared_values: Vec<f32> = (0..2 * 16)
        .map(|i| ((i * 23 % 61) as f32 - 30.0) * 0.0021)
        .collect();
    let shared_output = Array::from_slice(&shared_values, &[2, 16]);
    let routed_then_shared: Vec<f32> = fused_sum
        .iter()
        .zip(&shared_values)
        .map(|(routed, shared)| routed + shared)
        .collect();
    let fused_sum_shared = moe_gather_qmv_sum8_addr_add_shared(
        &down_address_table,
        &routed_x,
        &selected8,
        &down_keepalive,
        &shared_output,
        16,
        Dtype::Float32,
        None,
        &stream,
    )
    .expect("selected-expert down sum8 with shared add should dispatch")
    .to_vec1::<f32>()
    .expect("read selected-expert down sum8 with shared add");
    for (i, (&actual, &expected)) in fused_sum_shared.iter().zip(&routed_then_shared).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "down sum8 plus shared[{i}]"
        );
    }

    let down_s_bf16 = down_s
        .as_dtype(Dtype::Bfloat16)
        .expect("cast down scales to checkpoint BF16");
    let down_b_bf16 = down_b
        .as_dtype(Dtype::Bfloat16)
        .expect("cast down biases to checkpoint BF16");
    let mut down_addresses_bf16 = vec![0i64; experts * 9];
    for expert in 0..experts {
        for (field, tensor) in [
            (6, down_w.index((expert as i32, .., ..))),
            (7, down_s_bf16.index((expert as i32, .., ..))),
            (8, down_b_bf16.index((expert as i32, .., ..))),
        ] {
            down_addresses_bf16[expert * 9 + field] =
                tensor.gpu_address().expect("BF16 down tensor address") as i64;
        }
    }
    let down_address_table_bf16 = Array::from_slice(&down_addresses_bf16, &[(experts * 9) as i32]);
    let down_keepalive_bf16 = [&down_w, &down_s_bf16, &down_b_bf16];
    let fused_sum_bf16 = moe_gather_qmv_sum8_addr(
        &down_address_table_bf16,
        &routed_x,
        &selected8,
        &down_keepalive_bf16,
        16,
        Dtype::Bfloat16,
        None,
        &stream,
    )
    .expect("BF16 selected down sum8 should dispatch")
    .to_vec1::<f32>()
    .expect("read BF16 selected down sum8");
    let ds4_fused_sum_bf16 = moe_gather_qmv_sum8_addr_ds4(
        &down_address_table_bf16,
        &routed_x,
        &selected8,
        &down_keepalive_bf16,
        None,
        16,
        Dtype::Bfloat16,
        None,
        &stream,
    )
    .expect("BF16 DS4-exact down sum8 should dispatch")
    .to_vec1::<f32>()
    .expect("read BF16 DS4-exact down sum8");
    for (i, (&actual, &expected)) in fused_sum_bf16.iter().zip(&ds4_fused_sum_bf16).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "BF16 generic and DS4-exact down sum8[{i}]"
        );
    }

    let pair_values: Vec<f32> = (0..experts * 2 * mid * hidden)
        .map(|i| ((i * 31 % 97) as f32 - 48.0) * 0.0013)
        .collect();
    let pair_dense = Array::from_slice(
        &pair_values,
        &[experts as i32, (2 * mid) as i32, hidden as i32],
    );
    let (pair_w, pair_s, pair_b) =
        crate::ops::quantize(&pair_dense, 128, 2).expect("quantize synthetic gate/up weights");
    let route = 0.73f32;
    let route_weights = Array::from_slice(&[route], &[1]);
    let fused = moe_pair_swiglu(
        &pair_w,
        &pair_s,
        &pair_b,
        &x,
        &selected,
        &route_weights,
        mid,
        1,
        10.0,
        &stream,
    )
    .expect("gathered pair-SwiGLU should dispatch");
    let expert_w = pair_w
        .index((1, .., ..))
        .contiguous()
        .expect("slice pair weights");
    let expert_s = pair_s
        .index((1, .., ..))
        .contiguous()
        .expect("slice pair scales");
    let expert_b = pair_b
        .index((1, .., ..))
        .contiguous()
        .expect("slice pair biases");
    let gu = crate::ops::quantized_matmul(&x, &expert_w, &expert_s, Some(&expert_b), true, 128, 2)
        .expect("reference gate/up projection")
        .to_vec1::<f32>()
        .expect("read gate/up projection");
    let fused = fused.to_vec1::<f32>().expect("read fused pair output");
    for i in 0..mid {
        let gate = gu[i].min(10.0);
        let up = gu[mid + i].clamp(-10.0, 10.0);
        let want = (gate / (1.0 + (-gate).exp())) * up * route;
        assert!(
            (fused[i] - want).abs() <= 3e-4,
            "pair[{i}]: {} vs {want}",
            fused[i]
        );
    }

    // The address path must consume the same packed affine tensors
    // without stacking or copying their expert axis. Each table row stores
    // the separate gate/up tensors plus the down projection addresses.
    let mut addresses = Vec::<i64>::with_capacity(experts * 9);
    for expert in 0..experts {
        let gate_w = pair_w.index((expert as i32, 0..mid as i32, ..));
        let gate_s = pair_s.index((expert as i32, 0..mid as i32, ..));
        let gate_b = pair_b.index((expert as i32, 0..mid as i32, ..));
        let up_w = pair_w.index((expert as i32, mid as i32..(2 * mid) as i32, ..));
        let up_s = pair_s.index((expert as i32, mid as i32..(2 * mid) as i32, ..));
        let up_b = pair_b.index((expert as i32, mid as i32..(2 * mid) as i32, ..));
        let expert_down_w = down_w.index((expert as i32, .., ..));
        let expert_down_s = down_s.index((expert as i32, .., ..));
        let expert_down_b = down_b.index((expert as i32, .., ..));
        for tensor in [
            &gate_w,
            &gate_s,
            &gate_b,
            &up_w,
            &up_s,
            &up_b,
            &expert_down_w,
            &expert_down_s,
            &expert_down_b,
        ] {
            addresses.push(tensor.gpu_address().expect("tensor GPU address") as i64);
        }
    }
    let address_table = Array::from_slice(&addresses, &[(experts * 9) as i32]);
    let keepalive = [&pair_w, &pair_s, &pair_b, &down_w, &down_s, &down_b];
    let collect_mapped_fields = |pair_w: &Array,
                                 pair_s: &Array,
                                 pair_b: &Array,
                                 down_w: &Array,
                                 down_s: &Array,
                                 down_b: &Array| {
        let mut fields = Vec::with_capacity(experts * 9);
        for expert in 0..experts {
            fields.extend([
                pair_w.index((expert as i32, 0..mid as i32, ..)),
                pair_s.index((expert as i32, 0..mid as i32, ..)),
                pair_b.index((expert as i32, 0..mid as i32, ..)),
                pair_w.index((expert as i32, mid as i32..(2 * mid) as i32, ..)),
                pair_s.index((expert as i32, mid as i32..(2 * mid) as i32, ..)),
                pair_b.index((expert as i32, mid as i32..(2 * mid) as i32, ..)),
                down_w.index((expert as i32, .., ..)),
                down_s.index((expert as i32, .., ..)),
                down_b.index((expert as i32, .., ..)),
            ]);
        }
        fields
    };
    let mapped_fields = collect_mapped_fields(&pair_w, &pair_s, &pair_b, &down_w, &down_s, &down_b);
    let mapped_field_refs: Vec<&Array> = mapped_fields.iter().collect();
    let (mapped_table, mapped_resources) =
        moe_mapped_resource_table(&mapped_field_refs).expect("mapped resource table");
    let address_pair = moe_pair_swiglu_addr(
        &address_table,
        &x,
        &selected,
        &route_weights,
        &keepalive,
        mid,
        1,
        10.0,
        pair_s.dtype(),
        None,
        &stream,
    )
    .expect("GPU-address gate/up path should dispatch")
    .to_vec1::<f32>()
    .expect("read GPU-address pair output");
    for (i, (&actual, &expected)) in address_pair.iter().zip(&fused).enumerate() {
        assert!(
            (actual - expected).abs() <= 3e-4,
            "address pair[{i}]: {actual} vs {expected}"
        );
    }
    let mapped_pair = moe_pair_swiglu_mapped(
        &mapped_table,
        &x,
        &selected,
        &route_weights,
        &mapped_resources,
        mid,
        1,
        10.0,
        pair_s.dtype(),
        &stream,
    )
    .expect("encoder-bound mapped pair kernel should dispatch")
    .to_vec1::<f32>()
    .expect("read encoder-bound mapped pair output");
    for (i, (&actual, &expected)) in mapped_pair.iter().zip(&address_pair).enumerate() {
        assert_eq!(actual.to_bits(), expected.to_bits(), "mapped pair[{i}]");
    }
    let mapped_sum8 = moe_gather_qmv_sum8_mapped(
        &mapped_table,
        &routed_x,
        &selected8,
        &mapped_resources,
        16,
        Dtype::Float32,
        &stream,
    )
    .expect("encoder-bound mapped down sum8 kernel should dispatch")
    .to_vec1::<f32>()
    .expect("read encoder-bound mapped down sum8 output");
    let ds4_fused_sum = moe_gather_qmv_sum8_addr_ds4(
        &address_table,
        &routed_x,
        &selected8,
        &keepalive,
        None,
        16,
        Dtype::Float32,
        None,
        &stream,
    )
    .expect("DS4-exact selected-expert down sum8 should dispatch")
    .to_vec1::<f32>()
    .expect("read DS4-exact selected-expert down sum8");
    for (i, (&actual, &expected)) in mapped_sum8.iter().zip(&ds4_fused_sum).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "mapped down sum8[{i}]"
        );
    }

    let address_down = moe_gather_qmv_addr(
        &address_table,
        &x,
        &selected,
        &keepalive,
        16,
        false,
        down_s.dtype(),
        None,
        &stream,
    )
    .expect("GPU-address down path should dispatch")
    .to_vec1::<f32>()
    .expect("read GPU-address down output");
    let gathered_down = down.to_vec1::<f32>().expect("read gathered down output");
    for (i, (&actual, &expected)) in address_down.iter().zip(&gathered_down).enumerate() {
        assert!(
            (actual - expected).abs() <= 3e-4,
            "address down[{i}]: {actual} vs {expected}"
        );
    }

    // The checkpoint stores Q2 scales and biases as BF16. Read them from
    // their native buffers and promote each value in-kernel, matching the
    // F32 metadata staging used by the previous gathered implementation.
    let pair_s_bf16 = pair_s
        .as_dtype(Dtype::Bfloat16)
        .expect("cast gate/up scales to checkpoint BF16");
    let pair_b_bf16 = pair_b
        .as_dtype(Dtype::Bfloat16)
        .expect("cast gate/up biases to checkpoint BF16");
    let down_s_bf16 = down_s
        .as_dtype(Dtype::Bfloat16)
        .expect("cast down scales to checkpoint BF16");
    let down_b_bf16 = down_b
        .as_dtype(Dtype::Bfloat16)
        .expect("cast down biases to checkpoint BF16");
    let mut bf16_addresses = Vec::<i64>::with_capacity(experts * 9);
    for expert in 0..experts {
        let gate_w = pair_w.index((expert as i32, 0..mid as i32, ..));
        let gate_s = pair_s_bf16.index((expert as i32, 0..mid as i32, ..));
        let gate_b = pair_b_bf16.index((expert as i32, 0..mid as i32, ..));
        let up_w = pair_w.index((expert as i32, mid as i32..(2 * mid) as i32, ..));
        let up_s = pair_s_bf16.index((expert as i32, mid as i32..(2 * mid) as i32, ..));
        let up_b = pair_b_bf16.index((expert as i32, mid as i32..(2 * mid) as i32, ..));
        let expert_down_w = down_w.index((expert as i32, .., ..));
        let expert_down_s = down_s_bf16.index((expert as i32, .., ..));
        let expert_down_b = down_b_bf16.index((expert as i32, .., ..));
        for tensor in [
            &gate_w,
            &gate_s,
            &gate_b,
            &up_w,
            &up_s,
            &up_b,
            &expert_down_w,
            &expert_down_s,
            &expert_down_b,
        ] {
            bf16_addresses.push(tensor.gpu_address().expect("BF16 tensor GPU address") as i64);
        }
    }
    let bf16_address_table = Array::from_slice(&bf16_addresses, &[(experts * 9) as i32]);
    let bf16_keepalive = [
        &pair_w,
        &pair_s_bf16,
        &pair_b_bf16,
        &down_w,
        &down_s_bf16,
        &down_b_bf16,
    ];
    let bf16_fields = collect_mapped_fields(
        &pair_w,
        &pair_s_bf16,
        &pair_b_bf16,
        &down_w,
        &down_s_bf16,
        &down_b_bf16,
    );
    let bf16_field_refs: Vec<&Array> = bf16_fields.iter().collect();
    let (bf16_mapped_table, bf16_mapped_resources) =
        moe_mapped_resource_table(&bf16_field_refs).expect("BF16 mapped resource table");
    let pair_s_f32 = pair_s_bf16
        .as_dtype(Dtype::Float32)
        .expect("promote checkpoint gate/up scales");
    let pair_b_f32 = pair_b_bf16
        .as_dtype(Dtype::Float32)
        .expect("promote checkpoint gate/up biases");
    let down_s_f32 = down_s_bf16
        .as_dtype(Dtype::Float32)
        .expect("promote checkpoint down scales");
    let down_b_f32 = down_b_bf16
        .as_dtype(Dtype::Float32)
        .expect("promote checkpoint down biases");
    let bf16_pair_reference = moe_pair_swiglu(
        &pair_w,
        &pair_s_f32,
        &pair_b_f32,
        &x,
        &selected,
        &route_weights,
        mid,
        1,
        10.0,
        &stream,
    )
    .expect("F32-staged BF16 reference pair should dispatch")
    .to_vec1::<f32>()
    .expect("read BF16 reference pair output");
    let bf16_pair = moe_pair_swiglu_addr(
        &bf16_address_table,
        &x,
        &selected,
        &route_weights,
        &bf16_keepalive,
        mid,
        1,
        10.0,
        Dtype::Bfloat16,
        None,
        &stream,
    )
    .expect("BF16 GPU-address pair should dispatch")
    .to_vec1::<f32>()
    .expect("read BF16 GPU-address pair");
    for (i, (&actual, &expected)) in bf16_pair.iter().zip(&bf16_pair_reference).enumerate() {
        assert!(
            (actual - expected).abs() <= 3e-4,
            "BF16 address pair[{i}]: {actual} vs {expected}"
        );
    }
    let bf16_mapped_pair = moe_pair_swiglu_mapped(
        &bf16_mapped_table,
        &x,
        &selected,
        &route_weights,
        &bf16_mapped_resources,
        mid,
        1,
        10.0,
        Dtype::Bfloat16,
        &stream,
    )
    .expect("BF16 encoder-bound mapped pair should dispatch")
    .to_vec1::<f32>()
    .expect("read BF16 encoder-bound mapped pair");
    for (i, (&actual, &expected)) in bf16_mapped_pair.iter().zip(&bf16_pair).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "BF16 mapped pair[{i}]"
        );
    }
    let bf16_mapped_sum8 = moe_gather_qmv_sum8_mapped(
        &bf16_mapped_table,
        &routed_x,
        &selected8,
        &bf16_mapped_resources,
        16,
        Dtype::Bfloat16,
        &stream,
    )
    .expect("BF16 encoder-bound mapped down sum8 should dispatch")
    .to_vec1::<f32>()
    .expect("read BF16 encoder-bound mapped down sum8");
    let bf16_ds4_fused_sum = moe_gather_qmv_sum8_addr_ds4(
        &bf16_address_table,
        &routed_x,
        &selected8,
        &bf16_keepalive,
        None,
        16,
        Dtype::Bfloat16,
        None,
        &stream,
    )
    .expect("BF16 DS4-exact selected-expert down sum8 should dispatch")
    .to_vec1::<f32>()
    .expect("read BF16 DS4-exact selected-expert down sum8");
    for (i, (&actual, &expected)) in bf16_mapped_sum8.iter().zip(&bf16_ds4_fused_sum).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "BF16 mapped down sum8[{i}]"
        );
    }

    let bf16_down_reference = moe_gather_qmv(
        &down_w,
        &down_s_f32,
        &down_b_f32,
        &x,
        &selected,
        16,
        false,
        Dtype::Float32,
        &stream,
    )
    .expect("F32-staged BF16 reference down should dispatch")
    .to_vec1::<f32>()
    .expect("read BF16 reference down output");
    let bf16_down = moe_gather_qmv_addr(
        &bf16_address_table,
        &x,
        &selected,
        &bf16_keepalive,
        16,
        false,
        Dtype::Bfloat16,
        None,
        &stream,
    )
    .expect("BF16 GPU-address down should dispatch")
    .to_vec1::<f32>()
    .expect("read BF16 GPU-address down output");
    for (i, (&actual, &expected)) in bf16_down.iter().zip(&bf16_down_reference).enumerate() {
        assert!(
            (actual - expected).abs() <= 3e-4,
            "BF16 address down[{i}]: {actual} vs {expected}"
        );
    }
}

#[test]
fn mhc_expand_and_swiglu_compile_and_match() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-mhc-expand] skipping: no Metal device available");
        return;
    }
    let stream = Stream::thread_local_or_default();
    let (rows, hidden) = (2usize, 4096usize);
    let y_values: Vec<f32> = (0..rows * hidden)
        .map(|i| ((i * 13 % 41) as f32 - 20.0) * 0.001)
        .collect();
    let residual_values: Vec<f32> = (0..rows * 4 * hidden)
        .map(|i| ((i * 7 % 37) as f32 - 18.0) * 0.002)
        .collect();
    let post_values: Vec<f32> = (0..rows * 4).map(|i| 0.2 + i as f32 * 0.03).collect();
    let comb_values: Vec<f32> = (0..rows * 16)
        .map(|i| ((i * 5 % 23) as f32 - 11.0) * 0.007)
        .collect();
    let y = Array::from_slice(&y_values, &[rows as i32, 1, hidden as i32]);
    let residual = Array::from_slice(&residual_values, &[rows as i32, 1, 4, hidden as i32]);
    let post = Array::from_slice(&post_values, &[rows as i32, 1, 4]);
    let comb = Array::from_slice(&comb_values, &[rows as i32, 1, 4, 4]);
    let got = hc_expand4(&y, &residual, &post, &comb, &stream)
        .expect("HC expand kernel should compile and dispatch")
        .to_vec1::<f32>()
        .expect("read HC expand output");
    for row in 0..rows {
        for dst in 0..4 {
            for d in 0..hidden {
                let mut want = post_values[row * 4 + dst] * y_values[row * hidden + d];
                for src in 0..4 {
                    want += comb_values[row * 16 + src * 4 + dst]
                        * residual_values[(row * 4 + src) * hidden + d];
                }
                let index = (row * 4 + dst) * hidden + d;
                assert!((got[index] - want).abs() <= 1e-6, "expand[{index}]");
            }
        }
    }

    let gate_values = vec![-12.0f32, -2.0, -0.4, 0.0, 1.3, 8.0, 14.0];
    let up_values = vec![12.0f32, -3.0, -0.5, 0.0, 2.1, -14.0, 18.0];
    let gate = Array::from_slice(&gate_values, &[7]);
    let up = Array::from_slice(&up_values, &[7]);
    let got = swiglu_fused(&gate, &up, 10.0, &stream)
        .expect("SwiGLU kernel should compile and dispatch")
        .to_vec1::<f32>()
        .expect("read SwiGLU output");
    for i in 0..gate_values.len() {
        let g = gate_values[i].min(10.0);
        let u = up_values[i].clamp(-10.0, 10.0);
        let want = g / (1.0 + (-g).exp()) * u;
        assert!((got[i] - want).abs() <= 2e-6, "SwiGLU[{i}]");
    }
}

#[test]
fn shared_ds4_q8_schedule_handles_odd_output_extent() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-shared-q8-ds4-tail] skipping: no Metal device available");
        return;
    }
    const IN: usize = 64;
    const OUT: usize = 5;
    const GROUP: usize = 64;
    let pack_row = |q: i8| {
        let code = (i16::from(q) + 128) as u8;
        vec![u32::from_le_bytes([code; 4]); IN / 4]
    };
    let mut gate_packed = Vec::with_capacity(OUT * IN / 4);
    let mut up_packed = Vec::with_capacity(OUT * IN / 4);
    let gate_q = [-2i8, -1, 0, 1, 2];
    let up_q = [2i8, 1, 0, -1, -2];
    for row in 0..OUT {
        gate_packed.extend(pack_row(gate_q[row]));
        up_packed.extend(pack_row(up_q[row]));
    }
    gate_packed.extend(up_packed);
    let scales = Array::from_slice(&[1.0f32; 2 * OUT], &[(2 * OUT) as i32, 1]);
    let biases = Array::from_slice(&[-128.0f32; 2 * OUT], &[(2 * OUT) as i32, 1]);
    let x = Array::from_slice(&vec![0.125f32; IN], &[IN as i32]);
    let gate_up_weights = Array::from_slice(&gate_packed, &[(2 * OUT) as i32, (IN / 4) as i32]);
    let stream = Stream::thread_local_or_default();
    let (gate, up, mid) = shared_q8_swiglu_ds4(
        &x,
        &gate_up_weights,
        &scales,
        &biases,
        IN,
        OUT,
        GROUP,
        10.0,
        &stream,
    )
    .expect("odd output extent should dispatch");
    let gate = gate.to_vec1::<f32>().expect("read gate");
    let up = up.to_vec1::<f32>().expect("read up");
    let mid = mid.to_vec1::<f32>().expect("read mid");
    for row in 0..OUT {
        let expected_gate = 8.0 * f32::from(gate_q[row]);
        let expected_up = 8.0 * f32::from(up_q[row]);
        let clipped_gate = expected_gate.min(10.0);
        let clipped_up = expected_up.clamp(-10.0, 10.0);
        let expected_mid = clipped_gate / (1.0 + (-clipped_gate).exp()) * clipped_up;
        assert!((gate[row] - expected_gate).abs() <= 1e-5);
        assert!((up[row] - expected_up).abs() <= 1e-5);
        assert!((mid[row] - expected_mid).abs() <= 1e-4);
    }
}

#[test]
fn hc_mix_rms_bf16_gemv_matches_separate_chain_bitwise() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-hc-rms-gemv] skipping: no Metal device available");
        return;
    }
    let (width, out_dim) = (16_384usize, 24usize);
    let weight_values: Vec<f32> = (0..out_dim * width)
        .map(|i| (((i * 31 + 11) % 1009) as f32 - 504.0) * 0.00003)
        .collect();
    let weights = Array::from_slice(&weight_values, &[out_dim as i32, width as i32])
        .as_dtype(Dtype::Bfloat16)
        .expect("cast mHC projection weights to BF16");
    let stream = Stream::thread_local_or_default();
    for rows in [1usize, 2] {
        let input_values: Vec<f32> = (0..rows * width)
            .map(|i| (((i * 29 + 17) % 4093) as f32 - 2046.0) * 0.0007)
            .collect();
        let input = Array::from_slice(&input_values, &[rows as i32, width as i32]);
        let normalized =
            hc_mix_rms(&input, 1e-6, &stream).expect("separate mHC RMSNorm should dispatch");
        let expected = bf16_gemv(&weights, &normalized, &stream)
            .expect("separate BF16 GEMV should dispatch")
            .to_vec1::<f32>()
            .expect("read separate projection");
        let actual = hc_mix_rms_bf16_gemv(&weights, &input, 1e-6, &stream)
            .expect("fused mHC RMSNorm and BF16 GEMV should dispatch")
            .to_vec1::<f32>()
            .expect("read fused projection");
        assert_eq!(actual.len(), expected.len());
        for (i, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "rows={rows}: fused/separate projection mismatch at output {i}"
            );
        }
    }
}

#[test]
fn indexer_block_topk_matches_unique_score_order() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-indexer-topk] skipping: no Metal device available");
        return;
    }
    let stream = Stream::thread_local_or_default();
    for &(n_comp, top_k) in &[(13usize, 7usize), (520, 31), (1200, 512)] {
        let n_rows = 2usize;
        let mut values = Vec::with_capacity(n_rows * n_comp);
        for row in 0..n_rows {
            for pool in 0..n_comp {
                let score = if row == 0 {
                    pool as f32
                } else {
                    (n_comp - pool) as f32
                };
                values.push(score);
            }
        }
        let scores = Array::from_slice(&values, &[n_rows as i32, n_comp as i32]);
        let selected =
            dsa_indexer_topk(&scores, top_k, &stream).expect("ds4 blockwise top-k should dispatch");
        let shuffled = dsa_indexer_topk_shuffle(&scores, top_k, &stream)
            .expect("SIMD-shuffle blockwise top-k should dispatch");
        let got = selected
            .to_vec1::<i32>()
            .expect("selected pool IDs should be readable");
        let shuffled = shuffled
            .to_vec1::<i32>()
            .expect("shuffled selected pool IDs should be readable");
        assert_eq!(shuffled, got, "shuffle top-k differs for {n_comp} pools");
        assert_eq!(got.len(), n_rows * top_k);
        for rank in 0..top_k {
            assert_eq!(got[rank], (n_comp - 1 - rank) as i32);
            assert_eq!(got[top_k + rank], rank as i32);
        }

        // The reference's bitonic stages leave equal elements in their input order,
        // and its merge chooses the left run on equal scores. Zero-score
        // ties are common after the indexer's ReLU.
        let ties = Array::from_slice(
            &vec![0.0f32; n_rows * n_comp],
            &[n_rows as i32, n_comp as i32],
        );
        let tied = dsa_indexer_topk(&ties, top_k, &stream)
            .expect("ds4 blockwise top-k should dispatch for ties")
            .to_vec1::<i32>()
            .expect("tied pool IDs should be readable");
        let shuffled_tied = dsa_indexer_topk_shuffle(&ties, top_k, &stream)
            .expect("shuffle blockwise top-k should dispatch for ties")
            .to_vec1::<i32>()
            .expect("shuffled tied pool IDs should be readable");
        assert_eq!(shuffled_tied, tied, "shuffle top-k changed tie order");
        for row in 0..n_rows {
            let expected: Vec<i32> = (0..top_k as i32).collect();
            assert_eq!(&tied[row * top_k..(row + 1) * top_k], expected.as_slice());
        }
    }
}

#[test]
fn indexer_pool_expansion_matches_causal_batched_layout() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-indexer-expand] skipping: no Metal device available");
        return;
    }
    let stream = Stream::thread_local_or_default();
    // Two batch rows, each with two query positions. The selected pool
    // list includes one future pool so the kernel must preserve its -1
    // padding while advancing causal visibility only along the token axis.
    let pools = Array::from_slice(
        &[
            0i32, 1, 2, // batch 0, token 0
            0, 1, 2, // batch 0, token 1
            0, 1, 2, // batch 1, token 0
            0, 1, 2, // batch 1, token 1
        ],
        &[4, 3],
    );
    let selected = expand_pool_selection(&pools, 2, 8, 4, true, &stream)
        .expect("pool expansion should dispatch");
    let got = selected
        .to_vec1::<i32>()
        .expect("expanded token IDs should be readable");
    let row0 = vec![0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1, -1, 8, -1, -1];
    let row1 = vec![0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1, -1, 8, 9, -1];
    let expected = [
        row0.as_slice(),
        row1.as_slice(),
        row0.as_slice(),
        row1.as_slice(),
    ]
    .concat();
    assert_eq!(got, expected);

    let no_tail = expand_pool_selection(&pools, 2, 8, 4, false, &stream)
        .expect("pool expansion without tail should dispatch")
        .to_vec1::<i32>()
        .expect("expanded token IDs should be readable");
    for row in no_tail.as_chunks::<15>().0.iter() {
        assert_eq!(&row[12..], &[-1, -1, -1]);
    }

    let causal = expand_pool_selection_for_attention(&pools, 2, 8, 4, true, &stream)
        .expect("causal-prefix pool expansion should dispatch")
        .to_vec1::<i32>()
        .expect("causal-prefix token IDs should be readable");
    let row0 = (0..9).chain(std::iter::repeat_n(-1, 6)).collect::<Vec<_>>();
    let row1 = (0..10)
        .chain(std::iter::repeat_n(-1, 5))
        .collect::<Vec<_>>();
    let expected_causal = [
        row0.as_slice(),
        row1.as_slice(),
        row0.as_slice(),
        row1.as_slice(),
    ]
    .concat();
    assert_eq!(causal, expected_causal);
}

#[test]
fn indexer_score_paths_preserve_scores_and_visibility() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-indexer-scores] skipping: no Metal device available");
        return;
    }
    let stream = Stream::thread_local_or_default();
    let (batches, heads, dim, pools) = (1usize, 32usize, 128usize, 4usize);
    for tokens in [1usize, 3, 8] {
        let mut q = vec![0.0f32; batches * heads * tokens * dim];
        for head in 0..heads {
            for token in 0..tokens {
                q[(head * tokens + token) * dim] = 64.0;
            }
        }
        let weights = vec![1.0f32 / heads as f32; batches * tokens * heads];
        let mut pooled = vec![0.0f32; batches * pools * dim];
        for pool in 0..pools {
            pooled[pool * dim] = (pool + 1) as f32;
        }
        let q = Array::from_slice(
            &q,
            &[batches as i32, heads as i32, tokens as i32, dim as i32],
        );
        let weights = Array::from_slice(&weights, &[batches as i32, tokens as i32, heads as i32]);
        let pooled = Array::from_slice(&pooled, &[batches as i32, pools as i32, dim as i32])
            .as_dtype(Dtype::Float16)
            .expect("F16 indexer cache conversion should succeed");
        let pos0 = 8usize;
        let scores = indexer_scores(&q, &weights, &pooled, pos0, 4, 1.0 / 64.0, &stream)
            .expect("ds4-shaped indexer score path should dispatch");
        let got = scores
            .to_vec1::<f32>()
            .expect("indexer scores should be readable");
        if tokens == 1 {
            let rows4 = indexer_scores_rows4(&q, &weights, &pooled, pos0, 4, 1.0 / 64.0, &stream)
                .expect("DS4 rows4 scorer should dispatch")
                .to_vec1::<f32>()
                .expect("rows4 scores should be readable");
            assert_eq!(rows4, got, "rows4 decode score path changed output");
        }
        for token in 0..tokens {
            let visible = if tokens == 1 {
                pools
            } else {
                ((pos0 + token + 1) / 4).min(pools)
            };
            for pool in 0..pools {
                let value = got[token * pools + pool];
                if pool < visible {
                    assert!(
                        (value - (pool + 1) as f32).abs() <= 1e-5,
                        "tokens={tokens} token={token} pool={pool}: got {value}"
                    );
                } else {
                    assert_eq!(value, f32::NEG_INFINITY);
                }
            }
        }
    }
}

#[test]
fn hc_split_matches_ds4_float4_equations() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-hc-split] skipping: no Metal device available");
        return;
    }
    let stream = Stream::thread_local_or_default();
    let rows = 2usize;
    let eps = 1e-6f32;
    let iters = 4usize;
    let scale_values = [0.7f32, -0.3, 0.5];
    let mut mix_values = vec![0.0f32; rows * 24];
    let mut base = vec![0.0f32; 24];
    for (i, value) in base.iter_mut().enumerate() {
        *value = (i as f32 - 11.0) * 0.013;
    }
    for (i, value) in mix_values.iter_mut().enumerate() {
        *value = ((i * 17 % 43) as f32 - 21.0) * 0.019;
    }
    let mixes = Array::from_slice(&mix_values, &[rows as i32, 24]);
    let scale = Array::from_slice(&scale_values, &[3]);
    let base_array = Array::from_slice(&base, &[24]);
    let got = hc_split_sinkhorn(&mixes, &scale, &base_array, eps, iters, &stream)
        .expect("mHC split kernel should dispatch")
        .to_vec1::<f32>()
        .expect("mHC split output should be readable");

    for row in 0..rows {
        let mut expected = [0.0f32; 24];
        for i in 0..4 {
            let z = mix_values[row * 24 + i] * scale_values[0] + base[i];
            expected[i] = 1.0 / (1.0 + (-z).exp()) + eps;
            let z = mix_values[row * 24 + 4 + i] * scale_values[1] + base[4 + i];
            expected[4 + i] = 2.0 / (1.0 + (-z).exp());
        }
        let mut comb = [[0.0f32; 4]; 4];
        for dst in 0..4 {
            let mut logits = [0.0f32; 4];
            for src in 0..4 {
                logits[src] = mix_values[row * 24 + 8 + dst * 4 + src] * scale_values[2]
                    + base[8 + dst * 4 + src];
            }
            let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0.0f32;
            for src in 0..4 {
                comb[dst][src] = (logits[src] - maximum).exp();
                denom += comb[dst][src];
            }
            for src in 0..4 {
                comb[dst][src] = comb[dst][src] / denom + eps;
            }
        }
        for src in 0..4 {
            let mut denom = 0.0f32;
            for dst in 0..4 {
                denom += comb[dst][src];
            }
            let inv = 1.0 / (denom + eps);
            for dst in 0..4 {
                comb[dst][src] *= inv;
            }
        }
        for _ in 1..iters {
            for dst in 0..4 {
                let denom = comb[dst][0] + comb[dst][1] + comb[dst][2] + comb[dst][3] + eps;
                let inv = 1.0 / denom;
                for src in 0..4 {
                    comb[dst][src] *= inv;
                }
            }
            for src in 0..4 {
                let denom = comb[0][src] + comb[1][src] + comb[2][src] + comb[3][src] + eps;
                let inv = 1.0 / denom;
                for dst in 0..4 {
                    comb[dst][src] *= inv;
                }
            }
        }
        for dst in 0..4 {
            for src in 0..4 {
                expected[8 + dst * 4 + src] = comb[dst][src];
            }
        }
        for i in 0..24 {
            assert!(
                (got[row * 24 + i] - expected[i]).abs() <= 3e-6,
                "row={row} index={i}: got {}, expected {}",
                got[row * 24 + i],
                expected[i]
            );
        }
    }
}

#[test]
fn hc_split_collapse_norm4_matches_separate_reference() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-hc-fused] skipping: no Metal device available");
        return;
    }
    let stream = Stream::thread_local_or_default();
    let (batches, tokens, hidden) = (2usize, 1usize, 4096usize);
    let rows = batches * tokens;
    let eps = 1e-6f32;
    let norm_eps = 1e-6f32;
    let iters = 4usize;
    let scales = Array::from_slice(&[0.7f32, 0.4, 0.2], &[3]);
    let base_values: Vec<f32> = (0..24).map(|i| (i as f32 - 11.0) * 0.011).collect();
    let base = Array::from_slice(&base_values, &[24]);
    let mix_values: Vec<f32> = (0..rows * 24)
        .map(|i| ((i * 13 % 41) as f32 - 20.0) * 0.017)
        .collect();
    let mixes = Array::from_slice(&mix_values, &[rows as i32, 24]);
    let x_values: Vec<f32> = (0..rows * 4 * hidden)
        .map(|i| ((i * 7 % 37) as f32 - 18.0) * 0.003)
        .collect();
    let x = Array::from_slice(
        &x_values,
        &[batches as i32, tokens as i32, 4, hidden as i32],
    );
    let norm_values: Vec<f32> = (0..hidden).map(|i| 0.75 + (i % 19) as f32 * 0.01).collect();
    let norm_weight = Array::from_slice(&norm_values, &[hidden as i32]);

    let reference_split = hc_split_sinkhorn(&mixes, &scales, &base, eps, iters, &stream)
        .expect("standalone mHC split should dispatch")
        .to_vec1::<f32>()
        .expect("standalone mHC split should be readable");
    let (split, post, comb, collapsed, normalized) = hc_split_weighted_sum_norm4(
        &mixes,
        &scales,
        &base,
        &x,
        &norm_weight,
        eps,
        norm_eps,
        iters,
        &stream,
    )
    .expect("fused mHC split/collapse/norm should dispatch");
    let split = split
        .to_vec1::<f32>()
        .expect("fused split should be readable");
    let post = post
        .to_vec1::<f32>()
        .expect("fused post coefficients should be readable");
    let comb = comb
        .to_vec1::<f32>()
        .expect("fused comb coefficients should be readable");
    let collapsed = collapsed
        .to_vec1::<f32>()
        .expect("fused collapsed rows should be readable");
    let normalized = normalized
        .to_vec1::<f32>()
        .expect("fused normalized rows should be readable");
    for (i, (&got, &want)) in split.iter().zip(&reference_split).enumerate() {
        assert!((got - want).abs() <= 3e-6, "split[{i}]: {got} vs {want}");
    }
    for row in 0..rows {
        for i in 0..4 {
            assert_eq!(
                post[row * 4 + i].to_bits(),
                split[row * 24 + 4 + i].to_bits(),
                "post row={row} index={i}"
            );
        }
        for i in 0..16 {
            assert_eq!(
                comb[row * 16 + i].to_bits(),
                split[row * 24 + 8 + i].to_bits(),
                "comb row={row} index={i}"
            );
        }
    }

    for row in 0..rows {
        let mut collapse_sq = 0.0f32;
        for d in 0..hidden {
            let mut expected = 0.0f32;
            for stream_id in 0..4 {
                expected += x_values[(row * 4 + stream_id) * hidden + d]
                    * reference_split[row * 24 + stream_id];
            }
            let got = collapsed[row * hidden + d];
            assert!((got - expected).abs() <= 2e-6, "collapse row={row} d={d}");
            collapse_sq += got * got;
        }
        let norm_scale = 1.0 / (collapse_sq / hidden as f32 + norm_eps).sqrt();
        for d in 0..hidden {
            let want = (collapsed[row * hidden + d] * norm_scale) * norm_values[d];
            let got = normalized[row * hidden + d];
            assert!(
                (got - want).abs() <= 2e-5,
                "norm row={row} d={d}: {got} vs {want}"
            );
        }
    }
}

#[test]
fn kda_prepare_matches_reference() {
    let rt = Stream::thread_local_or_default();
    let heads = 64usize;
    let d = 128usize;
    let t = 2usize;
    let proj = heads * d;
    let conv_dim = 3 * proj;
    let mut rng_val = 12345u32;
    let mut next = || {
        rng_val = rng_val.wrapping_mul(1664525).wrapping_add(1013904223);
        (rng_val >> 8) as f32 / 8_388_608.0f32 - 0.5
    };
    let q: Vec<f32> = (0..t * proj).map(|_| next()).collect();
    let k: Vec<f32> = (0..t * proj).map(|_| next()).collect();
    let v: Vec<f32> = (0..t * proj).map(|_| next()).collect();
    let gate: Vec<f32> = (0..t * proj).map(|_| next()).collect();
    let cq: Vec<f32> = (0..proj * 4).map(|_| next() * 0.05).collect();
    let ck: Vec<f32> = (0..proj * 4).map(|_| next() * 0.05).collect();
    let cv: Vec<f32> = (0..proj * 4).map(|_| next() * 0.05).collect();
    let alog: Vec<f32> = (0..heads)
        .map(|i| -0.25f32 + (i % 7) as f32 * 0.01)
        .collect();
    let dtb: Vec<f32> = (0..proj).map(|_| next()).collect();
    let cs: Vec<f32> = (0..3 * conv_dim)
        .map(|i| ((i * 11 % 89) as f32 - 44.0) / 320.0)
        .collect();

    let qa = Array::from_slice(&q, &[1i32, t as i32, proj as i32]);
    let ka = Array::from_slice(&k, &[1i32, t as i32, proj as i32]);
    let va = Array::from_slice(&v, &[1i32, t as i32, proj as i32]);
    let ga = Array::from_slice(&gate, &[1i32, t as i32, proj as i32]);
    let cqa = Array::from_slice(&cq, &[proj as i32, 4]);
    let cka = Array::from_slice(&ck, &[proj as i32, 4]);
    let cva = Array::from_slice(&cv, &[proj as i32, 4]);
    let ala = Array::from_slice(&alog, &[heads as i32]);
    let dtba = Array::from_slice(&dtb, &[proj as i32]);
    let csa = Array::from_slice(&cs, &[3, conv_dim as i32]);

    let (qo, ko, vo, go, cso) = kda_prepare(
        &qa, &ka, &va, &ga, &cqa, &cka, &cva, &ala, &dtba, &csa, heads, -5.0, 1e-5, &rt,
    )
    .expect("kda_prepare declined");
    let _ = cso.eval();
    let qo_v = qo.to_vec1::<f32>().expect("q readback");
    let ko_v = ko.to_vec1::<f32>().expect("k readback");
    let vo_v = vo.to_vec1::<f32>().expect("v readback");
    let go_v = go.to_vec1::<f32>().expect("g readback");
    let cso_v = cso.to_vec1::<f32>().expect("cs readback");

    let silu = |x: f32| x / (1.0 + (-x).exp());
    let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
    // The convolution state stores three q rows, then three k rows, then
    // three v rows. Each new token consumes the corresponding branch row.
    let window = |tok: usize, w: usize, ch: usize| -> (f32, f32, f32) {
        let r = tok + w;
        if r < 3 {
            (
                cs[r * proj + ch],
                cs[3 * proj + r * proj + ch],
                cs[6 * proj + r * proj + ch],
            )
        } else {
            let ti = r - 3;
            (q[ti * proj + ch], k[ti * proj + ch], v[ti * proj + ch])
        }
    };
    let mut q_ref = vec![0f32; t * proj];
    let mut k_ref = vec![0f32; t * proj];
    for tok in 0..t {
        for h in 0..heads {
            let (mut q_sq, mut k_sq) = (0.0f32, 0.0f32);
            let mut qq = vec![0f32; d];
            let mut kk = vec![0f32; d];
            for c in 0..d {
                let ch = h * d + c;
                let tap = ch * 4;
                let (mut qa, mut ka) = (0.0f32, 0.0f32);
                for w in 0..4usize {
                    let (wq, wk, _) = window(tok, w, ch);
                    qa += wq * cq[tap + w];
                    ka += wk * ck[tap + w];
                }
                let (qv, kvv) = (silu(qa), silu(ka));
                q_sq += qv * qv;
                k_sq += kvv * kvv;
                q_ref[tok * proj + ch] = qv;
                k_ref[tok * proj + ch] = kvv;

                qq[c] = qv;
                kk[c] = kvv;
            }
            let qn = (q_sq + 1e-6f32).sqrt().recip() * (128f32).powf(-0.5);
            let kn = (k_sq + 1e-6f32).sqrt().recip();
            for c in 0..d {
                q_ref[tok * proj + h * d + c] = qq[c] * qn;
                k_ref[tok * proj + h * d + c] = kk[c] * kn;
            }
        }
    }

    // q/k: l2-normalized and q scaled by D^-0.5.
    for i in 0..t * proj {
        assert!(
            (qo_v[i] - q_ref[i]).abs() < 2e-4,
            "q mismatch at {i}: got {} want {}",
            qo_v[i],
            q_ref[i]
        );
        assert!(
            (ko_v[i] - k_ref[i]).abs() < 2e-4,
            "k mismatch at {i}: got {} want {}",
            ko_v[i],
            k_ref[i]
        );
    }
    // v: silu(conv)
    for tok in 0..t {
        for h in 0..heads {
            for c in 0..d {
                let ch = h * d + c;
                let tap = ch * 4;
                let mut va = 0.0f32;
                for w in 0..4usize {
                    let (_, _, wv) = window(tok, w, ch);
                    va += wv * cv[tap + w];
                }
                let want = silu(va);
                let got = vo_v[tok * proj + ch];
                assert!(
                    (got - want).abs() < 1e-4,
                    "v mismatch tok={tok} ch={ch}: got {got} want {want}"
                );
            }
        }
    }
    // g: compute_g_safe
    for i in 0..t * proj {
        let gt = gate[i] + dtb[i % proj];
        let want = (-5.0f32 * sig(alog[(i % proj) / d].exp() * gt)).exp();
        let got = go_v[i];
        assert!(
            (got - want).abs() < 1e-4 + 1e-3 * want,
            "g mismatch at {i}: got {got} want {want}"
        );
    }
    // The result remains branch-major: 3 q rows, 3 k rows, then 3 v rows.
    for w in 0..3usize {
        for ch in 0..conv_dim {
            let r = t + w;
            let branch = ch / proj;
            let channel = ch % proj;
            let want = if r >= 3 {
                let ti = r - 3;
                match branch {
                    0 => q[ti * proj + channel],
                    1 => k[ti * proj + channel],
                    _ => v[ti * proj + channel],
                }
            } else {
                cs[branch * 3 * proj + r * proj + channel]
            };
            let got = cso_v[branch * 3 * proj + w * proj + channel];
            assert!(
                (got - want).abs() < 1e-5,
                "conv_state mismatch [{w},{ch}]: got {got} want {want}"
            );
        }
    }
}

#[test]
fn kda_prepare_two_calls() {
    // Replicates the real forward: two back-to-back prepare calls with
    // fresh large inputs, t=5 then t=6, real-shaped tensors.
    let rt = Stream::thread_local_or_default();
    let heads = 64usize;
    let d = 128usize;
    let proj = heads * d;
    let conv_dim = 3 * proj;
    let mut rng_val = 12345u32;
    let mut next = || {
        rng_val = rng_val.wrapping_mul(1664525).wrapping_add(1013904223);
        (rng_val >> 8) as f32 / 8388608.0 - 0.5
    };
    // warmup: one wrapper-op eval before the first kernel call (mirrors
    // the reference test, which computes wrapper ops before kda_prepare)
    let warm = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);
    let _ = warm.exp().unwrap().to_vec1::<f32>().unwrap();
    for &t in &[5usize, 6usize] {
        let n = t * proj;
        let q: Vec<f32> = (0..n).map(|_| next()).collect();
        let k: Vec<f32> = (0..n).map(|_| next()).collect();
        let v: Vec<f32> = (0..n).map(|_| next()).collect();
        let gate: Vec<f32> = (0..n).map(|_| next()).collect();
        let cq: Vec<f32> = (0..proj * 4).map(|_| next()).collect();
        let ck: Vec<f32> = (0..proj * 4).map(|_| next()).collect();
        let cv: Vec<f32> = (0..proj * 4).map(|_| next()).collect();
        let alog: Vec<f32> = (0..heads)
            .map(|i| -0.25f32 + (i % 7) as f32 * 0.01)
            .collect();
        let dtb: Vec<f32> = (0..proj).map(|_| next()).collect();
        let cs: Vec<f32> = (0..3 * conv_dim).map(|_| next()).collect();
        let qa = Array::from_slice(&q, &[1, t as i32, proj as i32]);
        let ka = Array::from_slice(&k, &[1, t as i32, proj as i32]);
        let va = Array::from_slice(&v, &[1, t as i32, proj as i32]);
        let ga = Array::from_slice(&gate, &[1, t as i32, proj as i32]);
        let cqa = Array::from_slice(&cq, &[proj as i32, 4]);
        let cka = Array::from_slice(&ck, &[proj as i32, 4]);
        let cva = Array::from_slice(&cv, &[proj as i32, 4]);
        let ala = Array::from_slice(&alog, &[heads as i32]);
        let dtba = Array::from_slice(&dtb, &[proj as i32]);
        let csa = Array::from_slice(&cs, &[3, conv_dim as i32]);
        let out = kda_prepare(
            &qa, &ka, &va, &ga, &cqa, &cka, &cva, &ala, &dtba, &csa, heads, -5.0, 1.0e-5, &rt,
        )
        .expect("prepare call");
        let (qf, _, _, _, _) = out;
        let _ = qf.eval();
        let qv = qf.to_vec1::<f32>().unwrap();
        let nz = qv.iter().filter(|x| x.abs() > 1e-9).count();
        assert!(
            nz > n / 2,
            "call t={t}: only {nz}/{n} nonzero — lane coverage broken"
        );
    }
}

#[test]
fn kda_output_norm_matches_naive() {
    let rt = Stream::thread_local_or_default();
    let heads = 64usize;
    let d = 128usize;
    let t = 5usize;
    let proj = heads * d;
    let mut rng_val = 777u32;
    let mut next = || {
        rng_val = rng_val.wrapping_mul(1664525).wrapping_add(1013904223);
        (rng_val >> 8) as f32 / 8388608.0 - 0.5
    };
    let y: Vec<f32> = (0..t * proj).map(|_| next() * 3.0).collect();
    let gate: Vec<f32> = (0..t * proj).map(|_| next() * 2.0).collect();
    let w: Vec<f32> = (0..d).map(|_| next()).collect();
    let ya = Array::from_slice(&y, &[t as i32, proj as i32]);
    let ga = Array::from_slice(&gate, &[t as i32, proj as i32]);
    let wa = Array::from_slice(&w, &[d as i32]);
    let out = kda_output_norm(&ya, &ga, &wa, heads, 1e-5, &rt).expect("output kernel");
    let _ = out.eval();
    let ov = out.to_vec1::<f32>().unwrap();
    // naive: per (token, head): rms over d, then w * sigmoid(gate)
    let mut bad = 0usize;
    for tok in 0..t {
        for h in 0..heads {
            let base = tok * proj + h * d;
            let ss: f32 = y[base..base + d].iter().map(|x| x * x).sum();
            let scale = (ss / d as f32 + 1e-5).sqrt().recip();
            for c in 0..d {
                let want = y[base + c] * scale * w[c] / (1.0 + (-gate[base + c]).exp());
                let got = ov[base + c];
                if (got - want).abs() > 1e-3 + 1e-2 * want.abs() {
                    bad += 1;
                    if bad < 4 {
                        eprintln!("[out-norm] tok={tok} h={h} c={c}: got {got} want {want}");
                    }
                }
            }
        }
    }
    eprintln!("[out-norm] bad: {bad}");
    for tok in 0..t {
        let nz: Vec<usize> = (0..heads)
            .filter(|&h| ov[tok * proj + h * d].abs() > 1e-9)
            .collect();
        eprintln!(
            "[out-norm] tok={tok} nonzero heads: {}..{}",
            nz.first().unwrap_or(&0),
            nz.last().unwrap_or(&0),
        );
    }
    assert_eq!(bad, 0);
}

#[test]
fn weighted_clamped_swiglu_matches_materialized_chain_bits() -> anyhow::Result<()> {
    let stream = Stream::thread_local_or_default();
    // Every finite BF16 value in both input roles, plus F32 values around
    // the clamp edges. The actual standalone GPU chain is the oracle.
    let mut values = (0..=u16::MAX)
        .map(|v| f32::from_bits((v as u32) << 16))
        .filter(|v| v.is_finite())
        .collect::<Vec<_>>();
    for edge in [-10.0f32, 10.0, 0.0] {
        for delta in -16i64..=16 {
            values.push(f32::from_bits(
                (edge.to_bits() as i64 + delta).clamp(0, u32::MAX as i64) as u32,
            ));
        }
    }
    values.retain(|v| v.is_finite());
    for width in [1usize, 257, 1536] {
        let rows = values.len().div_ceil(width);
        let n = rows * width;
        let gate = (0..n).map(|i| values[i % values.len()]).collect::<Vec<_>>();
        let up = (0..n)
            .map(|i| values[(i * 7919 + 11) % values.len()])
            .collect::<Vec<_>>();
        let weights = (0..rows)
            .map(|i| [0., -0., 0.125, 0.31, 1., 2.5][i % 6])
            .collect::<Vec<f32>>();
        // Contiguous views with nonzero backing offsets must bind correctly.
        let gate = Array::from_slice(
            &[gate.clone(), gate].concat(),
            &[2, rows as i32, width as i32],
        )
        .index((1..2, .., ..));
        let up = Array::from_slice(&[up.clone(), up].concat(), &[2, rows as i32, width as i32])
            .index((1..2, .., ..));
        let weights = Array::from_slice(&[weights.clone(), weights].concat(), &[2, rows as i32])
            .index((1..2, ..));
        for limit in [0., 10.] {
            let activation =
                swiglu_fused(&gate, &up, limit, &stream).expect("standalone activation");
            let expected = activation
                .multiply(&weights.reshape(&[1, rows as i32, 1])?)?
                .contiguous()?;
            let actual = weighted_clamped_swiglu(&gate, &up, &weights, limit, &stream)
                .expect("weighted activation");
            let expected = expected.to_vec1::<f32>()?;
            let actual = actual.to_vec1::<f32>()?;
            for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan(),
                    "width={width}, limit={limit}, i={i}: {a:?} != {b:?}"
                );
            }
        }
    }
    Ok(())
}
