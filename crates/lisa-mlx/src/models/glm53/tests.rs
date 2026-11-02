use super::*;
use crate::ops::indexing::IndexOp;
use crate::{Array, Stream};

#[test]
fn glm53_moe_expert_route_map_matches_token_major_oracle() {
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
        .chunks_exact(N_TOKENS)
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
fn glm53_affine_q2_mpp_matches_half_weight_oracle() {
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
    let actual = crate::jit::glm53_affine_q2_mpp_prefill(
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

#[cfg(feature = "glm53-fixtures")]
#[test]
fn glm53_dsa_qkv_project_pair_matches_separate_mlx_and_ds4_paths() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-dsa-qkv-pair] skipping: no Metal device available");
        return;
    }
    const BATCH_FIXTURE: &[u8] = include_bytes!("fixtures/dsa_qkv_batch_ds4_f32.bin");
    const IN: usize = 4096;
    const Q_ROWS: usize = 1536;
    const KV_ROWS: usize = 512;
    let batch_fixture: Vec<f32> = BATCH_FIXTURE
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("fixture float width")))
        .collect();
    assert_eq!(batch_fixture.len(), 2 * KV_ROWS);
    let stream = Stream::thread_local_or_default();
    let packed = |rows: usize, salt: u32| {
        (0..rows * (IN / 4))
            .map(|i| (i as u32).wrapping_mul(0x9e37_79b9).wrapping_add(salt))
            .collect::<Vec<_>>()
    };
    let scales = |rows: usize, salt: usize| {
        (0..rows * (IN / 64))
            .map(|i| 0.0005 + ((i * 13 + salt * 7) % 97) as f32 * 0.0001)
            .collect::<Vec<_>>()
    };
    let biases = |rows: usize, salt: usize| {
        (0..rows * (IN / 64))
            .map(|i| (((i * 19 + salt * 11) % 61) as f32 - 30.0) * 0.0001)
            .collect::<Vec<_>>()
    };
    let q_w = Array::from_slice(&packed(Q_ROWS, 0x1234_5678), &[Q_ROWS as i32, 1024]);
    let q_s = Array::from_slice(&scales(Q_ROWS, 1), &[Q_ROWS as i32, 64]);
    let q_b = Array::from_slice(&biases(Q_ROWS, 2), &[Q_ROWS as i32, 64]);
    let kv_w = Array::from_slice(&packed(KV_ROWS, 0x8765_4321), &[KV_ROWS as i32, 1024]);
    let kv_s = Array::from_slice(&scales(KV_ROWS, 3), &[KV_ROWS as i32, 64]);
    let kv_b = Array::from_slice(&biases(KV_ROWS, 4), &[KV_ROWS as i32, 64]);
    let x_values: Vec<f32> = (0..IN)
        .map(|i| ((i * 29 % 127) as f32 - 63.0) * 0.0017)
        .collect();
    let x = Array::from_slice(&x_values, &[1, 1, IN as i32]);

    let (q_actual, kv_actual) =
        glm53_dsa_qkv_project_pair(&q_w, &q_s, &q_b, &kv_w, &kv_s, &kv_b, &x, &stream)
            .expect("paired DSA q_a/kv_a projection should dispatch");
    let q_expected = crate::ops::quantized_matmul(
        &x.reshape(&[1, IN as i32]).expect("reshape query input"),
        &q_w,
        &q_s,
        Some(&q_b),
        true,
        64,
        8,
    )
    .expect("stock MLX qmv_fast query projection");
    let kv_expected = glm53_affine_q8_mv_ds4(
        &kv_w,
        &kv_s,
        &kv_b,
        &x.reshape(&[1, IN as i32]).expect("reshape kv input"),
        &stream,
    )
    .expect("DS4 graph MLX-affine kv_a matvec");
    let compare = |actual: &Array, expected: &Array| {
        let actual = actual.to_vec1::<f32>().expect("read paired projection");
        let expected = expected.to_vec1::<f32>().expect("read separate projection");
        let (max_abs, mismatch_count) = actual.iter().zip(&expected).fold(
            (0.0f32, 0usize),
            |(max_abs, mismatch_count), (&got, &want)| {
                (
                    max_abs.max((got - want).abs()),
                    mismatch_count + usize::from(got.to_bits() != want.to_bits()),
                )
            },
        );
        (max_abs, mismatch_count)
    };
    let (q_abs, q_mismatches) = compare(&q_actual, &q_expected);
    let (kv_abs, kv_mismatches) = compare(&kv_actual, &kv_expected);
    assert!(
        q_mismatches == 0,
        "paired q_a must match stock MLX qmv_fast bit-for-bit (max abs {q_abs:e})"
    );
    assert!(
        kv_mismatches == 0,
        "paired kv_a must match DS4's graph MLX-affine matvec bit-for-bit (max abs {kv_abs:e})"
    );

    let x2_values: Vec<f32> = (0..2 * IN)
        .map(|i| (((i * 43 + 5) % 239) as f32 - 119.0) * 0.0007)
        .collect();
    let x2 = Array::from_slice(&x2_values, &[2, 1, IN as i32]);
    let (q2, kv2) = glm53_dsa_qkv_project_pair(&q_w, &q_s, &q_b, &kv_w, &kv_s, &kv_b, &x2, &stream)
        .expect("two-row paired DSA q_a/kv_a projection should dispatch");
    if let Some(dir) = std::env::var_os("LISA_GLM53_DSA_QKV_CAPTURE_DIR") {
        let dir = std::path::PathBuf::from(dir);
        std::fs::create_dir_all(&dir).expect("create DSA QKV capture directory");
        let write = |name: &str, values: Vec<u8>| {
            std::fs::write(dir.join(name), values).expect("write DSA QKV capture file");
        };
        let kv_packed = kv_w.t.to_vec::<u32>().expect("read packed kv weights");
        let kv_scale = kv_s.t.to_vec::<f32>().expect("read kv scales");
        let kv_bias = kv_b.t.to_vec::<f32>().expect("read kv biases");
        let paired = kv2.t.to_vec::<f32>().expect("read paired kv output");
        write(
            "input.f32",
            x2_values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        );
        write(
            "kv_weight.u32",
            kv_packed.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        );
        write(
            "kv_scales.f32",
            kv_scale.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        );
        write(
            "kv_biases.f32",
            kv_bias.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        );
        write(
            "lisa_output.f32",
            paired.iter().flat_map(|v| v.to_ne_bytes()).collect(),
        );
    }
    for row in 0..2 {
        let row_index = row as usize;
        let x_row = x2.index((row..row + 1, ..));
        let q_expected = crate::ops::quantized_matmul(
            &x_row.reshape(&[1, IN as i32]).expect("reshape query row"),
            &q_w,
            &q_s,
            Some(&q_b),
            true,
            64,
            8,
        )
        .expect("serial MLX qmv_fast query row");
        let q_actual = q2
            .index((row..row + 1, ..))
            .reshape(&[1, Q_ROWS as i32])
            .expect("reshape batched query row")
            .to_vec1::<f32>()
            .expect("read batched query row");
        let q_expected = q_expected.to_vec1::<f32>().expect("read serial query row");
        let kv_actual = kv2
            .index((row..row + 1, ..))
            .reshape(&[1, KV_ROWS as i32])
            .expect("reshape batched kv row")
            .to_vec1::<f32>()
            .expect("read batched kv row");
        assert_eq!(q_actual, q_expected, "paired q_a batch row {row}");
        let kv_expected = &batch_fixture[row_index * KV_ROWS..(row_index + 1) * KV_ROWS];
        assert_eq!(kv_actual, kv_expected, "paired kv_a DS4 batch row {row}");
    }
}

#[cfg(feature = "glm53-fixtures")]
#[test]
fn glm53_dsa_value_project_matches_ds4_f32_mv4_fixture() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-dsa-value] skipping: no Metal device available");
        return;
    }
    const FIXTURE: &[u8] = include_bytes!("fixtures/dsa_value_project_ds4_f32_mv4.bin");
    const INPUT_BYTES: usize = 2 * 512 * std::mem::size_of::<f32>();
    const WEIGHT_BYTES: usize = 2 * 512 * 128 * std::mem::size_of::<u32>();
    const AFFINE_BYTES: usize = 2 * 512 * 8 * std::mem::size_of::<u16>();
    const OUTPUT_BYTES: usize = 2 * 256 * std::mem::size_of::<f32>();
    assert_eq!(
        FIXTURE.len(),
        INPUT_BYTES + WEIGHT_BYTES + 2 * AFFINE_BYTES + OUTPUT_BYTES
    );

    let f32s = |bytes: &[u8]| {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("F32 fixture word")))
            .collect::<Vec<_>>()
    };
    let input = f32s(&FIXTURE[..INPUT_BYTES]);
    let weights_start = INPUT_BYTES;
    let weights_end = weights_start + WEIGHT_BYTES;
    let weights = FIXTURE[weights_start..weights_end]
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("U32 fixture word")))
        .collect::<Vec<_>>();
    let scales_start = weights_end;
    let scales_end = scales_start + AFFINE_BYTES;
    let expand_bf16 = |bytes: &[u8]| {
        bytes
            .chunks_exact(2)
            .map(|chunk| {
                let bits = u16::from_le_bytes(chunk.try_into().expect("BF16 fixture word"));
                f32::from_bits(u32::from(bits) << 16)
            })
            .collect::<Vec<_>>()
    };
    let scales = expand_bf16(&FIXTURE[scales_start..scales_end]);
    let biases_start = scales_end;
    let biases_end = biases_start + AFFINE_BYTES;
    let biases = expand_bf16(&FIXTURE[biases_start..biases_end]);
    let expected = f32s(&FIXTURE[biases_end..biases_end + OUTPUT_BYTES]);

    let input = Array::from_slice(&input, &[2, 512]);
    let weights = Array::from_slice(&weights, &[1024, 128]);
    let scales = Array::from_slice(&scales, &[1024, 8]);
    let biases = Array::from_slice(&biases, &[1024, 8]);
    let stream = Stream::thread_local_or_default();
    let actual = dsa_value_project_rows(&input, &weights, &scales, &biases, 64, 2, &stream)
        .expect("DS4 F32 _4 value projection should dispatch")
        .to_vec1::<f32>()
        .expect("value projection readback");
    assert_eq!(actual.len(), expected.len());
    let first_mismatch = actual
        .iter()
        .zip(&expected)
        .position(|(got, want)| got.to_bits() != want.to_bits());
    assert!(
        first_mismatch.is_none(),
        "DS4 F32 _4 value projection differs at {first_mismatch:?}"
    );
}

#[test]
fn glm53_router_mv_small_matches_matmul() {
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
fn glm53_router_mv_plain_matches_matmul() {
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

fn glm53_bf16_row_reference(weights: &[f32], x: &[f32], in_dim: usize) -> Vec<f32> {
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
    let actual = glm53_bf16_gemv(weights, x, stream).expect("two-row BF16 GEMV should dispatch");
    assert_eq!(actual.shape(), &[2, out_dim as i32]);
    let actual = actual.to_vec1::<f32>().expect("read two-row BF16 GEMV");
    for row in 0..2 {
        // A one-row call retains the original DS4 GEMV kernel, so the
        // reference never enters the new shared-weight two-row dispatch.
        let reference_row = Array::from_slice(
            &reference_x[row * in_dim..(row + 1) * in_dim],
            &[1, in_dim as i32],
        );
        let expected = glm53_bf16_gemv(reference_weights, &reference_row, stream)
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
fn glm53_bf16_rows2_gemv_matches_single_rows_bitwise_at_tails() {
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
fn glm53_bf16_rows2_gemv_matches_single_rows_bitwise_at_kda_shapes() {
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
fn glm53_bf16_rows2_gemv_matches_single_rows_bitwise_with_views() {
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
fn glm53_bf16_projection_dispatch_covers_all_rows() {
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
    let gemv = glm53_bf16_gemv(&gemv_w, &x_batch, &stream)
        .expect("BF16 GEMV should dispatch for every row");
    let gemv_out = gemv.to_vec1::<f32>().expect("read GEMV output");
    let mut gemv_want = Vec::with_capacity(rows * out_dim);
    for row in 0..rows {
        gemv_want.extend(glm53_bf16_row_reference(
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
        assert!(glm53_bf16_qkv(&q_w, &k_w, &v_w, &x_decode, &stream).is_none());
    }
    let (q, k, v) = glm53_bf16_qkv_inner(&q_w, &k_w, &v_w, &x_decode, &stream)
        .expect("fused BF16 qkv should dispatch for every output row");
    let x_row = &x_values[..in_dim];
    for (name, actual, weight) in [("Q", q, q_ref), ("K", k, k_ref), ("V", v, v_ref)] {
        let actual = actual.to_vec1::<f32>().expect("read qkv output");
        let expected = glm53_bf16_row_reference(&weight, x_row, in_dim);
        assert_bf16_row_close(name, &actual, &expected);
    }
}

#[test]
fn glm53_bf16_fused_kda_inputs_match_six_gemvs_bitwise() {
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
            let (packed, fused_fa, fused_ga) = glm53_bf16_kda_project_with_nsg(
                &q,
                &k,
                &v,
                &fa,
                &ga,
                &beta,
                &x,
                nsg_override,
                &stream,
            )
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
            glm53_bf16_gemv(weights, &x, &stream)
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
        let expected_fa = glm53_bf16_gemv(&fa, &x, &stream)
            .expect("reference f_a GEMV should dispatch")
            .to_vec1::<f32>()
            .expect("read reference f_a");
        let expected_ga = glm53_bf16_gemv(&ga, &x, &stream)
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
fn glm53_bf16_pair_gemv_matches_separate_ds4_rows_bitwise() {
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
        let (paired_a, paired_b) =
            glm53_bf16_pair_gemv(&weights_a, &x_a, &weights_b, &x_b, &stream)
                .expect("paired BF16 GEMV should dispatch");
        assert_eq!(
            crate::runtime_dispatch_count() - before,
            1,
            "rows={rows}: paired projections should use one dispatch"
        );
        let separate_a = glm53_bf16_gemv(&weights_a, &x_a, &stream)
            .expect("reference A GEMV should dispatch")
            .to_vec1::<f32>()
            .expect("read reference A GEMV");
        let separate_b = glm53_bf16_gemv(&weights_b, &x_b, &stream)
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
fn glm53_bf16_tiled_gemm_matches_ds4_operand_staging() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-bf16-gemm] skipping: no Metal device available");
        return;
    }
    // Cross both the 64-feature and 32-row tile edges: the first tile
    // takes ds4's direct store path and the remaining tiles exercise its
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
    let result = glm53_bf16_gemm(&weights, &x, &stream)
        .expect("tiled BF16 GEMM should dispatch for more than eight rows");
    let actual = result.to_vec1::<f32>().expect("read GEMM output");
    let rounded_weights = weights
        .as_dtype(Dtype::Float32)
        .expect("promote rounded BF16 weights")
        .to_vec1::<f32>()
        .expect("read rounded BF16 weights");

    // ds4 stages both BF16 weights and F32 activations through F16 before
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
fn glm53_bf16_tiled_gemm_reuses_long_k_tiles_safely() {
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
    let result =
        glm53_bf16_gemm(&weights, &x, &stream).expect("long-K tiled BF16 GEMM should dispatch");
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
fn glm53_router_selection_matches_reference_and_compiles() {
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
fn glm53_moe_sum8_matches_ds4_slot_order() {
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
fn glm53_moe_sum8_shared_q8_down_inline_scalars_matches_stock_qmv() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-moe-shared-down] skipping: no Metal device available");
        return;
    }
    #[cfg(feature = "glm53-fixtures")]
    use crate::ops::indexing::IndexOp;

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
fn glm53_hc_weighted_sum_matches_ds4_decode_order() {
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
fn glm53_hc_mix_rms_matches_reference() {
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
    let actual = glm53_hc_mix_rms(&input, 1e-6, &stream)
        .expect("ds4-shaped plain RMSNorm should dispatch")
        .to_vec1::<f32>()
        .expect("read normalized mHC input");
    let ones = Array::from_slice(&vec![1.0f32; width], &[width as i32]);
    let weighted = glm53_mtp_rms_norm(&input, &ones, 1e-6, &stream)
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

#[cfg(feature = "glm53-fixtures")]
#[test]
fn glm53_dsa_norms_match_ds4_metal_fixture() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-dsa-norm] skipping: no Metal device available");
        return;
    }
    const ROWS: usize = 8;
    const Q_DIM: usize = 1536;
    const KV_DIM: usize = 512;
    const Q_COUNT: usize = ROWS * Q_DIM;
    const KV_COUNT: usize = ROWS * KV_DIM;
    const FIXTURE: &[u8] = include_bytes!("fixtures/dsa_norm_ds4_metal.bin");
    assert_eq!(FIXTURE.len(), (Q_COUNT + KV_COUNT) * 4);

    let q_input: Vec<f32> = (0..Q_COUNT)
        .map(|i| (((i * 37 + 13) % 251) as f32 - 125.0) * 0.000244140625)
        .collect();
    let q_weight: Vec<f32> = (0..Q_DIM)
        .map(|i| 1.0 + ((i * 7 + 11) % 101) as f32 * 0.0078125)
        .collect();
    let kv_input: Vec<f32> = (0..KV_COUNT)
        .map(|i| (((i * 19 + 7) % 257) as f32 - 128.0) * 0.00048828125)
        .collect();
    let kv_weight: Vec<f32> = (0..KV_DIM)
        .map(|i| 0.75 + ((i * 11 + 5) % 89) as f32 * 0.00390625)
        .collect();
    let q_input = Array::from_slice(&q_input, &[ROWS as i32, Q_DIM as i32]);
    let q_weight = Array::from_slice(&q_weight, &[Q_DIM as i32]);
    let kv_input = Array::from_slice(&kv_input, &[ROWS as i32, KV_DIM as i32]);
    let kv_weight = Array::from_slice(&kv_weight, &[KV_DIM as i32]);
    let stream = Stream::thread_local_or_default();

    let actual_q = glm53_mtp_rms_norm(&q_input, &q_weight, 1e-6, &stream)
        .expect("DS4-order weighted q_a RMSNorm")
        .to_vec1::<f32>()
        .expect("read q_a RMSNorm output");
    let actual_kv = glm53_mtp_kv_lora_rms_norm(&kv_input, &kv_weight, 1e-6, &stream)
        .expect("DS4-order KV-LoRA RMSNorm")
        .to_vec1::<f32>()
        .expect("read KV-LoRA RMSNorm output");

    let expected_q = FIXTURE[..Q_COUNT * 4]
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("f32 fixture")));
    for (index, (actual, expected)) in actual_q.iter().zip(expected_q).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "weighted q_a RMSNorm differs from DS4 at {index}: {actual} vs {expected}"
        );
    }
    let expected_kv = FIXTURE[Q_COUNT * 4..]
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("f32 fixture")));
    for (index, (actual, expected)) in actual_kv.iter().zip(expected_kv).enumerate() {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "KV-LoRA RMSNorm differs from DS4 at {index}: {actual} vs {expected}"
        );
    }
}

#[cfg(feature = "glm53-fixtures")]
#[test]
fn glm53_qkv_norm_fusion_matches_ds4_fused_qkv_norm_store_fixture() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-qkv-norm-store] skipping: no Metal device available");
        return;
    }
    const ROWS: usize = 8;
    const Q_DIM: usize = 1536;
    const KV_DIM: usize = 512;
    const Q_COUNT: usize = ROWS * Q_DIM;
    const KV_COUNT: usize = ROWS * KV_DIM;
    const Q_BYTES: usize = Q_COUNT * 4;
    const FIXTURE: &[u8] = include_bytes!("fixtures/qkv_norm_store_ds4_metal.bin");
    assert_eq!(FIXTURE.len(), Q_BYTES + KV_COUNT * 2);

    let q_input: Vec<f32> = (0..Q_COUNT)
        .map(|i| (((i * 37 + 13) % 251) as f32 - 125.0) * 0.000244140625)
        .collect();
    let q_weight: Vec<f32> = (0..Q_DIM)
        .map(|i| 1.0 + ((i * 7 + 11) % 101) as f32 * 0.0078125)
        .collect();
    let kv_input: Vec<f32> = (0..KV_COUNT)
        .map(|i| (((i * 19 + 7) % 257) as f32 - 128.0) * 0.00048828125)
        .collect();
    let kv_weight: Vec<f32> = (0..KV_DIM)
        .map(|i| 0.75 + ((i * 11 + 5) % 89) as f32 * 0.00390625)
        .collect();
    let q_input = Array::from_slice(&q_input, &[ROWS as i32, Q_DIM as i32]);
    let q_weight = Array::from_slice(&q_weight, &[Q_DIM as i32]);
    let kv_input = Array::from_slice(&kv_input, &[ROWS as i32, KV_DIM as i32]);
    let kv_weight = Array::from_slice(&kv_weight, &[KV_DIM as i32]);
    let stream = Stream::thread_local_or_default();

    let actual_q = glm53_mtp_rms_norm(&q_input, &q_weight, 1e-6, &stream)
        .expect("DS4-order q_a norm should dispatch")
        .to_vec1::<f32>()
        .expect("read q_a norm output");
    let actual_kv = glm53_mtp_kv_lora_rms_norm(&kv_input, &kv_weight, 1e-6, &stream)
        .expect("DS4-order kv_a norm should dispatch")
        .as_dtype(Dtype::Float16)
        .expect("cast compact cache rows to F16")
        .to_vec1::<half::f16>()
        .expect("read compact cache rows");
    let (fused_q, fused_kv) =
        glm53_qkv_norm_compact(&q_input, &q_weight, &kv_input, &kv_weight, 1e-6, &stream)
            .expect("fused DS4-order q_a/kv_a norm should dispatch");
    let fused_q = fused_q.to_vec1::<f32>().expect("read fused q_a norm");
    let fused_kv = fused_kv
        .to_vec1::<half::f16>()
        .expect("read fused compact KV rows");

    for (index, actual) in actual_q.iter().enumerate() {
        let start = index * 4;
        let expected = f32::from_le_bytes(
            FIXTURE[start..start + 4]
                .try_into()
                .expect("f32 fixture value"),
        );
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "q_a output differs from DS4 fused norm/store at {index}: {actual} vs {expected}"
        );
        assert_eq!(
            fused_q[index].to_bits(),
            actual.to_bits(),
            "fused q_a norm differs from standalone DS4-order norm at {index}"
        );
    }
    for (index, actual) in actual_kv.iter().enumerate() {
        let start = Q_BYTES + index * 2;
        let expected = u16::from_le_bytes([FIXTURE[start], FIXTURE[start + 1]]);
        assert_eq!(
            actual.to_bits(),
            expected,
            "compact F16 cache output differs from DS4 fused norm/store at {index}"
        );
        assert_eq!(
            fused_kv[index].to_bits(),
            actual.to_bits(),
            "fused compact norm differs from standalone DS4-order norm at {index}"
        );
    }

    // The production decode/prefill route binds the compact cache as a
    // preallocated kernel output, so exercise its nonzero absolute row
    // offset and make sure rows outside the write remain untouched.
    const CACHE_ROWS: usize = 16;
    const CACHE_POS: usize = 4;
    let q_input_3d = q_input.reshape(&[1, ROWS as i32, Q_DIM as i32]).unwrap();
    let kv_input_3d = kv_input.reshape(&[1, ROWS as i32, KV_DIM as i32]).unwrap();
    let cache_seed = half::f16::from_f32(-3.0);
    let cache = Array::from_slice(
        &vec![cache_seed; CACHE_ROWS * KV_DIM],
        &[1, CACHE_ROWS as i32, KV_DIM as i32],
    );
    let direct_q = glm53_qkv_norm_compact_into(
        &q_input_3d,
        &q_weight,
        &kv_input_3d,
        &kv_weight,
        &cache,
        CACHE_POS,
        1e-6,
        &stream,
    )
    .expect("fused DS4-order norm should write compact rows into cache");
    let direct_q = direct_q.to_vec1::<f32>().expect("read direct q norm");
    let direct_cache = cache
        .to_vec1::<half::f16>()
        .expect("read direct compact cache");
    for (index, actual) in direct_q.iter().enumerate() {
        let start = index * 4;
        let expected = f32::from_le_bytes(
            FIXTURE[start..start + 4]
                .try_into()
                .expect("f32 fixture value"),
        );
        assert_eq!(actual.to_bits(), expected.to_bits(), "direct q_a[{index}]");
    }
    for row in 0..CACHE_ROWS {
        for col in 0..KV_DIM {
            let index = row * KV_DIM + col;
            if (CACHE_POS..CACHE_POS + ROWS).contains(&row) {
                let latent_index = (row - CACHE_POS) * KV_DIM + col;
                let fixture_index = Q_BYTES + latent_index * 2;
                let expected =
                    u16::from_le_bytes([FIXTURE[fixture_index], FIXTURE[fixture_index + 1]]);
                assert_eq!(
                    direct_cache[index].to_bits(),
                    expected,
                    "direct compact cache differs from DS4 at row {row}, col {col}"
                );
            } else {
                assert_eq!(
                    direct_cache[index].to_bits(),
                    cache_seed.to_bits(),
                    "direct compact cache modified untouched row {row}, col {col}"
                );
            }
        }
    }
}

#[test]
fn glm53_mtp_add3_matches_left_associated_f32_reference() {
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
    let got = glm53_mtp_add3(&a_array, &b_array, &c_array, &stream)
        .expect("DS4-ordered MTP add3 should dispatch")
        .to_vec1::<f32>()
        .expect("read MTP add3 result");
    for i in 0..n {
        let want = a[i] + b[i] + c[i];
        assert_eq!(got[i].to_bits(), want.to_bits(), "MTP add3[{i}]");
    }
}

#[test]
fn glm53_moe_gather_kernels_compile_on_packed_q2_layout() {
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
fn glm53_moe_nonzero_q2_gathers_match_affine_qmv() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-moe-q2] skipping: no Metal device available");
        return;
    }
    #[cfg(feature = "glm53-fixtures")]
    use crate::ops::indexing::IndexOp;

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

    // The DS4-shaped down kernel loops over the eight selected experts
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
fn glm53_mhc_expand_and_swiglu_compile_and_match() {
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

#[cfg(feature = "glm53-fixtures")]
#[test]
fn glm53_shared_affine_q8_matches_ds4_q8_0_schedule_fixture() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-shared-q8-ds4] skipping: no Metal device available");
        return;
    }
    const IN: usize = 64;
    const OUT: usize = 2;
    const GROUP: usize = 64;
    const FIXTURE: &[u8] = include_bytes!("fixtures/shared_q8_swiglu_ds4.bin");
    assert_eq!(FIXTURE.len(), 6 * std::mem::size_of::<f32>());
    let expected: Vec<f32> = FIXTURE
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("fixture f32")))
        .collect();

    let pack_affine = |signed: &[i8]| -> Vec<u32> {
        signed
            .chunks_exact(4)
            .map(|values| {
                let codes = [
                    (i16::from(values[0]) + 128) as u8,
                    (i16::from(values[1]) + 128) as u8,
                    (i16::from(values[2]) + 128) as u8,
                    (i16::from(values[3]) + 128) as u8,
                ];
                u32::from_le_bytes(codes)
            })
            .collect()
    };
    let mut gate_signed = vec![1i8; IN];
    gate_signed.extend(vec![-1i8; 32]);
    gate_signed.extend(vec![0i8; 32]);
    let mut up_signed = vec![-1i8; IN];
    up_signed.extend(std::iter::repeat_n(1i8, IN));
    let gate_packed = pack_affine(&gate_signed);
    let up_packed = pack_affine(&up_signed);
    let mut gate_up_packed = gate_packed;
    gate_up_packed.extend(up_packed);
    let scales = Array::from_slice(&[1.0f32; 2 * OUT], &[(2 * OUT) as i32, 1]);
    let biases = Array::from_slice(&[-128.0f32; 2 * OUT], &[(2 * OUT) as i32, 1]);
    let x = Array::from_slice(&[0.125f32; IN], &[IN as i32]);
    let gate_up_weights = Array::from_slice(&gate_up_packed, &[(2 * OUT) as i32, (IN / 4) as i32]);
    let stream = Stream::thread_local_or_default();
    let (gate, up, mid) = shared_q8_swiglu_ds4(
        &x,
        &gate_up_weights,
        &scales,
        &biases,
        IN,
        OUT,
        GROUP,
        4.0,
        &stream,
    )
    .expect("DS4-schedule affine-Q8 fixture kernel should dispatch");
    let actual = [
        gate.to_vec1::<f32>().expect("read gate"),
        up.to_vec1::<f32>().expect("read up"),
        mid.to_vec1::<f32>().expect("read mid"),
    ]
    .concat();
    eprintln!("[glm53-shared-q8-ds4] actual={actual:?} expected={expected:?}");
    for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "DS4 Q8_0 schedule fixture value {index}: {got} != {want}"
        );
    }
}

#[test]
fn glm53_shared_ds4_q8_schedule_handles_odd_output_extent() {
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
    let scales = Array::from_slice(&vec![1.0f32; 2 * OUT], &[(2 * OUT) as i32, 1]);
    let biases = Array::from_slice(&vec![-128.0f32; 2 * OUT], &[(2 * OUT) as i32, 1]);
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

#[cfg(feature = "glm53-fixtures")]
#[test]
fn glm53_affine_q8_nax_prefill_matches_ds4_q8_0_tensor_fixture() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-q8-nax-prefill] skipping: no Metal device available");
        return;
    }
    const IN: usize = 4096;
    const OUT: usize = 4096;
    const TOKENS: usize = 96;
    const GROUP: usize = 64;
    const FIXTURE: &[u8] = include_bytes!("fixtures/q8_prefill_ds4_m96_k4096_n4096.bin");
    assert_eq!(FIXTURE.len(), TOKENS * OUT * std::mem::size_of::<f32>());
    let expected: Vec<f32> = FIXTURE
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("fixture f32")))
        .collect();

    let mut packed = Vec::with_capacity(OUT * IN / 4);
    for row in 0..OUT {
        let mut signed = Vec::with_capacity(IN);
        for k in 0..IN {
            let block = k / 32;
            let local_k = k % 32;
            signed.push(((row * 13 + block * 17 + local_k * 7) % 9) as i8 - 4);
        }
        for values in signed.chunks_exact(4) {
            let codes = [
                (i16::from(values[0]) + 128) as u8,
                (i16::from(values[1]) + 128) as u8,
                (i16::from(values[2]) + 128) as u8,
                (i16::from(values[3]) + 128) as u8,
            ];
            packed.push(u32::from_le_bytes(codes));
        }
    }
    let scales = Array::from_slice(
        &[1.0f32 / 128.0; OUT * (IN / GROUP)],
        &[OUT as i32, (IN / GROUP) as i32],
    );
    let biases = Array::from_slice(
        &[-1.0f32; OUT * (IN / GROUP)],
        &[OUT as i32, (IN / GROUP) as i32],
    );
    let x_values: Vec<f32> = (0..TOKENS)
        .flat_map(|token| {
            (0..IN).map(move |k| (((token * 31 + k * 17) % 257) as f32 - 128.0) * 0.00037f32)
        })
        .collect();
    let x = Array::from_slice(&x_values, &[TOKENS as i32, IN as i32]);
    let weights = Array::from_slice(&packed, &[OUT as i32, (IN / 4) as i32]);
    let actual = crate::jit::qmm_nax(
        x.t.device(),
        &x.t,
        &weights.t,
        &scales.t,
        &biases.t,
        GROUP as i32,
        8,
    )
    .expect("affine Q8 NAX prefill should dispatch");
    let actual = Array::new(actual)
        .to_vec1::<f32>()
        .expect("read NAX prefill output");
    let mut max_abs = 0.0f32;
    let mut mismatch_count = 0usize;
    let mut max_index = 0usize;
    for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
        if got.to_bits() != want.to_bits() {
            mismatch_count += 1;
        }
        let error = (got - want).abs();
        if error > max_abs {
            max_abs = error;
            max_index = index;
        }
    }
    eprintln!(
        "[glm53-q8-nax-prefill] NAX vs DS4 MPP: mismatches={mismatch_count}/{} max_abs={max_abs:e} at {max_index}",
        expected.len()
    );
    eprintln!(
        "[glm53-q8-nax-prefill] actual[0..8]={:?} expected[0..8]={:?}",
        &actual[..8],
        &expected[..8]
    );
    assert_eq!(mismatch_count, 0, "NAX prefill should match DS4 Q8_0 MPP");
}

#[test]
fn glm53_dsa_prefill_value_partition_matches_ds4() {
    assert_eq!(dsa_prefill_value_slices(34, 3), [(0, 3), (3, 31)]);
    assert_eq!(
        dsa_prefill_value_slices(4096, 2051),
        [(0, 2048), (2048, 3), (2051, 2045)]
    );
    assert_eq!(dsa_prefill_value_slices(2052, 0), [(0, 2048), (2048, 4)]);
    assert_eq!(dsa_prefill_value_slices(2048, 2051), [(0, 2048)]);
    assert!(dsa_prefill_value_slices(0, 2051).is_empty());
}

#[test]
fn glm53_dsa_value_native_splits_at_dense_sparse_boundary() {
    if !crate::device::metal_available() {
        return;
    }
    const HEADS: usize = 64;
    const KV: usize = 512;
    const TOKENS: usize = 34;
    let mut packed = vec![0u32; HEADS * KV * KV / 4];
    for row in 0..HEADS * KV {
        for col in 0..KV {
            let code = 128 + ((row * 13 + col * 3) % 17) as i32 - 8;
            packed[row * (KV / 4) + col / 4] |= (code as u32) << ((col % 4) * 8);
        }
    }
    let weights = Array::from_slice(&packed, &[(HEADS * KV) as i32, (KV / 4) as i32]);
    let scale = 0.34375f32;
    let scales = Array::from_slice(&vec![scale; HEADS * KV * 8], &[(HEADS * KV) as i32, 8]);
    let biases = Array::from_slice(
        &vec![-128.0 * scale; HEADS * KV * 8],
        &[(HEADS * KV) as i32, 8],
    );
    let input = (0..TOKENS * HEADS * KV)
        .map(|index| (((index * 13 + (index / KV) * 29) % 251) as f32 - 125.0) * 0.00021)
        .collect::<Vec<_>>();
    let input = Array::from_slice(&input, &[(TOKENS * HEADS) as i32, KV as i32]);
    let stream = Stream::thread_local_or_default();
    let _native = Glm53DsaNativeScope::new(true);
    let project = |input: &Array| {
        dsa_value_project_rows(input, &weights, &scales, &biases, 64, HEADS, &stream)
            .expect("native DSA value projection")
    };
    let first = project(&Array::new(input.t.narrow(0, 0, 3 * HEADS).unwrap()));
    let second = project(&Array::new(
        input.t.narrow(0, 3 * HEADS, 31 * HEADS).unwrap(),
    ));
    let expected = Array::concatenate(&[&first, &second], 0)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let unsliced = project(&input).to_vec1::<f32>().unwrap();
    assert!(
        unsliced
            .iter()
            .zip(&expected)
            .any(|(a, b)| a.to_bits() != b.to_bits()),
        "fixture must distinguish the 34-row matrix schedule from 3+31 scalar rows"
    );
    let actual = {
        let _slice = Glm53DsaValueSliceScope::new(2048, 2051);
        project(&input).to_vec1::<f32>().unwrap()
    };
    assert_eq!(
        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    let restored = project(&input).to_vec1::<f32>().unwrap();
    assert_eq!(
        restored.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        unsliced.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "slice scope must restore"
    );
}

#[test]
fn glm53_dsa_fused_decode_matches_native_ds4_fixtures() {
    let Some(root) = std::env::var_os("LISA_GLM53_DSA_NATIVE_FIXTURE_DIR") else {
        return;
    };
    if !crate::device::metal_available() {
        return;
    }
    const HEADS: usize = 64;
    const KV: usize = 512;
    let scale = half::f16::from_bits(0x3580).to_f32();
    let mut packed = vec![0u32; HEADS * KV * KV / 4];
    for row in 0..HEADS * KV {
        for col in 0..KV {
            let code = 128 + (((row / KV) * 7 + (row % KV) * 13 + col * 3) % 17) as i32 - 8;
            packed[row * (KV / 4) + col / 4] |= (code as u32) << ((col % 4) * 8);
        }
    }
    let weights = Array::from_slice(&packed, &[(HEADS * KV) as i32, (KV / 4) as i32]);
    let scales = Array::from_slice(&vec![scale; HEADS * KV * 8], &[(HEADS * KV) as i32, 8]);
    let biases = Array::from_slice(
        &vec![-128.0 * scale; HEADS * KV * 8],
        &[(HEADS * KV) as i32, 8],
    );
    let root = std::path::Path::new(&root);
    let stream = Stream::thread_local_or_default();
    let _scope = Glm53DsaNativeScope::new(true);
    for count in [1usize, 31, 32, 256, 2048, 2051] {
        let read = |stage: &str| {
            let path = root.join(format!("dsa-native-fused-{count}.{stage}.f32"));
            std::fs::read(&path)
                .unwrap_or_else(|error| panic!("{}: {error}", path.display()))
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let q_low = Array::from_slice(&read("qk-native"), &[1, 1, HEADS as i32, KV as i32]);
        let cache_cap = count + 7;
        let cache = (0..cache_cap * KV)
            .map(|index| {
                half::f16::from_f32(
                    (((index * 7 + (index / KV) * 19) % 251) as f32 - 125.0) * 0.0021f32,
                )
            })
            .collect::<Vec<_>>();
        let cache = Array::from_slice(&cache, &[1, cache_cap as i32, KV as i32]);
        let mut selected = (0..count)
            .map(|index| ((index * 17) % count + 3) as i32)
            .collect::<Vec<_>>();
        selected.resize(count.max(8), -1);
        let selected = Array::from_slice(&selected, &[1, 1, selected.len() as i32]);
        let expected = read("fused-native");
        for capture in [false, true] {
            let (actual, _) = dsa_indexed_decode_fused(
                &q_low, &cache, &selected, count, false, &weights, &scales, &biases, 64, 0.0625,
                &stream, capture,
            )
            .expect("native fused DSA dispatch");
            let actual = actual.to_vec1::<f32>().expect("native fused DSA output");
            assert_eq!(actual.len(), expected.len());
            let mismatches = actual
                .iter()
                .zip(&expected)
                .filter(|(actual, expected)| {
                    !actual.is_finite() || actual.to_bits() != expected.to_bits()
                })
                .count();
            eprintln!(
                "[glm53-dsa-fused-native] selected={count} capture={capture} exact={}/{}",
                actual.len() - mismatches,
                actual.len()
            );
            assert_eq!(
                mismatches, 0,
                "native fused DSA mismatch at selected={count}, capture={capture}"
            );
        }
    }
}

#[test]
fn glm53_dsa_projections_match_native_ds4_boundary_fixtures() {
    let varied_root = std::env::var_os("LISA_GLM53_DSA_NATIVE_VARIED_FIXTURE_DIR");
    let varied = varied_root.is_some();
    let Some(root) = varied_root.or_else(|| std::env::var_os("LISA_GLM53_DSA_NATIVE_FIXTURE_DIR"))
    else {
        eprintln!("[glm53-dsa-native] skipping: LISA_GLM53_DSA_NATIVE_FIXTURE_DIR is unset");
        return;
    };
    if !crate::device::metal_available() {
        return;
    }
    const HEADS: usize = 64;
    const KV: usize = 512;
    const DIM: usize = 256;
    const GROUP: usize = 64;
    let scale = half::f16::from_bits(0x3580).to_f32();
    let mut packed = vec![0u32; HEADS * KV * KV / 4];
    for row in 0..HEADS * KV {
        let head = row / KV;
        let local = row % KV;
        for col in 0..KV {
            let code = 128 + ((head * 7 + local * 13 + col * 3) % 17) as i32 - 8;
            packed[row * (KV / 4) + col / 4] |= (code as u32) << ((col % 4) * 8);
        }
    }
    let weights = Array::from_slice(&packed, &[(HEADS * KV) as i32, (KV / 4) as i32]);
    let metadata = (0..HEADS * KV * KV / GROUP)
        .map(|index| {
            if varied {
                let row = index / (KV / GROUP);
                let head = row / KV;
                let local = row % KV;
                let group = index % (KV / GROUP);
                let bits = 0x3000 + ((head * 3 + (local / 32) * 5 + group * 7) % 8) * 0x180;
                half::f16::from_bits(bits as u16).to_f32()
            } else {
                scale
            }
        })
        .collect::<Vec<_>>();
    let scales = Array::from_slice(&metadata, &[(HEADS * KV) as i32, (KV / GROUP) as i32]);
    let biases = Array::from_slice(
        &metadata
            .iter()
            .map(|scale| -128.0 * scale)
            .collect::<Vec<_>>(),
        &[(HEADS * KV) as i32, (KV / GROUP) as i32],
    );
    let root = std::path::Path::new(&root);
    let stream = Stream::thread_local_or_default();
    let _scope = Glm53DsaNativeScope::new(true);
    for tokens in [1usize, 31, 32, 33, 96] {
        // The independent native DS4 probe stores queries in [T,H,D].
        // LISA's wrapper takes [B,H,T,D], so only permute their addresses.
        let mut query = vec![0.0f32; tokens * HEADS * DIM];
        for head in 0..HEADS {
            for token in 0..tokens {
                for col in 0..DIM {
                    let index = (token * HEADS + head) * DIM + col;
                    query[(head * tokens + token) * DIM + col] =
                        (((index * 17 + (index / DIM) * 31) % 257) as f32 - 128.0) * 0.00037f32;
                }
            }
        }
        let query = Array::from_slice(&query, &[1, HEADS as i32, tokens as i32, DIM as i32]);
        let lora = (0..tokens * HEADS * KV)
            .map(|index| (((index * 13 + (index / KV) * 29) % 251) as f32 - 125.0) * 0.00021f32)
            .collect::<Vec<_>>();
        let lora = Array::from_slice(&lora, &[(tokens * HEADS) as i32, KV as i32]);
        let qk = dsa_qk_lowrank_project(&query, &weights, &scales, &biases, GROUP, &stream)
            .expect("native DSA QK dispatch");
        let value =
            dsa_value_project_rows(&lora, &weights, &scales, &biases, GROUP, HEADS, &stream)
                .expect("native DSA value dispatch");
        for (name, result) in [("qk", qk), ("value", value)] {
            let prefix = if varied {
                "dsa-native-affine-varied"
            } else {
                "dsa-native-affine-fixed"
            };
            let path = root.join(format!("{prefix}-{tokens}.{name}-native.f32"));
            let expected =
                std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            let actual = result.to_vec1::<f32>().expect("native DSA output");
            assert_eq!(expected.len(), actual.len() * 4);
            let mismatches = actual
                .iter()
                .zip(expected.chunks_exact(4))
                .filter(|(actual, bytes)| {
                    !actual.is_finite()
                        || actual.to_bits() != u32::from_le_bytes((*bytes).try_into().unwrap())
                })
                .count();
            eprintln!(
                "[glm53-dsa-native] stage={name} tokens={tokens} exact={}/{}",
                actual.len() - mismatches,
                actual.len()
            );
            assert_eq!(
                mismatches, 0,
                "native DS4 DSA {name} mismatch at tokens={tokens}"
            );
        }
    }
}

#[test]
fn glm53_affine_q8_prefill_matches_native_ds4_boundary_fixtures() {
    let Some(root) = std::env::var_os("LISA_GLM53_Q8_NATIVE_FIXTURE_DIR") else {
        eprintln!("[glm53-q8-native] skipping: LISA_GLM53_Q8_NATIVE_FIXTURE_DIR is unset");
        return;
    };
    if !crate::device::metal_available() {
        return;
    }
    const IN: usize = 4096;
    const OUT: usize = 4096;
    const GROUP: usize = 64;
    // capture_ds4_q8_prefill.c receives this F16 bit pattern for the
    // independent native-Q8/BF16-affine boundary fixtures.
    let scale = half::f16::from_bits(0x3580).to_f32();
    assert_eq!(scale, 0.34375);
    let mut packed = vec![0u32; OUT * IN / 4];
    for row in 0..OUT {
        for k in 0..IN {
            let signed = ((row * 13 + (k / 32) * 17 + (k % 32) * 7) % 9) as i32 - 4;
            packed[row * (IN / 4) + k / 4] |= ((signed + 128) as u32) << ((k % 4) * 8);
        }
    }
    let weights = Array::from_slice(&packed, &[OUT as i32, (IN / 4) as i32]);
    let scales = Array::from_slice(
        &vec![scale; OUT * IN / GROUP],
        &[OUT as i32, (IN / GROUP) as i32],
    );
    let biases = Array::from_slice(
        &vec![-128.0 * scale; OUT * IN / GROUP],
        &[OUT as i32, (IN / GROUP) as i32],
    );
    let root = std::path::Path::new(&root);
    for metadata in [Dtype::Float32, Dtype::Bfloat16] {
        let scales = scales.as_dtype(metadata).expect("Q8 metadata cast");
        let biases = biases.as_dtype(metadata).expect("Q8 bias cast");
        for rows in [17usize, 31, 32, 64, 96, 128, 191, 192, 193, 256] {
            let path = root.join(format!("q8-native-affine-{rows}.f32"));
            let expected =
                std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
            assert_eq!(expected.len(), rows * OUT * 4);
            let input = (0..rows)
                .flat_map(|token| {
                    (0..IN)
                        .map(move |k| (((token * 31 + k * 17) % 257) as f32 - 128.0) * 0.00037f32)
                })
                .collect::<Vec<_>>();
            let input = Array::from_slice(&input, &[rows as i32, IN as i32]);
            let result = crate::jit::glm53_affine_q8_prefill(
                input.t.device(),
                &input.t,
                &weights.t,
                &scales.t,
                &biases.t,
            )
            .expect("native Q8 crossover candidate dispatch");
            let result = Array::new(result)
                .to_vec1::<f32>()
                .expect("native Q8 output");
            let mismatches = result
                .iter()
                .zip(expected.chunks_exact(4))
                .filter(|(actual, bytes)| {
                    !actual.is_finite()
                        || actual.to_bits() != u32::from_le_bytes((*bytes).try_into().unwrap())
                })
                .count();
            eprintln!(
                "[glm53-q8-native] metadata={metadata:?} rows={rows} exact={}/{}",
                result.len() - mismatches,
                result.len()
            );
            assert_eq!(
                mismatches, 0,
                "native DS4 Q8 mismatch at rows={rows}, metadata={metadata:?}"
            );
        }
    }
}

#[cfg(feature = "glm53-fixtures")]
#[test]
fn glm53_affine_q8_mpp_prefill_matches_ds4_with_half_weight_rounding() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-q8-mpp-prefill] skipping: no Metal device available");
        return;
    }
    const IN: usize = 4096;
    const OUT: usize = 4096;
    const TOKENS: usize = 96;
    const GROUP: usize = 64;
    const SCALE: f32 = 0.333251953125; // half bits 0x3555
    const FIXTURE: &[u8] = include_bytes!("fixtures/q8_prefill_ds4_m96_k4096_n4096_scale3555.bin");
    assert_eq!(FIXTURE.len(), TOKENS * OUT * std::mem::size_of::<f32>());
    let expected: Vec<f32> = FIXTURE
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("fixture f32")))
        .collect();

    let mut packed = Vec::with_capacity(OUT * IN / 4);
    for row in 0..OUT {
        for k in 0..IN {
            let block = k / 32;
            let local_k = k % 32;
            let signed = ((row * 13 + block * 17 + local_k * 7) % 9) as i8 - 4;
            let code = (i16::from(signed) + 128) as u8;
            let word_index = row * (IN / 4) + k / 4;
            let byte_shift = (k & 3) * 8;
            if k & 3 == 0 {
                packed.push(u32::from(code));
            } else {
                packed[word_index] |= u32::from(code) << byte_shift;
            }
        }
    }
    let scales = Array::from_slice(
        &[SCALE; OUT * (IN / GROUP)],
        &[OUT as i32, (IN / GROUP) as i32],
    );
    let biases = Array::from_slice(
        &[-128.0 * SCALE; OUT * (IN / GROUP)],
        &[OUT as i32, (IN / GROUP) as i32],
    );
    let x_values: Vec<f32> = (0..TOKENS)
        .flat_map(|token| {
            (0..IN).map(move |k| (((token * 31 + k * 17) % 257) as f32 - 128.0) * 0.00037f32)
        })
        .collect();
    let x = Array::from_slice(&x_values, &[TOKENS as i32, IN as i32]);
    let weights = Array::from_slice(&packed, &[OUT as i32, (IN / 4) as i32]);
    let actual = crate::jit::glm53_affine_q8_mpp_prefill(
        x.t.device(),
        &x.t,
        &weights.t,
        &scales.t,
        &biases.t,
    )
    .expect("DS4-schedule affine-Q8 MPP prefill should dispatch");
    let actual = Array::new(actual)
        .to_vec1::<f32>()
        .expect("read MPP prefill output");
    let mismatch_count = actual
        .iter()
        .zip(&expected)
        .filter(|(got, want)| got.to_bits() != want.to_bits())
        .count();
    let max_abs = actual
        .iter()
        .zip(&expected)
        .map(|(got, want)| (got - want).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "[glm53-q8-mpp-prefill] half-rounded candidate vs DS4: mismatches={mismatch_count}/{} max_abs={max_abs:e}",
        expected.len()
    );
    assert_eq!(mismatch_count, 0, "MPP prefill should match DS4 Q8_0");

    let stock_nax = crate::jit::qmm_nax(
        x.t.device(),
        &x.t,
        &weights.t,
        &scales.t,
        &biases.t,
        GROUP as i32,
        8,
    )
    .expect("stock affine Q8 NAX prefill should dispatch");
    let stock_nax = Array::new(stock_nax)
        .to_vec1::<f32>()
        .expect("read stock NAX prefill output");
    let stock_mismatches = stock_nax
        .iter()
        .zip(&expected)
        .filter(|(got, want)| got.to_bits() != want.to_bits())
        .count();
    eprintln!(
        "[glm53-q8-mpp-prefill] stock NAX vs DS4: mismatches={stock_mismatches}/{}",
        expected.len()
    );
    assert!(
        stock_mismatches > 0,
        "fixture must expose the stock path's missing half-weight rounding"
    );
}

#[cfg(feature = "glm53-fixtures")]
#[test]
fn glm53_affine_q8_mpp_prefill_matches_ds4_wide_token_tiles() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-q8-mpp-wide] skipping: no Metal device available");
        return;
    }
    const IN: usize = 4096;
    const OUT: usize = 4096;
    const GROUP: usize = 64;
    const SCALE: f32 = 0.333251953125;
    const FIXTURES: &[(usize, &[u8])] = &[
        (
            64,
            include_bytes!("fixtures/q8_prefill_ds4_m64_k4096_n4096_scale3555.bin"),
        ),
        (
            128,
            include_bytes!("fixtures/q8_prefill_ds4_m128_k4096_n4096_scale3555.bin"),
        ),
    ];
    let mut packed = Vec::with_capacity(OUT * IN / 4);
    for row in 0..OUT {
        for k in 0..IN {
            let block = k / 32;
            let local_k = k % 32;
            let signed = ((row * 13 + block * 17 + local_k * 7) % 9) as i8 - 4;
            let code = (i16::from(signed) + 128) as u8;
            let word_index = row * (IN / 4) + k / 4;
            if k & 3 == 0 {
                packed.push(u32::from(code));
            } else {
                packed[word_index] |= u32::from(code) << ((k & 3) * 8);
            }
        }
    }
    let scales = Array::from_slice(
        &[SCALE; OUT * (IN / GROUP)],
        &[OUT as i32, (IN / GROUP) as i32],
    );
    let biases = Array::from_slice(
        &[-128.0 * SCALE; OUT * (IN / GROUP)],
        &[OUT as i32, (IN / GROUP) as i32],
    );
    let weights = Array::from_slice(&packed, &[OUT as i32, (IN / 4) as i32]);
    for (tokens, fixture) in FIXTURES {
        assert_eq!(fixture.len(), tokens * OUT * std::mem::size_of::<f32>());
        let expected: Vec<f32> = fixture
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("fixture f32")))
            .collect();
        let inputs: Vec<f32> = (0..*tokens)
            .flat_map(|token| {
                (0..IN).map(move |k| (((token * 31 + k * 17) % 257) as f32 - 128.0) * 0.00037f32)
            })
            .collect();
        let x = Array::from_slice(&inputs, &[*tokens as i32, IN as i32]);
        let actual = crate::jit::glm53_affine_q8_mpp_prefill(
            x.t.device(),
            &x.t,
            &weights.t,
            &scales.t,
            &biases.t,
        )
        .expect("DS4 wide-token MPP specialization should dispatch");
        let actual = Array::new(actual)
            .to_vec1::<f32>()
            .expect("read wide-token MPP output");
        let mismatches = actual
            .iter()
            .zip(&expected)
            .filter(|(got, want)| got.to_bits() != want.to_bits())
            .count();
        eprintln!(
            "[glm53-q8-mpp-wide] M={tokens}: mismatches={mismatches}/{}",
            expected.len()
        );
        assert_eq!(mismatches, 0, "DS4 MPP token tile M={tokens} differs");
    }
}

#[test]
fn glm53_affine_q8_mpp_accepts_real_checkpoint_affine_groups() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-q8-mpp-checkpoint] skipping: no Metal device available");
        return;
    }
    let Some(root) = std::env::var_os("LISA_GLM53_AFFINE_MPP_CAPTURE_DIR") else {
        eprintln!("[glm53-q8-mpp-checkpoint] skipping: capture dir is unset");
        return;
    };
    const TOKENS: usize = 96;
    const GROUP: usize = 64;
    let root = std::path::Path::new(&root);
    let read = |name: &str| -> Vec<u8> {
        std::fs::read(root.join(name)).unwrap_or_else(|error| {
            panic!(
                "read checkpoint affine capture {}: {error}",
                root.join(name).display()
            )
        })
    };
    let packed_bytes = read("weight.u32");
    let scale_bytes = read("scales.bf16");
    let bias_bytes = read("biases.bf16");
    let input_bytes = root.join("input.f32");
    let input_width = if input_bytes.exists() {
        let bytes = read("input.f32");
        assert_eq!(bytes.len() % (TOKENS * 4), 0);
        bytes.len() / (TOKENS * 4)
    } else {
        4096
    };
    assert!(input_width >= 64 && input_width % GROUP == 0);
    assert_eq!(packed_bytes.len() % input_width, 0);
    let out = packed_bytes.len() / input_width;
    assert!(
        out >= 64 && out % 64 == 0,
        "invalid affine output width {out}"
    );
    assert_eq!(packed_bytes.len(), out * input_width);
    assert_eq!(scale_bytes.len(), out * (input_width / GROUP) * 2);
    assert_eq!(bias_bytes.len(), scale_bytes.len());
    let packed: Vec<u32> = packed_bytes
        .chunks_exact(4)
        .map(|bytes| u32::from_le_bytes(bytes.try_into().expect("U32 word")))
        .collect();
    let decode_bf16 = |bytes: &[u8]| -> Vec<f32> {
        bytes
            .chunks_exact(2)
            .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16))
            .collect()
    };
    let scales_f32 = decode_bf16(&scale_bytes);
    let biases_f32 = decode_bf16(&bias_bytes);
    let captured_input = root.join("input.f32").exists();
    let input: Vec<f32> = if captured_input {
        let bytes = read("input.f32");
        assert_eq!(bytes.len(), TOKENS * input_width * 4);
        bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("input f32")))
            .collect()
    } else {
        let one_row: Vec<f32> = (0..input_width)
            .map(|i| (((i * 31 + 17) % 251) as f32 - 125.0) * 0.0007)
            .collect();
        (0..TOKENS).flat_map(|_| one_row.iter().copied()).collect()
    };
    if captured_input {
        assert_ne!(
            &input[..input_width],
            &input[input_width..2 * input_width],
            "captured prefill activation must contain distinct token rows"
        );
        eprintln!(
            "[glm53-q8-mpp-checkpoint] using a varied, model-captured {TOKENS}-row layer-3 MoE activation"
        );
    }
    let x = Array::from_slice(&input, &[TOKENS as i32, input_width as i32]);
    let weights = Array::from_slice(&packed, &[out as i32, (input_width / 4) as i32]);
    let scales = Array::from_slice(&scales_f32, &[out as i32, (input_width / GROUP) as i32]);
    let biases = Array::from_slice(&biases_f32, &[out as i32, (input_width / GROUP) as i32]);
    let candidate = crate::jit::glm53_affine_q8_mpp_prefill(
        x.t.device(),
        &x.t,
        &weights.t,
        &scales.t,
        &biases.t,
    )
    .expect("checkpoint affine Q8 should dispatch through the DS4 schedule");
    let candidate = Array::new(candidate)
        .to_vec1::<f32>()
        .expect("read MPP output");
    let production_capture = root.join("production.f32");
    if production_capture.exists() {
        let bytes = read("production.f32");
        let expected: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("production F32")))
            .collect();
        assert_eq!(expected.len(), candidate.len());
        let mismatches = candidate
            .iter()
            .zip(&expected)
            .filter(|(got, want)| got.to_bits() != want.to_bits())
            .count();
        let half_out = out / 2;
        let segment_mismatches = |start: usize, width: usize| {
            (0..TOKENS * width)
                .filter(|&index| {
                    let row = index / width;
                    let col = index % width;
                    candidate[row * out + start + col].to_bits()
                        != expected[row * out + start + col].to_bits()
                })
                .count()
        };
        let mut error_sq = 0.0f64;
        let mut expected_sq = 0.0f64;
        let mut max_abs = 0.0f32;
        for (&got, &want) in candidate.iter().zip(&expected) {
            let error = (got - want).abs();
            max_abs = max_abs.max(error);
            error_sq += f64::from(error) * f64::from(error);
            expected_sq += f64::from(want) * f64::from(want);
        }
        let rel_l2 = error_sq.sqrt() / expected_sq.sqrt().max(1e-30);
        let dump = candidate
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        std::fs::write(root.join("candidate.f32"), dump)
            .expect("write isolated MPP candidate for differential analysis");
        eprintln!(
            "[glm53-q8-mpp-checkpoint] isolated MPP vs full-model captured projection: mismatches={mismatches}/{}, first_half={}, second_half={}, max_abs={max_abs:e}, rel_l2={rel_l2:e}",
            candidate.len(),
            segment_mismatches(0, half_out),
            segment_mismatches(half_out, half_out),
        );
    }
    let ds4_f16_dir = root.join("ds4-f16");
    std::fs::create_dir_all(&ds4_f16_dir).expect("create DS4 F16 capture directory");
    let write_capture = |name: &str, bytes: &[u8]| {
        std::fs::write(ds4_f16_dir.join(name), bytes)
            .unwrap_or_else(|error| panic!("write DS4 F16 capture {name}: {error}"));
    };
    let input_bytes: Vec<u8> = input.iter().flat_map(|value| value.to_le_bytes()).collect();
    let candidate_bytes: Vec<u8> = candidate
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    write_capture("input.f32", &input_bytes);
    write_capture("candidate.f32", &candidate_bytes);

    let mut staged_fma = Vec::with_capacity(out * input_width * 2);
    let mut staged_separate = Vec::with_capacity(out * input_width * 2);
    let mut half_rounding_differences = 0usize;
    for row in 0..out {
        for k in 0..input_width {
            let word = packed[row * (input_width / 4) + k / 4];
            let code = ((word >> ((k % 4) * 8)) & 0xff) as f32;
            let group_index = row * (input_width / GROUP) + k / GROUP;
            let scale = scales_f32[group_index];
            let bias = biases_f32[group_index];
            let fma_half = half::f16::from_f32(code.mul_add(scale, bias));
            let separate_half = half::f16::from_f32(code * scale + bias);
            if fma_half.to_bits() != separate_half.to_bits() {
                half_rounding_differences += 1;
            }
            staged_fma.extend_from_slice(&fma_half.to_bits().to_le_bytes());
            staged_separate.extend_from_slice(&separate_half.to_bits().to_le_bytes());
        }
    }
    write_capture("weights-fma.f16", &staged_fma);
    write_capture("weights-separate.f16", &staged_separate);
    eprintln!(
        "[glm53-q8-mpp-checkpoint] DS4 F16 oracle weights: {} affine values round differently between FMA and separate F32 dequantization",
        half_rounding_differences
    );
    let stock = crate::jit::affine_qmm(
        x.t.device(),
        &x.t,
        &weights.t,
        &scales.t,
        &biases.t,
        GROUP as i32,
        8,
    )
    .expect("stock affine Q8 prefill should dispatch");
    let stock = Array::new(stock)
        .to_vec1::<f32>()
        .expect("read stock output");
    let mut error_sq = 0.0f64;
    let mut reference_sq = 0.0f64;
    let mut max_abs = 0.0f32;
    let mismatches = candidate
        .iter()
        .zip(&stock)
        .filter(|(got, expected)| got.to_bits() != expected.to_bits())
        .count();
    for (&got, &expected) in candidate.iter().zip(&stock) {
        let error = (got - expected).abs();
        max_abs = max_abs.max(error);
        error_sq += f64::from(error) * f64::from(error);
        reference_sq += f64::from(expected) * f64::from(expected);
    }
    let rel_l2 = error_sq.sqrt() / reference_sq.sqrt().max(1e-30);
    eprintln!(
        "[glm53-q8-mpp-checkpoint] DS4 schedule vs stock MLX: mismatches={mismatches}/{}, max_abs={max_abs:e}, rel_l2={rel_l2:e}",
        stock.len()
    );
    if production_capture.exists() {
        let bytes = read("production.f32");
        let expected: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("production F32")))
            .collect();
        let stock_mismatches = stock
            .iter()
            .zip(&expected)
            .filter(|(got, want)| got.to_bits() != want.to_bits())
            .count();
        let mut stock_error_sq = 0.0f64;
        let mut expected_sq = 0.0f64;
        let mut stock_max_abs = 0.0f32;
        for (&got, &want) in stock.iter().zip(&expected) {
            let error = (got - want).abs();
            stock_max_abs = stock_max_abs.max(error);
            stock_error_sq += f64::from(error) * f64::from(error);
            expected_sq += f64::from(want) * f64::from(want);
        }
        let stock_rel_l2 = stock_error_sq.sqrt() / expected_sq.sqrt().max(1e-30);
        let mpp_mismatches = candidate
            .iter()
            .zip(&expected)
            .filter(|(got, want)| got.to_bits() != want.to_bits())
            .count();
        eprintln!(
            "[glm53-q8-mpp-checkpoint] stock vs full-model captured projection: mismatches={stock_mismatches}/{}, max_abs={stock_max_abs:e}, rel_l2={stock_rel_l2:e}; MPP mismatches={mpp_mismatches}",
            stock.len()
        );
        assert_eq!(mpp_mismatches, 0, "isolated MPP differs from model capture");
    }
    assert!(candidate.iter().all(|value| value.is_finite()));
}

#[cfg(feature = "glm53-fixtures")]
#[test]
fn glm53_mhc_expand_layer21_capture_matches_ds4_fixture() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-mhc-ds4-expand] skipping: no Metal device available");
        return;
    }
    const ROWS: usize = 4;
    const HIDDEN: usize = 4096;
    const STREAMS: usize = 4;
    const Y_VALUES: usize = ROWS * HIDDEN;
    const RESIDUAL_VALUES: usize = ROWS * STREAMS * HIDDEN;
    const POST_VALUES: usize = ROWS * STREAMS;
    const COMB_VALUES: usize = ROWS * STREAMS * STREAMS;
    const OUTPUT_VALUES: usize = RESIDUAL_VALUES;
    let fixture = include_bytes!("fixtures/mhc_expand_ds4_layer21_ids1234.f32");
    let values = fixture
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte float")))
        .collect::<Vec<_>>();
    assert_eq!(fixture.len() % 4, 0);
    assert_eq!(
        values.len(),
        Y_VALUES + RESIDUAL_VALUES + POST_VALUES + COMB_VALUES + OUTPUT_VALUES
    );
    let mut offset = 0;
    let y = Array::from_slice(
        &values[offset..offset + Y_VALUES],
        &[1, ROWS as i32, HIDDEN as i32],
    );
    offset += Y_VALUES;
    let residual = Array::from_slice(
        &values[offset..offset + RESIDUAL_VALUES],
        &[1, ROWS as i32, STREAMS as i32, HIDDEN as i32],
    );
    offset += RESIDUAL_VALUES;
    let post = Array::from_slice(
        &values[offset..offset + POST_VALUES],
        &[1, ROWS as i32, STREAMS as i32],
    );
    offset += POST_VALUES;
    let comb = Array::from_slice(
        &values[offset..offset + COMB_VALUES],
        &[1, ROWS as i32, STREAMS as i32, STREAMS as i32],
    );
    offset += COMB_VALUES;
    let expected = &values[offset..offset + OUTPUT_VALUES];
    let stream = Stream::thread_local_or_default();
    let actual = hc_expand4(&y, &residual, &post, &comb, &stream)
        .expect("mHC expand kernel should dispatch")
        .to_vec1::<f32>()
        .expect("read mHC expand output");
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "captured layer-21 mHC output {index}: {got} vs DS4 {want}"
        );
    }
}

#[cfg(feature = "glm53-fixtures")]
#[test]
fn glm53_mhc_ffn_expand_layer21_capture_matches_ds4_fixture() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-mhc-ds4-ffn-expand] skipping: no Metal device available");
        return;
    }
    const ROWS: usize = 4;
    const HIDDEN: usize = 4096;
    const STREAMS: usize = 4;
    const Y_VALUES: usize = ROWS * HIDDEN;
    const RESIDUAL_VALUES: usize = ROWS * STREAMS * HIDDEN;
    const POST_VALUES: usize = ROWS * STREAMS;
    const COMB_VALUES: usize = ROWS * STREAMS * STREAMS;
    const OUTPUT_VALUES: usize = RESIDUAL_VALUES;
    let fixture = include_bytes!("fixtures/mhc_ffn_expand_ds4_layer21_ids1234.f32");
    let values = fixture
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte float")))
        .collect::<Vec<_>>();
    assert_eq!(fixture.len() % 4, 0);
    assert_eq!(
        values.len(),
        Y_VALUES + RESIDUAL_VALUES + POST_VALUES + COMB_VALUES + OUTPUT_VALUES
    );
    let mut offset = 0;
    let y = Array::from_slice(
        &values[offset..offset + Y_VALUES],
        &[1, ROWS as i32, HIDDEN as i32],
    );
    offset += Y_VALUES;
    let residual = Array::from_slice(
        &values[offset..offset + RESIDUAL_VALUES],
        &[1, ROWS as i32, STREAMS as i32, HIDDEN as i32],
    );
    offset += RESIDUAL_VALUES;
    let post = Array::from_slice(
        &values[offset..offset + POST_VALUES],
        &[1, ROWS as i32, STREAMS as i32],
    );
    offset += POST_VALUES;
    let comb = Array::from_slice(
        &values[offset..offset + COMB_VALUES],
        &[1, ROWS as i32, STREAMS as i32, STREAMS as i32],
    );
    offset += COMB_VALUES;
    let expected = &values[offset..offset + OUTPUT_VALUES];
    let stream = Stream::thread_local_or_default();
    let actual = hc_expand4(&y, &residual, &post, &comb, &stream)
        .expect("FFN mHC expand kernel should dispatch")
        .to_vec1::<f32>()
        .expect("read FFN mHC expand output");
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "layer-21 FFN expanded mHC output {index}: {got} vs DS4 {want}"
        );
    }
}

#[test]
fn glm53_hc_mix_rms_bf16_gemv_matches_separate_chain_bitwise() {
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
            glm53_hc_mix_rms(&input, 1e-6, &stream).expect("separate mHC RMSNorm should dispatch");
        let expected = glm53_bf16_gemv(&weights, &normalized, &stream)
            .expect("separate BF16 GEMV should dispatch")
            .to_vec1::<f32>()
            .expect("read separate projection");
        let actual = glm53_hc_mix_rms_bf16_gemv(&weights, &input, 1e-6, &stream)
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

#[cfg(feature = "glm53-fixtures")]
#[test]
fn glm53_mhc_mix_rms_layer21_capture_matches_ds4_fixture() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-mhc-ds4-mix-rms] skipping: no Metal device available");
        return;
    }
    const ROWS: usize = 4;
    const WIDTH: usize = 4 * 4096;
    const VALUES: usize = ROWS * WIDTH;
    let fixture = include_bytes!("fixtures/mhc_mix_norm_ds4_layer21_ids1234.f32");
    assert_eq!(fixture.len(), VALUES * 4 * std::mem::size_of::<f32>());
    let values = fixture
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte float")))
        .collect::<Vec<_>>();
    let stream = Stream::thread_local_or_default();
    let mut offset = 0;
    for half in ["attention", "FFN"] {
        let input = Array::from_slice(
            &values[offset..offset + VALUES],
            &[ROWS as i32, WIDTH as i32],
        );
        offset += VALUES;
        let expected = &values[offset..offset + VALUES];
        offset += VALUES;
        let actual = glm53_hc_mix_rms(&input, 1e-5, &stream)
            .expect("mHC mixer RMSNorm kernel should dispatch")
            .to_vec1::<f32>()
            .expect("read normalized mHC mixer input");
        for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "layer-21 {half} mHC mixer RMSNorm[{index}]: {got} vs DS4 {want}"
            );
        }
    }
}

#[cfg(feature = "glm53-fixtures")]
#[test]
fn glm53_mhc_mix_projection_layer21_capture_matches_ds4_fixture() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-mhc-ds4-mix-projection] skipping: no Metal device available");
        return;
    }
    const ROWS: usize = 4;
    const WIDTH: usize = 4 * 4096;
    const MIXES: usize = ROWS * 24;
    const WEIGHT_BYTES: usize = 24 * WIDTH * std::mem::size_of::<u16>();
    const OUTPUT_BYTES: usize = MIXES * std::mem::size_of::<f32>();
    let norm_fixture = include_bytes!("fixtures/mhc_mix_norm_ds4_layer21_ids1234.f32");
    let projection_fixture = include_bytes!("fixtures/mhc_mix_projection_ds4_layer21_ids1234.bin");
    assert_eq!(
        norm_fixture.len(),
        ROWS * WIDTH * 4 * std::mem::size_of::<f32>()
    );
    assert_eq!(projection_fixture.len(), 2 * (WEIGHT_BYTES + OUTPUT_BYTES));
    let norm_values = norm_fixture
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte float")))
        .collect::<Vec<_>>();
    let stream = Stream::thread_local_or_default();
    let mut norm_offset = ROWS * WIDTH;
    let mut projection_offset = 0;
    for half in ["attention", "FFN"] {
        let mut weight_values = Vec::with_capacity(24 * WIDTH);
        for bytes in projection_fixture[projection_offset..projection_offset + WEIGHT_BYTES]
            .chunks_exact(std::mem::size_of::<u16>())
        {
            let bf16 = u16::from_le_bytes(bytes.try_into().expect("two-byte BF16"));
            weight_values.push(f32::from_bits(u32::from(bf16) << 16));
        }
        projection_offset += WEIGHT_BYTES;
        let weights = Array::from_slice(&weight_values, &[24, WIDTH as i32])
            .as_dtype(Dtype::Bfloat16)
            .expect("cast checkpoint mixer weights to BF16");
        let normed = Array::from_slice(
            &norm_values[norm_offset..norm_offset + ROWS * WIDTH],
            &[ROWS as i32, WIDTH as i32],
        );
        norm_offset += ROWS * WIDTH * 2;
        let expected = projection_fixture[projection_offset..projection_offset + OUTPUT_BYTES]
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte float")))
            .collect::<Vec<_>>();
        projection_offset += OUTPUT_BYTES;
        let actual = glm53_bf16_gemv(&weights, &normed, &stream)
            .expect("BF16 mixer GEMV should dispatch")
            .to_vec1::<f32>()
            .expect("read BF16 mixer projection");
        for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "layer-21 {half} mHC mixer projection[{index}]: {got} vs DS4 {want}"
            );
        }
    }
}

#[cfg(feature = "glm53-fixtures")]
#[test]
fn glm53_mhc_mix_prefill_gemm_layer21_rows9_matches_ds4_fixture() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-mhc-ds4-mix-gemm] skipping: no Metal device available");
        return;
    }
    const ROWS: usize = 9;
    const WIDTH: usize = 4 * 4096;
    const OUTPUTS: usize = ROWS * 24;
    const WEIGHT_BYTES: usize = 24 * WIDTH * std::mem::size_of::<u16>();
    const INPUT_BYTES: usize = ROWS * WIDTH * std::mem::size_of::<f32>();
    const OUTPUT_BYTES: usize = OUTPUTS * std::mem::size_of::<f32>();
    let weights_fixture = include_bytes!("fixtures/mhc_mix_projection_ds4_layer21_ids1234.bin");
    let fixture = include_bytes!("fixtures/mhc_mix_gemm_ds4_layer21_rows9.bin");
    assert_eq!(weights_fixture.len(), 2 * (WEIGHT_BYTES + 4 * 24 * 4));
    assert_eq!(fixture.len(), INPUT_BYTES + OUTPUT_BYTES);
    let mut weight_values = Vec::with_capacity(24 * WIDTH);
    for bytes in weights_fixture[..WEIGHT_BYTES].chunks_exact(2) {
        let bf16 = u16::from_le_bytes(bytes.try_into().expect("two-byte BF16"));
        weight_values.push(f32::from_bits(u32::from(bf16) << 16));
    }
    let weights = Array::from_slice(&weight_values, &[24, WIDTH as i32])
        .as_dtype(Dtype::Bfloat16)
        .expect("cast checkpoint mixer weights to BF16");
    let input_values = fixture[..INPUT_BYTES]
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte float")))
        .collect::<Vec<_>>();
    let expected = fixture[INPUT_BYTES..]
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte float")))
        .collect::<Vec<_>>();
    let input = Array::from_slice(&input_values, &[ROWS as i32, WIDTH as i32]);
    let stream = Stream::thread_local_or_default();
    let actual = glm53_bf16_gemm(&weights, &input, &stream)
        .expect("BF16 mixer prefill GEMM should dispatch above eight rows")
        .to_vec1::<f32>()
        .expect("read BF16 mixer prefill GEMM");
    for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "layer-21 mHC prefill mixer GEMM[{index}]: {got} vs DS4 {want}"
        );
    }
}

#[cfg(feature = "glm53-fixtures")]
#[test]
fn glm53_mhc_split_norm_layer21_capture_matches_ds4_fixture() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-mhc-ds4-split-norm] skipping: no Metal device available");
        return;
    }
    const ROWS: usize = 4;
    const HIDDEN: usize = 4096;
    const SPLIT_VALUES: usize = ROWS * 24;
    const RESIDUAL_VALUES: usize = ROWS * 4 * HIDDEN;
    const NORM_VALUES: usize = ROWS * HIDDEN;
    const FIXTURE_VALUES: usize =
        SPLIT_VALUES + 3 + 24 + RESIDUAL_VALUES + HIDDEN + SPLIT_VALUES + NORM_VALUES + NORM_VALUES;
    let fixture = include_bytes!("fixtures/mhc_split_norm_ds4_layer21_ids1234.f32");
    assert_eq!(fixture.len(), FIXTURE_VALUES * std::mem::size_of::<f32>());
    let values = fixture
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte float")))
        .collect::<Vec<_>>();
    let mut offset = 0;
    let mixes = Array::from_slice(&values[offset..offset + SPLIT_VALUES], &[ROWS as i32, 24]);
    offset += SPLIT_VALUES;
    let scale = Array::from_slice(&values[offset..offset + 3], &[3]);
    offset += 3;
    let base = Array::from_slice(&values[offset..offset + 24], &[24]);
    offset += 24;
    let residual = Array::from_slice(
        &values[offset..offset + RESIDUAL_VALUES],
        &[1, ROWS as i32, 4, HIDDEN as i32],
    );
    offset += RESIDUAL_VALUES;
    let norm_weight = Array::from_slice(&values[offset..offset + HIDDEN], &[HIDDEN as i32]);
    offset += HIDDEN;
    let expected_split = &values[offset..offset + SPLIT_VALUES];
    offset += SPLIT_VALUES;
    let expected_collapsed = &values[offset..offset + NORM_VALUES];
    offset += NORM_VALUES;
    let expected_normed = &values[offset..offset + NORM_VALUES];

    let stream = Stream::thread_local_or_default();
    let (split, post, comb, collapsed, normed) = hc_split_weighted_sum_norm4(
        &mixes,
        &scale,
        &base,
        &residual,
        &norm_weight,
        1e-6,
        1e-5,
        20,
        &stream,
    )
    .expect("DS4 HC split/collapse/norm kernel should dispatch");
    let assert_bits = |name: &str, got: &[f32], want: &[f32]| {
        assert_eq!(got.len(), want.len(), "{name} length");
        for (index, (&got, &want)) in got.iter().zip(want).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "layer-21 {name}[{index}]: {got} vs DS4 {want}"
            );
        }
    };
    let actual_split = split.to_vec1::<f32>().expect("read mHC split");
    let actual_post = post.to_vec1::<f32>().expect("read mHC post weights");
    let actual_comb = comb.to_vec1::<f32>().expect("read mHC combine weights");
    let actual_collapsed = collapsed
        .to_vec1::<f32>()
        .expect("read mHC collapsed state");
    let actual_normed = normed.to_vec1::<f32>().expect("read mHC normalized state");
    let mut expected_post = Vec::with_capacity(ROWS * 4);
    let mut expected_comb = Vec::with_capacity(ROWS * 16);
    for row in expected_split.chunks_exact(24) {
        expected_post.extend_from_slice(&row[4..8]);
        expected_comb.extend_from_slice(&row[8..24]);
    }
    assert_bits("split", &actual_split, expected_split);
    assert_bits("post", &actual_post, &expected_post);
    assert_bits("comb", &actual_comb, &expected_comb);
    assert_bits("collapsed", &actual_collapsed, expected_collapsed);
    assert_bits("normalized", &actual_normed, expected_normed);
}

#[cfg(feature = "glm53-fixtures")]
#[test]
fn glm53_mhc_ffn_split_norm_layer21_capture_matches_ds4_fixture() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-mhc-ds4-ffn-split-norm] skipping: no Metal device available");
        return;
    }
    const ROWS: usize = 4;
    const HIDDEN: usize = 4096;
    const SPLIT_VALUES: usize = ROWS * 24;
    const RESIDUAL_VALUES: usize = ROWS * 4 * HIDDEN;
    const NORM_VALUES: usize = ROWS * HIDDEN;
    const FIXTURE_VALUES: usize =
        SPLIT_VALUES + 3 + 24 + RESIDUAL_VALUES + HIDDEN + SPLIT_VALUES + NORM_VALUES + NORM_VALUES;
    let fixture = include_bytes!("fixtures/mhc_ffn_split_norm_ds4_layer21_ids1234.f32");
    assert_eq!(fixture.len(), FIXTURE_VALUES * std::mem::size_of::<f32>());
    let values = fixture
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte float")))
        .collect::<Vec<_>>();
    let mut offset = 0;
    let mixes = Array::from_slice(&values[offset..offset + SPLIT_VALUES], &[ROWS as i32, 24]);
    offset += SPLIT_VALUES;
    let scale = Array::from_slice(&values[offset..offset + 3], &[3]);
    offset += 3;
    let base = Array::from_slice(&values[offset..offset + 24], &[24]);
    offset += 24;
    let residual = Array::from_slice(
        &values[offset..offset + RESIDUAL_VALUES],
        &[1, ROWS as i32, 4, HIDDEN as i32],
    );
    offset += RESIDUAL_VALUES;
    let norm_weight = Array::from_slice(&values[offset..offset + HIDDEN], &[HIDDEN as i32]);
    offset += HIDDEN;
    let expected_split = &values[offset..offset + SPLIT_VALUES];
    offset += SPLIT_VALUES;
    let expected_collapsed = &values[offset..offset + NORM_VALUES];
    offset += NORM_VALUES;
    let expected_normed = &values[offset..offset + NORM_VALUES];

    let stream = Stream::thread_local_or_default();
    let (split, post, comb, collapsed, normed) = hc_split_weighted_sum_norm4(
        &mixes,
        &scale,
        &base,
        &residual,
        &norm_weight,
        1e-6,
        1e-5,
        20,
        &stream,
    )
    .expect("DS4 FFN HC split/collapse/norm kernel should dispatch");
    let assert_bits = |name: &str, got: &[f32], want: &[f32]| {
        assert_eq!(got.len(), want.len(), "{name} length");
        for (index, (&got, &want)) in got.iter().zip(want).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "layer-21 FFN {name}[{index}]: {got} vs DS4 {want}"
            );
        }
    };
    let actual_split = split.to_vec1::<f32>().expect("read mHC split");
    let actual_post = post.to_vec1::<f32>().expect("read mHC post weights");
    let actual_comb = comb.to_vec1::<f32>().expect("read mHC combine weights");
    let actual_collapsed = collapsed
        .to_vec1::<f32>()
        .expect("read mHC collapsed state");
    let actual_normed = normed.to_vec1::<f32>().expect("read mHC normalized state");
    let mut expected_post = Vec::with_capacity(ROWS * 4);
    let mut expected_comb = Vec::with_capacity(ROWS * 16);
    for row in expected_split.chunks_exact(24) {
        expected_post.extend_from_slice(&row[4..8]);
        expected_comb.extend_from_slice(&row[8..24]);
    }
    assert_bits("split", &actual_split, expected_split);
    assert_bits("post", &actual_post, &expected_post);
    assert_bits("comb", &actual_comb, &expected_comb);
    assert_bits("collapsed", &actual_collapsed, expected_collapsed);
    assert_bits("normalized", &actual_normed, expected_normed);
}

#[test]
fn glm53_indexer_block_topk_matches_unique_score_order() {
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

        // ds4's bitonic stages leave equal elements in their input order,
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

#[cfg(feature = "glm53-fixtures")]
#[test]
fn glm53_indexer_topk_matches_ds4_sparse_tail_fixture() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-indexer-topk] skipping: no Metal device available");
        return;
    }
    const FIXTURE: &[u8] = include_bytes!("fixtures/indexer_topk_ds4_sparse_tail.bin");
    const SCORE_COUNT: usize = 4 * 513;
    const SCORE_BYTES: usize = SCORE_COUNT * 4;
    const TOPK_BYTES: usize = 4 * 512 * 4;
    const EXPANDED_START: usize = SCORE_BYTES + TOPK_BYTES;
    let scores: Vec<f32> = FIXTURE[..SCORE_BYTES]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("F32 score fixture word")))
        .collect();
    let expected: Vec<i32> = FIXTURE[SCORE_BYTES..EXPANDED_START]
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("I32 ID fixture word")))
        .collect();
    let scores = Array::from_slice(&scores, &[4, 513]);
    let stream = Stream::thread_local_or_default();
    let actual =
        dsa_indexer_topk(&scores, 512, &stream).expect("DS4-shaped block top-k should dispatch");
    let shuffled = dsa_indexer_topk_shuffle(&scores, 512, &stream)
        .expect("DS4-shaped shuffle top-k should dispatch");
    let actual_ids = actual
        .to_vec1::<i32>()
        .expect("selected pool IDs should be readable");
    assert_eq!(
        actual_ids, expected,
        "sparse-boundary pool IDs must match DS4 order"
    );
    assert_eq!(
        shuffled
            .to_vec1::<i32>()
            .expect("shuffled pool IDs should be readable"),
        expected,
        "shuffled sparse-boundary pool IDs must match DS4 order"
    );

    let expanded = glm53_expand_pool_selection(&actual, 4, 2048, 4, true, &stream)
        .expect("DS4-shaped pool expansion should dispatch")
        .to_vec1::<i32>()
        .expect("expanded token IDs should be readable");
    let expected_expanded: Vec<i32> = FIXTURE[EXPANDED_START..]
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("I32 expanded ID word")))
        .collect();
    assert_eq!(
        expanded, expected_expanded,
        "sparse-boundary pool expansion and causal tail must match DS4"
    );
}

#[cfg(feature = "glm53-fixtures")]
#[test]
fn glm53_indexer_topk_model_ties_match_ds4_order() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-indexer-topk-ties] skipping: no Metal device available");
        return;
    }
    const ROWS: usize = 3;
    const POOLS: usize = 1024;
    const TOP_K: usize = 512;
    const SCORE_BYTES: usize = ROWS * POOLS * std::mem::size_of::<f32>();
    const FIXTURE: &[u8] = include_bytes!("fixtures/indexer_topk_tied_model_rows_ds4.bin");
    let scores = FIXTURE[..SCORE_BYTES]
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("F32 score fixture word")))
        .collect::<Vec<_>>();
    let expected = FIXTURE[SCORE_BYTES..]
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("I32 ID fixture word")))
        .collect::<Vec<_>>();
    let scores = Array::from_slice(&scores, &[ROWS as i32, POOLS as i32]);
    let stream = Stream::thread_local_or_default();
    let threadgroup = dsa_indexer_topk(&scores, TOP_K, &stream)
        .expect("threadgroup model-tie top-k should dispatch")
        .to_vec1::<i32>()
        .expect("threadgroup model-tie pool IDs should be readable");
    let shuffle = dsa_indexer_topk_shuffle(&scores, TOP_K, &stream)
        .expect("shuffle model-tie top-k should dispatch")
        .to_vec1::<i32>()
        .expect("shuffle model-tie pool IDs should be readable");
    let mut mismatch_counts = Vec::new();
    for (name, actual) in [("threadgroup", threadgroup), ("shuffle", shuffle)] {
        let mismatches = actual
            .iter()
            .zip(&expected)
            .enumerate()
            .filter_map(|(index, (&got, &want))| (got != want).then_some(index))
            .collect::<Vec<_>>();
        eprintln!(
            "[glm53-indexer-topk-ties] {name}: {} mismatches, first {:?}",
            mismatches.len(),
            mismatches.first().map(|&index| {
                (
                    index,
                    actual[index],
                    expected[index],
                    actual[index.saturating_sub(2)..=(index + 2).min(actual.len() - 1)].to_vec(),
                    expected[index.saturating_sub(2)..=(index + 2).min(expected.len() - 1)]
                        .to_vec(),
                )
            })
        );
        mismatch_counts.push((name, mismatches.len()));
    }
    for (name, count) in mismatch_counts {
        assert_eq!(count, 0, "{name} top-k order differs from DS4");
    }
}

#[cfg(feature = "glm53-fixtures")]
#[test]
fn glm53_indexed_lora_matches_ds4_sparse_boundary_fixture() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-indexed-lora] skipping: no Metal device available");
        return;
    }
    const FIXTURE: &[u8] = include_bytes!("fixtures/dsa_indexed_lora_ds4_boundary.bin");
    const Q_LOW_BYTES: usize = 64 * 512 * 4;
    const CACHE_START: usize = Q_LOW_BYTES;
    const CACHE_BYTES: usize = 2052 * 512 * 2;
    const SELECTED_START: usize = CACHE_START + CACHE_BYTES;
    const SELECTED_BYTES: usize = 2051 * 4;
    const EXPECTED_START: usize = SELECTED_START + SELECTED_BYTES;
    const OUTPUT_COUNT: usize = 64 * 512;
    assert_eq!(FIXTURE.len(), EXPECTED_START + OUTPUT_COUNT * 4);

    let read_f32 = |bytes: &[u8]| -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("F32 fixture word")))
            .collect()
    };
    let q_low = read_f32(&FIXTURE[..CACHE_START]);
    let latents: Vec<half::f16> = FIXTURE[CACHE_START..SELECTED_START]
        .chunks_exact(2)
        .map(|chunk| half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect();
    let selected: Vec<i32> = FIXTURE[SELECTED_START..EXPECTED_START]
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("I32 fixture word")))
        .collect();
    assert_eq!(selected.iter().filter(|&&row| row < 0).count(), 2);
    let expected = read_f32(&FIXTURE[EXPECTED_START..]);

    let q_low = Array::from_slice(&q_low, &[1, 1, 64, 512]);
    let latents = Array::from_slice(&latents, &[1, 2052, 512]);
    let selected = Array::from_slice(&selected, &[1, 1, 2051]);
    let n_batches = Array::from_int(1);
    let n_tokens = Array::from_int(1);
    let n_heads = Array::from_int(64);
    let n_selected = Array::from_int(2051);
    let selected_valid_prefix = Array::from_int(2048);
    let dense_all = Array::from_int(0);
    let tc = Array::from_int(2052);
    let scale = Array::from_f32_as(1.0 / 16.0, Dtype::Float32);
    let inputs: [&Array; 11] = [
        &n_batches,
        &n_tokens,
        &n_heads,
        &n_selected,
        &selected_valid_prefix,
        &dense_all,
        &tc,
        &scale,
        &q_low,
        &latents,
        &selected,
    ];
    let actual = dsa_indexed_lora_kernel()
        .as_ref()
        .expect("DSA indexed-lora kernel should compile")
        .apply(
            &inputs,
            &[],
            (8 * 256, 1, 1),
            (256, 1, 1),
            &[OutputArg {
                shape: vec![1, 1, 64, 512],
                dtype: Dtype::Float32,
            }],
            &Stream::thread_local_or_default(),
        )
        .expect("DSA indexed-lora fixture should dispatch")
        .pop()
        .expect("DSA indexed-lora output should exist")
        .to_vec1::<f32>()
        .expect("DSA indexed-lora output should be readable");
    assert_eq!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "captured sparse-boundary DSA attention output must match DS4 exactly"
    );
}

#[cfg(feature = "glm53-fixtures")]
#[test]
fn glm53_indexer_pool_matches_ds4_f16_midpoint_fixture() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-indexer-pool] skipping: no Metal device available");
        return;
    }
    const FIXTURE: &[u8] = include_bytes!("fixtures/indexer_pool_ds4_midpoints.bin");
    const RAW_BYTES: usize = 2 * 4 * 128 * 4;
    const NORM_W_START: usize = RAW_BYTES;
    const GATE_START: usize = NORM_W_START + 128 * 4;
    const NORM_B_START: usize = GATE_START + RAW_BYTES;
    const APE_START: usize = NORM_B_START + 128 * 4;
    const EXPECTED_START: usize = APE_START + 4 * 128 * 2;
    fn read_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("F32 fixture word")))
            .collect()
    }
    let raw = read_f32(&FIXTURE[..RAW_BYTES]);
    let norm_w = read_f32(&FIXTURE[NORM_W_START..GATE_START]);
    let gate = read_f32(&FIXTURE[GATE_START..NORM_B_START]);
    let norm_b = read_f32(&FIXTURE[NORM_B_START..APE_START]);
    let ape: Vec<half::bf16> = FIXTURE[APE_START..EXPECTED_START]
        .chunks_exact(2)
        .map(|chunk| half::bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect();
    let expected: Vec<u16> = FIXTURE[EXPECTED_START..]
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();

    let raw = Array::from_slice(&raw, &[1, 2, 4, 128]);
    let gate = Array::from_slice(&gate, &[1, 2, 4, 128]);
    let norm_w = Array::from_slice(&norm_w, &[128]);
    let norm_b = Array::from_slice(&norm_b, &[128]);
    let ape = Array::from_slice(&ape, &[4, 128]);
    let actual = indexer_pool_fused(
        &raw,
        &gate,
        &ape,
        &norm_w,
        &norm_b,
        1e-6,
        &Stream::thread_local_or_default(),
    )
    .expect("DS4-shaped DSA pool should dispatch")
    .to_vec1::<half::f16>()
    .expect("F16 pooled cache should be readable")
    .into_iter()
    .map(half::f16::to_bits)
    .collect::<Vec<_>>();
    assert_eq!(
        actual, expected,
        "two DS4 F16 midpoint pool rows must match bit-for-bit"
    );
}

#[cfg(feature = "glm53-fixtures")]
#[test]
fn glm53_indexer_pool_matches_ds4_layer11_fixture() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-indexer-pool] skipping: no Metal device available");
        return;
    }
    const FIXTURE: &[u8] = include_bytes!("fixtures/indexer_pool_ds4_layer11.bin");
    const RAW_BYTES: usize = 4 * 128 * 4;
    const NORM_W_START: usize = RAW_BYTES;
    const GATE_START: usize = NORM_W_START + 128 * 4;
    const NORM_B_START: usize = GATE_START + RAW_BYTES;
    const APE_START: usize = NORM_B_START + 128 * 4;
    const EXPECTED_START: usize = APE_START + 4 * 128 * 2;
    assert_eq!(FIXTURE.len(), EXPECTED_START + 128 * 2);

    fn read_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("F32 fixture word")))
            .collect()
    }
    let raw = read_f32(&FIXTURE[..NORM_W_START]);
    let norm_w = read_f32(&FIXTURE[NORM_W_START..GATE_START]);
    let gate = read_f32(&FIXTURE[GATE_START..NORM_B_START]);
    let norm_b = read_f32(&FIXTURE[NORM_B_START..APE_START]);
    let ape: Vec<half::bf16> = FIXTURE[APE_START..EXPECTED_START]
        .chunks_exact(2)
        .map(|chunk| half::bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect();
    let expected: Vec<u16> = FIXTURE[EXPECTED_START..]
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();

    let raw = Array::from_slice(&raw, &[1, 1, 4, 128]);
    let gate = Array::from_slice(&gate, &[1, 1, 4, 128]);
    let norm_w = Array::from_slice(&norm_w, &[128]);
    let norm_b = Array::from_slice(&norm_b, &[128]);
    let ape = Array::from_slice(&ape, &[4, 128]);
    let actual = indexer_pool_fused(
        &raw,
        &gate,
        &ape,
        &norm_w,
        &norm_b,
        1e-6,
        &Stream::thread_local_or_default(),
    )
    .expect("DS4-shaped layer-11 pool should dispatch")
    .to_vec1::<half::f16>()
    .expect("F16 pooled cache should be readable")
    .into_iter()
    .map(half::f16::to_bits)
    .collect::<Vec<_>>();
    assert_eq!(actual, expected, "layer-11 pool must match DS4 bit-for-bit");
}

#[cfg(feature = "glm53-fixtures")]
#[test]
fn glm53_indexer_model_long_decode_projection_matches_ds4() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-indexer-projection-model-long] skipping: no Metal device available");
        return;
    }
    const IN_DIM: usize = 4096;
    const OUT_DIM: usize = 128;
    const HEADER_BYTES: usize = 4 * std::mem::size_of::<u32>();
    const INPUT_BYTES: usize = IN_DIM * std::mem::size_of::<f32>();
    const WEIGHT_BYTES: usize = OUT_DIM * IN_DIM * std::mem::size_of::<u16>();
    const OUTPUT_BYTES: usize = 2 * OUT_DIM * std::mem::size_of::<f32>();
    const FIXTURE: &[u8] = include_bytes!("fixtures/indexer_projection_model_long_ds4.bin");
    assert_eq!(
        FIXTURE.len(),
        HEADER_BYTES + INPUT_BYTES + 2 * WEIGHT_BYTES + OUTPUT_BYTES
    );
    assert_eq!(
        FIXTURE[..HEADER_BYTES],
        [
            1u32.to_le_bytes(),
            (IN_DIM as u32).to_le_bytes(),
            (OUT_DIM as u32).to_le_bytes(),
            2u32.to_le_bytes()
        ]
        .concat()
    );

    let read_f32 = |bytes: &[u8]| {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("F32 fixture word")))
            .collect::<Vec<_>>()
    };
    let read_bf16 = |bytes: &[u8]| {
        bytes
            .chunks_exact(2)
            .map(|chunk| half::bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
            .collect::<Vec<_>>()
    };
    let input_start = HEADER_BYTES;
    let key_start = input_start + INPUT_BYTES;
    let gate_start = key_start + WEIGHT_BYTES;
    let output_start = gate_start + WEIGHT_BYTES;
    let input = read_f32(&FIXTURE[input_start..key_start]);
    let key_weight = read_bf16(&FIXTURE[key_start..gate_start]);
    let gate_weight = read_bf16(&FIXTURE[gate_start..output_start]);
    let expected = read_f32(&FIXTURE[output_start..]);

    let input = Array::from_slice(&input, &[1, IN_DIM as i32]);
    let key_weight = Array::from_slice(&key_weight, &[OUT_DIM as i32, IN_DIM as i32]);
    let gate_weight = Array::from_slice(&gate_weight, &[OUT_DIM as i32, IN_DIM as i32]);
    let stream = Stream::thread_local_or_default();
    for (name, weight, expected) in [
        ("wk", &key_weight, &expected[..OUT_DIM]),
        ("pool gate", &gate_weight, &expected[OUT_DIM..]),
    ] {
        let actual = glm53_bf16_gemv(weight, &input, &stream)
            .expect("production DS4-order BF16 GEMV should dispatch")
            .to_vec1::<f32>()
            .expect("model-derived projection output should be readable");
        assert_eq!(actual.len(), OUT_DIM);
        for (column, (&got, &want)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "model-derived long-window {name} projection[{column}]: LISA {got}, DS4 {want}"
            );
        }
    }
}

#[cfg(feature = "glm53-fixtures")]
#[test]
fn glm53_indexer_model_long_pool_update_matches_ds4() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-indexer-pool-model-long] skipping: no Metal device available");
        return;
    }
    const HISTORY: usize = 16_384;
    const POOLS: usize = HISTORY / 4;
    const DIM: usize = 128;
    const RAW_BYTES: usize = HISTORY * DIM * std::mem::size_of::<f32>();
    const NORM_BYTES: usize = DIM * std::mem::size_of::<f32>();
    const APE_BYTES: usize = 4 * DIM * std::mem::size_of::<u16>();
    const POOL_BYTES: usize = POOLS * DIM * std::mem::size_of::<u16>();
    const FIXTURE: &[u8] = include_bytes!("fixtures/indexer_pool_model_long_ds4.bin");
    assert_eq!(
        FIXTURE.len(),
        8 + 2 * RAW_BYTES + 2 * NORM_BYTES + APE_BYTES + POOL_BYTES
    );
    assert_eq!(
        FIXTURE[..8],
        [(HISTORY as u32).to_le_bytes(), (POOLS as u32).to_le_bytes()].concat()
    );

    let read_f32 = |bytes: &[u8]| {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("F32 fixture word")))
            .collect::<Vec<_>>()
    };
    let raw_start = 8;
    let gate_start = raw_start + RAW_BYTES;
    let norm_weight_start = gate_start + RAW_BYTES;
    let norm_bias_start = norm_weight_start + NORM_BYTES;
    let ape_start = norm_bias_start + NORM_BYTES;
    let pool_start = ape_start + APE_BYTES;
    let raw = read_f32(&FIXTURE[raw_start..gate_start]);
    let gate = read_f32(&FIXTURE[gate_start..norm_weight_start]);
    let norm_weight = read_f32(&FIXTURE[norm_weight_start..norm_bias_start]);
    let norm_bias = read_f32(&FIXTURE[norm_bias_start..ape_start]);
    let ape = FIXTURE[ape_start..pool_start]
        .chunks_exact(2)
        .map(|chunk| half::bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect::<Vec<_>>();
    let expected = FIXTURE[pool_start..]
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect::<Vec<_>>();

    let raw = Array::from_slice(&raw, &[1, POOLS as i32, 4, DIM as i32]);
    let gate = Array::from_slice(&gate, &[1, POOLS as i32, 4, DIM as i32]);
    let norm_weight = Array::from_slice(&norm_weight, &[DIM as i32]);
    let norm_bias = Array::from_slice(&norm_bias, &[DIM as i32]);
    let ape = Array::from_slice(&ape, &[4, DIM as i32]);
    let actual = indexer_pool_fused(
        &raw,
        &gate,
        &ape,
        &norm_weight,
        &norm_bias,
        1e-6,
        &Stream::thread_local_or_default(),
    )
    .expect("16K checkpoint-derived pool update should dispatch")
    .to_vec1::<half::f16>()
    .expect("F16 model-derived pooled keys should be readable")
    .into_iter()
    .map(half::f16::to_bits)
    .collect::<Vec<_>>();
    assert_eq!(
        actual, expected,
        "all 4,096 model-derived DS4 pools must match"
    );
}

#[cfg(feature = "glm53-fixtures")]
#[test]
fn glm53_indexer_pool_matches_ds4_incremental_rollback_replay_fixture() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-indexer-pool-lifecycle] skipping: no Metal device available");
        return;
    }
    const POOLS: usize = 2;
    const POOL_SIZE: usize = 4;
    const DIM: usize = 128;
    const RAW_VALUES: usize = POOLS * POOL_SIZE * DIM;
    const RAW_BYTES: usize = RAW_VALUES * std::mem::size_of::<f32>();
    const NORM_BYTES: usize = DIM * std::mem::size_of::<f32>();
    const GATE_BYTES: usize = RAW_BYTES;
    const APE_BYTES: usize = POOL_SIZE * DIM * std::mem::size_of::<u16>();
    const OUTPUT_BYTES: usize = POOLS * DIM * std::mem::size_of::<u16>();
    const FIXTURE: &[u8] = include_bytes!("fixtures/indexer_pool_lifecycle_ds4.bin");
    assert_eq!(
        FIXTURE.len(),
        RAW_BYTES + NORM_BYTES + GATE_BYTES + NORM_BYTES + APE_BYTES + OUTPUT_BYTES
    );

    let decode_f32 = |bytes: &[u8]| {
        bytes
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("F32 fixture word")))
            .collect::<Vec<_>>()
    };
    let decode_bf16 = |bytes: &[u8]| {
        bytes
            .chunks_exact(std::mem::size_of::<u16>())
            .map(|chunk| half::bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
            .collect::<Vec<_>>()
    };
    let decode_f16_bits = |bytes: &[u8]| {
        bytes
            .chunks_exact(std::mem::size_of::<u16>())
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect::<Vec<_>>()
    };
    let raw = decode_f32(&FIXTURE[..RAW_BYTES]);
    let norm_start = RAW_BYTES;
    let norm_weight = decode_f32(&FIXTURE[norm_start..norm_start + NORM_BYTES]);
    let gate_start = norm_start + NORM_BYTES;
    let gate = decode_f32(&FIXTURE[gate_start..gate_start + GATE_BYTES]);
    let bias_start = gate_start + GATE_BYTES;
    let norm_bias = decode_f32(&FIXTURE[bias_start..bias_start + NORM_BYTES]);
    let ape_start = bias_start + NORM_BYTES;
    let ape = decode_bf16(&FIXTURE[ape_start..ape_start + APE_BYTES]);
    let expected = decode_f16_bits(&FIXTURE[ape_start + APE_BYTES..]);

    let raw = Array::from_slice(&raw, &[1, POOLS as i32, POOL_SIZE as i32, DIM as i32]);
    let gate = Array::from_slice(&gate, &[1, POOLS as i32, POOL_SIZE as i32, DIM as i32]);
    let norm_weight = Array::from_slice(&norm_weight, &[DIM as i32]);
    let norm_bias = Array::from_slice(&norm_bias, &[DIM as i32]);
    let ape = Array::from_slice(&ape, &[POOL_SIZE as i32, DIM as i32]);
    let actual = indexer_pool_fused(
        &raw,
        &gate,
        &ape,
        &norm_weight,
        &norm_bias,
        1e-6,
        &Stream::thread_local_or_default(),
    )
    .expect("DS4-shaped fused pool should dispatch")
    .to_vec1::<half::f16>()
    .expect("pooled replay rows should be readable")
    .into_iter()
    .map(half::f16::to_bits)
    .collect::<Vec<_>>();
    assert_eq!(
        actual, expected,
        "LISA fused pool rows must equal DS4's incremental update after rollback and replay"
    );
}

#[test]
fn glm53_indexer_pool_expansion_matches_causal_batched_layout() {
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
    let selected = glm53_expand_pool_selection(&pools, 2, 8, 4, true, &stream)
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

    let no_tail = glm53_expand_pool_selection(&pools, 2, 8, 4, false, &stream)
        .expect("pool expansion without tail should dispatch")
        .to_vec1::<i32>()
        .expect("expanded token IDs should be readable");
    for row in no_tail.chunks_exact(15) {
        assert_eq!(&row[12..], &[-1, -1, -1]);
    }

    let causal = glm53_expand_pool_selection_for_attention(&pools, 2, 8, 4, true, &stream)
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
fn glm53_indexer_score_paths_preserve_scores_and_visibility() {
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
        let scores = glm53_indexer_scores(&q, &weights, &pooled, pos0, 4, 1.0 / 64.0, &stream)
            .expect("ds4-shaped indexer score path should dispatch");
        let got = scores
            .to_vec1::<f32>()
            .expect("indexer scores should be readable");
        if tokens == 1 {
            let rows4 =
                glm53_indexer_scores_rows4(&q, &weights, &pooled, pos0, 4, 1.0 / 64.0, &stream)
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

#[cfg(feature = "glm53-fixtures")]
#[test]
fn glm53_indexer_prefill_tail_scores_match_ds4_natural_8k_fixture() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-indexer-scores] skipping: no Metal device available");
        return;
    }
    const HEADS: usize = 32;
    const DIM: usize = 128;
    const POOLS: usize = 2048;
    const Q_BYTES: usize = HEADS * DIM * std::mem::size_of::<f32>();
    const WEIGHT_BYTES: usize = HEADS * std::mem::size_of::<f32>();
    const POOL_BYTES: usize = POOLS * DIM * std::mem::size_of::<u16>();
    const SCORE_BYTES: usize = POOLS * std::mem::size_of::<f32>();
    const SELECTED_BYTES: usize = (2048 / 4 * 4 + 3) * std::mem::size_of::<i32>();
    const FIXTURE: &[u8] = include_bytes!("fixtures/indexer_score_prefill_tail_ds4.bin");
    assert_eq!(
        FIXTURE.len(),
        Q_BYTES + WEIGHT_BYTES + POOL_BYTES + SCORE_BYTES + SELECTED_BYTES
    );

    let read_f32 = |bytes: &[u8]| {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("F32 fixture word")))
            .collect::<Vec<_>>()
    };
    let q = read_f32(&FIXTURE[..Q_BYTES]);
    let weights = read_f32(&FIXTURE[Q_BYTES..Q_BYTES + WEIGHT_BYTES]);
    let pool_start = Q_BYTES + WEIGHT_BYTES;
    let pooled = FIXTURE[pool_start..pool_start + POOL_BYTES]
        .chunks_exact(2)
        .map(|chunk| half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect::<Vec<_>>();
    let score_start = pool_start + POOL_BYTES;
    let expected = read_f32(&FIXTURE[score_start..score_start + SCORE_BYTES]);
    let selected_start = score_start + SCORE_BYTES;
    let expected_selected = FIXTURE[selected_start..]
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("I32 fixture word")))
        .collect::<Vec<_>>();

    let q = Array::from_slice(&q, &[1, HEADS as i32, 1, DIM as i32]);
    let weights = Array::from_slice(&weights, &[1, 1, HEADS as i32]);
    let pooled = Array::from_slice(&pooled, &[1, POOLS as i32, DIM as i32]);
    let stream = Stream::thread_local_or_default();
    let actual = glm53_indexer_scores_batch(&q, &weights, &pooled, 8192, 4, 1.0 / 64.0, &stream)
        .expect("DS4 prefill-tail indexer score path should dispatch")
        .to_vec1::<f32>()
        .expect("prefill-tail indexer scores should be readable");
    assert_eq!(actual.len(), expected.len());
    for (pool, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "natural 8K prefill-tail score[{pool}]: LISA {got}, DS4 {want}"
        );
    }
    assert!(
        actual[711] > actual[1684],
        "DS4 score order at the natural 8K top-k boundary changed"
    );

    let score_rows = Array::from_slice(&actual, &[1, POOLS as i32]);
    let top = dsa_indexer_topk(&score_rows, 512, &stream)
        .expect("DS4 prefill-tail top-k should dispatch");
    let expanded = glm53_expand_pool_selection_for_attention(&top, 1, 8192, 4, true, &stream)
        .expect("DS4 prefill-tail pool expansion should dispatch")
        .to_vec1::<i32>()
        .expect("prefill-tail selected token IDs should be readable");
    assert_eq!(
        expanded, expected_selected,
        "natural 8K prefill-tail selected token IDs differ from DS4"
    );
}

#[cfg(feature = "glm53-fixtures")]
#[test]
fn glm53_indexer_decode_scores_match_ds4_sparse_boundary_fixture() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-indexer-scores] skipping: no Metal device available");
        return;
    }
    const BATCHES: usize = 2;
    const HEADS: usize = 32;
    const DIM: usize = 128;
    const POOLS: usize = 513;
    const Q_BYTES: usize = BATCHES * HEADS * DIM * std::mem::size_of::<f32>();
    const WEIGHT_BYTES: usize = BATCHES * HEADS * std::mem::size_of::<f32>();
    const POOL_BYTES: usize = BATCHES * POOLS * DIM * std::mem::size_of::<u16>();
    const SCORE_BYTES: usize = BATCHES * POOLS * std::mem::size_of::<f32>();
    const FIXTURE: &[u8] = include_bytes!("fixtures/indexer_score_decode_ds4_sparse_boundary.bin");
    assert_eq!(
        FIXTURE.len(),
        Q_BYTES + WEIGHT_BYTES + POOL_BYTES + SCORE_BYTES
    );

    let read_f32 = |bytes: &[u8]| {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("F32 fixture word")))
            .collect::<Vec<_>>()
    };
    let q = read_f32(&FIXTURE[..Q_BYTES]);
    let weights = read_f32(&FIXTURE[Q_BYTES..Q_BYTES + WEIGHT_BYTES]);
    let pool_start = Q_BYTES + WEIGHT_BYTES;
    let pooled = FIXTURE[pool_start..pool_start + POOL_BYTES]
        .chunks_exact(2)
        .map(|chunk| half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect::<Vec<_>>();
    let expected = read_f32(&FIXTURE[pool_start + POOL_BYTES..]);

    let q = Array::from_slice(&q, &[BATCHES as i32, HEADS as i32, 1, DIM as i32]);
    let weights = Array::from_slice(&weights, &[BATCHES as i32, 1, HEADS as i32]);
    let pooled = Array::from_slice(&pooled, &[BATCHES as i32, POOLS as i32, DIM as i32]);
    let stream = Stream::thread_local_or_default();
    let actual = glm53_indexer_scores(&q, &weights, &pooled, 2051, 4, 1.0 / 64.0, &stream)
        .expect("DS4 direct-decode indexer score path should dispatch")
        .to_vec1::<f32>()
        .expect("DSA decode scores should be readable");
    let rows4 = glm53_indexer_scores_rows4(&q, &weights, &pooled, 2051, 4, 1.0 / 64.0, &stream)
        .expect("DS4 direct-decode rows4 scorer should dispatch")
        .to_vec1::<f32>()
        .expect("DSA rows4 decode scores should be readable");
    assert_eq!(rows4, actual, "rows4 boundary scores differ from DS4");
    assert_eq!(
        actual
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "B=2 one-token 513-pool scores must match DS4's direct Metal scorer"
    );
}

#[cfg(feature = "glm53-fixtures")]
#[test]
fn glm53_indexer_long_decode_scores_and_topk_match_ds4_fixture() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-indexer-long] skipping: no Metal device available");
        return;
    }
    const BATCHES: usize = 2;
    const HEADS: usize = 32;
    const DIM: usize = 128;
    const POOLS: usize = 4096;
    const TOP_K: usize = 512;
    const Q_BYTES: usize = BATCHES * HEADS * DIM * std::mem::size_of::<f32>();
    const WEIGHT_BYTES: usize = BATCHES * HEADS * std::mem::size_of::<f32>();
    const POOL_BYTES: usize = BATCHES * POOLS * DIM * std::mem::size_of::<u16>();
    const SCORE_BYTES: usize = BATCHES * POOLS * std::mem::size_of::<f32>();
    const IDS_BYTES: usize = BATCHES * TOP_K * std::mem::size_of::<i32>();
    const FIXTURE: &[u8] = include_bytes!("fixtures/indexer_score_decode_ds4_long_window.bin");
    assert_eq!(
        FIXTURE.len(),
        Q_BYTES + WEIGHT_BYTES + POOL_BYTES + SCORE_BYTES + IDS_BYTES
    );

    let read_f32 = |bytes: &[u8]| {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("F32 fixture word")))
            .collect::<Vec<_>>()
    };
    let q = read_f32(&FIXTURE[..Q_BYTES]);
    let weights = read_f32(&FIXTURE[Q_BYTES..Q_BYTES + WEIGHT_BYTES]);
    let pool_start = Q_BYTES + WEIGHT_BYTES;
    let pooled = FIXTURE[pool_start..pool_start + POOL_BYTES]
        .chunks_exact(2)
        .map(|chunk| half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect::<Vec<_>>();
    let score_start = pool_start + POOL_BYTES;
    let expected_scores = read_f32(&FIXTURE[score_start..score_start + SCORE_BYTES]);
    let expected_ids = FIXTURE[score_start + SCORE_BYTES..]
        .chunks_exact(4)
        .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("I32 fixture word")))
        .collect::<Vec<_>>();

    let q = Array::from_slice(&q, &[BATCHES as i32, HEADS as i32, 1, DIM as i32]);
    let weights = Array::from_slice(&weights, &[BATCHES as i32, 1, HEADS as i32]);
    let pooled = Array::from_slice(&pooled, &[BATCHES as i32, POOLS as i32, DIM as i32]);
    let stream = Stream::thread_local_or_default();
    let actual_scores =
        glm53_indexer_scores(&q, &weights, &pooled, POOLS * 4 - 1, 4, 1.0 / 64.0, &stream)
            .expect("DS4 direct-decode scorer should dispatch over 4096 pools")
            .to_vec1::<f32>()
            .expect("long-window DSA scores should be readable");
    let rows4_scores =
        glm53_indexer_scores_rows4(&q, &weights, &pooled, POOLS * 4 - 1, 4, 1.0 / 64.0, &stream)
            .expect("DS4 long-window rows4 scorer should dispatch")
            .to_vec1::<f32>()
            .expect("long-window rows4 scores should be readable");
    assert_eq!(actual_scores.len(), expected_scores.len());
    assert_eq!(
        rows4_scores, actual_scores,
        "rows4 long-window scores differ"
    );
    for (index, (&got, &want)) in actual_scores.iter().zip(&expected_scores).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "B=2 one-token score[{index}] across 4096 pools: LISA {got}, DS4 {want}"
        );
    }

    let score_rows = Array::from_slice(&actual_scores, &[BATCHES as i32, POOLS as i32]);
    let actual_ids = dsa_indexer_topk(&score_rows, TOP_K, &stream)
        .expect("DS4 blockwise top-k should dispatch over 4096 pools")
        .to_vec1::<i32>()
        .expect("long-window selected pool IDs should be readable");
    let shuffled_ids = dsa_indexer_topk_shuffle(&score_rows, TOP_K, &stream)
        .expect("DS4 shuffle top-k should dispatch over 4096 pools")
        .to_vec1::<i32>()
        .expect("long-window shuffled pool IDs should be readable");
    assert_eq!(actual_ids.len(), expected_ids.len());
    for (rank, (&got, &want)) in actual_ids.iter().zip(&expected_ids).enumerate() {
        assert_eq!(
            got, want,
            "B=2 top-512 pool ID at rank {rank}: LISA {got}, DS4 {want}"
        );
    }
    assert_eq!(
        shuffled_ids, expected_ids,
        "shuffle top-k IDs differ from DS4"
    );
}

#[cfg(feature = "glm53-fixtures")]
#[test]
fn glm53_indexer_model_long_decode_scores_topk_and_expansion_match_ds4() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-indexer-model-long] skipping: no Metal device available");
        return;
    }
    const HEADS: usize = 32;
    const DIM: usize = 128;
    const POOLS: usize = 4096;
    const TOP_POOLS: usize = 512;
    const SELECTED_WIDTH: usize = TOP_POOLS * 4 + 3;
    const Q_BYTES: usize = HEADS * DIM * std::mem::size_of::<f32>();
    const WEIGHT_BYTES: usize = HEADS * std::mem::size_of::<f32>();
    const POOL_BYTES: usize = POOLS * DIM * std::mem::size_of::<u16>();
    const SCORE_BYTES: usize = POOLS * std::mem::size_of::<f32>();
    const POOL_IDS_BYTES: usize = TOP_POOLS * std::mem::size_of::<i32>();
    const EXPANDED_BYTES: usize = SELECTED_WIDTH * std::mem::size_of::<i32>();
    const FIXTURE: &[u8] = include_bytes!("fixtures/indexer_score_model_long_ds4.bin");
    assert_eq!(
        FIXTURE.len(),
        Q_BYTES + WEIGHT_BYTES + POOL_BYTES + SCORE_BYTES + POOL_IDS_BYTES + EXPANDED_BYTES
    );

    let read_f32 = |bytes: &[u8]| {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("F32 fixture word")))
            .collect::<Vec<_>>()
    };
    let read_i32 = |bytes: &[u8]| {
        bytes
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("I32 fixture word")))
            .collect::<Vec<_>>()
    };

    let q = read_f32(&FIXTURE[..Q_BYTES]);
    let weights = read_f32(&FIXTURE[Q_BYTES..Q_BYTES + WEIGHT_BYTES]);
    let pool_start = Q_BYTES + WEIGHT_BYTES;
    let pooled = FIXTURE[pool_start..pool_start + POOL_BYTES]
        .chunks_exact(2)
        .map(|chunk| half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect::<Vec<_>>();
    let score_start = pool_start + POOL_BYTES;
    let expected_scores = read_f32(&FIXTURE[score_start..score_start + SCORE_BYTES]);
    let pool_ids_start = score_start + SCORE_BYTES;
    let expected_pool_ids = read_i32(&FIXTURE[pool_ids_start..pool_ids_start + POOL_IDS_BYTES]);
    let expected_expanded = read_i32(&FIXTURE[pool_ids_start + POOL_IDS_BYTES..]);

    let q = Array::from_slice(&q, &[1, HEADS as i32, 1, DIM as i32]);
    let weights = Array::from_slice(&weights, &[1, 1, HEADS as i32]);
    let pooled = Array::from_slice(&pooled, &[1, POOLS as i32, DIM as i32]);
    let stream = Stream::thread_local_or_default();
    let actual_scores = glm53_indexer_scores(&q, &weights, &pooled, 16_384, 4, 1.0 / 64.0, &stream)
        .expect("model-produced long-window scorer should dispatch")
        .to_vec1::<f32>()
        .expect("model-produced scores should be readable");
    let rows4_scores =
        glm53_indexer_scores_rows4(&q, &weights, &pooled, 16_384, 4, 1.0 / 64.0, &stream)
            .expect("model-produced rows4 scorer should dispatch")
            .to_vec1::<f32>()
            .expect("model-produced rows4 scores should be readable");
    assert_eq!(actual_scores.len(), POOLS);
    assert_eq!(rows4_scores, actual_scores, "rows4 model scores differ");
    for (pool, (&got, &want)) in actual_scores.iter().zip(&expected_scores).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "model-produced long-window score[{pool}]: LISA {got}, DS4 {want}"
        );
    }

    let score_rows = Array::from_slice(&actual_scores, &[1, POOLS as i32]);
    let actual_pool_ids = dsa_indexer_topk(&score_rows, TOP_POOLS, &stream)
        .expect("model-produced DS4 blockwise top-k should dispatch")
        .to_vec1::<i32>()
        .expect("model-produced pool IDs should be readable");
    assert_eq!(actual_pool_ids, expected_pool_ids);
    let shuffled_pool_ids = dsa_indexer_topk_shuffle(&score_rows, TOP_POOLS, &stream)
        .expect("model-produced DS4 shuffle top-k should dispatch")
        .to_vec1::<i32>()
        .expect("model-produced shuffled pool IDs should be readable");
    assert_eq!(shuffled_pool_ids, expected_pool_ids);
    let pool_rows = Array::from_slice(&actual_pool_ids, &[1, TOP_POOLS as i32]);
    let actual_expanded = glm53_expand_pool_selection(&pool_rows, 1, 16_384, 4, true, &stream)
        .expect("model-produced DS4 pool expansion should dispatch")
        .to_vec1::<i32>()
        .expect("model-produced selected token IDs should be readable");
    assert_eq!(actual_expanded, expected_expanded);
}

#[cfg(feature = "glm53-fixtures")]
#[test]
fn glm53_indexed_lora_model_long_decode_matches_ds4() {
    if !crate::device::metal_available() {
        eprintln!("[glm53-indexed-lora-model-long] skipping: no Metal device available");
        return;
    }
    const BATCHES: usize = 1;
    const TOKENS: usize = 1;
    const HEADS: usize = 64;
    const QDIM: usize = 256;
    const WIDTH: usize = 512;
    const CACHE_ROWS: usize = 16_385;
    const SELECTED: usize = 2_051;
    const VALID_PREFIX: usize = 2_048;
    const Q_BYTES: usize = HEADS * QDIM * std::mem::size_of::<f32>();
    const LOW_BYTES: usize = HEADS * WIDTH * std::mem::size_of::<f32>();
    const CACHE_BYTES: usize = CACHE_ROWS * WIDTH * std::mem::size_of::<u16>();
    const SELECTED_BYTES: usize = SELECTED * std::mem::size_of::<i32>();
    const OUTPUT_BYTES: usize = HEADS * WIDTH * std::mem::size_of::<f32>();
    const FIXTURE: &[u8] = include_bytes!("fixtures/dsa_indexed_lora_model_long_ds4.bin");
    assert_eq!(
        FIXTURE.len(),
        Q_BYTES + LOW_BYTES + CACHE_BYTES + SELECTED_BYTES + OUTPUT_BYTES
    );

    let read_f32 = |bytes: &[u8]| {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("F32 fixture word")))
            .collect::<Vec<_>>()
    };
    let read_i32 = |bytes: &[u8]| {
        bytes
            .chunks_exact(4)
            .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("I32 fixture word")))
            .collect::<Vec<_>>()
    };

    let _query = read_f32(&FIXTURE[..Q_BYTES]);
    let low_start = Q_BYTES;
    let q_low = read_f32(&FIXTURE[low_start..low_start + LOW_BYTES]);
    let cache_start = low_start + LOW_BYTES;
    let compact = FIXTURE[cache_start..cache_start + CACHE_BYTES]
        .chunks_exact(2)
        .map(|chunk| half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect::<Vec<_>>();
    let selected_start = cache_start + CACHE_BYTES;
    let selected = read_i32(&FIXTURE[selected_start..selected_start + SELECTED_BYTES]);
    let expected = read_f32(&FIXTURE[selected_start + SELECTED_BYTES..]);

    let q_low = Array::from_slice(
        &q_low,
        &[BATCHES as i32, TOKENS as i32, HEADS as i32, WIDTH as i32],
    );
    let compact = Array::from_slice(&compact, &[BATCHES as i32, CACHE_ROWS as i32, WIDTH as i32]);
    let selected = Array::from_slice(&selected, &[BATCHES as i32, TOKENS as i32, SELECTED as i32]);

    let n_batches = Array::from_int(BATCHES as i32);
    let n_tokens = Array::from_int(TOKENS as i32);
    let n_heads = Array::from_int(HEADS as i32);
    let n_selected = Array::from_int(SELECTED as i32);
    let valid_prefix = Array::from_int(VALID_PREFIX as i32);
    let dense_all = Array::from_int(0);
    let cache_rows = Array::from_int(CACHE_ROWS as i32);
    let scale = Array::from_f32_as(1.0 / 16.0, Dtype::Float32);
    let inputs: [&Array; 11] = [
        &n_batches,
        &n_tokens,
        &n_heads,
        &n_selected,
        &valid_prefix,
        &dense_all,
        &cache_rows,
        &scale,
        &q_low,
        &compact,
        &selected,
    ];
    let actual = dsa_indexed_lora_kernel()
        .as_ref()
        .expect("production indexed-lora kernel should compile")
        .apply(
            &inputs,
            &[],
            (
                (HEADS.div_ceil(8) * 256) as i32,
                TOKENS as i32,
                BATCHES as i32,
            ),
            (256, 1, 1),
            &[OutputArg {
                shape: vec![BATCHES as i32, TOKENS as i32, HEADS as i32, WIDTH as i32],
                dtype: Dtype::Float32,
            }],
            &Stream::thread_local_or_default(),
        )
        .expect("production indexed-lora kernel should dispatch")
        .pop()
        .expect("production indexed-lora output");
    let actual = actual
        .to_vec1::<f32>()
        .expect("model-derived indexed-lora output should be readable");
    assert_eq!(actual.len(), expected.len());
    for (index, (&got, &want)) in actual.iter().zip(&expected).enumerate() {
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "model-derived long-window indexed-lora[{index}]: LISA {got}, DS4 {want}"
        );
    }
}

#[test]
fn glm53_hc_split_matches_ds4_float4_equations() {
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
fn glm53_hc_split_collapse_norm4_matches_separate_reference() {
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
fn kda_prepare_real_inputs() {
    // Loads the real-model inputs dumped by LISA_GLM53_DUMP and validates
    // the kernel against the naive reference formulas on REAL data.
    let path = "/tmp/kda-real-inputs.bin";
    let Ok(bytes) = std::fs::read(path) else {
        eprintln!("[real-inputs] {path} missing — skipping");
        return;
    };
    let mut off = 0usize;
    let read_vec = |bytes: &Vec<u8>, off: &mut usize| -> Vec<f32> {
        let n = u32::from_le_bytes(bytes[*off..*off + 4].try_into().unwrap()) as usize;
        *off += 4;
        let mut v = vec![0f32; n];
        for (i, x) in v.iter_mut().enumerate() {
            *x = f32::from_le_bytes(bytes[*off + i * 4..*off + i * 4 + 4].try_into().unwrap());
        }
        *off += n * 4;
        v
    };
    let q = read_vec(&bytes, &mut off);
    let k = read_vec(&bytes, &mut off);
    let v = read_vec(&bytes, &mut off);
    let gate = read_vec(&bytes, &mut off);
    let cq = read_vec(&bytes, &mut off);
    let ck = read_vec(&bytes, &mut off);
    let cv = read_vec(&bytes, &mut off);
    let alog = read_vec(&bytes, &mut off);
    let dtb = read_vec(&bytes, &mut off);
    let cs = read_vec(&bytes, &mut off);
    let heads = 64usize;
    let d = 128usize;
    let proj = heads * d;
    let t = q.len() / proj;
    let conv_dim = 3 * proj;
    eprintln!(
        "[real-inputs] t={t} proj={proj} cs_len={} alog_len={}",
        cs.len(),
        alog.len()
    );
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
    let rt = Stream::thread_local_or_default();
    let (qo, _ko, vo, go, cso) = kda_prepare(
        &qa, &ka, &va, &ga, &cqa, &cka, &cva, &ala, &dtba, &csa, heads, -5.0, 1e-5, &rt,
    )
    .expect("kda_prepare declined");
    let _ = cso.eval();
    let qo_v = qo.to_vec1::<f32>().unwrap();
    let vo_v = vo.to_vec1::<f32>().unwrap();
    let go_v = go.to_vec1::<f32>().unwrap();

    let silu = |x: f32| x / (1.0 + (-x).exp());
    let sig = |x: f32| 1.0 / (1.0 + (-x).exp());
    let window = |tok: usize, w: usize, ch: usize| -> (f32, f32, f32) {
        let r = tok + w;
        if r < 3 {
            (
                cs[r * conv_dim + ch],
                cs[r * conv_dim + proj + ch],
                cs[r * conv_dim + 2 * proj + ch],
            )
        } else {
            let ti = r - 3;
            (q[ti * proj + ch], k[ti * proj + ch], v[ti * proj + ch])
        }
    };
    // per-head norm check on q
    let mut bad_q = 0usize;
    let mut worst = (0usize, 0.0f32, 0.0f32);
    for tok in 0..t {
        for h in 0..heads {
            let (mut q_sq, mut qq) = (0.0f32, [0f32; 128]);
            for c in 0..d {
                let ch = h * d + c;
                let tap = ch * 4;
                let mut qa2 = 0.0f32;
                for w in 0..4usize {
                    let (wq, _, _) = window(tok, w, ch);
                    qa2 += wq * cq[tap + w];
                }
                let qv = silu(qa2);
                q_sq += qv * qv;
                qq[c] = qv;
            }
            let qn = (q_sq + 1e-6f32).sqrt().recip() * (128f32).powf(-0.5);
            for c in 0..d {
                let want = qq[c] * qn;
                let got = qo_v[tok * proj + h * d + c];
                if (got - want).abs() > 2e-4 {
                    bad_q += 1;
                    if (got - want).abs() > worst.2 {
                        worst = (h, got, want);
                    }
                }
            }
        }
    }
    eprintln!(
        "[real-inputs] bad q values: {bad_q} worst head {} got {} want {}",
        worst.0, worst.1, worst.2
    );
    // v check (first 2048 channels)
    let mut bad_v = 0usize;
    for tok in 0..t.min(1) {
        for ch in 0..2048 {
            let mut va2 = 0.0f32;
            for w in 0..4usize {
                let (_, _, wv) = window(tok, w, ch);
                va2 += wv * cv[ch * 4 + w];
            }
            let want = silu(va2);
            let got = vo_v[tok * proj + ch];
            if (got - want).abs() > 1e-4 {
                bad_v += 1;
                if bad_v < 4 {
                    eprintln!("[real-inputs] v mismatch tok={tok} ch={ch}: got {got} want {want}");
                }
            }
        }
    }
    eprintln!("[real-inputs] bad v values: {bad_v}");
    let mut bad_g = 0usize;
    for i in 0..(t * proj).min(2048) {
        let gt = gate[i] + dtb[i];
        let want = (-5.0f32 * sig(alog[i / d].exp() * gt)).exp();
        let got = go_v[i];
        if (got - want).abs() > 1e-3 + 1e-2 * want {
            bad_g += 1;
            if bad_g < 4 {
                eprintln!("[real-inputs] g mismatch at {i}: got {got} want {want}");
            }
        }
    }
    eprintln!("[real-inputs] bad g values: {bad_g}");
    assert_eq!(bad_q, 0, "kernel q diverges on real inputs");
    assert_eq!(bad_v, 0, "kernel v diverges on real inputs");
    assert_eq!(bad_g, 0, "kernel g diverges on real inputs");
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
fn dsa_absorbed_attention_reads_packed_affine_kv_b() {
    if !crate::device::metal_available() {
        eprintln!("[dsa-packed-kv] skipping: no Metal device available");
        return;
    }
    let rt = Stream::thread_local_or_default();
    let batches = 2usize;
    let heads = 64usize;
    let queries = 2usize;
    let tc = 3usize;
    let rows = heads * 512;
    let group_size = 64usize;
    let groups = 512 / group_size;
    let mut packed = vec![0u32; rows * 128];
    let mut scales = vec![0.0f32; rows * groups];
    let mut biases = vec![0.0f32; rows * groups];
    for row in 0..rows {
        for group in 0..groups {
            scales[row * groups + group] = 0.00012 + (row % 7) as f32 * 0.000003;
            biases[row * groups + group] = -0.012 + (group % 3) as f32 * 0.0004;
        }
        for word in 0..128 {
            let mut value = 0u32;
            for lane in 0..4 {
                let j = word * 4 + lane;
                let code = ((row * 13 + j * 29 + 17) % 251) as u32;
                value |= code << (lane * 8);
            }
            packed[row * 128 + word] = value;
        }
    }
    let q: Vec<f32> = (0..batches * heads * queries * 256)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 200.0)
        .collect();
    let latent_f32: Vec<f32> = (0..batches * tc * 512)
        .map(|i| ((i * 19 % 97) as f32 - 48.0) / 240.0)
        .collect();
    let latent_f16: Vec<half::f16> = latent_f32
        .iter()
        .copied()
        .map(half::f16::from_f32)
        .collect();
    let q_arr = Array::from_slice(&q, &[batches as i32, heads as i32, queries as i32, 256]);
    let w_arr = Array::from_slice(&packed, &[rows as i32, 128]);
    let s_arr = Array::from_slice(&scales, &[rows as i32, groups as i32]);
    let b_arr = Array::from_slice(&biases, &[rows as i32, groups as i32]);
    let lat_arr = Array::from_slice(&latent_f16, &[batches as i32, tc as i32, 512]);
    let selected_ids = [0i32, 1, 2, 2, 1, 0, 1, 0, 2, 2, 0, 1];
    let selected = Array::from_slice(&selected_ids, &[batches as i32, queries as i32, 3]);

    let out = dsa_absorbed_attention(
        &q_arr,
        &w_arr,
        &s_arr,
        &b_arr,
        group_size,
        &lat_arr,
        &selected,
        1.0 / 16.0,
        &rt,
    )
    .expect("packed affine absorbed DSA kernel should accept the fixture");
    let got = out.to_vec1::<f32>().expect("output readback");

    // DS4's sparse-boundary expansion pads incomplete causal tails with
    // -1. Those rows must be ignored by the indexed-lora softmax.
    let mut padded_ids = Vec::with_capacity(batches * queries * 4);
    for row in selected_ids.chunks_exact(3) {
        padded_ids.extend_from_slice(row);
        padded_ids.push(-1);
    }
    let padded_selected = Array::from_slice(&padded_ids, &[batches as i32, queries as i32, 4]);
    let padded_out = dsa_absorbed_attention(
        &q_arr,
        &w_arr,
        &s_arr,
        &b_arr,
        group_size,
        &lat_arr,
        &padded_selected,
        1.0 / 16.0,
        &rt,
    )
    .expect("invalid sparse-tail padding should be ignored");
    assert_eq!(
        got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        padded_out
            .to_vec1::<f32>()
            .expect("padded output readback")
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        "-1 sparse-tail entries must not affect indexed attention"
    );

    let prefix_out = dsa_absorbed_attention_with_valid_prefix(
        &q_arr,
        &w_arr,
        &s_arr,
        &b_arr,
        group_size,
        &lat_arr,
        &padded_selected,
        3,
        1.0 / 16.0,
        &rt,
        false,
    )
    .expect("guaranteed-prefix indexed attention should accept padded tail ids");
    assert_eq!(
        got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        prefix_out
            .to_vec1::<f32>()
            .expect("valid-prefix output readback")
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        "DS4-style valid-prefix fast path must preserve padded-selection output"
    );

    let dense_ids: Vec<i32> = (0..batches * queries).flat_map(|_| 0..tc as i32).collect();
    let dense_selected =
        Array::from_slice(&dense_ids, &[batches as i32, queries as i32, tc as i32]);
    let explicit_dense = dsa_absorbed_attention(
        &q_arr,
        &w_arr,
        &s_arr,
        &b_arr,
        group_size,
        &lat_arr,
        &dense_selected,
        1.0 / 16.0,
        &rt,
    )
    .expect("explicit contiguous selection should dispatch");
    let implicit_dense = dsa_absorbed_attention_dense(
        &q_arr,
        &w_arr,
        &s_arr,
        &b_arr,
        group_size,
        &lat_arr,
        1.0 / 16.0,
        &rt,
    )
    .expect("implicit contiguous rows should dispatch");
    assert_eq!(
        explicit_dense
            .to_vec1::<f32>()
            .expect("explicit dense readback")
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>(),
        implicit_dense
            .to_vec1::<f32>()
            .expect("implicit dense readback")
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );

    let affine_weight = |row: usize, j: usize| -> f32 {
        let word = packed[row * 128 + j / 4];
        let code = ((word >> ((j % 4) * 8)) & 0xff) as f32;
        let group = row * groups + j / group_size;
        code.mul_add(scales[group], biases[group])
    };
    let latent_at = |token: usize, j: usize| -> f32 { latent_f16[token * 512 + j].to_f32() };
    assert_eq!(
        out.shape(),
        [batches as i32, heads as i32, queries as i32, 256]
    );
    let mut max_error = 0.0f32;
    for batch in 0..batches {
        for query in 0..queries {
            let sel_base = (batch * queries + query) * 3;
            let mut weights = [0.0f32; 3];
            let mut scores = [0.0f32; 3];
            for head in 0..heads {
                let mut q_low = vec![0.0f32; 512];
                for (j, dst) in q_low.iter_mut().enumerate() {
                    for d in 0..256 {
                        let row = head * 512 + d;
                        let q_row = ((batch * heads + head) * queries + query) * 256;
                        *dst = q[q_row + d].mul_add(affine_weight(row, j), *dst);
                    }
                }
                for slot in 0..3 {
                    let token = selected_ids[sel_base + slot] as usize;
                    let mut dot = 0.0f32;
                    for j in 0..512 {
                        dot = q_low[j].mul_add(latent_at(batch * tc + token, j), dot);
                    }
                    scores[slot] = dot / 16.0;
                }
                let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exps = scores.map(|s| (s - max_score).exp());
                let denom: f32 = exps.iter().sum();
                for slot in 0..3 {
                    weights[slot] = exps[slot] / denom;
                }
                let mut weighted = vec![0.0f32; 512];
                for j in 0..512 {
                    for slot in 0..3 {
                        let token = selected_ids[sel_base + slot] as usize;
                        weighted[j] += weights[slot] * latent_at(batch * tc + token, j);
                    }
                }
                for d in 0..256 {
                    let row = head * 512 + 256 + d;
                    let mut want = 0.0f32;
                    for j in 0..512 {
                        want = weighted[j].mul_add(affine_weight(row, j), want);
                    }
                    let out_row = ((batch * heads + head) * queries + query) * 256;
                    let error = (got[out_row + d] - want).abs();
                    max_error = max_error.max(error);
                }
            }
        }
    }
    assert!(max_error < 2e-4, "packed affine DSA max error {max_error}");
}

#[test]
fn dsa_fused_serial_decode_matches_staged_affine_path() {
    if !crate::device::metal_available() {
        eprintln!("[dsa-fused-decode] skipping: no Metal device available");
        return;
    }
    let stream = Stream::thread_local_or_default();
    let heads = 64usize;
    let tc = 2051usize;
    let group_size = 64usize;
    let groups = 512 / group_size;
    let weight_rows = heads * 512;
    let mut packed = vec![0u32; weight_rows * 128];
    let mut scales = vec![0.0f32; weight_rows * groups];
    let mut biases = vec![0.0f32; weight_rows * groups];
    for row in 0..weight_rows {
        for group in 0..groups {
            scales[row * groups + group] = 0.00008 + (row % 11) as f32 * 0.000002;
            biases[row * groups + group] = -0.008 + (group % 5) as f32 * 0.0003;
        }
        for word in 0..128 {
            let mut value = 0u32;
            for lane in 0..4 {
                let code = ((row * 17 + word * 7 + lane * 31 + 3) % 251) as u32;
                value |= code << (lane * 8);
            }
            packed[row * 128 + word] = value;
        }
    }
    let q: Vec<f32> = (0..heads * 256)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 300.0)
        .collect();
    let latent_f16: Vec<half::f16> = (0..tc * 512)
        .map(|i| half::f16::from_f32(((i * 19 % 97) as f32 - 48.0) / 320.0))
        .collect();
    let selected_ids: Vec<i32> = (0..tc as i32).collect();
    let query = Array::from_slice(&q, &[1, heads as i32, 1, 256]);
    let weight = Array::from_slice(&packed, &[weight_rows as i32, 128]);
    let scale = Array::from_slice(&scales, &[weight_rows as i32, groups as i32]);
    let bias = Array::from_slice(&biases, &[weight_rows as i32, groups as i32]);
    let latents = Array::from_slice(&latent_f16, &[1, tc as i32, 512]);
    let selected = Array::from_slice(&selected_ids, &[1, 1, tc as i32]);

    let fused = dsa_absorbed_attention(
        &query,
        &weight,
        &scale,
        &bias,
        group_size,
        &latents,
        &selected,
        1.0 / 16.0,
        &stream,
    )
    .expect("fused serial DSA kernel should dispatch")
    .to_vec1::<f32>()
    .expect("read fused serial DSA result");
    let staged = dsa_absorbed_attention_stages(
        &query,
        &weight,
        &scale,
        &bias,
        group_size,
        &latents,
        &selected,
        1.0 / 16.0,
        &stream,
        false,
    )
    .expect("staged serial DSA reference should dispatch")
    .0
    .to_vec1::<f32>()
    .expect("read staged serial DSA result");
    assert_eq!(fused.len(), staged.len());
    let mut max_abs = 0.0f32;
    let mut max_index = 0usize;
    for (index, (&got, &want)) in fused.iter().zip(&staged).enumerate() {
        let error = (got - want).abs();
        if error > max_abs {
            max_abs = error;
            max_index = index;
        }
    }
    eprintln!("[dsa-fused-decode] vs staged: max_abs={max_abs:e} at {max_index}");
    assert!(max_abs <= 3e-3, "fused DSA result differs from staged path");
}
