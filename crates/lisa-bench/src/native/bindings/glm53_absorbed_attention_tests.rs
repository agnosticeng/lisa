#![cfg(test)]
use super::*;
#[path = "glm53_absorbed_attention_oracle.rs"]
mod oracle;
#[test]
fn glm53_dsa_prefill_value_partition_matches_ds4() {
    assert_eq!(
        lisa_models::glm53::absorbed_attention::value_slices(34, 3),
        [(0, 3), (3, 31)]
    );
    assert_eq!(
        lisa_models::glm53::absorbed_attention::value_slices(4096, 2051),
        [(0, 2048), (2048, 3), (2051, 2045)]
    );
    assert_eq!(
        lisa_models::glm53::absorbed_attention::value_slices(2052, 0),
        [(0, 2048), (2048, 4)]
    );
    assert_eq!(
        lisa_models::glm53::absorbed_attention::value_slices(2048, 2051),
        [(0, 2048)]
    );
    assert!(lisa_models::glm53::absorbed_attention::value_slices(0, 2051).is_empty());
}

#[test]
fn glm53_dsa_value_native_splits_at_dense_sparse_boundary() {
    if !lisa_metal::device::metal_available() {
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
        value_rows(input, &weights, &scales, &biases, 64, HEADS, &stream)
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
fn dsa_absorbed_attention_reads_packed_affine_kv_b() {
    if !lisa_metal::device::metal_available() {
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

    // The reference's sparse-boundary expansion pads incomplete causal tails with
    // -1. Those rows must be ignored by the indexed-lora softmax.
    let mut padded_ids = Vec::with_capacity(batches * queries * 4);
    for row in selected_ids.as_chunks::<3>().0.iter() {
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
    if !lisa_metal::device::metal_available() {
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

fn values(shape: &[i32], seed: usize) -> Array {
    Array::from_slice(
        &(0..shape.iter().product::<i32>() as usize)
            .map(|i| ((i * 17 + seed) % 127) as f32 / 151. - 0.3)
            .collect::<Vec<_>>(),
        shape,
    )
}
fn equal(a: &Array, b: &Array) -> anyhow::Result<()> {
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.dtype(), b.dtype());
    assert_eq!(a.to_vec1::<u32>()?, b.to_vec1::<u32>()?);
    Ok(())
}
#[test]
fn absorbed_plans_preserve_all_schedules_and_boundaries() -> anyhow::Result<()> {
    use lisa_metal::Dtype;
    let stream = Stream::thread_local_or_default();
    let w = Array::from_slice(
        &(0..32768 * 128)
            .map(|i| (i as u32).wrapping_mul(1729).wrapping_add(781))
            .collect::<Vec<_>>(),
        &[32768, 128],
    );
    let scales = values(&[32768, 8], 11).multiply(Array::from_f32_as(0.002, Dtype::Float32))?;
    let biases = values(&[32768, 8], 17).multiply(Array::from_f32_as(0.001, Dtype::Float32))?;
    for native in [false, true] {
        let _mode = Glm53DsaNativeScope::new(native);
        for (batch, tokens, n, boundaries) in [
            (1, 1, 1, false),
            (1, 1, 7, false),
            (1, 1, 2051, false),
            (1, 1, 2052, false),
            (1, 1, 4097, false),
            (2, 1, 7, false),
            (2, 3, 8, false),
            (1, 8, 15, true),
            (1, 34, 40, true),
        ] {
            let q = values(&[batch, 64, tokens, 256], 23);
            let cache = values(&[batch, n, 512], 31).as_dtype(Dtype::Float16)?;
            let ids = Array::from_slice(
                &(0..batch * tokens)
                    .flat_map(|_| (0..n).map(|i| if i % 7 == 6 { -1 } else { i }))
                    .collect::<Vec<_>>(),
                &[batch, tokens, n],
            );
            let _slice = boundaries.then(|| Glm53DsaValueSliceScope::new(2048, 2051));
            for (capture, force, capture_fused) in [
                (false, false, false),
                (true, false, false),
                (false, true, true),
            ] {
                let a = oracle::dsa_absorbed_attention_stages_inner(
                    &q,
                    &w,
                    &scales,
                    &biases,
                    64,
                    &cache,
                    Some(&ids),
                    0,
                    0.0625,
                    &stream,
                    capture,
                    force,
                    capture_fused,
                    boundaries,
                )
                .expect("original attention");
                let b = dsa_absorbed_attention_stages_inner(
                    &q,
                    &w,
                    &scales,
                    &biases,
                    64,
                    &cache,
                    Some(&ids),
                    0,
                    0.0625,
                    &stream,
                    capture,
                    force,
                    capture_fused,
                    boundaries,
                )
                .expect("planned attention");
                equal(&a.0, &b.0)?;
                for (a, b) in [(&a.1, &b.1), (&a.2, &b.2)] {
                    assert_eq!(a.is_some(), b.is_some());
                    if let (Some(a), Some(b)) = (a, b) {
                        equal(a, b)?;
                    }
                }
            }
        }
        for tokens in [24, 31, 32, 34] {
            let q = values(&[1, 64, tokens, 256], 41);
            let cache = values(&[1, tokens + 3, 512], 43).as_dtype(Dtype::Float16)?;
            let _slice = Glm53DsaValueSliceScope::new(2048, 2051);
            let a = oracle::dsa_absorbed_attention_flash_causal_prefill_stages(
                &q, &w, &scales, &biases, 64, &cache, 3, 0.0625, &stream, true,
            )
            .unwrap();
            let b = dsa_absorbed_attention_flash_causal_prefill_stages(
                &q, &w, &scales, &biases, 64, &cache, 3, 0.0625, &stream, true,
            )
            .unwrap();
            equal(&a.0, &b.0)?;
            equal(&a.1, &b.1)?;
            equal(&a.2, &b.2)?;
        }
    }
    Ok(())
}
