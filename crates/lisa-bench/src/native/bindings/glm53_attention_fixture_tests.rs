#![cfg(test)]
use super::*;

#[test]
fn glm53_indexer_projection_uses_bf16_tiled_prefill() -> anyhow::Result<()> {
    if !lisa_metal::device::metal_available() {
        eprintln!("[glm53-indexer-projection] skipping: no Metal device available");
        return Ok(());
    }
    let (in_dim, out_dim, rows) = (323usize, 13usize, 9usize);
    let source_weights: Vec<f32> = (0..out_dim * in_dim)
        .map(|i| (((i * 17 + 5) % 113) as f32 - 56.0) * 0.001)
        .collect();
    let weights = Array::from_slice(&source_weights, &[out_dim as i32, in_dim as i32])
        .as_dtype(Dtype::Bfloat16)?;
    let input_values: Vec<f32> = (0..rows * in_dim)
        .map(|i| (((i * 11 + 9) % 97) as f32 - 48.0) * 0.002)
        .collect();
    let input = Array::from_slice(&input_values, &[1, rows as i32, in_dim as i32]);
    let projection = Indexer {
        context: crate::metal_context::execution_context(),
        wq_b_w: weights.clone(),
        wk_w: weights.clone(),
        k_norm_w: Array::from_slice(&vec![1.0f32; 128], &[128]),
        k_norm_b: Array::from_slice(&vec![0.0f32; 128], &[128]),
        weights_w: weights.clone(),
        gate_w: weights.clone(),
        ape: Array::from_slice(&vec![0.0f32; 512], &[4, 128]),
        heads: 32,
        head_dim: 128,
        topk: 2048,
        kpool: 4,
        always_tail: true,
    };
    let actual = projection.project(&input, &weights)?.to_vec1::<f32>()?;
    let rounded_weights = weights.as_dtype(Dtype::Float32)?.to_vec1::<f32>()?;
    for row in 0..rows {
        for col in 0..out_dim {
            let mut expected = 0.0f64;
            for k in 0..in_dim {
                let w = half::f16::from_f32(rounded_weights[col * in_dim + k]).to_f32();
                let x = half::f16::from_f32(input_values[row * in_dim + k]).to_f32();
                expected += f64::from(w) * f64::from(x);
            }
            assert!(
                (f64::from(actual[row * out_dim + col]) - expected).abs() < 2e-4,
                "indexer projection ({row},{col}): {} vs {expected}",
                actual[row * out_dim + col]
            );
        }
    }
    Ok(())
}

#[test]
fn glm53_indexer_pair_projection_matches_separate_gemv() -> anyhow::Result<()> {
    if !lisa_metal::device::metal_available() {
        eprintln!("[glm53-indexer-pair] skipping: no Metal device available");
        return Ok(());
    }
    let make_weights = |out_dim: usize, in_dim: usize, salt: usize| {
        let values: Vec<f32> = (0..out_dim * in_dim)
            .map(|i| (((i * 19 + salt * 23) % 127) as f32 - 63.0) * 0.001)
            .collect();
        Array::from_slice(&values, &[out_dim as i32, in_dim as i32])
            .as_dtype(Dtype::Bfloat16)
            .expect("cast paired indexer weights to BF16")
    };
    let weights_a = make_weights(17, 323, 3);
    let weights_b = make_weights(9, 197, 7);
    let x_a_values: Vec<f32> = (0..2 * 323)
        .map(|i| (((i * 11 + 5) % 97) as f32 - 48.0) * 0.002)
        .collect();
    let x_b_values: Vec<f32> = (0..2 * 197)
        .map(|i| (((i * 13 + 9) % 103) as f32 - 51.0) * 0.0015)
        .collect();
    let x_a = Array::from_slice(&x_a_values, &[1, 2, 323]);
    let x_b = Array::from_slice(&x_b_values, &[1, 2, 197]);
    let projection = Indexer {
        context: crate::metal_context::execution_context(),
        wq_b_w: weights_a.clone(),
        wk_w: weights_a.clone(),
        k_norm_w: Array::from_slice(&vec![1.0f32; 128], &[128]),
        k_norm_b: Array::from_slice(&vec![0.0f32; 128], &[128]),
        weights_w: weights_b.clone(),
        gate_w: weights_b.clone(),
        ape: Array::from_slice(&vec![0.0f32; 512], &[4, 128]),
        heads: 32,
        head_dim: 128,
        topk: 2048,
        kpool: 4,
        always_tail: true,
    };

    let before = lisa_metal::runtime_dispatch_count();
    let (paired_a, paired_b) = projection.project_pair(&x_a, &weights_a, &x_b, &weights_b)?;
    assert_eq!(
        lisa_metal::runtime_dispatch_count() - before,
        1,
        "paired indexer projections should use one dispatch"
    );
    let separate_a = projection.project(&x_a, &weights_a)?.to_vec1::<f32>()?;
    let separate_b = projection.project(&x_b, &weights_b)?.to_vec1::<f32>()?;
    let paired_a = paired_a.to_vec1::<f32>()?;
    let paired_b = paired_b.to_vec1::<f32>()?;
    assert_eq!(
        paired_a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        separate_a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "paired A projection changed the DS4 GEMV result"
    );
    assert_eq!(
        paired_b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        separate_b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "paired B projection changed the DS4 GEMV result"
    );
    Ok(())
}
