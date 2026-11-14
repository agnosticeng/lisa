#![cfg(test)]
use super::*;
use lisa_metal::{Dtype, Stream};
#[path = "qwen4_indexer_oracle.rs"]
mod original;
fn pair() -> (original::QsaIndexer, QsaIndexer) {
    let projection = QuantizedLinear {
        weight: Array::from_slice(
            &(0..640 * 32)
                .map(|i| (i as u32).wrapping_mul(2654435761))
                .collect::<Vec<_>>(),
            &[640, 32],
        ),
        scales: Array::from_slice(&vec![half::bf16::from_f32(0.01); 640 * 8], &[640, 8]),
        biases: Array::from_slice(&vec![half::bf16::from_f32(-0.07); 640 * 8], &[640, 8]),
        group_size: 32,
        bits: 4,
    };
    let norm = RmsNorm {
        weight: Array::from_slice(&vec![half::bf16::ONE; 128], &[128]),
        eps: 1e-6,
        group_size: None,
    };
    (
        original::QsaIndexer {
            index_qk_proj: projection.clone(),
            q_layer_norm: norm.clone(),
            k_layer_norm: norm.clone(),
            heads: 4,
            head_dim: 128,
            token_budget: 2048,
            compress_ratio: 4,
            block_top_k: 512,
        },
        QsaIndexer {
            context: crate::metal_context::execution_context(),
            index_qk_proj: projection,
            q_layer_norm: norm.clone(),
            k_layer_norm: norm,
            heads: 4,
            head_dim: 128,
            token_budget: 2048,
            compress_ratio: 4,
            block_top_k: 512,
            select_available: true,
        },
    )
}
fn same(a: &Array, b: &Array) -> anyhow::Result<()> {
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.dtype(), b.dtype());
    match a.dtype() {
        Dtype::Bfloat16 => assert_eq!(a.to_vec1::<u16>()?, b.to_vec1::<u16>()?),
        Dtype::Int32 | Dtype::Uint32 => assert_eq!(a.to_vec1::<u32>()?, b.to_vec1::<u32>()?),
        Dtype::Bool | Dtype::Uint8 => assert_eq!(a.to_vec1::<bool>()?, b.to_vec1::<bool>()?),
        Dtype::Float32 => assert_eq!(
            a.to_vec1::<f32>()?
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            b.to_vec1::<f32>()?
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        ),
        _ => panic!("unexpected fixture dtype"),
    }
    Ok(())
}
#[test]
fn indexer_plans_preserve_pooling_selection_and_tape_bits() -> anyhow::Result<()> {
    use crate::bindings::deepseek41_moe::tests::{counts, difference};
    let stream = Stream::gpu();
    let rope = Rotary::new(64, 10000000.);
    for (batch, cached, steps) in [
        (1, true, vec![2048, 1, 3, 8, 9]),
        (2, true, vec![2048, 1, 3]),
        (1, false, vec![2053]),
        (2, false, vec![2053]),
    ] {
        let (old, new) = pair();
        let (mut a, mut b) = (
            IndexerTape::new(crate::metal_context::execution_context()),
            IndexerTape::new(crate::metal_context::execution_context()),
        );
        let mut offset = 0;
        for rows in steps {
            let x = Array::from_slice(
                &(0..batch * rows * 256)
                    .map(|i| {
                        half::bf16::from_f32((((i + offset * 256) * 13 % 251) as f32 - 125.) / 311.)
                    })
                    .collect::<Vec<_>>(),
                &[batch, rows, 256],
            );
            stream.runtime().commands.flush_and_wait()?;
            let first = counts();
            let left = old.forward(&x, &rope, cached.then_some(&mut a), offset as usize)?;
            stream.runtime().commands.flush_and_wait()?;
            let middle = counts();
            let right = new.forward(&x, &rope, cached.then_some(&mut b), offset as usize)?;
            stream.runtime().commands.flush_and_wait()?;
            let last = counts();
            assert_eq!(left.is_some(), right.is_some());
            if let (Some(left), Some(right)) = (left, right) {
                same(&left.block_ids, &right.block_ids)?;
                same(&left.block_valid, &right.block_valid)?;
                same(&left.complete, &right.complete)?;
                same(&left.q_pos, &right.q_pos)?;
                assert_eq!((left.cr, left.kv_len), (right.cr, right.kv_len));
            }
            if cached {
                assert_eq!((a.total, a.pooled_upto), (b.total, b.pooled_upto));
                same(
                    &a.raw_live_for_test().unwrap(),
                    &b.raw_live_for_test().unwrap(),
                )?;
                if let Some(ap) = a.pooled_view() {
                    same(&ap, &b.pooled_view().unwrap())?;
                }
            }
            if !last.is_empty() {
                assert_eq!(
                    difference(&first, &middle),
                    difference(&middle, &last),
                    "batch={batch} rows={rows} offset={offset} cached={cached}"
                );
            }
            offset += rows;
        }
    }
    Ok(())
}
