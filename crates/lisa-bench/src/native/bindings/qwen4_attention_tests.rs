#![cfg(test)]
use super::*;
use anyhow::Context;
use lisa_metal::{Dtype, Stream};
#[path = "qwen4_attention_oracle.rs"]
mod original;
pub(crate) struct Source(std::collections::HashMap<String, Array>);
impl lisa_engine::weights::source::TensorSource<Array> for Source {
    fn get(&mut self, name: &str) -> anyhow::Result<Array> {
        self.0
            .get(name)
            .cloned()
            .with_context(|| format!("missing fixture {name}"))
    }
}
pub(crate) fn source(hidden: i32) -> Source {
    let mut src = Source(Default::default());
    for (name, out, input) in [
        ("q_proj", 12288, hidden),
        ("k_proj", 512, hidden),
        ("v_proj", 512, hidden),
        ("o_proj", hidden, 6144),
        ("indexer.index_qk_proj", 640, hidden),
    ] {
        let values = (0..out * input / 8)
            .map(|i| (i as u32).wrapping_mul(2654435761))
            .collect::<Vec<_>>();
        src.0.insert(
            format!("a.{name}.weight"),
            Array::from_slice(&values, &[out, input / 8]),
        );
        for (suffix, val) in [("scales", 0.001), ("biases", -0.007)] {
            src.0.insert(
                format!("a.{name}.{suffix}"),
                Array::from_slice(
                    &vec![half::bf16::from_f32(val); (out * input / 32) as usize],
                    &[out, input / 32],
                ),
            );
        }
    }
    for (name, n) in [
        ("q_norm", 256),
        ("k_norm", 256),
        ("indexer.q_layernorm", 128),
        ("indexer.k_layernorm", 128),
    ] {
        src.0.insert(
            format!("a.{name}.weight"),
            Array::from_slice(&vec![half::bf16::ONE; n], &[n as i32]),
        );
    }
    src
}
fn same(a: &Array, b: &Array) -> anyhow::Result<()> {
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.dtype(), b.dtype());
    let a = a.to_vec1::<u16>()?;
    let b = b.to_vec1::<u16>()?;
    if let Some(i) = a.iter().zip(&b).position(|(a, b)| a != b) {
        anyhow::bail!(
            "first bit mismatch at {i}: {} vs {} ({} elements)",
            a[i],
            b[i],
            a.len()
        );
    }
    Ok(())
}
#[test]
fn attention_plans_preserve_causal_sparse_ragged_and_cache_bits() -> anyhow::Result<()> {
    use crate::bindings::deepseek41_moe::tests::{counts, difference};
    let stream = Stream::gpu();
    let mut src = source(256);
    let old = original::Attention::load(&mut src, "a", 1e-6, 24, 2, 256)?;
    let new = Attention::load_on(
        crate::metal_context::execution_context(),
        &mut src,
        "a",
        1e-6,
        24,
        2,
        256,
    )?;
    let rope = Rotary::new(64, 10000000.);
    for (batch, steps, ragged) in [
        (1, vec![1, 8, 9, 2030, 1, 3, 8], false),
        (1, vec![2053], false),
        (2, vec![1, 8, 9], false),
        (2, vec![9, 1], true),
    ] {
        let (mut a, mut b) = (
            FullAttentionCache::new(crate::metal_context::execution_context(), 2, 256),
            FullAttentionCache::new(crate::metal_context::execution_context(), 2, 256),
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
            let pos = Array::from_slice(
                &(offset..offset + rows)
                    .map(|i| i as f32)
                    .collect::<Vec<_>>(),
                &[1, rows],
            );
            if ragged {
                let mask = Array::from_slice(
                    &vec![true; (batch * rows * (offset + rows)) as usize],
                    &[batch, 1, rows, offset + rows],
                );
                a.attn_mask = Some(mask.clone());
                b.attn_mask = Some(mask);
                a.next_pos = Some(vec![offset as usize; batch as usize]);
                b.next_pos = a.next_pos.clone();
            }
            stream.runtime().commands.flush_and_wait()?;
            let first = counts();
            let left = old.forward(&x, &rope, Some(&mut a), offset as usize, &pos)?;
            stream.runtime().commands.flush_and_wait()?;
            let middle = counts();
            let right = new.forward(&x, &rope, Some(&mut b), offset as usize, &pos)?;
            stream.runtime().commands.flush_and_wait()?;
            let last = counts();
            same(&left, &right)?;
            assert_eq!(left.dtype(), Dtype::Bfloat16);
            assert_eq!((a.offset, a.capacity), (b.offset, b.capacity));
            // Compare live rows only; reserved capacity is intentionally uninitialized.
            use lisa_metal::ops::indexing::IndexOp;
            same(
                &a.keys
                    .as_ref()
                    .unwrap()
                    .index((.., .., 0..a.offset as i32, ..)),
                &b.keys
                    .as_ref()
                    .unwrap()
                    .index((.., .., 0..b.offset as i32, ..)),
            )?;
            same(
                &a.values
                    .as_ref()
                    .unwrap()
                    .index((.., .., 0..a.offset as i32, ..)),
                &b.values
                    .as_ref()
                    .unwrap()
                    .index((.., .., 0..b.offset as i32, ..)),
            )?;
            if !last.is_empty() {
                assert_eq!(
                    difference(&first, &middle),
                    difference(&middle, &last),
                    "batch={batch} rows={rows} offset={offset} ragged={ragged}"
                );
            }
            offset += rows;
        }
    }
    Ok(())
}
