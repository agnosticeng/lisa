#![cfg(test)]
use super::*;
use anyhow::Context;
use lisa_metal::{Dtype, Stream};
#[path = "qwen4_moe_oracle.rs"]
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
pub(crate) fn source(e: i32, h: i32, n: i32) -> anyhow::Result<Source> {
    let mut source = Source(Default::default());
    let router = Array::from_slice(
        &(0..e * h)
            .map(|i| half::bf16::from_f32(((i * 7 % 67) as f32 - 33.) / 307.))
            .collect::<Vec<_>>(),
        &[e, h],
    );
    source.0.insert("block.gate.weight".into(), router);
    for (name, shape) in [
        ("switch_mlp.gate_proj", vec![e, n, h]),
        ("switch_mlp.up_proj", vec![e, n, h]),
        ("switch_mlp.down_proj", vec![e, h, n]),
        ("shared_expert.gate_proj", vec![n, h]),
        ("shared_expert.up_proj", vec![n, h]),
        ("shared_expert.down_proj", vec![h, n]),
        ("shared_expert_gate", vec![1, h]),
    ] {
        let mut ws = shape.clone();
        *ws.last_mut().unwrap() /= 8;
        let count = ws.iter().map(|&d| d as usize).product();
        let data = (0..count)
            .map(|i| {
                (i as u32)
                    .wrapping_mul(2654435761)
                    .wrapping_add(name.len() as u32)
            })
            .collect::<Vec<_>>();
        source.0.insert(
            format!("block.{name}.weight"),
            Array::from_slice(&data, &ws),
        );
        drop(data);
        let mut ms = shape;
        *ms.last_mut().unwrap() /= 32;
        let count = ms.iter().map(|&d| d as usize).product();
        for (suffix, value) in [("scales", 0.001), ("biases", -0.007)] {
            source.0.insert(
                format!("block.{name}.{suffix}"),
                Array::from_slice(&vec![half::bf16::from_f32(value); count], &ms),
            );
        }
    }
    Ok(source)
}
#[test]
fn expert_plans_preserve_decode_gather_indirect_and_batch_bits() -> anyhow::Result<()> {
    use crate::bindings::deepseek41_moe::tests::{counts, difference};
    let stream = Stream::gpu();
    let mut src = source(512, 2560, 640)?;
    let old = original::SparseMoeBlock::load(&mut src, "block", 10)?;
    let new = SparseMoeBlock::load_on(
        crate::metal_context::execution_context(),
        &mut src,
        "block",
        10,
    )?;
    drop(src);
    for (batch, rows) in [
        (1, 1),
        (1, 2),
        (1, 8),
        (1, 9),
        (2, 1),
        (2, 8),
        (1, 32),
        (1, 204),
        (1, 205),
        (1, 256),
    ] {
        let x = Array::from_slice(
            &(0..batch * rows * 2560)
                .map(|i| half::bf16::from_f32(((i * 13 % 251) as f32 - 125.) / 311.))
                .collect::<Vec<_>>(),
            &[batch, rows, 2560],
        );
        stream.runtime().commands.flush_and_wait()?;
        let before = counts();
        let a = old.forward(&x)?;
        stream.runtime().commands.flush_and_wait()?;
        let middle = counts();
        let b = new.forward(&x)?;
        stream.runtime().commands.flush_and_wait()?;
        let after = counts();
        assert_eq!(a.shape(), b.shape());
        assert_eq!(a.dtype(), Dtype::Bfloat16);
        assert_eq!(
            a.to_vec1::<u16>()?,
            b.to_vec1::<u16>()?,
            "batch={batch} rows={rows}"
        );
        if !after.is_empty() {
            assert_eq!(
                difference(&before, &middle),
                difference(&middle, &after),
                "batch={batch} rows={rows}"
            );
        }
    }
    Ok(())
}
