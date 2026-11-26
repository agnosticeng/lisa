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

#[test]
fn captured_expert_plan_matches_independent_serial_rows() -> anyhow::Result<()> {
    use lisa_metal::ops::indexing::IndexOp;
    let (h, n) = (2560, 640);
    for experts in [512, 32] {
        // The smaller expert bank exercises the generic single-row fallback.
        let mut src = source(experts, h, n)?;
        let model = SparseMoeBlock::load_on(
            crate::metal_context::execution_context(),
            &mut src,
            "block",
            10,
        )?;
        drop(src);
        for (batch, rows) in [(1, 1), (1, 2), (1, 7), (1, 8), (1, 9), (2, 1), (2, 7)] {
            // Retain a nonzero storage offset and padded row strides.
            let backing = Array::from_slice(
                &(0..batch * rows * (h + 32))
                    .map(|i| half::bf16::from_f32(((i * 13 % 251) as f32 - 125.) / 311.))
                    .collect::<Vec<_>>(),
                &[batch, rows, h + 32],
            );
            let x = backing.index((.., .., 8..8 + h));
            let captured = model
                .forward_captured(&x)
                .with_context(|| format!("captured E={experts} B={batch} S={rows}"))?
                .to_vec1::<u16>()?;
            let mut expected = Vec::with_capacity((batch * rows * h) as usize);
            for b in 0..batch {
                for row in 0..rows {
                    let single = x.index((b..b + 1, row..row + 1, ..)).contiguous()?;
                    expected.extend(
                        model
                            .forward(&single)
                            .with_context(|| {
                                format!("serial E={experts} B={batch} S={rows} row={b}/{row}")
                            })?
                            .to_vec1::<u16>()?,
                    );
                }
            }
            assert_eq!(
                captured, expected,
                "experts={experts} batch={batch} rows={rows}"
            );
            // Exercise both cached graph identities at the same shape.
            let ordinary = model
                .forward(&x)
                .with_context(|| format!("ordinary padded E={experts} B={batch} S={rows}"))?
                .to_vec1::<u16>()?;
            let ordinary_dense = model
                .forward(&x.contiguous()?)
                .with_context(|| format!("ordinary dense E={experts} B={batch} S={rows}"))?
                .to_vec1::<u16>()?;
            assert_eq!(
                ordinary, ordinary_dense,
                "ordinary input layout E={experts} B={batch} S={rows}"
            );
            let different = ordinary
                .iter()
                .zip(&expected)
                .filter(|(a, b)| a != b)
                .count();
            let maximum = ordinary
                .iter()
                .zip(&expected)
                .map(|(&a, &b)| {
                    (half::bf16::from_bits(a).to_f32() - half::bf16::from_bits(b).to_f32()).abs()
                })
                .fold(0.0f32, f32::max);
            eprintln!(
                "expert serial differential E={experts} B={batch} S={rows}: ordinary={different}/{} max_abs={maximum} captured=0",
                expected.len(),
            );
            assert_eq!(
                model
                    .forward_captured(&x)
                    .with_context(|| format!("cached captured E={experts} B={batch} S={rows}"))?
                    .to_vec1::<u16>()?,
                expected,
                "captured graph after ordinary execution: experts={experts} B={batch} S={rows}"
            );
        }
    }
    Ok(())
}
