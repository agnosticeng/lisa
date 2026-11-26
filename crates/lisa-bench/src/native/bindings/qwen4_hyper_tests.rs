#![cfg(test)]
use super::*;
use lisa_metal::ops::indexing::IndexOp;
use lisa_metal::{Dtype, Stream};
#[path = "qwen4_hyper_oracle.rs"]
mod original;
pub(crate) fn linear(output: i32, input: i32, seed: u32) -> anyhow::Result<QuantizedLinear> {
    let weight = Array::from_slice(
        &(0..output * input / 8)
            .map(|i| (i as u32).wrapping_mul(2654435761).wrapping_add(seed))
            .collect::<Vec<_>>(),
        &[output, input / 8],
    );
    let scales = Array::from_slice(
        &(0..output * input / 32)
            .map(|i| 0.0001 + (i % 7) as f32 * 0.00001)
            .collect::<Vec<_>>(),
        &[output, input / 32],
    )
    .as_dtype(Dtype::Bfloat16)?;
    let biases = Array::from_slice(
        &vec![-0.0007f32; (output * input / 32) as usize],
        &[output, input / 32],
    )
    .as_dtype(Dtype::Bfloat16)?;
    Ok(QuantizedLinear {
        weight,
        scales,
        biases,
        group_size: 32,
        bits: 4,
    })
}
fn same(
    old: impl FnOnce() -> anyhow::Result<Vec<Array>>,
    new: impl FnOnce() -> anyhow::Result<Vec<Array>>,
) -> anyhow::Result<()> {
    use crate::bindings::deepseek41_moe::tests::{counts, difference};
    let stream = Stream::gpu();
    stream.runtime().commands.flush_and_wait()?;
    let before = counts();
    let old = old()?;
    stream.runtime().commands.flush_and_wait()?;
    let middle = counts();
    let new = new()?;
    stream.runtime().commands.flush_and_wait()?;
    let after = counts();
    assert_eq!(old.len(), new.len());
    for (a, b) in old.iter().zip(new.iter()) {
        assert_eq!(a.shape(), b.shape());
        assert_eq!(a.dtype(), b.dtype());
        if a.dtype() == Dtype::Uint32 {
            assert_eq!(a.to_vec1::<u32>()?, b.to_vec1::<u32>()?);
        } else {
            assert_eq!(a.to_vec1::<u16>()?, b.to_vec1::<u16>()?);
        }
    }
    if !after.is_empty() {
        assert_eq!(difference(&before, &middle), difference(&middle, &after));
    }
    Ok(())
}
#[test]
fn gated_residual_plans_preserve_packed_decode_and_batched_bits() -> anyhow::Result<()> {
    // Both implementations share this immutable LUT. Initialize it before
    // counting per-invocation launches so the first oracle call has no bias.
    lisa_metal::moe_decode::bf16_sigmoid_table(&Stream::gpu()).expect("sigmoid table");
    let (hidden, rank, hc) = (2560, 320, 4);
    let up = linear(hidden * hc, rank, 13)?;
    let order = model::decode_order(hidden as usize, hc as usize, (hidden * hc) as usize).unwrap();
    let idx = Array::from_slice(&order, &[hidden * hc]);
    let packed = QuantizedLinear {
        weight: up.weight.take_axis(&idx, 0)?,
        scales: up.scales.take_axis(&idx, 0)?,
        biases: up.biases.take_axis(&idx, 0)?,
        group_size: 32,
        bits: 4,
    };
    let prepared = PreparedGraph::new(
        MetalDevice::system_default(),
        model::pack_rows(&order),
        vec![],
    )?;
    let out = prepared.run(&[&up.weight, &up.scales, &up.biases], &mut [])?;
    for (a, b) in [&packed.weight, &packed.scales, &packed.biases]
        .into_iter()
        .zip(out)
    {
        assert_eq!(a.shape(), b.shape());
        if a.dtype() == Dtype::Uint32 {
            assert_eq!(a.to_vec1::<u32>()?, b.to_vec1::<u32>()?);
        } else {
            assert_eq!(a.to_vec1::<u16>()?, b.to_vec1::<u16>()?);
        }
    }
    for has_head in [false, true] {
        let new = GatedResidual {
            context: crate::metal_context::execution_context(),
            hc_norm: RmsNorm {
                weight: Array::from_slice(&vec![1f32; (hidden * hc) as usize], &[hidden * hc])
                    .as_dtype(Dtype::Bfloat16)?,
                eps: 1e-6,
                group_size: Some(hidden as usize),
            },
            mix_down: linear(rank, hidden * hc, 29)?,
            mix_up: up.clone(),
            decode_up: Some(packed.clone()),
            block_inject: has_head.then(|| linear(hc, hidden * hc, 37)).transpose()?,
            hc_count: hc as usize,
            hidden: (hidden * hc) as usize,
            force_norm_operations: false,
        };
        let old = original::GatedResidual {
            hc_norm: new.hc_norm.clone(),
            mix_down: new.mix_down.clone(),
            mix_up: new.mix_up.clone(),
            decode_up: new.decode_up.clone(),
            block_inject: new.block_inject.clone(),
            hc_count: new.hc_count,
            hidden: new.hidden,
        };
        same(
            || Ok(vec![old.norm_scale_q()?]),
            || Ok(vec![new.norm_scale_q()?]),
        )?;
        for (b, s) in [(1, 1), (1, 2), (1, 8), (1, 9), (2, 2)] {
            let x = Array::from_slice(
                &(0..b * s * hidden * hc)
                    .map(|i| ((i * 31 % 127) as f32 - 63.) / 89.)
                    .collect::<Vec<_>>(),
                &[b, s, hidden * hc],
            )
            .as_dtype(Dtype::Bfloat16)?;
            same(
                || Ok(vec![old.hc_norm_quarter(&x)?]),
                || Ok(vec![new.hc_norm_quarter(&x)?]),
            )?;
            same(
                || Ok(vec![old.mixed_from_normed(&x)?]),
                || Ok(vec![new.mixed_from_normed(&x)?]),
            )?;
            same(
                || {
                    let (a, b) = old.mix_from_normed(&x, has_head)?;
                    Ok(vec![a, b])
                },
                || {
                    let (a, b) = new.mix_from_normed(&x, has_head)?;
                    Ok(vec![a, b])
                },
            )?;
            if has_head {
                same(
                    || {
                        let (a, b, c) = old.mix_with_inject(&x)?;
                        Ok(vec![a, b, c])
                    },
                    || {
                        let (a, b, c) = new.mix_with_inject(&x)?;
                        Ok(vec![a, b, c])
                    },
                )?;
                let (y, _, i) = old.mix_with_inject(&x)?;
                same(
                    || Ok(vec![original::inject(&x, &y, &i)?]),
                    || Ok(vec![inject(&x, &y, &i)?]),
                )?;
            }
        }
    }
    Ok(())
}

#[test]
fn captured_hyper_mixer_matches_independent_serial_rows() -> anyhow::Result<()> {
    let (hidden, rank, hc) = (2560, 320, 4);
    let up = linear(hidden * hc, rank, 13)?;
    let order = model::decode_order(hidden as usize, hc as usize, (hidden * hc) as usize).unwrap();
    let idx = Array::from_slice(&order, &[hidden * hc]);
    let mixer = GatedResidual {
        context: crate::metal_context::execution_context(),
        hc_norm: RmsNorm {
            weight: Array::from_slice(
                &vec![half::bf16::ONE; (hidden * hc) as usize],
                &[hidden * hc],
            ),
            eps: 1e-6,
            group_size: Some(hidden as usize),
        },
        mix_down: linear(rank, hidden * hc, 29)?,
        decode_up: Some(QuantizedLinear {
            weight: up.weight.take_axis(&idx, 0)?,
            scales: up.scales.take_axis(&idx, 0)?,
            biases: up.biases.take_axis(&idx, 0)?,
            group_size: 32,
            bits: 4,
        }),
        mix_up: up,
        block_inject: Some(linear(hc, hidden * hc, 37)?),
        hc_count: hc as usize,
        hidden: (hidden * hc) as usize,
        force_norm_operations: false,
    };
    for (batch, rows) in [(1, 1), (1, 2), (1, 7), (2, 2)] {
        let make = |width| {
            Array::from_slice(
                &(0..batch * rows * width)
                    .map(|i| half::bf16::from_f32(((i * 13 % 251) as f32 - 125.) / 311.))
                    .collect::<Vec<_>>(),
                &[batch, rows, width],
            )
        };
        let x = make(hidden * hc);
        let out = make(hidden);
        let inj = make(hc);
        for finish in [false, true] {
            let mut left = vec![];
            let mut right = vec![];
            for b in 0..batch {
                for s in 0..rows {
                    let row = |a: &Array| a.index((b..b + 1, s..s + 1, ..)).contiguous();
                    let (a, c) = if finish {
                        mixer.finish(&row(&x)?, Some(&row(&out)?), Some(&row(&inj)?), 1e-6)?
                    } else {
                        mixer.mix_from_normed(&row(&x)?, true)?
                    };
                    left.extend(a.to_vec1::<u16>()?);
                    right.extend(c.to_vec1::<u16>()?);
                }
            }
            let ordinary = if finish {
                mixer.finish(&x, Some(&out), Some(&inj), 1e-6)?
            } else {
                mixer.mix_from_normed(&x, true)?
            };
            let prior = ordinary.0.to_vec1::<u16>()?;
            let differing = prior.iter().zip(&left).filter(|(a, b)| a != b).count();
            eprintln!(
                "hyper B={batch} S={rows} finish={finish}: ordinary wide/serial differences {differing}/{}",
                left.len()
            );
            let captured = if finish {
                mixer.finish_captured(&x, Some(&out), Some(&inj), 1e-6)?
            } else {
                mixer.mix_from_normed_captured(&x, true)?
            };
            assert_eq!(
                captured.0.to_vec1::<u16>()?,
                left,
                "mixed B={batch} S={rows} finish={finish}"
            );
            assert_eq!(
                captured.1.to_vec1::<u16>()?,
                right,
                "state/inject B={batch} S={rows} finish={finish}"
            );
        }
    }
    Ok(())
}
