#![cfg(test)]
use super::*;
use crate::core::graph::PreparedGraph;
use lisa_backend::ScalarType;
use lisa_metal::device::MetalDevice;
use lisa_metal::{ops, ops::indexing::IndexOp};
use lisa_models::qwen3_5::entry::Plan;
fn same(a: &Array, b: &Array) {
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.to_vec1::<u16>().unwrap(), b.to_vec1::<u16>().unwrap());
}
#[test]
fn entry_plan_preserves_text_ragged_and_image_inputs() -> anyhow::Result<()> {
    let width = 128;
    for bits in [2, 4, 8] {
        for batch in [1, 2] {
            for rows in [1, 7] {
                for multimodal in [false, true] {
                    if multimodal && batch != 1 {
                        continue;
                    }
                    let weight = Array::from_slice(
                        &(0..64 * width * bits / 32)
                            .map(|i| (i as u32).wrapping_mul(0x1234567))
                            .collect::<Vec<_>>(),
                        &[64, width * bits / 32],
                    );
                    let scales =
                        Array::from_slice(&vec![half::bf16::from_f32(1. / 128.); 64 * 2], &[64, 2]);
                    let biases =
                        Array::from_slice(&vec![half::bf16::from_f32(-0.0625); 64 * 2], &[64, 2]);
                    let ids = Array::from_slice(
                        &(0..batch * rows).map(|i| i as u32 % 64).collect::<Vec<_>>(),
                        &[batch, rows],
                    );
                    let shape = if multimodal {
                        vec![1, rows, 32]
                    } else {
                        vec![batch, rows]
                    };
                    let positions = Array::from_slice(
                        &(0..shape.iter().product::<i32>())
                            .map(|i| ((i * 11 + 3) % 97) as f32)
                            .collect::<Vec<_>>(),
                        &shape,
                    );
                    let rope = Rotary::new(64, 10000.);
                    let (cos, sin) = if multimodal {
                        rope.cos_sin_multimodal(&positions)?
                    } else {
                        rope.cos_sin(&positions)?
                    };
                    let cos = cos.expand_dims(1)?.as_dtype(lisa_metal::Dtype::Bfloat16)?;
                    let sin = sin.expand_dims(1)?.as_dtype(lisa_metal::Dtype::Bfloat16)?;
                    let mut x = ops::dequantize(
                        &weight.take_axis(&ids, 0)?,
                        &scales.take_axis(&ids, 0)?,
                        &biases.take_axis(&ids, 0)?,
                        64,
                        bits,
                    )?;
                    let features = Array::from_slice(
                        &(0..3 * width)
                            .map(|i| half::bf16::from_f32(i as f32 / 64.))
                            .collect::<Vec<_>>(),
                        &[1, 3, width],
                    );
                    let splice = (batch == 1).then_some(lisa_models::vision::prompt::Splice {
                        shape: vec![batch, rows, width],
                        dtype: ScalarType::BF16,
                        offset: 2,
                        spans: vec![(1, 3)],
                    });
                    if splice.is_some() {
                        let parts = (0..rows)
                            .map(|i| {
                                if i + 2 < 4 {
                                    features.index((.., i + 1..i + 2, ..))
                                } else {
                                    x.index((.., i..i + 1, ..))
                                }
                            })
                            .collect::<Vec<_>>();
                        x = Array::concatenate(&parts.iter().collect::<Vec<_>>(), 1)?;
                    }
                    let spec = Plan {
                        positions_shape: shape,
                        multimodal,
                        rotary: 64,
                        base: 10000f32.to_bits(),
                        embedding: (64, bits),
                        splice,
                    };
                    let mut inputs = vec![&ids, &weight, &scales, &biases, &positions];
                    if batch == 1 {
                        inputs.push(&features);
                    }
                    let mut out =
                        PreparedGraph::new(MetalDevice::system_default(), spec.graph()?, vec![])?
                            .run(&inputs, &mut [])?
                            .into_iter();
                    same(&out.next().unwrap(), &x);
                    same(&out.next().unwrap(), &cos);
                    same(&out.next().unwrap(), &sin);
                }
            }
        }
    }
    Ok(())
}
