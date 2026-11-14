//! Application adapters and native oracles for the portable recurrent bindings.
pub use crate::core::quant::QuantizedLinearExecution;
use lisa_engine::recurrent::Replay;
use lisa_metal::Array;
pub fn capture(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
) -> anyhow::Result<Option<(Array, Array, Replay<Array>)>> {
    lisa_engine::recurrent::capture(
        &crate::metal_context::execution_context(),
        q,
        k,
        v,
        g,
        beta,
        state,
    )
}
pub fn snapshot_prefix(tensor: &Array, prefix: usize) -> anyhow::Result<Array> {
    lisa_engine::recurrent::snapshot_prefix(
        &crate::metal_context::execution_context(),
        tensor,
        prefix,
    )
}
pub fn convolution_prefix(tensor: &Array, prefix: usize, history: usize) -> anyhow::Result<Array> {
    lisa_engine::recurrent::convolution_prefix(
        &crate::metal_context::execution_context(),
        tensor,
        prefix,
        history,
    )
}
#[cfg(test)]
use lisa_engine::core::graph::PreparedGraph;
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::quant::QuantizedLinear;
    use lisa_metal::{Dtype, ops};
    pub(super) fn tensor(shape: &[i32], seed: usize, dtype: Dtype) -> Array {
        let n = shape.iter().product::<i32>() as usize;
        let data: Vec<f32> = (0..n)
            .map(|i| (((i * 37 + seed * 13) % 251) as f32 - 125.) / 512.)
            .collect();
        Array::from_slice(&data, shape).as_dtype(dtype).unwrap()
    }
    pub(super) fn linear(input: i32, output: i32, seed: usize) -> QuantizedLinear {
        let dense = tensor(&[output, input], seed, Dtype::Bfloat16);
        let (weight, scales, biases) = ops::quantize(&dense, 64, 4).unwrap();
        QuantizedLinear {
            weight,
            scales,
            biases,
            group_size: 64,
            bits: 4,
        }
    }
    /// Independent pre-migration operation sequence, including nullary state
    /// allocation and recurrence across subsequent one-token calls.
    #[test]
    fn delta_decode_plan_preserves_activation_and_state_bits() -> anyhow::Result<()> {
        use lisa_backend::DeltaDecode;
        use lisa_models::gated_delta::DecodePlan;
        for batch in [1, 2] {
            for swish in [false, true] {
                for (hk, hv) in [(1, 2), (4, 4)] {
                    let channels = (2 * hk + hv) * 128;
                    let projection = channels + hv * 128 + 2 * hv;
                    let spec = DeltaDecode {
                        key_heads: hk,
                        value_heads: hv,
                        key_width: 128,
                        value_width: 128,
                        convolution: 4,
                        channels,
                        projection,
                        beta_offset: channels + hv * 128,
                        decay_offset: channels + hv * 128 + hv,
                        gate_offset: channels,
                        epsilon: 1e-6f32.to_bits(),
                        swish,
                    };
                    let input = linear(128, projection, 1);
                    let output = linear(hv * 128, 128, 2);
                    let weight = tensor(&[channels, 4, 1], 3, Dtype::Bfloat16);
                    let decay = Array::from_slice(&vec![-0.5f32; hv as usize], &[hv]);
                    let bias = tensor(&[hv], 4, Dtype::Bfloat16);
                    let norm = tensor(&[128], 5, Dtype::Bfloat16);
                    let weights = vec![
                        input.weight.clone(),
                        input.scales.clone(),
                        input.biases.clone(),
                        weight.clone(),
                        decay.clone(),
                        bias.clone(),
                        norm.clone(),
                        output.weight.clone(),
                        output.scales.clone(),
                        output.biases.clone(),
                    ];
                    let mut reference_conv = ops::zeros::<half::bf16>(&[batch, 3, channels])?;
                    let mut reference_state = ops::zeros::<f32>(&[batch, hv, 128, 128])?;
                    let mut actual_conv = None;
                    let mut actual_state = None;
                    for step in 0..4 {
                        let x = tensor(&[batch, 1, 128], step + 10, Dtype::Bfloat16);
                        let projected = input.forward(&x)?;
                        let (y, ssm, conv) = lisa_metal::kernels::gated_delta::gdn_decode_complete(
                            &projected,
                            &reference_conv,
                            &weight.reshape(&[channels, 4])?,
                            &decay,
                            &bias,
                            &reference_state,
                            &norm,
                            hk,
                            hv,
                            128,
                            128,
                            4,
                            channels,
                            projection,
                            spec.beta_offset,
                            spec.decay_offset,
                            spec.gate_offset,
                            1e-6,
                            swish,
                            &lisa_metal::Stream::thread_local_or_default(),
                        )
                        .expect("supported reference kernel");
                        let expected = output.forward(&y.reshape(&[batch, 1, hv * 128])?)?;
                        let plan = DecodePlan {
                            kernel: spec,
                            batch,
                            convolution_present: step > 0,
                            state_present: step > 0,
                            projections: [(64, 4); 2],
                        };
                        let graph = PreparedGraph::new(
                            crate::metal_context::device(),
                            plan.graph()?,
                            weights.clone(),
                        )?;
                        let mut inputs = vec![&x];
                        if let Some(c) = actual_conv.as_ref() {
                            inputs.push(c);
                        }
                        if let Some(s) = actual_state.as_ref() {
                            inputs.push(s);
                        }
                        let mut values = graph.run(&inputs, &mut [])?.into_iter();
                        let got = values.next().unwrap();
                        let got_conv = values.next().unwrap();
                        let got_state = values.next().unwrap();
                        assert_eq!(
                            got.to_vec1::<u16>()?,
                            expected.to_vec1::<u16>()?,
                            "activation B={batch} H={hk}/{hv} step={step} swish={swish}"
                        );
                        assert_eq!(
                            got_conv.to_vec1::<u16>()?,
                            conv.to_vec1::<u16>()?,
                            "convolution"
                        );
                        assert_eq!(
                            got_state.to_vec1::<u32>()?,
                            ssm.to_vec1::<u32>()?,
                            "recurrent state"
                        );
                        reference_conv = conv;
                        reference_state = ssm;
                        actual_conv = Some(got_conv);
                        actual_state = Some(got_state);
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod preparation_tests {
    use super::tests::{linear, tensor};
    use super::*;
    use lisa_metal::{Dtype, ops};
    #[test]
    fn delta_preparation_plan_matches_packed_and_separate_chains() -> anyhow::Result<()> {
        use lisa_models::gated_delta::PrepPlan;
        for (batch, tokens, capture) in [(1, 1, false), (1, 7, true), (1, 9, false), (2, 9, false)]
        {
            for present in [false, true] {
                let (hk, hv) = (1, 2);
                let channels = (2 * hk + hv) * 128;
                let pw = channels + hv * 128 + hv * 2;
                let projections = [
                    linear(128, pw, 1),
                    linear(128, channels, 2),
                    linear(128, hv * 128, 3),
                    linear(128, hv, 4),
                    linear(128, hv, 5),
                ];
                let weight = tensor(&[channels, 4, 1], 6, Dtype::Bfloat16);
                let decay = Array::from_slice(&vec![-0.5f32; hv as usize], &[hv]);
                let bias = tensor(&[hv], 7, Dtype::Bfloat16);
                let conv = if present {
                    tensor(&[batch, 3, channels], 8, Dtype::Bfloat16)
                } else {
                    ops::zeros::<half::bf16>(&[batch, 3, channels])?
                };
                let x = tensor(&[batch, tokens, 128], 9, Dtype::Bfloat16);
                let separate = PrepPlan::separate(batch, tokens, capture);
                let spec = PrepPlan {
                    kernel: lisa_backend::DeltaPrep {
                        key_heads: hk,
                        value_heads: hv,
                        key_width: 128,
                        value_width: 128,
                        convolution: 4,
                        channels,
                        projection: pw,
                        beta_offset: channels + hv * 128,
                        decay_offset: channels + hv * 128 + hv,
                        separate,
                    },
                    batch,
                    tokens,
                    gate_offset: channels,
                    capture,
                    convolution_present: present,
                    projections: [(64, 4); 5],
                };
                let stream = lisa_metal::Stream::thread_local_or_default();
                let expected = if separate {
                    let qkv = projections[1].forward(&x)?;
                    let z = projections[2].forward(&x)?;
                    let b = projections[3].forward(&x)?;
                    let a = projections[4].forward(&x)?;
                    let (q, k, v, g, beta, c) = lisa_metal::kernels::gdn_prep(
                        &qkv, &conv, &weight, &decay, &bias, &b, &a, tokens, hk, hv, 128, 128, 4,
                        channels, &stream,
                    )
                    .unwrap();
                    vec![q, k, v, g, beta, c, z.reshape(&[batch, tokens, hv, 128])?]
                } else {
                    use lisa_metal::ops::indexing::IndexOp;
                    let proj = projections[0].forward(&x)?;
                    let z = proj.index((.., .., channels..channels + hv * 128));
                    let (q, k, v, g, beta, c, history) = lisa_metal::kernels::gdn_prep_fused(
                        &proj,
                        &conv,
                        &weight,
                        &decay,
                        &bias,
                        spec.kernel.beta_offset,
                        spec.kernel.decay_offset,
                        tokens,
                        hk,
                        hv,
                        128,
                        128,
                        4,
                        pw,
                        channels,
                        &stream,
                    )
                    .unwrap();
                    vec![q, k, v, g, beta, c, z, history]
                };
                let mut weights = vec![];
                for p in projections {
                    weights.extend([p.weight, p.scales, p.biases]);
                }
                weights.extend([weight, decay, bias]);
                let plan =
                    PreparedGraph::new(crate::metal_context::device(), spec.graph()?, weights)?;
                let bindings = if present { vec![&x, &conv] } else { vec![&x] };
                let actual = plan.run(&bindings, &mut [])?;
                assert_eq!(actual.len(), expected.len());
                for (i, (a, e)) in actual.into_iter().zip(expected).enumerate() {
                    assert_eq!(a.shape(), e.shape());
                    if a.dtype() == Dtype::Float32 {
                        assert_eq!(a.to_vec1::<u32>()?, e.to_vec1::<u32>()?, "prep output {i}");
                    } else {
                        assert_eq!(
                            a.contiguous()?.to_vec1::<u16>()?,
                            e.contiguous()?.to_vec1::<u16>()?,
                            "prep output {i}"
                        );
                    }
                }
            }
        }
        Ok(())
    }
    #[test]
    fn delta_recurrence_plan_preserves_reused_snapshots_and_replay() -> anyhow::Result<()> {
        use lisa_metal::ops::indexing::IndexOp;
        use lisa_models::gated_delta::{CaptureKind, RecurrencePlan, ReplayGeometry};
        for (batch, tokens) in [(1, 1), (1, 2), (1, 7), (1, 16), (1, 17), (2, 7)] {
            for (capture, replay, swish, fusion) in [
                (false, false, false, true),
                (true, false, true, true),
                (true, false, true, false),
                (true, true, true, true),
            ] {
                let (hk, hv) = (4, 16);
                let geometry = ReplayGeometry {
                    batch,
                    tokens,
                    key_heads: hk,
                    value_heads: hv,
                    key_width: 128,
                    value_width: 128,
                };
                let q = tensor(&[batch, tokens, hk, 128], 1, Dtype::Bfloat16);
                let k = tensor(&[batch, tokens, hk, 128], 2, Dtype::Bfloat16);
                let v = tensor(&[batch, tokens, hv, 128], 3, Dtype::Bfloat16);
                let decay = Array::from_slice(
                    &vec![0.9f32; (batch * tokens * hv) as usize],
                    &[batch, tokens, hv],
                );
                let beta = tensor(&[batch, tokens, hv], 4, Dtype::Float32);
                let z_backing = tensor(&[batch, tokens, hv * 128 + 32], 5, Dtype::Bfloat16);
                let z = z_backing.index((.., .., 16..16 + hv * 128));
                let norm = tensor(&[128], 6, Dtype::Bfloat16);
                let projection = linear(hv * 128, 128, 7);
                let mut expected_state = tensor(&[batch, hv, 128, 128], 8, Dtype::Float32);
                let mut actual_state = expected_state.clone();
                let weights = vec![
                    norm.clone(),
                    projection.weight.clone(),
                    projection.scales.clone(),
                    projection.biases.clone(),
                ];
                let stream = lisa_metal::Stream::thread_local_or_default();
                let mut workspace: Option<(Array, Array)> = None;
                // A second call exercises base-state aliases into the reused tape.
                // Multi-batch snapshots retain the original single-stream law and
                // are tested for one capture only, not resumed as a batched stream.
                for round in 0..if batch == 1 { 2 } else { 1 } {
                    let mut spec = RecurrencePlan {
                        geometry,
                        capture,
                        replay,
                        cache: true,
                        swish,
                        verify_fusion: fusion,
                        epsilon: 1e-6f32.to_bits(),
                        state_present: true,
                        reuse: false,
                        projection: (64, 4),
                    };
                    spec.reuse = spec.accepts_workspace(
                        workspace.as_ref().map(|(y, _)| y.shape()),
                        workspace.as_ref().map(|(_, s)| s.shape()),
                    );
                    let fused = capture && !replay && swish && fusion && (2..=16).contains(&tokens);
                    let (y, states) = if fused {
                        lisa_metal::kernels::gated_delta::gdn_verify_fused(
                            &q,
                            &k,
                            &v,
                            &decay,
                            &beta,
                            &expected_state,
                            &z,
                            &norm,
                            1e-6,
                            None,
                            &stream,
                        )
                        .unwrap()
                    } else if tokens > 8 && !capture {
                        lisa_metal::kernels::gdn_rows(
                            &q,
                            &k,
                            &v,
                            &decay,
                            &beta,
                            &expected_state,
                            false,
                            &stream,
                        )
                        .unwrap()
                    } else if tokens > 1 {
                        lisa_metal::kernels::gdn_two_row(
                            &q,
                            &k,
                            &v,
                            &decay,
                            &beta,
                            &expected_state,
                            capture,
                            &stream,
                        )
                        .unwrap()
                    } else {
                        lisa_metal::kernels::gated_delta_kernel(
                            &q,
                            &k,
                            &v,
                            &decay,
                            &beta,
                            &expected_state,
                            capture,
                            &stream,
                        )
                        .unwrap()
                    };
                    let normed = if fused {
                        y.clone()
                    } else {
                        let f = if swish {
                            lisa_metal::kernels::gated_rms_silu
                        } else {
                            lisa_metal::kernels::gated_rms
                        };
                        f(&y, &z, &norm, hv, 128, 1e-6, &stream).unwrap()
                    };
                    let expected =
                        projection.forward(&normed.reshape(&[batch, tokens, hv * 128])?)?;
                    let next_state = if capture {
                        states.index((tokens - 1, .., .., ..)).contiguous()?
                    } else {
                        states.clone()
                    };
                    let mut inputs = vec![&q, &k, &v, &decay, &beta, &z, &actual_state];
                    if spec.reuse {
                        let (y, s) = workspace.as_ref().unwrap();
                        inputs.extend([y, s]);
                    }
                    let plan = PreparedGraph::new(
                        crate::metal_context::device(),
                        spec.graph()?,
                        weights.clone(),
                    )?;
                    let mut outputs = plan.run(&inputs, &mut [])?.into_iter();
                    let got = outputs.next().unwrap();
                    let raw_y = outputs.next().unwrap();
                    let raw_state = outputs.next().unwrap();
                    let committed = outputs.next().unwrap();
                    assert_eq!(
                        got.to_vec1::<u16>()?,
                        expected.to_vec1::<u16>()?,
                        "activation B={batch} S={tokens} capture={capture} replay={replay} round={round}"
                    );
                    assert_eq!(
                        committed.to_vec1::<u32>()?,
                        next_state.to_vec1::<u32>()?,
                        "committed state"
                    );
                    if spec.capture_kind() == CaptureKind::Replay {
                        let history = Replay {
                            geometry,
                            tensors: std::array::from_fn(|_| outputs.next().unwrap()),
                        };
                        for prefix in 0..=tokens as usize {
                            let expected = if prefix == 0 {
                                expected_state.clone()
                            } else {
                                states.index((prefix as i32 - 1, .., .., ..)).contiguous()?
                            };
                            assert_eq!(
                                history
                                    .restore(&crate::metal_context::execution_context(), prefix)?
                                    .to_vec1::<u32>()?,
                                expected.to_vec1::<u32>()?,
                                "prefix {prefix}"
                            );
                        }
                    } else {
                        assert_eq!(
                            raw_state.to_vec1::<u32>()?,
                            states.to_vec1::<u32>()?,
                            "snapshot tape"
                        );
                    }
                    if spec.capture_kind() == CaptureKind::Snapshots {
                        workspace = Some((raw_y, raw_state));
                    }
                    expected_state = next_state;
                    actual_state = committed;
                }
            }
        }
        Ok(())
    }
}
