//! Individual delta-rule history kernels; no transaction or model state.
use super::*;
const BF4: TensorConstraint = TensorConstraint {
    min_rank: 4,
    max_rank: Some(4),
    ..BF16
};
const FP4: TensorConstraint = TensorConstraint {
    dtypes: &[ScalarType::F32],
    min_rank: 4,
    max_rank: Some(4),
    alignment: 4,
    ..BF16
};
const FP3: TensorConstraint = TensorConstraint {
    min_rank: 3,
    max_rank: Some(3),
    ..FP4
};
const STATE: TensorConstraint = TensorConstraint {
    min_rank: 3,
    max_rank: Some(4),
    ..FP4
};
const GATE: TensorConstraint = TensorConstraint {
    min_rank: 3,
    max_rank: Some(4),
    layout: LayoutSupport::Strided,
    ..BF16
};
pub(super) fn capabilities() -> Vec<KernelCapability> {
    [
        (KernelKind::GatedRms, &[BF4, GATE, BF16][..]),
        (
            KernelKind::DeltaScan {
                gated: false,
                reuse: false,
            },
            &[BF4, BF4, BF4, FP3, FP3, STATE][..],
        ),
        (
            KernelKind::DeltaScan {
                gated: false,
                reuse: true,
            },
            &[BF4, BF4, BF4, FP3, FP3, STATE, BF4, FP4][..],
        ),
        (
            KernelKind::DeltaScan {
                gated: true,
                reuse: false,
            },
            &[BF4, BF4, BF4, FP3, FP3, STATE, GATE, BF16][..],
        ),
        (
            KernelKind::DeltaScan {
                gated: true,
                reuse: true,
            },
            &[BF4, BF4, BF4, FP3, FP3, STATE, GATE, BF16, BF4, FP4][..],
        ),
        (
            KernelKind::DeltaPrepPacked,
            &[
                BF16,
                BF16,
                BF16,
                TensorConstraint {
                    dtypes: &[ScalarType::F32],
                    alignment: 4,
                    ..BF16
                },
                BF16,
            ][..],
        ),
        (
            KernelKind::DeltaPrepSeparate,
            &[
                BF16,
                BF16,
                BF16,
                TensorConstraint {
                    dtypes: &[ScalarType::F32],
                    alignment: 4,
                    ..BF16
                },
                BF16,
                BF16,
                BF16,
            ][..],
        ),
        (
            KernelKind::DeltaDecode,
            &[
                BF16,
                BF16,
                BF16,
                TensorConstraint {
                    dtypes: &[ScalarType::F32],
                    ..BF16
                },
                BF16,
                FP4,
                BF16,
            ][..],
        ),
    ]
    .into_iter()
    .map(|(kind, inputs)| KernelCapability {
        kind,
        law: match kind {
            KernelKind::GatedRms => NumericalLaw::Bf16GatedRmsV1,
            KernelKind::DeltaPrepPacked | KernelKind::DeltaPrepSeparate => {
                NumericalLaw::DeltaPrepV1
            }
            KernelKind::DeltaDecode => NumericalLaw::DeltaDecodeV1,
            _ => NumericalLaw::DeltaSequentialV1,
        },
        inputs,
        variadic: false,
        shape: ShapeRelation::DeltaRecurrence,
        encoding: None,
        workspace: match kind {
            KernelKind::DeltaScan {
                gated: true,
                reuse: true,
            } => Workspace::CallerOutputs { inputs: &[8, 9] },
            KernelKind::DeltaScan {
                gated: false,
                reuse: true,
            } => Workspace::CallerOutputs { inputs: &[6, 7] },
            _ => Workspace::ProviderManaged,
        },
    })
    .collect()
}
pub(super) fn invoke(
    kernel: &Kernel,
    inputs: &[&Array],
    stream: &Stream,
) -> Result<Option<Outputs<Array>>> {
    let output = match kernel {
        Kernel::GatedRms {
            heads,
            width,
            epsilon,
            swish,
            serial_gate,
        } => {
            let [y, z, w] = inputs else { unreachable!() };
            if y.dim(2) != *heads
                || y.dim(3) != *width
                || w.shape() != [*width]
                || z.size() != y.size()
                || z.dim(0) != y.dim(0)
                || z.dim(1) != y.dim(1)
            {
                return Err(Error::msg("invalid gated RMS bindings"));
            }
            let gated = if *swish {
                crate::kernels::gated_rms_silu_with_serial_gate(
                    y,
                    z,
                    w,
                    *heads,
                    *width,
                    f32::from_bits(*epsilon),
                    *serial_gate,
                    stream,
                )
            } else {
                crate::kernels::gated_rms(y, z, w, *heads, *width, f32::from_bits(*epsilon), stream)
            };
            Outputs::One(gated.ok_or_else(|| Error::msg("gated RMS dispatch failed"))?)
        }
        Kernel::DeltaScan {
            mode,
            capture,
            reuse,
        } => {
            let (q, k, v, g, beta, state) = (
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
            );
            let (b, t, hk, dk) = (k.dim(0), k.dim(1), k.dim(2), k.dim(3));
            let (hv, dv) = (v.dim(2), v.dim(3));
            if b <= 0
                || t <= 0
                || hk <= 0
                || hv <= 0
                || hv % hk != 0
                || dk != 128
                || dv != 128
                || q.shape() != k.shape()
                || v.shape() != [b, t, hv, dv]
                || g.shape() != [b, t, hv]
                || beta.shape() != g.shape()
                || !(state.shape() == [b, hv, dv, dk] || b == 1 && state.shape() == [hv, dv, dk])
            {
                return Err(Error::msg("invalid delta scan bindings"));
            }
            let prealloc = if *reuse {
                let i = 6 + 2 * usize::from(mode.gated());
                let (y, s) = (inputs[i], inputs[i + 1]);
                if y.shape() != [b, t, hv, dv]
                    || s.shape()
                        != if *capture {
                            vec![b * t, hv, dv, dk]
                        } else {
                            state.shape().to_vec()
                        }
                {
                    return Err(Error::msg("invalid delta scan output storage"));
                }
                Some((y, s))
            } else {
                None
            };
            use crate::kernels::gated_delta as native;
            let result = match mode {
                DeltaScan::Scalar => {
                    native::gated_delta_kernel(q, k, v, g, beta, state, *capture, stream)
                }
                DeltaScan::TwoRows => match prealloc {
                    Some((y, s)) => {
                        native::gdn_two_row_into(q, k, v, g, beta, state, *capture, y, s, stream)
                    }
                    None => native::gdn_two_row(q, k, v, g, beta, state, *capture, stream),
                },
                DeltaScan::FourRows => native::gdn_rows(q, k, v, g, beta, state, *capture, stream),
                DeltaScan::Verify {
                    epsilon,
                    serial_gate,
                } => {
                    let z = inputs[6];
                    let norm = inputs[7];
                    if !*capture
                        || !(2..=16).contains(&t)
                        || norm.shape() != [dv]
                        || z.size() != v.size()
                        || z.dim(0) != b
                        || z.dim(1) != t
                    {
                        return Err(Error::msg("invalid fused delta scan bindings"));
                    }
                    native::gdn_verify_fused_with_serial_gate(
                        q,
                        k,
                        v,
                        g,
                        beta,
                        state,
                        z,
                        norm,
                        f32::from_bits(*epsilon),
                        prealloc,
                        *serial_gate,
                        stream,
                    )
                }
            }
            .ok_or_else(|| Error::msg("delta scan dispatch failed"))?;
            Outputs::Two([result.0, result.1])
        }
        Kernel::DeltaPrep(s) => {
            let proj = inputs[0];
            let conv = inputs[1];
            let weight = inputs[2];
            let decay = inputs[3];
            let bias = inputs[4];
            if !s.validate() || proj.rank() != 3 || proj.dim(0) <= 0 || proj.dim(1) <= 0 {
                return Err(Error::msg("invalid delta preparation geometry"));
            }
            let (b, t) = (proj.dim(0), proj.dim(1));
            if proj.dim(2) != if s.separate { s.channels } else { s.projection }
                || conv.shape() != [b, s.convolution - 1, s.channels]
                || weight.shape() != [s.channels, s.convolution, 1]
                || decay.shape() != [s.value_heads]
                || bias.shape() != [s.value_heads]
            {
                return Err(Error::msg("invalid delta preparation bindings"));
            }
            if s.separate {
                if inputs[5].shape() != [b, t, s.value_heads]
                    || inputs[6].shape() != inputs[5].shape()
                {
                    return Err(Error::msg("invalid delta gate bindings"));
                }
                let (q, k, v, g, beta, conv) = crate::kernels::gated_delta::gdn_prep(
                    proj,
                    conv,
                    weight,
                    decay,
                    bias,
                    inputs[5],
                    inputs[6],
                    t,
                    s.key_heads,
                    s.value_heads,
                    s.key_width,
                    s.value_width,
                    s.convolution,
                    s.channels,
                    stream,
                )
                .ok_or_else(|| Error::msg("delta preparation failed"))?;
                Outputs::Many(vec![q, k, v, g, beta, conv])
            } else {
                let (q, k, v, g, beta, conv, history) =
                    crate::kernels::gated_delta::gdn_prep_fused(
                        proj,
                        conv,
                        weight,
                        decay,
                        bias,
                        s.beta_offset,
                        s.decay_offset,
                        t,
                        s.key_heads,
                        s.value_heads,
                        s.key_width,
                        s.value_width,
                        s.convolution,
                        s.projection,
                        s.channels,
                        stream,
                    )
                    .ok_or_else(|| Error::msg("packed delta preparation failed"))?;
                Outputs::Many(vec![q, k, v, g, beta, conv, history])
            }
        }
        Kernel::DeltaDecode(spec) => {
            let [proj, conv, weight, decay, bias, state, norm] = inputs else {
                unreachable!()
            };
            let s = spec;
            if !s.validate()
                || proj.rank() != 3
                || proj.dim(0) <= 0
                || proj.shape() != [proj.dim(0), 1, s.projection]
                || conv.shape() != [proj.dim(0), s.convolution - 1, s.channels]
                || weight.shape() != [s.channels, s.convolution]
                || decay.shape() != [s.value_heads]
                || bias.shape() != [s.value_heads]
                || state.shape() != [proj.dim(0), s.value_heads, s.value_width, s.key_width]
                || norm.shape() != [s.value_width]
            {
                return Err(Error::msg("invalid delta decode bindings"));
            }
            let (y, state, conv) = crate::kernels::gated_delta::delta_decode_checked(
                proj,
                conv,
                weight,
                decay,
                bias,
                state,
                norm,
                s.key_heads,
                s.value_heads,
                s.key_width,
                s.value_width,
                s.convolution,
                s.channels,
                s.projection,
                s.beta_offset,
                s.decay_offset,
                s.gate_offset,
                f32::from_bits(s.epsilon),
                s.swish,
                stream,
            )?;
            Outputs::Many(vec![y, state, conv])
        }
        _ => unreachable!(),
    };
    Ok(Some(output))
}
