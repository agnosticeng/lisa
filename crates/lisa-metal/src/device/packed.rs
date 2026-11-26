//! Prepared original packed arithmetic. No checkpoint parsing or model traversal.
use super::*;
use crate::kernels::ggml as native;
const F32: TensorConstraint = TensorConstraint {
    dtypes: &[ScalarType::F32],
    min_rank: 1,
    max_rank: None,
    layout: LayoutSupport::RowContiguous,
    alignment: 4,
};
const DENSE: TensorConstraint = TensorConstraint {
    dtypes: &[ScalarType::F32, ScalarType::BF16, ScalarType::F16],
    min_rank: 2,
    max_rank: Some(2),
    layout: LayoutSupport::RowContiguous,
    alignment: 2,
};
const BYTES: TensorConstraint = TensorConstraint {
    dtypes: &[ScalarType::U8],
    min_rank: 2,
    max_rank: Some(3),
    layout: LayoutSupport::RowContiguous,
    alignment: 2,
};
/// Any storage, bound at its view's offset (vendored kernels' raw buffers).
const ANY_BUFFER: TensorConstraint = TensorConstraint {
    dtypes: &[
        ScalarType::U8,
        ScalarType::U16,
        ScalarType::U32,
        ScalarType::I8,
        ScalarType::I32,
        ScalarType::F16,
        ScalarType::BF16,
        ScalarType::F32,
    ],
    min_rank: 1,
    max_rank: None,
    // Kernels address views through their own strides from the view's base.
    layout: LayoutSupport::Strided,
    alignment: 1,
};
const VECTOR_I32: TensorConstraint = TensorConstraint {
    dtypes: &[ScalarType::I32],
    min_rank: 1,
    max_rank: None,
    layout: LayoutSupport::RowContiguous,
    alignment: 4,
};
const INDICES: TensorConstraint = TensorConstraint {
    dtypes: &[ScalarType::I32],
    min_rank: 2,
    max_rank: Some(2),
    layout: LayoutSupport::RowContiguous,
    alignment: 4,
};

/// Every packed capability, built once.
pub(super) fn capabilities() -> Vec<KernelCapability> {
    static CAPABILITIES: std::sync::OnceLock<Vec<KernelCapability>> = std::sync::OnceLock::new();
    CAPABILITIES.get_or_init(base_capabilities).clone()
}

fn base_capabilities() -> Vec<KernelCapability> {
    use PackedComputeKind as K;
    let cap = |kind, inputs| KernelCapability {
        kind: KernelKind::PackedCompute(kind),
        law: NumericalLaw::PackedScalarV1,
        inputs,
        variadic: false,
        shape: ShapeRelation::TensorOperation,
        encoding: None,
        workspace: Workspace::ProviderManaged,
    };
    let mut caps = vec![
        KernelCapability {
            law: NumericalLaw::PackedFastV1,
            ..cap(K::RepackPlanes, &[BYTES])
        },
        KernelCapability {
            law: NumericalLaw::PackedFastV1,
            ..cap(K::DraftTree, &[VECTOR_I32, F32, F32, VECTOR_I32, F32])
        },
        KernelCapability {
            law: NumericalLaw::PackedFastV1,
            inputs: &[ANY_BUFFER],
            variadic: true,
            ..cap(K::Vendored, &[])
        },
        KernelCapability {
            law: NumericalLaw::PackedFastV1,
            ..cap(K::KvPageTable, &[ANY_BUFFER])
        },
        KernelCapability {
            law: NumericalLaw::PackedFastV1,
            inputs: &[ANY_BUFFER],
            variadic: true,
            ..cap(K::PlaneAddresses, &[])
        },
        KernelCapability {
            inputs: &[TensorConstraint {
                layout: LayoutSupport::Strided,
                max_rank: Some(3),
                ..F32
            }],
            variadic: true,
            ..cap(K::ConcatF32, &[])
        },
        KernelCapability {
            workspace: Workspace::CallerOutputs { inputs: &[0] },
            ..cap(K::GatedResidual4(false), &[F32, F32, F32, F32])
        },
        KernelCapability {
            workspace: Workspace::CallerOutputs { inputs: &[0] },
            ..cap(K::GatedResidual4(true), &[F32, F32, F32, F32, F32])
        },
        cap(K::RmsNorm(false), &[F32]),
        cap(K::RmsNorm(true), &[F32, F32]),
        cap(K::RmsNormPair, &[F32, F32, F32, F32]),
        cap(K::ResidualInput4, &[F32, F32, F32, F32, F32, F32]),
        cap(K::ResidualExpand4(false), &[F32, F32, F32]),
        cap(K::ResidualExpand4(true), &[F32, F32, F32, F32]),
        cap(K::ResidualCollapse4, &[F32, F32]),
        cap(K::PlainLinear, &[DENSE, F32]),
        cap(K::Q8Linear, &[BYTES, F32]),
        cap(K::Q8ExpertLinear, &[BYTES, F32, INDICES]),
        cap(K::Q8Pair, &[BYTES, BYTES, F32]),
        cap(K::Router, &[F32, F32]),
        cap(K::SharedSwiGlu, &[F32, F32]),
        cap(K::WeightedSwiGlu, &[F32, F32, F32]),
        cap(K::RowGateBlendPair, &[F32, F32, F32, F32, F32, F32]),
        cap(K::DifferentialSiluGate, &[F32, F32, F32]),
        cap(K::SmallRowMatmul, &[F32, F32]),
        cap(K::ReluSquared, &[F32]),
        cap(K::WeightedSumSix, &[F32, F32]),
        cap(K::SigmoidRouter256, &[F32, F32]),
        cap(K::SumSix, &[F32]),
    ];
    use PackedEncoding::*;
    for e in [Iq2Xxs, Iq2Xs, Iq1M, Q5K, Q6K, Q2K, Q4K, Nvfp4] {
        caps.extend([
            cap(K::BlockLinear(e), &[BYTES, F32]),
            cap(K::GroupedLinear(e), &[BYTES, F32]),
            cap(K::ExpertLinear(e), &[BYTES, F32, INDICES]),
        ]);
    }
    caps
}
fn encoding(e: PackedEncoding) -> native::BlockFormat {
    use native::BlockFormat as B;
    match e {
        PackedEncoding::Iq2Xxs => B::Iq2Xxs,
        PackedEncoding::Iq2Xs => B::Iq2Xs,
        PackedEncoding::Iq1M => B::Iq1M,
        PackedEncoding::Q5K => B::Q5K,
        PackedEncoding::Q6K => B::Q6K,
        PackedEncoding::Q2K => B::Q2K,
        PackedEncoding::Q4K => B::Q4K,
        PackedEncoding::Nvfp4 => B::Nvfp4,
    }
}
fn round(mut x: Array, enabled: bool, stream: &Stream) -> Result<Array> {
    if enabled {
        let shape = x.shape().to_vec();
        if x.rank() != 2 {
            x = x.reshape(&[-1, x.dim(-1)])?;
        }
        native::quantize_in_place(&mut x, native::ActivationFormat::Bf16, stream)?;
        if x.shape() != shape {
            x = x.reshape(&shape)?;
        }
    }
    Ok(x)
}
pub(super) fn invoke(
    kernel: &Kernel,
    a: &[&Array],
    stream: &Stream,
) -> Result<Option<Outputs<Array>>> {
    let Kernel::PackedCompute(op) = kernel else {
        return Ok(None);
    };
    use PackedCompute as K;
    let out = match op {
        K::RepackPlanes {
            ggml_type,
            first_row,
            rows,
            columns,
        } => {
            let w = a[0];
            if w.rank() != 3 || !w.t.buffer_and_layout().1.is_contiguous() {
                return Err(Error::msg(
                    "expert planes need a contiguous [E, rows, bytes] tensor",
                ));
            }
            // Read in place: a host copy of every expert tensor would cost
            // seconds and stay cached beside the mapped weights.
            let source = native::GgufExpertTensor {
                bytes: w.t.contiguous_bytes()?,
                ggml_type: *ggml_type,
                experts: w.dim(0) as usize,
                rows: w.dim(1) as usize,
                columns: *columns as usize,
            };
            let planes =
                native::repack_planes(&source, *first_row as usize, *rows as usize, stream)?;
            Outputs::One(planes.storage().clone())
        }
        K::Vendored(d) => Outputs::One(native::vendored_dispatch(d, a, stream)?),
        K::PlaneAddresses { offsets } => Outputs::One(native::plane_addresses(a, offsets)?),
        K::KvPageTable { extent_pages } => {
            Outputs::One(native::kv_page_table(a[0], *extent_pages as usize)?)
        }
        K::DraftTree {
            anchor,
            depth,
            chain,
            children,
            rows,
            lane_cap,
            model,
            selected,
        } => {
            let f = |i: usize| f32::from_bits(model[i]);
            let mut intercepts = [0f32; native::TREE_POSITIONS];
            for (p, v) in intercepts.iter_mut().enumerate() {
                *v = f(2 + p);
            }
            let params = native::TreeParams {
                anchor: *anchor,
                depth: *depth,
                chain: *chain,
                children: *children,
                softcap: f(0),
                slope: f(1),
                intercepts,
                rows: *rows,
                lane_cap: *lane_cap,
                floor: f(9),
                selected: *selected,
            };
            let t = native::draft_tree(a[0], a[1], a[2], a[3], a[4], &params, stream)?;
            Outputs::Many(vec![
                t.ids,
                t.tree,
                t.sources,
                t.tokens,
                t.rows,
                t.scores,
                t.caps,
                t.size,
                t.candidates,
                t.verify_tree,
                t.lane_rows,
            ])
        }
        K::ConcatF32 { axis, .. } => Outputs::One(native::concat_f32(a, *axis as usize, stream)?),
        K::GatedResidual4 { epsilon, masked } => {
            let mut x = a[0].clone();
            native::engram_in_place(
                &mut x,
                a[1],
                a[2],
                a[3],
                if *masked { Some(a[4]) } else { None },
                f32::from_bits(*epsilon),
                stream,
            )?;
            Outputs::One(x)
        }
        K::RmsNorm {
            weighted,
            round_output,
            epsilon,
        } => Outputs::One(native::rms_norm(
            a[0],
            if *weighted { Some(a[1]) } else { None },
            f32::from_bits(*epsilon),
            *round_output,
            stream,
        )?),
        K::RmsNormPair { epsilon } => {
            let (x, y) =
                native::rms_norm_pair(a[0], a[1], a[2], a[3], f32::from_bits(*epsilon), stream)?;
            Outputs::Two([x, y])
        }
        K::ResidualInput4 {
            epsilon,
            norm_epsilon,
        } => {
            let x = native::hc_input(
                a[0],
                a[1],
                a[2],
                a[3],
                a[4],
                a[5],
                f32::from_bits(*epsilon),
                f32::from_bits(*norm_epsilon),
                stream,
            )?;
            Outputs::Many(vec![x.split, x.collapsed, x.norm])
        }
        K::ResidualExpand4 { add, round_block } => Outputs::One(if *add {
            native::hc_expand(a[0], Some(a[1]), a[2], a[3], *round_block, stream)?
        } else {
            native::hc_expand(a[0], None, a[1], a[2], *round_block, stream)?
        }),
        K::ResidualCollapse4 => Outputs::One(native::hc_collapse(a[0], a[1], stream)?),
        K::PlainLinear { round_output } => Outputs::One(round(
            native::plain_matvec(a[0], a[1], stream)?,
            *round_output,
            stream,
        )?),
        K::Q8Linear {
            round_output,
            round_input,
        } => Outputs::One(native::q8_matvec(
            a[0],
            a[1],
            *round_output,
            *round_input,
            stream,
        )?),
        K::BlockLinear {
            encoding: e,
            round_output,
        } => Outputs::One(round(
            native::packed_matvec(encoding(*e), a[0], a[1], stream)?,
            *round_output,
            stream,
        )?),
        K::GroupedLinear {
            encoding: e,
            round_output,
        } => Outputs::One(round(
            native::grouped_packed_matvec(encoding(*e), a[0], a[1], stream)?,
            *round_output,
            stream,
        )?),
        K::ExpertLinear { encoding: e } => Outputs::One(native::expert_matvec(
            encoding(*e),
            a[0],
            a[1],
            a[2],
            stream,
        )?),
        K::Q8ExpertLinear => Outputs::One(native::q8_expert_matvec(a[0], a[1], a[2], stream)?),
        K::Q8Pair => {
            let (x, y) = native::q8_matvec_pair(a[0], a[1], a[2], stream)?;
            Outputs::Two([x, y])
        }
        K::Router { top_k, scale } => {
            let r = native::router(a[0], a[1], *top_k, f32::from_bits(*scale), stream)?;
            Outputs::Many(vec![r.probabilities, r.selected, r.weights])
        }
        K::SharedSwiGlu { round_output } => Outputs::One(round(
            native::shared_swiglu(a[0], a[1], stream)?,
            *round_output,
            stream,
        )?),
        K::WeightedSwiGlu { clamp } => Outputs::One(native::swiglu_weight(
            a[0],
            a[1],
            a[2],
            f32::from_bits(*clamp),
            stream,
        )?),
        K::RowGateBlendPair { zero } => {
            let (k, v) = native::row_gate_blend_pair(a, *zero, stream)?;
            Outputs::Two([k, v])
        }
        K::DifferentialSiluGate { bias } => Outputs::One(native::differential_silu_gate(
            a[0],
            a[1],
            a[2],
            f32::from_bits(*bias),
            stream,
        )?),
        K::SmallRowMatmul { transpose_rhs } => Outputs::One(native::small_row_matmul(
            a[0],
            a[1],
            *transpose_rhs,
            stream,
        )?),
        K::ReluSquared => Outputs::One(native::relu_squared(a[0], stream)?),
        K::WeightedSumSix => Outputs::One(native::weighted_sum6(a[0], a[1], stream)?),
        K::SigmoidRouter256 { scale } => {
            let (ids, weights) =
                native::sigmoid_router256(a[0], a[1], f32::from_bits(*scale), stream)?;
            Outputs::Two([ids, weights])
        }
        K::SumSix => Outputs::One(native::sum_six(a[0], stream)?),
    };
    Ok(Some(out))
}
