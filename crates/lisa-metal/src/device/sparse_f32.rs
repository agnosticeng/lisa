//! Prepared sparse arithmetic primitives; model graphs choose their order.
use super::*;
use crate::kernels::sparse::kernels as native;
const F32: TensorConstraint = TensorConstraint {
    dtypes: &[ScalarType::F32],
    min_rank: 1,
    max_rank: None,
    layout: LayoutSupport::RowContiguous,
    alignment: 4,
};
pub(super) fn capabilities(triple_linear: bool) -> Vec<KernelCapability> {
    use SparseF32Kind as K;
    let cap = |kind, inputs| KernelCapability {
        kind: KernelKind::SparseF32(kind),
        law: NumericalLaw::SparseF32V1,
        inputs,
        variadic: false,
        shape: ShapeRelation::TensorOperation,
        encoding: None,
        workspace: if kind == K::QueryLatentRmsInto {
            Workspace::CallerOutputs { inputs: &[4] }
        } else {
            Workspace::ProviderManaged
        },
    };
    const F16: TensorConstraint = TensorConstraint {
        dtypes: &[ScalarType::F16],
        alignment: 2,
        ..F32
    };
    const I32: TensorConstraint = TensorConstraint {
        dtypes: &[ScalarType::I32],
        ..F32
    };
    const BYTES: TensorConstraint = TensorConstraint {
        dtypes: &[ScalarType::U8],
        alignment: 1,
        ..F32
    };
    const PACKED: TensorConstraint = TensorConstraint {
        dtypes: &[ScalarType::U32],
        min_rank: 2,
        max_rank: Some(2),
        ..F32
    };
    const STRIDED_F32: TensorConstraint = TensorConstraint {
        layout: LayoutSupport::Strided,
        ..F32
    };
    let mut caps = vec![
        cap(
            K::QueryLatentRmsRows,
            &[STRIDED_F32, FLOAT, STRIDED_F32, FLOAT],
        ),
        cap(
            K::QueryLatentRmsCompact,
            &[STRIDED_F32, FLOAT, STRIDED_F32, FLOAT],
        ),
        cap(
            K::QueryLatentRmsInto,
            &[STRIDED_F32, FLOAT, STRIDED_F32, FLOAT, F16],
        ),
        cap(K::AttentionOutput, &[F32, PACKED, F32, F32]),
        cap(K::AttentionScores, &[F32, F32, F32]),
        cap(K::AttentionWeightedValue, &[F32, F32]),
        cap(K::AffineHeadQuery, &[F32, PACKED, F32, F32]),
        cap(
            K::AffineHeadValue,
            &[
                TensorConstraint {
                    layout: LayoutSupport::Strided,
                    ..F32
                },
                PACKED,
                F32,
                F32,
            ],
        ),
        cap(K::AffineIndexedDecode, &[F32, F16, I32, PACKED, F32, F32]),
        cap(K::IndexedLatentShort, &[F32, F16, I32]),
        cap(K::FlashLatent, &[F32, F16]),
        cap(
            K::AffineGateUp,
            &[F32, PACKED, FLOAT, FLOAT, PACKED, FLOAT, FLOAT],
        ),
        cap(K::OrderedRms, &[F32, FLOAT]),
        cap(K::OrderedAddRms, &[F32, F32, FLOAT]),
        cap(K::OrderedAddThree, &[F32, F32, F32]),
        cap(K::Q8HeadQuery, &[F32, BYTES]),
        cap(K::Q8HeadValue, &[F32, BYTES]),
        cap(K::Q8IndexedDecode, &[F32, F16, I32, BYTES]),
        cap(K::IndexedLatent, &[F32, F32, F16, I32]),
        cap(K::Pool128, &[F32, F32, BF16, FLOAT, FLOAT]),
        cap(
            K::IndexScores128,
            &[
                TensorConstraint {
                    layout: LayoutSupport::Strided,
                    ..F32
                },
                TensorConstraint {
                    layout: LayoutSupport::Strided,
                    ..F32
                },
                TensorConstraint {
                    dtypes: &[ScalarType::F16],
                    alignment: 2,
                    layout: LayoutSupport::Strided,
                    ..F32
                },
            ],
        ),
        cap(K::TopPools, &[F32]),
        cap(
            K::ExpandPools,
            &[TensorConstraint {
                dtypes: &[ScalarType::I32],
                ..F32
            }],
        ),
        cap(K::WeightedRms(false), &[F32, FLOAT]),
        cap(K::WeightedRms(true), &[F32, FLOAT]),
        cap(K::DeltaDecode, &[F32; 14]),
        cap(K::DeltaPrepare, &[F32; 10]),
        cap(K::DeltaRecurrence, &[F32; 6]),
        cap(K::DeltaOutput, &[F32; 3]),
        cap(K::Bf16PairLinear, &[BF16, F32, BF16, F32]),
        cap(K::Bf16PackedSix, &[BF16, BF16, BF16, BF16, BF16, BF16, F32]),
        cap(K::MixRms, &[F32]),
        cap(K::MixRmsBf16Linear, &[BF16, F32]),
        cap(K::SplitSinkhorn4, &[F32, F32, F32]),
        cap(K::SplitNorm4, &[F32, F32, F32, FLOAT, FLOAT]),
        cap(K::WeightedSum4, &[F32, F32]),
        cap(K::Expand4, &[F32, F32, F32, F32]),
        cap(K::Bf16Linear(false), &[BF16, F32]),
        cap(K::Bf16Linear(true), &[BF16, F32]),
        cap(K::RouterLinear, &[F32, F32]),
        cap(K::RouteSigmoid, &[F32, FLOAT]),
        cap(K::ClampedSwiGlu, &[F32, F32]),
        cap(K::WeightedClampedSwiGlu, &[F32, F32, F32]),
        cap(K::SumEight, &[F32]),
    ];
    for schedule in [
        lisa_backend::AffineSchedule::Matrix,
        lisa_backend::AffineSchedule::Tiny,
        lisa_backend::AffineSchedule::Vector,
        lisa_backend::AffineSchedule::Generic,
        lisa_backend::AffineSchedule::IndependentRows,
    ] {
        caps.push(cap(
            K::AffineProjection(schedule),
            &[
                TensorConstraint {
                    layout: LayoutSupport::Strided,
                    ..FLOAT
                },
                TensorConstraint {
                    dtypes: &[ScalarType::U32],
                    layout: LayoutSupport::Strided,
                    ..F32
                },
                TensorConstraint {
                    layout: LayoutSupport::Strided,
                    ..FLOAT
                },
                TensorConstraint {
                    layout: LayoutSupport::Strided,
                    ..FLOAT
                },
            ],
        ));
    }
    if triple_linear {
        caps.push(cap(K::Bf16TripleLinear, &[BF16, BF16, BF16, F32]));
    }
    caps
}
pub(super) fn invoke(
    kernel: &Kernel,
    a: &[&Array],
    stream: &Stream,
) -> Result<Option<Outputs<Array>>> {
    let Kernel::SparseF32(op) = kernel else {
        return Ok(None);
    };
    if a.iter().any(|t| t.shape().iter().any(|&d| d <= 0)) {
        return Ok(None);
    }
    if matches!(op, SparseF32::Pool128 { .. }) && (a[3].size() != 128 || a[4].size() != 128) {
        return Ok(None);
    }
    let sizes = |expected: &[usize]| {
        a.len() == expected.len() && a.iter().zip(expected).all(|(a, n)| a.size() == *n)
    };
    match op {
        SparseF32::DeltaDecode { heads, .. } => {
            let h = *heads;
            let n = h * 128;
            if !sizes(&[
                n,
                n,
                n,
                n,
                h,
                n,
                n * 4,
                n * 4,
                n * 4,
                h,
                n,
                128,
                n * 9,
                n * 128,
            ]) {
                return Ok(None);
            }
        }
        SparseF32::DeltaPrepare { heads, .. } => {
            let h = *heads;
            let n = h * 128;
            if a[0].rank() != 3 || a[0].dim(0) != 1 || a[0].dim(2) != n as i32 {
                return Ok(None);
            }
            let t = a[0].dim(1) as usize;
            if !sizes(&[t * n, t * n, t * n, t * n, n * 4, n * 4, n * 4, h, n, n * 9]) {
                return Ok(None);
            }
        }
        SparseF32::DeltaRecurrence { .. } => {
            if a[0].rank() != 4
                || a[0].dim(3) != 128
                || a[..4].iter().any(|x| x.shape() != a[0].shape())
            {
                return Ok(None);
            }
            let s = a[0].shape();
            if a[4].shape() != &s[..3] || a[5].shape() != [s[0], s[2], 128, 128] {
                return Ok(None);
            }
        }
        SparseF32::DeltaOutput { heads, .. }
            if (a[0].rank() != 2
                || a[0].dim(1) != (*heads * 128) as i32
                || a[1].shape() != a[0].shape()
                || a[2].size() != 128) =>
        {
            return Ok(None);
        }
        _ => {}
    }
    match op {
        SparseF32::SplitSinkhorn4 { .. } | SparseF32::SplitNorm4 { .. }
            if a[0].rank() != 2 || a[0].dim(1) != 24 || a[1].size() != 3 || a[2].size() != 24 =>
        {
            return Ok(None);
        }
        SparseF32::WeightedSum4
            if a[0].rank() != 4 || a[0].dim(2) != 4 || a[1].shape() != &a[0].shape()[..3] =>
        {
            return Ok(None);
        }
        SparseF32::Expand4
            if a[0].rank() != 3
                || a[1].rank() != 4
                || a[1].dim(2) != 4
                || a[0].shape() != [a[1].dim(0), a[1].dim(1), a[1].dim(3)]
                || a[2].shape() != &a[1].shape()[..3]
                || a[3].shape() != [a[1].dim(0), a[1].dim(1), 4, 4] =>
        {
            return Ok(None);
        }
        SparseF32::RouteSigmoid { .. }
            if a[0].rank() != 2 || a[1].size() != a[0].dim(1) as usize =>
        {
            return Ok(None);
        }
        SparseF32::ClampedSwiGlu { .. }
            if a[0].shape() != a[1].shape() || a[0].size() > i32::MAX as usize =>
        {
            return Ok(None);
        }
        _ => {}
    }
    let out = match op {
        SparseF32::AffineGateUp { limit, capture } => {
            for tensor in a {
                tensor.eval_enqueue()?;
            }
            if *capture {
                let (g, u, y) = crate::jit::affine_q8_gate_up_native(
                    a[0].t.device(),
                    &a[0].t,
                    &a[1].t,
                    &a[2].t,
                    &a[3].t,
                    &a[4].t,
                    &a[5].t,
                    &a[6].t,
                    f32::from_bits(*limit),
                )?;
                return Ok(Some(Outputs::Many(vec![
                    Array::new(g),
                    Array::new(u),
                    Array::new(y),
                ])));
            }
            Some(Array::new(crate::jit::affine_q8_swiglu_native(
                a[0].t.device(),
                &a[0].t,
                &a[1].t,
                &a[2].t,
                &a[3].t,
                &a[4].t,
                &a[5].t,
                &a[6].t,
                f32::from_bits(*limit),
            )?))
        }

        SparseF32::AffineProjection {
            group_size,
            bits,
            schedule,
        } => {
            use lisa_backend::AffineSchedule as P;
            let (x, w, scales, biases) = (a[0], a[1], a[2], a[3]);
            if x.rank() == 0
                || w.rank() != 2
                || scales.rank() != 2
                || biases.shape() != scales.shape()
                || scales.dim(0) != w.dim(0)
                || x.dim(-1) % *group_size != 0
                || scales.dim(1) != x.dim(-1) / *group_size
                || i64::from(w.dim(1)) * 32 != i64::from(x.dim(-1)) * i64::from(*bits)
            {
                return Ok(None);
            }
            match schedule {
                P::Matrix => Some(Array::new(crate::jit::affine_q8_prefill(
                    x.t.device(),
                    &x.t,
                    &w.t,
                    &scales.t,
                    &biases.t,
                )?)),
                P::Tiny => Some(Array::new(crate::jit::affine_q8_tiny_native(
                    x.t.device(),
                    &x.t,
                    &w.t,
                    &scales.t,
                    &biases.t,
                )?)),
                P::Vector => native::affine_q8_mv_ds4(w, scales, biases, x, stream),
                P::Generic => Some(crate::ops::quantized_matmul(
                    x,
                    w,
                    scales,
                    Some(biases),
                    true,
                    *group_size,
                    *bits,
                )?),
                P::IndependentRows => Some(crate::ops::quantized_matmul_serial_rows(
                    x,
                    w,
                    scales,
                    Some(biases),
                    true,
                    *group_size,
                    *bits,
                )?),
            }
        }

        SparseF32::QueryLatentRmsRows { epsilon } => {
            return Ok(native::dsa_qkv_rms_norm_rows(
                a[0],
                a[1],
                a[2],
                a[3],
                f32::from_bits(*epsilon),
                stream,
            )
            .map(|(q, c)| Outputs::Two([q, c])));
        }
        SparseF32::QueryLatentRmsCompact { epsilon } => {
            return Ok(native::qkv_norm_compact(
                a[0],
                a[1],
                a[2],
                a[3],
                f32::from_bits(*epsilon),
                stream,
            )
            .map(|(q, c)| Outputs::Two([q, c])));
        }
        SparseF32::QueryLatentRmsInto { epsilon, position } => native::qkv_norm_compact_into(
            a[0],
            a[1],
            a[2],
            a[3],
            a[4],
            *position,
            f32::from_bits(*epsilon),
            stream,
        ),
        SparseF32::AttentionOutput { group } => {
            native::dsa_o_project(a[0], a[1], a[2], a[3], *group, stream)
        }
        SparseF32::AttentionScores { scale } => {
            native::dsa_decode_scores(a[0], a[1], a[2], f32::from_bits(*scale), stream)
        }
        SparseF32::AttentionWeightedValue => native::dsa_decode_weighted_v(a[0], a[1], stream),
        SparseF32::AffineHeadQuery { group, native: n } => {
            native::dsa_qk_lowrank_project(a[0], a[1], a[2], a[3], *group, stream, *n)
        }
        SparseF32::AffineHeadValue {
            group,
            heads,
            native: n,
            matrix,
        } => native::dsa_value_project_rows(
            a[0], a[1], a[2], a[3], *group, *heads, stream, *n, *matrix,
        ),
        SparseF32::AffineIndexedDecode {
            group,
            selected,
            dense,
            scale,
            native: n,
            capture,
        } => {
            return Ok(native::dsa_indexed_decode_fused(
                a[0],
                a[1],
                a[2],
                *selected,
                *dense,
                a[3],
                a[4],
                a[5],
                *group,
                f32::from_bits(*scale),
                stream,
                *capture,
                *n,
            )
            .map(|(out, latent)| {
                if *capture {
                    Outputs::Two([out, latent.expect("capture output")])
                } else {
                    Outputs::One(out)
                }
            }));
        }
        SparseF32::IndexedLatentShort {
            selected,
            prefix,
            dense,
            scale,
        } => native::dsa_indexed_lora_short(
            a[0],
            a[1],
            a[2],
            *selected,
            *prefix,
            *dense,
            f32::from_bits(*scale),
            stream,
        ),
        SparseF32::FlashLatent { position, scale } => Some(Array::new(
            crate::jit::ds4_flash_attention_prefill_raw_heads(
                &a[0].t,
                &a[1].t,
                *position,
                f32::from_bits(*scale),
            )?,
        )),
        SparseF32::OrderedRms { epsilon } => {
            native::ds4_rms_norm(a[0], a[1], f32::from_bits(*epsilon), stream)
        }
        SparseF32::OrderedAddRms { epsilon } => {
            return Ok(native::mtp_add_rms_norm(
                a[0],
                a[1],
                a[2],
                f32::from_bits(*epsilon),
                stream,
            )
            .map(|(a, b)| Outputs::Two([a, b])));
        }
        SparseF32::OrderedAddThree => native::mtp_add3(a[0], a[1], a[2], stream),
        SparseF32::Q8HeadQuery => Some(crate::kernels::sparse::gguf::qk(a[0], a[1], stream)?),
        SparseF32::Q8HeadValue => Some(crate::kernels::sparse::gguf::value(a[0], a[1], stream)?),
        SparseF32::Q8IndexedDecode { dense } => Some(crate::kernels::sparse::gguf::decode(
            a[0], a[1], a[2], a[3], *dense, stream,
        )?),
        SparseF32::IndexedLatent { scale } => native::dsa_indexed_lora_wide_native(
            a[0],
            a[1],
            a[2],
            a[3],
            f32::from_bits(*scale),
            stream,
        ),
        SparseF32::Pool128 { epsilon } => native::indexer_pool_fused(
            a[0],
            a[1],
            a[2],
            a[3],
            a[4],
            f32::from_bits(*epsilon),
            stream,
        ),
        SparseF32::IndexScores128 {
            position,
            pool_size,
            scale,
            schedule,
        } => {
            let f = match schedule {
                lisa_backend::ScoreSchedule::Scalar => native::indexer_scores,
                lisa_backend::ScoreSchedule::Rows4 => native::indexer_scores_rows4,
                lisa_backend::ScoreSchedule::Batch => native::indexer_scores_batch,
            };
            f(
                a[0],
                a[1],
                a[2],
                *position,
                *pool_size,
                f32::from_bits(*scale),
                stream,
            )
        }
        SparseF32::TopPools { count, shuffle } => {
            if *shuffle {
                native::dsa_indexer_topk_shuffle(a[0], *count, stream)
            } else {
                native::dsa_indexer_topk(a[0], *count, stream)
            }
        }
        SparseF32::ExpandPools {
            tokens,
            position,
            pool_size,
            tail,
        } => native::expand_pool_selection_for_attention(
            a[0], *tokens, *position, *pool_size, *tail, stream,
        ),
        SparseF32::WeightedRms { epsilon, latent } => {
            if *latent {
                native::mtp_kv_lora_rms_norm(a[0], a[1], f32::from_bits(*epsilon), stream)
            } else {
                native::mtp_rms_norm(a[0], a[1], f32::from_bits(*epsilon), stream)
            }
        }
        SparseF32::DeltaDecode {
            heads,
            lower_bound,
            epsilon,
        } => {
            return Ok(native::kda_decode(
                a[0],
                a[1],
                a[2],
                a[3],
                a[4],
                a[5],
                a[6],
                a[7],
                a[8],
                a[9],
                a[10],
                a[11],
                a[12],
                a[13],
                *heads,
                f32::from_bits(*lower_bound),
                f32::from_bits(*epsilon),
                stream,
            )
            .map(|(x, y, z)| Outputs::Many(vec![x, y, z])));
        }
        SparseF32::DeltaPrepare {
            heads,
            lower_bound,
            epsilon,
        } => {
            return Ok(native::kda_prepare(
                a[0],
                a[1],
                a[2],
                a[3],
                a[4],
                a[5],
                a[6],
                a[7],
                a[8],
                a[9],
                *heads,
                f32::from_bits(*lower_bound),
                f32::from_bits(*epsilon),
                stream,
            )
            .map(|(q, k, v, g, c)| Outputs::Many(vec![q, k, v, g, c])));
        }
        SparseF32::DeltaRecurrence { capture } => {
            return Ok(native::kda_recurrence(
                a[0], a[1], a[2], a[3], a[4], a[5], *capture, stream,
            )
            .map(|(y, s)| Outputs::Two([y, s])));
        }
        SparseF32::DeltaOutput { heads, epsilon } => {
            native::kda_output_norm(a[0], a[1], a[2], *heads, f32::from_bits(*epsilon), stream)
        }
        SparseF32::Bf16PairLinear => {
            return Ok(native::bf16_pair_gemv(a[0], a[1], a[2], a[3], stream)
                .map(|(x, y)| Outputs::Two([x, y])));
        }
        SparseF32::Bf16TripleLinear => {
            return Ok(native::bf16_qkv(a[0], a[1], a[2], a[3], stream)
                .map(|(x, y, z)| Outputs::Many(vec![x, y, z])));
        }
        SparseF32::Bf16PackedSix { groups } => {
            return Ok(native::bf16_kda_project_with_nsg(
                a[0], a[1], a[2], a[3], a[4], a[5], a[6], *groups, stream,
            )
            .map(|(x, y, z)| Outputs::Many(vec![x, y, z])));
        }
        SparseF32::MixRms { epsilon } => native::hc_mix_rms(a[0], f32::from_bits(*epsilon), stream),
        SparseF32::MixRmsBf16Linear { epsilon } => {
            native::hc_mix_rms_bf16_gemv(a[0], a[1], f32::from_bits(*epsilon), stream)
        }
        SparseF32::SplitSinkhorn4 {
            epsilon,
            iterations,
        } => native::hc_split_sinkhorn(
            a[0],
            a[1],
            a[2],
            f32::from_bits(*epsilon),
            *iterations,
            stream,
        ),
        SparseF32::SplitNorm4 {
            epsilon,
            norm_epsilon,
            iterations,
        } => {
            return Ok(native::hc_split_weighted_sum_norm4(
                a[0],
                a[1],
                a[2],
                a[3],
                a[4],
                f32::from_bits(*epsilon),
                f32::from_bits(*norm_epsilon),
                *iterations,
                stream,
            )
            .map(|(split, post, comb, collapsed, normed)| {
                Outputs::Many(vec![split, post, comb, collapsed, normed])
            }));
        }
        SparseF32::WeightedSum4 => native::hc_weighted_sum(a[0], a[1], stream),
        SparseF32::Expand4 => native::hc_expand4(a[0], a[1], a[2], a[3], stream),
        SparseF32::Bf16Linear { wide } => {
            if *wide {
                native::bf16_gemm(a[0], a[1], stream)
            } else {
                native::bf16_gemv(a[0], a[1], stream)
            }
        }
        SparseF32::RouterLinear => native::router_matvec_f32(a[0], a[1], stream),
        SparseF32::RouteSigmoid { top_k, scale } => {
            return Ok(native::router_select_fused(
                a[0],
                a[1],
                *top_k,
                f32::from_bits(*scale),
                stream,
            )
            .map(|(a, b)| Outputs::Two([a, b])));
        }
        SparseF32::ClampedSwiGlu { limit } => {
            native::swiglu_fused(a[0], a[1], f32::from_bits(*limit), stream)
        }
        SparseF32::WeightedClampedSwiGlu { limit } => {
            native::weighted_clamped_swiglu(a[0], a[1], a[2], f32::from_bits(*limit), stream)
        }
        SparseF32::SumEight { rows, width } => native::moe_sum8(a[0], *rows, *width, stream),
    };
    Ok(out.map(Outputs::One))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sparse_contracts_reject_invalid_parameters_and_bias_storage() -> Result<()> {
        let d = MetalDevice::system_default();
        let stream = d.stream();
        for op in [
            SparseF32::RouteSigmoid {
                top_k: 0,
                scale: 1f32.to_bits(),
            },
            SparseF32::ClampedSwiGlu {
                limit: f32::NAN.to_bits(),
            },
            SparseF32::WeightedClampedSwiGlu {
                limit: (-1f32).to_bits(),
            },
            SparseF32::SumEight {
                rows: usize::MAX,
                width: 1,
            },
        ] {
            assert!(d.prepare(Kernel::SparseF32(op))?.is_none());
        }
        let h = d
            .prepare(Kernel::SparseF32(SparseF32::RouteSigmoid {
                top_k: 8,
                scale: 2.5f32.to_bits(),
            }))?
            .unwrap();
        let x = Array::from_slice(&[1f32; 16], &[1, 16]);
        let short = Array::from_slice(&[0f32; 8], &[8]);
        assert!(d.dispatch(&h, &[&x, &short], &stream)?.is_none());
        let weighted = d
            .prepare(Kernel::SparseF32(SparseF32::WeightedClampedSwiGlu {
                limit: 10f32.to_bits(),
            }))?
            .unwrap();
        let gate = Array::from_slice(&[1f32; 32], &[1, 8, 4]);
        let weights = Array::from_slice(&[0.25f32; 8], &[1, 8]);
        assert!(
            d.dispatch(&weighted, &[&gate, &gate, &short], &stream)?
                .is_none()
        );
        let output = d
            .dispatch(&weighted, &[&gate, &gate, &weights], &stream)?
            .unwrap()
            .output;
        assert_eq!(
            output.into_one().map_err(anyhow::Error::msg)?.shape(),
            gate.shape()
        );
        let wrong = x.as_dtype(Dtype::Bfloat16)?;
        assert!(d.dispatch(&h, &[&wrong, &short], &stream).is_err());
        assert!(
            d.prepare(Kernel::PackedCompute(PackedCompute::Q8Linear {
                round_output: false,
                round_input: true
            }))?
            .is_none()
        );
        Ok(())
    }
}
