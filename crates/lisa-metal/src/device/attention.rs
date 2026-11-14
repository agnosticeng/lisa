//! Individual copy, normalization/rotation and gate kernels. No cache policy.
use super::*;
const STRIDED: TensorConstraint = TensorConstraint {
    layout: LayoutSupport::Strided,
    ..BF16
};
const FLOATS: TensorConstraint = TensorConstraint {
    dtypes: &[ScalarType::BF16, ScalarType::F32, ScalarType::F16],
    min_rank: 2,
    layout: LayoutSupport::Strided,
    ..BF16
};
pub(super) fn capabilities() -> Vec<KernelCapability> {
    [
        (
            KernelKind::KeyedGumbel,
            NumericalLaw::KeyedGumbelV1,
            &[TensorConstraint {
                dtypes: &[ScalarType::F32],
                min_rank: 1,
                max_rank: Some(1),
                layout: LayoutSupport::RowContiguous,
                alignment: 1,
            }][..],
            Workspace::ProviderManaged,
        ),
        (
            KernelKind::CopyRowsInto,
            NumericalLaw::StorageBytesV1,
            &[
                FLOATS,
                TensorConstraint {
                    layout: LayoutSupport::RowContiguous,
                    ..FLOATS
                },
            ][..],
            Workspace::CallerOutputs { inputs: &[1] },
        ),
        (
            KernelKind::CopyRowsConcatInto,
            NumericalLaw::StorageBytesV1,
            &[
                FLOATS,
                FLOATS,
                TensorConstraint {
                    layout: LayoutSupport::RowContiguous,
                    ..FLOATS
                },
            ][..],
            Workspace::CallerOutputs { inputs: &[2] },
        ),
        (
            KernelKind::QkNormRotary,
            NumericalLaw::Bf16QkRotaryV1,
            &[STRIDED, STRIDED, BF16, BF16, BF16, BF16][..],
            Workspace::ProviderManaged,
        ),
        (
            KernelKind::SigmoidGateTail,
            NumericalLaw::Bf16SigmoidGateV1,
            &[STRIDED, STRIDED][..],
            Workspace::ProviderManaged,
        ),
    ]
    .into_iter()
    .map(|(kind, law, inputs, workspace)| KernelCapability {
        kind,
        law,
        inputs,
        workspace,
        variadic: false,
        encoding: None,
        shape: ShapeRelation::TensorOperation,
    })
    .collect()
}
pub(super) fn invoke(
    kernel: &Kernel,
    inputs: &[&Array],
    stream: &Stream,
) -> Result<Option<Outputs<Array>>> {
    use crate::kernels::gated_pointwise as native;
    let result = match kernel {
        Kernel::KeyedGumbel {
            inverse_temperature,
            seed,
            position,
        } => {
            return Ok(native::keyed_gumbel_sample(
                inputs[0],
                f32::from_bits(*inverse_temperature),
                *seed,
                *position,
                stream,
            )
            .map(Outputs::One));
        }
        Kernel::CopyRowsInto { offset } => {
            let (src, dst) = (inputs[0], inputs[1]);
            let rank = src.rank();
            if rank < 2
                || dst.rank() != rank
                || src.dtype() != dst.dtype()
                || src.dim(-1) != dst.dim(-1)
                || src.shape()[..rank - 2] != dst.shape()[..rank - 2]
                || *offset < 0
                || i64::from(*offset) + i64::from(src.dim(-2)) > i64::from(dst.dim(-2))
            {
                return Err(Error::msg("invalid row-copy bindings"));
            }
            Outputs::One(
                crate::kernels::copy_rows_into(src, dst, *offset, stream)
                    .ok_or_else(|| Error::msg("row-copy dispatch failed"))?,
            )
        }
        Kernel::CopyRowsConcatInto { offset } => {
            return Ok(crate::kernels::copy_rows_concat_into(
                inputs[0], inputs[1], inputs[2], *offset, stream,
            )
            .map(Outputs::One));
        }
        Kernel::QkNormRotary {
            query_heads,
            key_heads,
            rows,
            epsilon,
            interleaved,
        } => {
            let [q, k, qw, kw, cos, sin] = inputs else {
                unreachable!()
            };
            if qw.shape() != [256]
                || kw.shape() != [256]
                || cos.size() != (*rows as usize) * 64
                || sin.size() != cos.size()
            {
                return Err(Error::msg("invalid normalized rotary bindings"));
            }
            if !*interleaved
                && (!q.t.buffer_and_layout().1.is_contiguous()
                    || !k.t.buffer_and_layout().1.is_contiguous())
            {
                return Err(Error::msg(
                    "normalized rotary decode requires contiguous vectors",
                ));
            }
            if *interleaved {
                let (q, k, g) = native::qk_norm_rope_rows(
                    q,
                    k,
                    qw,
                    kw,
                    cos,
                    sin,
                    f32::from_bits(*epsilon),
                    *query_heads,
                    *key_heads,
                    *rows,
                    stream,
                )
                .ok_or_else(|| Error::msg("interleaved normalized rotary dispatch failed"))?;
                Outputs::Many(vec![q, k, g])
            } else {
                let (q, k) = native::qk_norm_rope(
                    q,
                    k,
                    qw,
                    kw,
                    cos,
                    sin,
                    f32::from_bits(*epsilon),
                    *query_heads,
                    *key_heads,
                    stream,
                )
                .ok_or_else(|| Error::msg("normalized rotary dispatch failed"))?;
                Outputs::Two([q, k])
            }
        }
        Kernel::SigmoidGateTail {
            batch,
            rows,
            heads,
            width,
        } => {
            let [gate, up] = inputs else { unreachable!() };
            if gate.shape() != [*batch, *rows, *heads, *width]
                || up.shape() != [*batch, *heads, *rows, *width]
            {
                return Err(Error::msg("invalid sigmoid gate bindings"));
            }
            Outputs::One(
                native::sigmoid_mul_tail(gate, up, *batch, *rows, *heads, *width, stream)
                    .ok_or_else(|| Error::msg("sigmoid gate dispatch failed"))?,
            )
        }
        _ => unreachable!(),
    };
    Ok(Some(result))
}
