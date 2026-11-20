//! Prepared tensor primitives; no model, checkpoint or request policy.
use super::*;
use crate::ops::indexing::IndexElem;

const ANY: TensorConstraint = TensorConstraint {
    dtypes: &[
        ScalarType::Bool,
        ScalarType::U8,
        ScalarType::U32,
        ScalarType::I32,
        ScalarType::I64,
        ScalarType::F16,
        ScalarType::BF16,
        ScalarType::F32,
    ],
    min_rank: 0,
    max_rank: None,
    layout: LayoutSupport::Strided,
    alignment: 1,
};
const FLOATING: TensorConstraint = TensorConstraint {
    dtypes: &[ScalarType::F16, ScalarType::BF16, ScalarType::F32],
    ..ANY
};
const ROW: TensorConstraint = TensorConstraint {
    min_rank: 1,
    layout: LayoutSupport::RowContiguous,
    ..FLOATING
};
const SCALE: TensorConstraint = TensorConstraint {
    max_rank: Some(1),
    ..ROW
};
const MASK: TensorConstraint = TensorConstraint {
    dtypes: &[
        ScalarType::Bool,
        ScalarType::U8,
        ScalarType::BF16,
        ScalarType::F32,
    ],
    ..ANY
};

pub(super) fn capabilities() -> Vec<KernelCapability> {
    use TensorOperationKind as K;
    let mut result = Vec::new();
    let mut add = |kind, inputs: &'static [TensorConstraint], variadic| {
        result.push(KernelCapability {
            kind: KernelKind::Tensor(kind),
            law: NumericalLaw::TensorV1,
            inputs,
            variadic,
            shape: ShapeRelation::TensorOperation,
            encoding: None,
            workspace: Workspace::ProviderManaged,
        })
    };
    for k in [
        K::Reshape,
        K::SplitLastAxis,
        K::Transpose,
        K::Slice,
        K::Select,
        K::Contiguous,
        K::Cast,
        K::Broadcast,
        K::Tile,
        K::Repeat,
    ] {
        add(k, &[ANY], false);
    }
    add(K::Concatenate, &[ANY], true);
    add(K::AssignSlice, &[ANY, ANY], false);
    add(K::AssignRegion, &[ANY, ANY], false);
    for k in [
        Unary::Negate,
        Unary::Sigmoid,
        Unary::Silu,
        Unary::Exp,
        Unary::Log,
        Unary::Sqrt,
        Unary::Rsqrt,
        Unary::Erf,
        Unary::Log1p,
        Unary::Cos,
        Unary::Sin,
        Unary::Tanh,
    ] {
        add(K::Unary(k), &[FLOATING], false);
    }
    for k in [
        Binary::Add,
        Binary::Subtract,
        Binary::Multiply,
        Binary::Divide,
        Binary::Maximum,
        Binary::Minimum,
        Binary::Power,
    ] {
        add(K::Binary(k), &[FLOATING, FLOATING], false);
    }
    const STATE_F32: TensorConstraint = TensorConstraint { dtypes: &[ScalarType::F32], layout: LayoutSupport::RowContiguous, ..ANY };
    add(K::Mamba3Mimo, &[STATE_F32; 8], false);
    add(K::GeodesicResidual, &[STATE_F32; 4], false);
    add(K::Matmul, &[FLOATING, FLOATING], false);
    const SORTABLE: TensorConstraint = TensorConstraint {
        dtypes: &[
            ScalarType::I32,
            ScalarType::U32,
            ScalarType::F32,
            ScalarType::BF16,
            ScalarType::F16,
        ],
        min_rank: 1,
        ..ANY
    };
    add(K::Argsort, &[SORTABLE], false);
    add(
        K::ArgPartition,
        &[TensorConstraint {
            min_rank: 1,
            ..FLOATING
        }],
        false,
    );
    const INTEGER: TensorConstraint = TensorConstraint {
        dtypes: &[ScalarType::I32, ScalarType::U32],
        ..ANY
    };
    add(K::FloorDivide, &[INTEGER, INTEGER], false);
    const BOOL: TensorConstraint = TensorConstraint {
        dtypes: &[ScalarType::Bool, ScalarType::U8],
        ..ANY
    };
    for c in [
        Comparison::Less,
        Comparison::LessEqual,
        Comparison::Greater,
        Comparison::GreaterEqual,
    ] {
        add(
            K::Compare(c),
            &[
                TensorConstraint {
                    min_rank: 0,
                    ..SORTABLE
                },
                TensorConstraint {
                    min_rank: 0,
                    ..SORTABLE
                },
            ],
            false,
        );
    }
    add(K::Where, &[BOOL, FLOATING, FLOATING], false);
    add(K::LogicalAnd, &[BOOL, BOOL], false);
    add(K::LogicalOr, &[BOOL, BOOL], false);
    add(K::PutAlong, &[BOOL, INTEGER, BOOL], false);
    add(K::Rotary, &[FLOATING], false);
    add(K::Sum, &[FLOATING], false);
    add(K::SumAll, &[FLOATING], false);
    add(K::MaxAll, &[FLOATING], false);
    add(
        K::TakeAlong,
        &[
            ANY,
            TensorConstraint {
                dtypes: &[ScalarType::I32, ScalarType::U32],
                ..ANY
            },
        ],
        false,
    );
    add(K::Mean, &[FLOATING], false);
    add(K::Softmax, &[FLOATING], false);
    add(K::Top32Indices, &[FLOATING], false);
    add(K::Argmax, &[FLOATING], false);
    add(
        K::Take,
        &[
            ANY,
            TensorConstraint {
                dtypes: &[ScalarType::I32, ScalarType::U32],
                ..ANY
            },
        ],
        false,
    );
    add(K::LayerNorm, &[FLOATING, SCALE, SCALE], false);
    add(K::RmsNorm, &[ROW, SCALE], false);
    add(K::GroupedRms128, &[BF16], false);
    add(K::AddRmsNorm, &[ROW, ROW, SCALE], false);
    for mask in [AttentionMask::None, AttentionMask::Causal] {
        add(K::Attention(mask), &[FLOATING, FLOATING, FLOATING], false);
    }
    add(
        K::Attention(AttentionMask::Tensor),
        &[FLOATING, FLOATING, FLOATING, MASK],
        false,
    );
    result.push(KernelCapability {
        kind: KernelKind::Tensor(K::CopySliceInto),
        law: NumericalLaw::TensorV1,
        inputs: &[ANY, ANY],
        variadic: false,
        shape: ShapeRelation::TensorOperation,
        encoding: None,
        workspace: Workspace::CallerOutputs { inputs: &[1] },
    });
    result
}
pub(super) fn invoke(
    kernel: &Kernel,
    inputs: &[&Array],
    stream: &Stream,
) -> Result<Option<Outputs<Array>>> {
    let Kernel::Tensor(op) = kernel else {
        unreachable!()
    };
    let x = inputs[0];
    if matches!(
        op,
        TensorOperation::RmsNorm { .. } | TensorOperation::AddRmsNorm { .. }
    ) {
        let weight = inputs[inputs.len() - 1];
        if x.rank() == 0
            || x.dim(-1) <= 0
            || weight.rank() != 1
            || weight.dim(0) != x.dim(-1)
            || weight.dtype() != x.dtype()
        {
            return Err(Error::msg(
                "normalization: incompatible scale shape or dtype",
            ));
        }
        if matches!(op, TensorOperation::AddRmsNorm { .. })
            && (inputs[1].size() != x.size() || inputs[1].dtype() != x.dtype())
        {
            return Err(Error::msg("residual normalization: incompatible residual"));
        }
    }
    if matches!(op, TensorOperation::LayerNorm { .. })
        && (x.rank() == 0
            || x.dim(-1) <= 0
            || inputs[1].shape() != [x.dim(-1)]
            || inputs[2].shape() != [x.dim(-1)])
    {
        return Err(Error::msg("layer normalization: incompatible affine shape"));
    }
    if let TensorOperation::Transpose(axes) = op {
        if axes.len() != x.rank() {
            return Err(Error::msg("transpose: axis count differs from rank"));
        }
    }

    let output = match op {
        TensorOperation::Mamba3Mimo { heads, epsilon } => {
            let (y, state) = crate::kernels::recurrent_mimo::step(inputs, *heads as usize, f32::from_bits(*epsilon), stream)?;
            return Ok(Some(Outputs::Two([y, state])));
        }
        TensorOperation::GeodesicResidual { inverse_depth } => crate::kernels::recurrent_mimo::geodesic(inputs, f32::from_bits(*inverse_depth), stream)?,
        TensorOperation::SequentialLinear { .. }
        | TensorOperation::SequentialLayerNorm { .. }
        | TensorOperation::SequentialAttention => {
            return Err(Error::msg(
                "host arithmetic is unsupported by this provider",
            ));
        }
        TensorOperation::Reshape(shape) => x.reshape(shape)?,
        TensorOperation::SplitLastAxis(shape) => x.split_last_axis(shape)?,
        TensorOperation::CopySliceInto { axis, start, end } => {
            let dst = inputs[1];
            let rank = dst.rank() as i32;
            let axis = if *axis < 0 { rank + axis } else { *axis };
            if axis < 0
                || axis >= rank
                || *start < 0
                || end < start
                || *end > dst.dim(axis)
                || x.rank() != dst.rank()
                || x.dtype() != dst.dtype()
            {
                return Err(Error::msg("invalid byte-copy range"));
            }
            let mut shape = dst.shape().to_vec();
            shape[axis as usize] = end - start;
            if x.shape() != shape {
                return Err(Error::msg("byte-copy source shape differs"));
            }
            let slice = (*start..*end).apply(&dst.t, axis as usize)?.0;
            x.t.copy_strided_into(&slice)?;
            dst.clone()
        }
        TensorOperation::AssignRegion(region) => {
            if region.len() != x.rank()
                || region
                    .iter()
                    .zip(x.shape())
                    .any(|(&(start, end), &dim)| start < 0 || end < start || end > dim)
            {
                return Err(Error::msg("invalid assignment region"));
            }
            let ranges = region
                .iter()
                .map(|&(a, b)| a as usize..b as usize)
                .collect::<Vec<_>>();
            Array::new(x.t.slice_assign(&ranges, &inputs[1].t)?)
        }
        TensorOperation::AssignSlice { axis, start, end } => {
            let rank = x.rank() as i32;
            let axis = if *axis < 0 { rank + axis } else { *axis };
            if axis < 0 || axis >= rank || *start < 0 || end < start || *end > x.dim(axis) {
                return Err(Error::msg("invalid slice assignment range"));
            }
            let mut ranges = x.shape().iter().map(|&d| 0..d as usize).collect::<Vec<_>>();
            ranges[axis as usize] = *start as usize..*end as usize;
            Array::new(x.t.slice_assign(&ranges, &inputs[1].t)?)
        }
        TensorOperation::Transpose(axes) => x.transpose_axes(axes)?,
        TensorOperation::Broadcast(shape) => x.broadcast_to(shape)?,
        TensorOperation::Tile(reps) => crate::ops::tile(x, reps)?,
        TensorOperation::Repeat { count, axis } => x.repeat_axis(*count, *axis)?,
        TensorOperation::Slice { axis, start, end } => {
            let rank = x.rank() as i32;
            let axis = if *axis < 0 { rank + axis } else { *axis };
            if axis < 0 || axis >= rank {
                return Err(Error::msg("slice axis exceeds rank"));
            }
            Array::new((*start..*end).apply(&x.t, axis as usize)?.0)
        }
        TensorOperation::Select { axis, index } => {
            let rank = x.rank() as i32;
            let axis = if *axis < 0 { rank + axis } else { *axis };
            if axis < 0 || axis >= rank || *index < 0 || *index >= x.dim(axis) {
                return Err(Error::msg("invalid select index"));
            }
            Array::new(index.apply(&x.t, axis as usize)?.0.squeeze(axis as usize)?)
        }
        TensorOperation::Contiguous => x.contiguous()?,
        TensorOperation::Cast(dt) => x.as_dtype(dtype(*dt))?,
        TensorOperation::Concatenate { axis, .. } => crate::ops::concatenate(inputs, *axis)?,
        TensorOperation::Unary(op) => match op {
            Unary::Negate => x.neg()?,
            Unary::Sigmoid => x.sigmoid()?,
            Unary::Silu => x.silu()?,
            Unary::Erf => x.unary("Erf")?,
            Unary::Log1p => x.unary("Log1p")?,
            Unary::Cos => crate::ops::cos(x)?,
            Unary::Sin => crate::ops::sin(x)?,
            Unary::Tanh => x.unary("Tanh")?,
            Unary::Exp => x.exp()?,
            Unary::Log => x.log()?,
            Unary::Sqrt => x.sqrt()?,
            Unary::Rsqrt => x.rsqrt()?,
        },
        TensorOperation::Binary(op) => match op {
            Binary::Maximum => x.maximum(inputs[1])?,
            Binary::Minimum => x.minimum(inputs[1])?,
            Binary::Power => Array::new(crate::mlx_rt::binary_op(
                x.t.device(),
                &x.t,
                &inputs[1].t,
                "Power",
            )?),
            Binary::Add => x.add(inputs[1])?,
            Binary::Subtract => x.subtract(inputs[1])?,
            Binary::Multiply => x.multiply(inputs[1])?,
            Binary::Divide => x.divide(inputs[1])?,
        },
        TensorOperation::Matmul => x.matmul(inputs[1])?,
        TensorOperation::Rotary {
            dimensions,
            base,
            offset,
        } => crate::fast::rope(
            x,
            *dimensions,
            false,
            Some(f32::from_bits(*base)),
            1.,
            *offset,
            None,
        )?,
        TensorOperation::SumAll => x.sum(None)?,
        TensorOperation::MaxAll => x.max(None)?,
        TensorOperation::Sum { axis, keep_dims } => x.sum_axis(*axis, *keep_dims)?,
        TensorOperation::TakeAlong { axis } => x.take_along_axis(inputs[1], *axis)?,
        TensorOperation::Mean { axis, keep_dims } => x.mean_axis(*axis, *keep_dims)?,
        TensorOperation::Argsort => crate::ops::argsort(x)?,
        TensorOperation::ArgPartition { kth, axis } => {
            if x.rank() == 0 || (*axis != -1 && *axis != x.rank() as i32 - 1) || *kth >= x.dim(-1) {
                return Ok(None);
            }
            crate::ops::argpartition_axis(x, *kth, *axis)?
        }
        TensorOperation::Compare(c) => match c {
            Comparison::Less => x.lt(inputs[1])?,
            Comparison::Greater => x.gt(inputs[1])?,
            Comparison::LessEqual => x.le(inputs[1])?,
            Comparison::GreaterEqual => x.ge(inputs[1])?,
        },
        TensorOperation::Where => crate::ops::r#where(x, inputs[1], inputs[2])?,
        TensorOperation::LogicalAnd => x.logical_and(inputs[1])?,
        TensorOperation::LogicalOr => x.logical_or(inputs[1])?,
        TensorOperation::PutAlong { axis } => x.put_along_axis(inputs[1], inputs[2], *axis)?,
        TensorOperation::FloorDivide => crate::ops::floor_divide(x, inputs[1])?,
        TensorOperation::Top32Indices => crate::ops::top32_indices_on(x, stream)?,
        TensorOperation::Argmax { axis, keep_dims } => {
            crate::ops::indexing::argmax_axis(x, *axis, *keep_dims)?
        }
        TensorOperation::Softmax { axis } => x.softmax_axis(*axis)?,
        TensorOperation::Take { axis } => x.take_axis(inputs[1], *axis)?,
        TensorOperation::LayerNorm { epsilon } => {
            crate::fast::layer_norm(x, inputs[1], inputs[2], f32::from_bits(*epsilon))?
        }
        TensorOperation::GroupedRms128 { epsilon } => {
            if x.rank() == 0 || x.dim(-1) <= 0 || x.dim(-1) % 128 != 0 {
                return Ok(None);
            }
            return Ok(crate::kernels::sparse::kernels::rms_row_fused(
                x,
                f32::from_bits(*epsilon),
                128,
                stream,
            )
            .map(Outputs::One));
        }
        TensorOperation::RmsNorm { epsilon } => {
            crate::fast::rms_norm(x, Some(inputs[1]), f32::from_bits(*epsilon))?
        }
        TensorOperation::AddRmsNorm { epsilon } => {
            let (sum, normalized) =
                crate::fast::fused_add_rms_norm(x, inputs[1], inputs[2], f32::from_bits(*epsilon))?;
            return Ok(Some(Outputs::Two([sum, normalized])));
        }
        TensorOperation::Attention { scale, mask } => {
            let mask = match mask {
                AttentionMask::None => None,
                AttentionMask::Causal => Some(crate::fast::ScaledDotProductAttentionMask::Causal),
                AttentionMask::Tensor => {
                    Some(crate::fast::ScaledDotProductAttentionMask::Array(inputs[3]))
                }
            };
            crate::fast::scaled_dot_product_attention(
                x,
                inputs[1],
                inputs[2],
                f32::from_bits(*scale),
                mask,
                None,
            )?
        }
    };
    Ok(Some(Outputs::One(output)))
}
