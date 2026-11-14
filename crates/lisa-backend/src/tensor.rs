//! Model-independent tensor operation contracts. Scalar attributes carry their
//! exact IEEE bits so preparation cannot round or canonicalize a numerical law.
use crate::ScalarType;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unary {
    Negate,
    Sigmoid,
    Silu,
    Exp,
    Log,
    Sqrt,
    Rsqrt,
    Erf,
    Log1p,
    Cos,
    Sin,
    Tanh,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binary {
    Add,
    Subtract,
    Multiply,
    Divide,
    Maximum,
    Minimum,
    Power,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Comparison {
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionMask {
    None,
    Causal,
    Tensor,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TensorOperation {
    Reshape(Vec<i32>),
    /// Split the final contiguous axis while preserving leading row strides.
    SplitLastAxis(Vec<i32>),
    /// Return a copy with the supplied axis range replaced by the second input.
    /// Copy source bytes into an explicitly writable destination range.
    CopySliceInto {
        axis: i32,
        start: i32,
        end: i32,
    },
    /// Copy with one replaced rectangular region; one half-open range per axis.
    AssignRegion(Vec<(i32, i32)>),
    AssignSlice {
        axis: i32,
        start: i32,
        end: i32,
    },
    Transpose(Vec<i32>),
    Broadcast(Vec<i32>),
    Tile(Vec<i32>),
    Repeat {
        count: usize,
        axis: i32,
    },
    /// A view preserving strides. Negative axes are relative to input rank.
    Slice {
        axis: i32,
        start: i32,
        end: i32,
    },
    Select {
        axis: i32,
        index: i32,
    },
    Contiguous,
    Cast(ScalarType),
    Concatenate {
        axis: i32,
        inputs: usize,
    },
    Unary(Unary),
    Binary(Binary),
    SequentialLinear {
        bias: bool,
    },
    SequentialLayerNorm {
        epsilon: u32,
        bias: bool,
    },
    SequentialAttention,
    Matmul,
    /// Whole-tensor reductions retain their own reduction and cast order.
    SumAll,
    MaxAll,
    Sum {
        axis: i32,
        keep_dims: bool,
    },
    TakeAlong {
        axis: i32,
    },
    Mean {
        axis: i32,
        keep_dims: bool,
    },
    Rotary {
        dimensions: i32,
        base: u32,
        offset: i32,
    },
    Top32Indices,
    Argsort,
    ArgPartition {
        kth: i32,
        axis: i32,
    },
    FloorDivide,
    Compare(Comparison),
    Where,
    LogicalAnd,
    LogicalOr,
    PutAlong {
        axis: i32,
    },
    Argmax {
        axis: i32,
        keep_dims: bool,
    },
    Softmax {
        axis: i32,
    },
    Take {
        axis: i32,
    },
    LayerNorm {
        epsilon: u32,
    },
    RmsNorm {
        epsilon: u32,
    },
    /// BF16 groups of 128: four sequential F32 squares per lane, the original
    /// DS4 lane reduction, reciprocal square root, then BF16 output. No scale.
    GroupedRms128 {
        epsilon: u32,
    },
    /// The residual is a contiguous flat sequence with the same element count
    /// and dtype as the input; output geometry follows the first input.
    AddRmsNorm {
        epsilon: u32,
    },
    Attention {
        scale: u32,
        mask: AttentionMask,
    },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorOperationKind {
    Reshape,
    SplitLastAxis,
    CopySliceInto,
    AssignRegion,
    AssignSlice,
    Transpose,
    Broadcast,
    Tile,
    Repeat,
    Slice,
    Select,
    Contiguous,
    Cast,
    Concatenate,
    Unary(Unary),
    Binary(Binary),
    SequentialLinear(bool),
    SequentialLayerNorm(bool),
    SequentialAttention,
    Matmul,
    SumAll,
    MaxAll,
    Sum,
    TakeAlong,
    Mean,
    Rotary,
    Top32Indices,
    Argsort,
    ArgPartition,
    FloorDivide,
    Compare(Comparison),
    Where,
    LogicalAnd,
    LogicalOr,
    PutAlong,
    Argmax,
    Softmax,
    Take,
    LayerNorm,
    RmsNorm,
    GroupedRms128,
    AddRmsNorm,
    Attention(AttentionMask),
}
impl TensorOperation {
    pub fn validate(&self) -> bool {
        match self {
            Self::ArgPartition { kth, .. } => *kth >= 0,
            Self::Reshape(shape) => {
                shape.iter().all(|&d| d >= -1) && shape.iter().filter(|&&d| d == -1).count() <= 1
            }
            Self::SplitLastAxis(shape) => !shape.is_empty() && shape.iter().all(|&d| d > 0),
            Self::CopySliceInto { start, end, .. } | Self::AssignSlice { start, end, .. } => {
                *start >= 0 && end >= start
            }
            Self::AssignRegion(region) => region
                .iter()
                .all(|&(start, end)| start >= 0 && end >= start),
            Self::Transpose(axes) => {
                let rank = axes.len() as i32;
                let mut seen = vec![false; axes.len()];
                axes.iter().all(|&axis| {
                    let axis = if axis < 0 { rank + axis } else { axis };
                    axis >= 0 && axis < rank && !std::mem::replace(&mut seen[axis as usize], true)
                })
            }
            Self::Concatenate { inputs, .. } => *inputs > 0,
            Self::Broadcast(shape) | Self::Tile(shape) => shape.iter().all(|&d| d >= 0),
            Self::LayerNorm { epsilon }
            | Self::SequentialLayerNorm { epsilon, .. }
            | Self::RmsNorm { epsilon }
            | Self::GroupedRms128 { epsilon }
            | Self::AddRmsNorm { epsilon } => {
                let e = f32::from_bits(*epsilon);
                e.is_finite() && e >= 0.0
            }
            Self::Rotary {
                dimensions,
                base,
                offset,
            } => {
                *dimensions > 0
                    && *dimensions % 2 == 0
                    && *offset >= 0
                    && f32::from_bits(*base).is_finite()
                    && f32::from_bits(*base) > 0.
            }
            Self::Attention { scale, .. } => f32::from_bits(*scale).is_finite(),
            _ => true,
        }
    }
    pub fn kind(&self) -> TensorOperationKind {
        match self {
            Self::Reshape(_) => TensorOperationKind::Reshape,
            Self::SplitLastAxis(_) => TensorOperationKind::SplitLastAxis,
            Self::CopySliceInto { .. } => TensorOperationKind::CopySliceInto,
            Self::AssignRegion(_) => TensorOperationKind::AssignRegion,
            Self::AssignSlice { .. } => TensorOperationKind::AssignSlice,
            Self::Transpose(_) => TensorOperationKind::Transpose,
            Self::Broadcast(_) => TensorOperationKind::Broadcast,
            Self::Tile(_) => TensorOperationKind::Tile,
            Self::Repeat { .. } => TensorOperationKind::Repeat,
            Self::Slice { .. } => TensorOperationKind::Slice,
            Self::Select { .. } => TensorOperationKind::Select,
            Self::Contiguous => TensorOperationKind::Contiguous,
            Self::Cast(_) => TensorOperationKind::Cast,
            Self::Concatenate { .. } => TensorOperationKind::Concatenate,
            Self::Unary(v) => TensorOperationKind::Unary(*v),
            Self::Binary(v) => TensorOperationKind::Binary(*v),
            Self::SequentialLinear { bias } => TensorOperationKind::SequentialLinear(*bias),
            Self::SequentialLayerNorm { bias, .. } => {
                TensorOperationKind::SequentialLayerNorm(*bias)
            }
            Self::SequentialAttention => TensorOperationKind::SequentialAttention,
            Self::Matmul => TensorOperationKind::Matmul,
            Self::SumAll => TensorOperationKind::SumAll,
            Self::MaxAll => TensorOperationKind::MaxAll,
            Self::Sum { .. } => TensorOperationKind::Sum,
            Self::TakeAlong { .. } => TensorOperationKind::TakeAlong,
            Self::Mean { .. } => TensorOperationKind::Mean,
            Self::Rotary { .. } => TensorOperationKind::Rotary,
            Self::Top32Indices => TensorOperationKind::Top32Indices,
            Self::Argsort => TensorOperationKind::Argsort,
            Self::ArgPartition { .. } => TensorOperationKind::ArgPartition,
            Self::FloorDivide => TensorOperationKind::FloorDivide,
            Self::Compare(c) => TensorOperationKind::Compare(*c),
            Self::Where => TensorOperationKind::Where,
            Self::LogicalAnd => TensorOperationKind::LogicalAnd,
            Self::LogicalOr => TensorOperationKind::LogicalOr,
            Self::PutAlong { .. } => TensorOperationKind::PutAlong,
            Self::Argmax { .. } => TensorOperationKind::Argmax,
            Self::Softmax { .. } => TensorOperationKind::Softmax,
            Self::Take { .. } => TensorOperationKind::Take,
            Self::LayerNorm { .. } => TensorOperationKind::LayerNorm,
            Self::RmsNorm { .. } => TensorOperationKind::RmsNorm,
            Self::GroupedRms128 { .. } => TensorOperationKind::GroupedRms128,
            Self::AddRmsNorm { .. } => TensorOperationKind::AddRmsNorm,
            Self::Attention { mask, .. } => TensorOperationKind::Attention(*mask),
        }
    }
    pub fn arity(&self) -> usize {
        match self {
            Self::SequentialLinear { bias } | Self::SequentialLayerNorm { bias, .. } => {
                2 + usize::from(*bias)
            }
            Self::SequentialAttention => 4,
            Self::Where | Self::PutAlong { .. } => 3,
            Self::Compare(_) | Self::LogicalAnd | Self::LogicalOr => 2,
            Self::Concatenate { inputs, .. } => *inputs,
            Self::CopySliceInto { .. }
            | Self::AssignRegion(_)
            | Self::AssignSlice { .. }
            | Self::Binary(_)
            | Self::FloorDivide
            | Self::Matmul
            | Self::TakeAlong { .. }
            | Self::Take { .. }
            | Self::RmsNorm { .. } => 2,
            Self::AddRmsNorm { .. } | Self::LayerNorm { .. } => 3,
            Self::Attention {
                mask: AttentionMask::Tensor,
                ..
            } => 4,
            Self::Attention { .. } => 3,
            _ => 1,
        }
    }
    pub fn outputs(&self) -> usize {
        if matches!(self, Self::AddRmsNorm { .. }) {
            2
        } else {
            1
        }
    }
}

/// Single-output launches stay allocation-free at the contract boundary.
/// Multi-output operations retain their output collection as one owned value.
pub enum Outputs<T> {
    One(T),
    Two([T; 2]),
    Many(Vec<T>),
}
impl<T> Outputs<T> {
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Outputs<U> {
        match self {
            Self::One(t) => Outputs::One(f(t)),
            Self::Two([a, b]) => Outputs::Two([f(a), f(b)]),
            Self::Many(v) => Outputs::Many(v.into_iter().map(f).collect()),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::One(_) => 1,
            Self::Two(_) => 2,
            Self::Many(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn get(&self, index: usize) -> Option<&T> {
        match self {
            Self::One(t) => (index == 0).then_some(t),
            Self::Two(v) => v.get(index),
            Self::Many(v) => v.get(index),
        }
    }
    pub fn into_one(self) -> Result<T, &'static str> {
        match self {
            Self::One(t) => Ok(t),
            Self::Many(mut v) if v.len() == 1 => Ok(v.remove(0)),
            _ => Err("expected one kernel output"),
        }
    }
}
pub enum OutputIter<T> {
    One(std::option::IntoIter<T>),
    Two(std::array::IntoIter<T, 2>),
    Many(std::vec::IntoIter<T>),
}
impl<T> Iterator for OutputIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<T> {
        match self {
            Self::One(v) => v.next(),
            Self::Two(v) => v.next(),
            Self::Many(v) => v.next(),
        }
    }
}
impl<T> IntoIterator for Outputs<T> {
    type Item = T;
    type IntoIter = OutputIter<T>;
    fn into_iter(self) -> Self::IntoIter {
        match self {
            Self::One(t) => OutputIter::One(Some(t).into_iter()),
            Self::Two(v) => OutputIter::Two(v.into_iter()),
            Self::Many(v) => OutputIter::Many(v.into_iter()),
        }
    }
}
