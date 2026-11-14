//! Independent packed projection and ordered F32 primitive contracts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OrderedF32 {
    /// Original GGML block bytes `[outputs, row_bytes]`, activation `[rows, inputs]`.
    BlockLinear {
        encoding: u32,
    },
    UnitRms {
        epsilon: u32,
    },
    DepthwiseConv4Silu,
    DecayGates,
    SiluProduct {
        reverse: bool,
    },
    /// Sequential delta recurrence with 128-wide keys/values and F32 state.
    Delta128,
    /// Vector attention reduction for every query row, including wide verification.
    VectorAttention {
        scale: u32,
        causal: bool,
    },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderedF32Kind {
    BlockLinear(u32),
    UnitRms,
    DepthwiseConv4Silu,
    DecayGates,
    SiluProduct,
    Delta128,
    VectorAttention,
}
impl OrderedF32 {
    pub fn kind(&self) -> OrderedF32Kind {
        match self {
            Self::BlockLinear { encoding } => OrderedF32Kind::BlockLinear(*encoding),
            Self::DepthwiseConv4Silu => OrderedF32Kind::DepthwiseConv4Silu,
            Self::UnitRms { .. } => OrderedF32Kind::UnitRms,
            Self::DecayGates => OrderedF32Kind::DecayGates,
            Self::SiluProduct { .. } => OrderedF32Kind::SiluProduct,
            Self::Delta128 => OrderedF32Kind::Delta128,
            Self::VectorAttention { .. } => OrderedF32Kind::VectorAttention,
        }
    }
    pub fn arity(&self) -> usize {
        match self {
            Self::UnitRms { .. } => 1,
            Self::BlockLinear { .. } | Self::DepthwiseConv4Silu | Self::SiluProduct { .. } => 2,
            Self::DecayGates => 4,
            Self::Delta128 => 6,
            Self::VectorAttention { .. } => 3,
        }
    }
    pub fn outputs(&self) -> usize {
        if matches!(self, Self::DecayGates | Self::Delta128) {
            2
        } else {
            1
        }
    }
    pub fn validate(&self) -> bool {
        match self {
            Self::VectorAttention { scale, .. } => f32::from_bits(*scale).is_finite(),
            Self::UnitRms { epsilon } => {
                f32::from_bits(*epsilon).is_finite() && f32::from_bits(*epsilon) >= 0.
            }
            _ => true,
        }
    }
}
