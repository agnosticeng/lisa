//! Independent packed projection and ordered F32 primitive contracts.
/// Original block payload encodings supported by the ordered F32 projection.
/// These names describe arithmetic layouts, independent of any file container.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderedBlockEncoding {
    Q8_0,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Iq2Xxs,
    Iq2Xs,
    Iq3Xxs,
    Iq1S,
    Iq3S,
    Iq2S,
    Iq4Xs,
    Iq1M,
}
impl OrderedBlockEncoding {
    pub const ALL: [Self; 14] = [
        Self::Q8_0,
        Self::Q2K,
        Self::Q3K,
        Self::Q4K,
        Self::Q5K,
        Self::Q6K,
        Self::Iq2Xxs,
        Self::Iq2Xs,
        Self::Iq3Xxs,
        Self::Iq1S,
        Self::Iq3S,
        Self::Iq2S,
        Self::Iq4Xs,
        Self::Iq1M,
    ];
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OrderedF32 {
    /// Original packed block bytes `[outputs, row_bytes]`, activation `[rows, inputs]`.
    BlockLinear {
        encoding: OrderedBlockEncoding,
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
    BlockLinear(OrderedBlockEncoding),
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
