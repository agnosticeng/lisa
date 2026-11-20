//! Original packed-block projection and ordered routing arithmetic contracts.
//! Encodings describe payloads, never a checkpoint reader or model family.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedEncoding {
    Iq2Xxs,
    Iq2Xs,
    Iq1M,
    Q5K,
    Q6K,
    Q2K,
    Q4K,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PackedCompute {
    GatedResidual4 {
        epsilon: u32,
        masked: bool,
    },
    RmsNorm {
        weighted: bool,
        round_output: bool,
        epsilon: u32,
    },
    RmsNormPair {
        epsilon: u32,
    },
    ResidualInput4 {
        epsilon: u32,
        norm_epsilon: u32,
    },
    ResidualExpand4 {
        add: bool,
        round_block: bool,
    },
    ResidualCollapse4,
    PlainLinear {
        round_output: bool,
    },
    Q8Linear {
        round_output: bool,
        round_input: bool,
    },
    BlockLinear {
        encoding: PackedEncoding,
        round_output: bool,
    },
    GroupedLinear {
        encoding: PackedEncoding,
        round_output: bool,
    },
    /// Original Q8_0 experts with GPU indices and the unrounded dense dot-product law.
    Q8ExpertLinear,
    Q8Pair,
    ExpertLinear {
        encoding: PackedEncoding,
    },
    Router {
        top_k: u32,
        scale: u32,
    },
    /// Flat clamped SiLU product, optionally rounded to BF16 in F32 storage.
    SharedSwiGlu {
        round_output: bool,
    },
    WeightedSwiGlu {
        clamp: u32,
    },
    /// Two independent F32 row blends, with native sigmoid and separate products.
    RowGateBlendPair {
        zero: bool,
    },
    DifferentialSiluGate {
        bias: u32,
    },
    /// Batched F32 products with 2..=8 rows and the original eight-wide MMA order.
    SmallRowMatmul {
        transpose_rhs: bool,
    },
    ReluSquared,
    /// Six separately rounded products followed by five left-associated F32 adds.
    WeightedSumSix,
    /// Six stable routes from 256 finite logits/biases; outputs I32 ids and F32 weights.
    SigmoidRouter256 {
        scale: u32,
    },
    SumSix,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedComputeKind {
    GatedResidual4(bool),
    RmsNorm(bool),
    RmsNormPair,
    ResidualInput4,
    ResidualExpand4(bool),
    ResidualCollapse4,
    PlainLinear,
    Q8Linear,
    BlockLinear(PackedEncoding),
    GroupedLinear(PackedEncoding),
    Q8ExpertLinear,
    Q8Pair,
    ExpertLinear(PackedEncoding),
    Router,
    SharedSwiGlu,
    WeightedSwiGlu,
    RowGateBlendPair,
    DifferentialSiluGate,
    SmallRowMatmul,
    ReluSquared,
    WeightedSumSix,
    SigmoidRouter256,
    SumSix,
}
impl PackedCompute {
    pub fn kind(&self) -> PackedComputeKind {
        use PackedComputeKind as K;
        match self {
            Self::GatedResidual4 { masked, .. } => K::GatedResidual4(*masked),
            Self::RmsNorm { weighted, .. } => K::RmsNorm(*weighted),
            Self::RmsNormPair { .. } => K::RmsNormPair,
            Self::ResidualInput4 { .. } => K::ResidualInput4,
            Self::ResidualExpand4 { add, .. } => K::ResidualExpand4(*add),
            Self::ResidualCollapse4 => K::ResidualCollapse4,
            Self::PlainLinear { .. } => K::PlainLinear,
            Self::Q8Linear { .. } => K::Q8Linear,
            Self::BlockLinear { encoding, .. } => K::BlockLinear(*encoding),
            Self::GroupedLinear { encoding, .. } => K::GroupedLinear(*encoding),
            Self::Q8ExpertLinear => K::Q8ExpertLinear,
            Self::Q8Pair => K::Q8Pair,
            Self::ExpertLinear { encoding } => K::ExpertLinear(*encoding),
            Self::Router { .. } => K::Router,
            Self::SharedSwiGlu { .. } => K::SharedSwiGlu,
            Self::WeightedSwiGlu { .. } => K::WeightedSwiGlu,
            Self::RowGateBlendPair { .. } => K::RowGateBlendPair,
            Self::DifferentialSiluGate { .. } => K::DifferentialSiluGate,
            Self::SmallRowMatmul { .. } => K::SmallRowMatmul,
            Self::ReluSquared => K::ReluSquared,
            Self::WeightedSumSix => K::WeightedSumSix,
            Self::SigmoidRouter256 { .. } => K::SigmoidRouter256,
            Self::SumSix => K::SumSix,
        }
    }
    pub fn arity(&self) -> usize {
        match self {
            Self::GatedResidual4 { masked, .. } => 4 + usize::from(*masked),
            Self::RmsNorm { weighted, .. } => 1 + usize::from(*weighted),
            Self::RmsNormPair { .. } => 4,
            Self::ResidualInput4 { .. } | Self::RowGateBlendPair { .. } => 6,
            Self::ResidualExpand4 { add, .. } => 3 + usize::from(*add),
            Self::SumSix | Self::ReluSquared => 1,
            Self::DifferentialSiluGate { .. }
            | Self::Q8ExpertLinear
            | Self::Q8Pair
            | Self::ExpertLinear { .. }
            | Self::WeightedSwiGlu { .. } => 3,
            _ => 2,
        }
    }
    pub fn outputs(&self) -> usize {
        match self {
            Self::RmsNormPair { .. }
            | Self::Q8Pair
            | Self::SigmoidRouter256 { .. }
            | Self::RowGateBlendPair { .. } => 2,
            Self::ResidualInput4 { .. } => 3,
            Self::Router { .. } => 3,
            _ => 1,
        }
    }
    pub fn validate(&self) -> bool {
        let positive = |bits| {
            let x = f32::from_bits(bits);
            x.is_finite() && x > 0.
        };
        match self {
            Self::GatedResidual4 { epsilon, .. } => positive(*epsilon),
            Self::RmsNorm {
                weighted,
                round_output,
                epsilon,
            } => positive(*epsilon) && (!round_output || *weighted),
            Self::RmsNormPair { epsilon } => positive(*epsilon),
            Self::ResidualInput4 {
                epsilon,
                norm_epsilon,
            } => positive(*epsilon) && positive(*norm_epsilon),
            Self::Q8Linear {
                round_output,
                round_input,
            } => !round_input || *round_output,
            Self::Router { top_k, scale } => {
                (1..=32).contains(top_k)
                    && f32::from_bits(*scale).is_finite()
                    && f32::from_bits(*scale) > 0.
            }
            Self::SigmoidRouter256 { scale } => positive(*scale),
            Self::DifferentialSiluGate { bias } => f32::from_bits(*bias).is_finite(),
            Self::WeightedSwiGlu { clamp } => {
                f32::from_bits(*clamp).is_finite() && f32::from_bits(*clamp) >= 0.
            }
            _ => true,
        }
    }
}
