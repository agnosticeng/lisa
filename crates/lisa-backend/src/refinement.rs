//! Independent normalization, causal-write and integer-selection contracts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refinement {
    Normalize { epsilon: u32, read: bool },
    NormalizeCodes { epsilon: u32 },
    FeatureMessage { batch: i32, rows: i32 },
    FeatureUpdate,
    Candidates { count: usize },
    GreedyChain,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RefinementKind {
    Normalize,
    NormalizeCodes,
    FeatureMessage,
    FeatureUpdate,
    Candidates(usize),
    GreedyChain,
}
impl Refinement {
    pub fn kind(&self) -> RefinementKind {
        match self {
            Self::Normalize { .. } => RefinementKind::Normalize,
            Self::NormalizeCodes { .. } => RefinementKind::NormalizeCodes,
            Self::FeatureMessage { .. } => RefinementKind::FeatureMessage,
            Self::FeatureUpdate => RefinementKind::FeatureUpdate,
            Self::Candidates { count } => RefinementKind::Candidates(*count),
            Self::GreedyChain => RefinementKind::GreedyChain,
        }
    }
    pub fn arity(&self) -> usize {
        match self {
            Self::Normalize { .. } | Self::FeatureMessage { .. } | Self::GreedyChain => 2,
            Self::NormalizeCodes { .. } | Self::FeatureUpdate => 3,
            Self::Candidates { .. } => 1,
        }
    }
    pub fn outputs(&self) -> usize {
        if matches!(self, Self::Normalize { .. }) {
            2
        } else {
            1
        }
    }
    pub fn validate(&self) -> bool {
        match self {
            Self::Normalize { epsilon, .. } | Self::NormalizeCodes { epsilon } => {
                f32::from_bits(*epsilon).is_finite() && f32::from_bits(*epsilon) > 0.
            }
            Self::FeatureMessage { batch, rows } => *batch > 0 && *rows > 0,
            Self::Candidates { count } => matches!(count, 16 | 128),
            _ => true,
        }
    }
}
