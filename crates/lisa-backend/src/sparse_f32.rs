//! Original F32 sparse-routing and projection reduction contracts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SparseF32 {
    QueryLatentRmsRows {
        epsilon: u32,
    },
    QueryLatentRmsCompact {
        epsilon: u32,
    },
    QueryLatentRmsInto {
        epsilon: u32,
        position: usize,
    },
    AttentionOutput {
        group: usize,
    },
    AttentionScores {
        scale: u32,
    },
    AttentionWeightedValue,
    AffineHeadQuery {
        group: usize,
        native: bool,
    },
    AffineHeadValue {
        group: usize,
        heads: usize,
        native: bool,
        matrix: bool,
    },
    AffineIndexedDecode {
        group: usize,
        selected: usize,
        dense: bool,
        scale: u32,
        native: bool,
        capture: bool,
    },
    IndexedLatentShort {
        selected: usize,
        prefix: usize,
        dense: bool,
        scale: u32,
    },
    FlashLatent {
        position: usize,
        scale: u32,
    },
    AffineGateUp {
        limit: u32,
        capture: bool,
    },
    AffineProjection {
        group_size: i32,
        bits: i32,
        schedule: AffineSchedule,
    },
    OrderedRms {
        epsilon: u32,
    },
    OrderedAddRms {
        epsilon: u32,
    },
    OrderedAddThree,
    Q8HeadQuery,
    Q8HeadValue,
    Q8IndexedDecode {
        dense: bool,
    },
    IndexedLatent {
        scale: u32,
    },
    Pool128 {
        epsilon: u32,
    },
    IndexScores128 {
        position: usize,
        pool_size: usize,
        scale: u32,
        schedule: ScoreSchedule,
    },
    TopPools {
        count: usize,
        shuffle: bool,
    },
    ExpandPools {
        tokens: usize,
        position: usize,
        pool_size: usize,
        tail: bool,
    },
    WeightedRms {
        epsilon: u32,
        latent: bool,
    },
    DeltaDecode {
        heads: usize,
        lower_bound: u32,
        epsilon: u32,
    },
    DeltaPrepare {
        heads: usize,
        lower_bound: u32,
        epsilon: u32,
    },
    DeltaRecurrence {
        capture: bool,
    },
    DeltaOutput {
        heads: usize,
        epsilon: u32,
    },

    Bf16PairLinear,
    Bf16TripleLinear,
    Bf16PackedSix {
        groups: Option<usize>,
    },
    MixRms {
        epsilon: u32,
    },
    MixRmsBf16Linear {
        epsilon: u32,
    },
    SplitSinkhorn4 {
        epsilon: u32,
        iterations: usize,
    },
    SplitNorm4 {
        epsilon: u32,
        norm_epsilon: u32,
        iterations: usize,
    },
    WeightedSum4,
    Expand4,
    Bf16Linear {
        wide: bool,
    },
    RouterLinear,
    RouteSigmoid {
        top_k: usize,
        scale: u32,
    },
    ClampedSwiGlu {
        limit: u32,
    },
    SumEight {
        rows: usize,
        width: usize,
    },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SparseF32Kind {
    QueryLatentRmsRows,
    QueryLatentRmsCompact,
    QueryLatentRmsInto,
    AttentionOutput,
    AttentionScores,
    AttentionWeightedValue,
    AffineHeadQuery,
    AffineHeadValue,
    AffineIndexedDecode,
    IndexedLatentShort,
    FlashLatent,
    AffineGateUp,
    AffineProjection(AffineSchedule),
    OrderedRms,
    OrderedAddRms,
    OrderedAddThree,
    Q8HeadQuery,
    Q8HeadValue,
    Q8IndexedDecode,
    IndexedLatent,
    Pool128,
    IndexScores128,
    TopPools,
    ExpandPools,
    WeightedRms(bool),
    DeltaDecode,
    DeltaPrepare,
    DeltaRecurrence,
    DeltaOutput,
    Bf16PairLinear,
    Bf16TripleLinear,
    Bf16PackedSix,
    MixRms,
    MixRmsBf16Linear,
    SplitSinkhorn4,
    SplitNorm4,
    WeightedSum4,
    Expand4,
    Bf16Linear(bool),
    RouterLinear,
    RouteSigmoid,
    ClampedSwiGlu,
    SumEight,
}
impl SparseF32 {
    pub fn kind(&self) -> SparseF32Kind {
        use SparseF32Kind as K;
        match self {
            Self::QueryLatentRmsRows { .. } => K::QueryLatentRmsRows,
            Self::QueryLatentRmsCompact { .. } => K::QueryLatentRmsCompact,
            Self::QueryLatentRmsInto { .. } => K::QueryLatentRmsInto,
            Self::AttentionOutput { .. } => K::AttentionOutput,
            Self::AttentionScores { .. } => K::AttentionScores,
            Self::AttentionWeightedValue => K::AttentionWeightedValue,
            Self::AffineHeadQuery { .. } => K::AffineHeadQuery,
            Self::AffineHeadValue { .. } => K::AffineHeadValue,
            Self::AffineIndexedDecode { .. } => K::AffineIndexedDecode,
            Self::IndexedLatentShort { .. } => K::IndexedLatentShort,
            Self::FlashLatent { .. } => K::FlashLatent,
            Self::AffineGateUp { .. } => K::AffineGateUp,
            Self::AffineProjection { schedule, .. } => K::AffineProjection(*schedule),
            Self::OrderedRms { .. } => K::OrderedRms,
            Self::OrderedAddRms { .. } => K::OrderedAddRms,
            Self::OrderedAddThree => K::OrderedAddThree,
            Self::Q8HeadQuery => K::Q8HeadQuery,
            Self::Q8HeadValue => K::Q8HeadValue,
            Self::Q8IndexedDecode { .. } => K::Q8IndexedDecode,
            Self::IndexedLatent { .. } => K::IndexedLatent,
            Self::Pool128 { .. } => K::Pool128,
            Self::IndexScores128 { .. } => K::IndexScores128,
            Self::TopPools { .. } => K::TopPools,
            Self::ExpandPools { .. } => K::ExpandPools,
            Self::WeightedRms { latent, .. } => K::WeightedRms(*latent),
            Self::DeltaDecode { .. } => K::DeltaDecode,
            Self::DeltaPrepare { .. } => K::DeltaPrepare,
            Self::DeltaRecurrence { .. } => K::DeltaRecurrence,
            Self::DeltaOutput { .. } => K::DeltaOutput,
            Self::Bf16PairLinear => K::Bf16PairLinear,
            Self::Bf16TripleLinear => K::Bf16TripleLinear,
            Self::Bf16PackedSix { .. } => K::Bf16PackedSix,
            Self::MixRms { .. } => K::MixRms,
            Self::MixRmsBf16Linear { .. } => K::MixRmsBf16Linear,
            Self::SplitSinkhorn4 { .. } => K::SplitSinkhorn4,
            Self::SplitNorm4 { .. } => K::SplitNorm4,
            Self::WeightedSum4 => K::WeightedSum4,
            Self::Expand4 => K::Expand4,
            Self::Bf16Linear { wide } => K::Bf16Linear(*wide),
            Self::RouterLinear => K::RouterLinear,
            Self::RouteSigmoid { .. } => K::RouteSigmoid,
            Self::ClampedSwiGlu { .. } => K::ClampedSwiGlu,
            Self::SumEight { .. } => K::SumEight,
        }
    }
    pub fn arity(&self) -> usize {
        match self {
            Self::QueryLatentRmsRows { .. }
            | Self::QueryLatentRmsCompact { .. }
            | Self::AttentionOutput { .. } => 4,
            Self::QueryLatentRmsInto { .. } => 5,
            Self::AttentionScores { .. } => 3,
            Self::AffineHeadQuery { .. } | Self::AffineHeadValue { .. } => 4,
            Self::AffineIndexedDecode { .. } => 6,
            Self::IndexedLatentShort { .. } => 3,
            Self::AffineGateUp { .. } => 7,
            Self::AffineProjection { .. } => 4,
            Self::OrderedAddRms { .. } | Self::OrderedAddThree => 3,
            Self::Q8IndexedDecode { .. } | Self::IndexedLatent { .. } => 4,
            Self::Pool128 { .. } => 5,
            Self::IndexScores128 { .. } => 3,
            Self::TopPools { .. } | Self::ExpandPools { .. } => 1,
            Self::DeltaDecode { .. } => 14,
            Self::DeltaPrepare { .. } => 10,
            Self::DeltaRecurrence { .. } => 6,
            Self::DeltaOutput { .. } => 3,
            Self::Bf16PairLinear | Self::Bf16TripleLinear => 4,
            Self::Bf16PackedSix { .. } => 7,
            Self::SumEight { .. } | Self::MixRms { .. } => 1,
            Self::SplitSinkhorn4 { .. } => 3,
            Self::SplitNorm4 { .. } => 5,
            Self::Expand4 => 4,
            _ => 2,
        }
    }
    pub fn outputs(&self) -> usize {
        match self {
            Self::QueryLatentRmsRows { .. } | Self::QueryLatentRmsCompact { .. } => 2,
            Self::AffineIndexedDecode { capture, .. } | Self::AffineGateUp { capture, .. } => {
                if *capture {
                    if matches!(self, Self::AffineIndexedDecode { .. }) {
                        2
                    } else {
                        3
                    }
                } else {
                    1
                }
            }
            Self::OrderedAddRms { .. } => 2,
            Self::DeltaDecode { .. } => 3,
            Self::DeltaPrepare { .. } => 5,
            Self::DeltaRecurrence { .. } => 2,
            Self::Bf16PairLinear => 2,
            Self::Bf16TripleLinear | Self::Bf16PackedSix { .. } => 3,
            Self::RouteSigmoid { .. } => 2,
            Self::SplitNorm4 { .. } => 5,
            _ => 1,
        }
    }
    pub fn validate(&self) -> bool {
        let positive = |bits| {
            let x = f32::from_bits(bits);
            x.is_finite() && x > 0.
        };
        match self {
            Self::QueryLatentRmsRows { epsilon } | Self::QueryLatentRmsCompact { epsilon } => {
                positive(*epsilon)
            }
            Self::QueryLatentRmsInto { epsilon, position } => {
                positive(*epsilon) && *position <= i32::MAX as usize
            }
            Self::AttentionOutput { group } => *group > 0 && 16384 % group == 0,
            Self::AttentionScores { scale } => positive(*scale),
            Self::AffineHeadQuery { group, native } => {
                *group > 0 && 512 % group == 0 && (!native || *group == 64)
            }
            Self::AffineHeadValue {
                group,
                heads,
                native,
                matrix,
            } => {
                *group > 0
                    && 512 % group == 0
                    && *heads > 0
                    && *heads <= i32::MAX as usize / 512
                    && (!native || *group == 64 && *heads == 64)
                    && (!matrix || *native)
            }
            Self::AffineIndexedDecode {
                group,
                selected,
                scale,
                native,
                ..
            } => {
                *group > 0
                    && 512 % group == 0
                    && (1..=2051).contains(selected)
                    && positive(*scale)
                    && (!native || *group == 64)
            }
            Self::IndexedLatentShort {
                selected,
                prefix,
                scale,
                ..
            } => {
                *selected > 0
                    && *selected <= i32::MAX as usize
                    && prefix <= selected
                    && positive(*scale)
            }
            Self::FlashLatent { position, scale } => {
                *position <= i32::MAX as usize && positive(*scale)
            }
            Self::AffineProjection {
                group_size,
                bits,
                schedule,
            } => {
                *group_size > 0
                    && matches!(bits, 2 | 4 | 8)
                    && (matches!(
                        schedule,
                        AffineSchedule::Generic | AffineSchedule::IndependentRows
                    ) || *group_size == 64 && *bits == 8)
            }
            Self::IndexedLatent { scale } => positive(*scale),
            Self::OrderedRms { epsilon } | Self::OrderedAddRms { epsilon } => positive(*epsilon),
            Self::Pool128 { epsilon } | Self::WeightedRms { epsilon, .. } => positive(*epsilon),
            Self::IndexScores128 {
                position,
                pool_size,
                scale,
                ..
            } => *position <= i32::MAX as usize && *pool_size == 4 && positive(*scale),
            Self::TopPools { count, .. } => *count > 0 && *count <= i32::MAX as usize,
            Self::ExpandPools {
                tokens,
                position,
                pool_size,
                ..
            } => {
                *tokens > 0
                    && *tokens <= i32::MAX as usize
                    && *position <= i32::MAX as usize
                    && *pool_size > 0
                    && *pool_size <= i32::MAX as usize
            }
            Self::DeltaDecode {
                heads,
                lower_bound,
                epsilon,
            }
            | Self::DeltaPrepare {
                heads,
                lower_bound,
                epsilon,
            } => {
                *heads > 0
                    && *heads <= i32::MAX as usize / 128
                    && f32::from_bits(*lower_bound).is_finite()
                    && f32::from_bits(*lower_bound) < 0.
                    && positive(*epsilon)
            }
            Self::DeltaOutput { heads, epsilon } => {
                *heads > 0 && *heads <= i32::MAX as usize / 128 && positive(*epsilon)
            }
            Self::Bf16PackedSix { groups } => groups.is_none_or(|n| matches!(n, 4 | 8 | 16)),
            Self::MixRms { epsilon } | Self::MixRmsBf16Linear { epsilon } => positive(*epsilon),
            Self::SplitSinkhorn4 {
                epsilon,
                iterations,
            } => positive(*epsilon) && *iterations > 0 && *iterations <= i32::MAX as usize,
            Self::SplitNorm4 {
                epsilon,
                norm_epsilon,
                iterations,
            } => {
                positive(*epsilon)
                    && positive(*norm_epsilon)
                    && *iterations > 0
                    && *iterations <= i32::MAX as usize
            }
            Self::RouteSigmoid { top_k, scale } => {
                (1..=16).contains(top_k)
                    && f32::from_bits(*scale).is_finite()
                    && f32::from_bits(*scale) > 0.
            }
            Self::AffineGateUp { limit, .. } | Self::ClampedSwiGlu { limit } => {
                f32::from_bits(*limit).is_finite() && f32::from_bits(*limit) >= 0.
            }
            Self::SumEight { rows, width } => {
                *rows > 0
                    && *width > 0
                    && *width <= i32::MAX as usize
                    && *rows <= i32::MAX as usize / 256
            }
            _ => true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreSchedule {
    Scalar,
    Rows4,
    Batch,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AffineSchedule {
    Matrix,
    Tiny,
    Vector,
    Generic,
    IndependentRows,
}
