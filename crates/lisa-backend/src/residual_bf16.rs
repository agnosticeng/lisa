//! BF16 gated residual primitives with the original affine group-32/Q4 law.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResidualBf16 {
    InjectNorm {
        hidden: i32,
        streams: i32,
        rows: i32,
        inject: bool,
        tile: bool,
        epsilon: u32,
    },
    DownInject {
        width: i32,
        rank: i32,
        streams: i32,
        rows: i32,
        inject: bool,
    },
    UpMix {
        hidden: i32,
        streams: i32,
        rows: i32,
        inject: bool,
        packed: bool,
    },
    Mix {
        hidden: i32,
        streams: i32,
        rows: i32,
        inject: bool,
    },
    SiluHead {
        width: i32,
    },
    NormalizedProduct {
        hidden: i32,
        streams: i32,
    },
    GatedNorm {
        hidden: i32,
        streams: i32,
    },
    DilatedConvAdd {
        dilation: i32,
    },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidualBf16Kind {
    InjectNorm,
    DownInject,
    UpMix,
    Mix,
    SiluHead,
    NormalizedProduct,
    GatedNorm,
    DilatedConvAdd,
}
impl ResidualBf16 {
    pub fn kind(&self) -> ResidualBf16Kind {
        match self {
            Self::InjectNorm { .. } => ResidualBf16Kind::InjectNorm,
            Self::DownInject { .. } => ResidualBf16Kind::DownInject,
            Self::UpMix { .. } => ResidualBf16Kind::UpMix,
            Self::Mix { .. } => ResidualBf16Kind::Mix,
            Self::SiluHead { .. } => ResidualBf16Kind::SiluHead,
            Self::NormalizedProduct { .. } => ResidualBf16Kind::NormalizedProduct,
            Self::GatedNorm { .. } => ResidualBf16Kind::GatedNorm,
            Self::DilatedConvAdd { .. } => ResidualBf16Kind::DilatedConvAdd,
        }
    }
    pub fn arity(&self) -> usize {
        match self {
            Self::InjectNorm { .. } => 4,
            Self::DownInject { .. } => 7,
            Self::UpMix { .. } => 6,
            Self::Mix { .. } => 3,
            Self::SiluHead { .. } => 1,
            Self::NormalizedProduct { .. } => 5,
            Self::GatedNorm { .. } => 6,
            Self::DilatedConvAdd { .. } => 3,
        }
    }
    pub fn outputs(&self) -> usize {
        match self {
            Self::DownInject { .. } => 3,
            Self::SiluHead { .. }
            | Self::NormalizedProduct { .. }
            | Self::DilatedConvAdd { .. } => 1,
            _ => 2,
        }
    }
    pub fn validate(&self) -> bool {
        match self {
            Self::InjectNorm {
                hidden,
                streams,
                rows,
                epsilon,
                ..
            } => {
                *hidden >= 128
                    && *hidden <= 4096
                    && hidden % 128 == 0
                    && *streams > 0
                    && *rows > 0
                    && hidden.checked_mul(*streams).is_some()
                    && f32::from_bits(*epsilon).is_finite()
                    && f32::from_bits(*epsilon) >= 0.
            }
            Self::DownInject {
                width,
                rank,
                streams,
                rows,
                ..
            } => {
                *width > 0
                    && width % 32 == 0
                    && *rank > 0
                    && rank % 8 == 0
                    && *streams == 4
                    && (1..=8).contains(rows)
            }
            Self::UpMix {
                hidden,
                streams,
                rows,
                packed,
                ..
            } => {
                *hidden > 0
                    && hidden % 2 == 0
                    && hidden.checked_mul(*streams).is_some()
                    && *streams == 4
                    && (1..=8).contains(rows)
                    && (!packed || *rows == 1)
            }
            Self::Mix {
                hidden,
                streams,
                rows,
                ..
            } => *hidden > 0 && *streams > 0 && *rows > 0 && hidden.checked_mul(*streams).is_some(),
            Self::SiluHead { width } => *width > 0,
            Self::NormalizedProduct { hidden, streams } | Self::GatedNorm { hidden, streams } => {
                *hidden >= 128
                    && *hidden <= 4096
                    && hidden % 128 == 0
                    && *streams > 0
                    && hidden.checked_mul(*streams).is_some()
            }
            Self::DilatedConvAdd { dilation } => *dilation > 0,
        }
    }
}
