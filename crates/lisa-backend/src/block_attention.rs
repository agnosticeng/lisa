//! Individual BF16 block-selection, normalized rotary and attention contracts.
//! Fixed native geometry is an operation constraint, not a model traversal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockAttention {
    Select {
        offset: i32,
        total: i32,
        blocks: i32,
    },
    Attend {
        offset: i32,
        total: i32,
        scale: u32,
    },
    Prepare {
        heads: i32,
        key_heads: i32,
        width: i32,
        rotary: i32,
        epsilon: u32,
    },
    Gate {
        heads: i32,
        width: i32,
    },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockAttentionKind {
    Select,
    Attend,
    Prepare,
    Gate,
}
impl BlockAttention {
    pub fn kind(&self) -> BlockAttentionKind {
        match self {
            Self::Select { .. } => BlockAttentionKind::Select,
            Self::Attend { .. } => BlockAttentionKind::Attend,
            Self::Prepare { .. } => BlockAttentionKind::Prepare,
            Self::Gate { .. } => BlockAttentionKind::Gate,
        }
    }
    pub fn arity(&self) -> usize {
        match self {
            Self::Select { .. } | Self::Gate { .. } => 2,
            Self::Attend { .. } => 5,
            Self::Prepare { .. } => 7,
        }
    }
    pub fn outputs(&self) -> usize {
        match self {
            Self::Select { .. } => 2,
            Self::Prepare { .. } => 3,
            _ => 1,
        }
    }
    pub fn validate(&self) -> bool {
        match *self {
            Self::Select {
                offset,
                total,
                blocks,
            } => {
                offset >= 0
                    && total > offset
                    && blocks == total / 4
                    && (1..=65536).contains(&blocks)
            }
            Self::Attend {
                offset,
                total,
                scale,
            } => offset >= 0 && total > offset && f32::from_bits(scale).is_finite(),
            Self::Prepare {
                heads,
                key_heads,
                width,
                rotary,
                epsilon,
            } => {
                heads > 0
                    && key_heads > 0
                    && heads % key_heads == 0
                    && heads.checked_mul(width).is_some()
                    && (128..=4096).contains(&width)
                    && width % 128 == 0
                    && rotary > 0
                    && rotary <= width
                    && rotary % 2 == 0
                    && f32::from_bits(epsilon).is_finite()
                    && f32::from_bits(epsilon) >= 0.
            }
            Self::Gate { heads, width } => {
                heads > 0 && width > 0 && heads.checked_mul(width).is_some()
            }
        }
    }
}
