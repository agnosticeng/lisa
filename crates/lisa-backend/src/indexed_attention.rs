//! Ordered scalar indexed-attention primitives, independent of checkpoint policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActivationEncoding {
    Fp8E8M0,
    Fp4E8M0,
    Fp4E4M3,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IndexedAttention {
    RotaryTable {
        position: u32,
        frequencies: [[u32; 32]; 2],
    },
    Rotary {
        position: u32,
        stride: u32,
        frequencies: [u32; 32],
        inverse: bool,
        round: bool,
    },
    RotaryStore {
        row: usize,
        position: u32,
        frequencies: [u32; 32],
        encoding: ActivationEncoding,
    },
    PoolPair,
    IndexQuery,
    IndexKeys,
    IndexScores,
    TopK {
        count: i32,
    },
    Candidates,
    CandidateSelect {
        rows: i32,
    },
    /// Inputs: Q, sinks, raw, rope, four writable scratch arrays, then optional
    /// compressed cache and selection. Scratch is explicit session storage.
    Decode {
        raw_count: u32,
        raw_start: u32,
        compressed: bool,
        selected: bool,
    },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexedAttentionKind {
    RotaryTable,
    Rotary,
    RotaryStore,
    PoolPair,
    IndexQuery,
    IndexKeys,
    IndexScores,
    TopK,
    Candidates,
    CandidateSelect,
    Decode(bool),
}
impl IndexedAttention {
    pub fn kind(&self) -> IndexedAttentionKind {
        use IndexedAttentionKind as K;
        match self {
            Self::RotaryTable { .. } => K::RotaryTable,
            Self::Rotary { .. } => K::Rotary,
            Self::RotaryStore { .. } => K::RotaryStore,
            Self::PoolPair => K::PoolPair,
            Self::IndexQuery => K::IndexQuery,
            Self::IndexKeys => K::IndexKeys,
            Self::IndexScores => K::IndexScores,
            Self::TopK { .. } => K::TopK,
            Self::Candidates => K::Candidates,
            Self::CandidateSelect { .. } => K::CandidateSelect,
            Self::Decode { selected, .. } => K::Decode(*selected),
        }
    }
    pub fn arity(&self) -> usize {
        match self {
            Self::RotaryTable { .. } => 0,
            Self::Rotary { .. } | Self::IndexKeys | Self::TopK { .. } | Self::Candidates => 1,
            Self::RotaryStore { .. } | Self::IndexQuery => 2,
            Self::PoolPair | Self::CandidateSelect { .. } => 4,
            Self::IndexScores => 3,
            Self::Decode { selected, .. } => 8 + 2 * usize::from(*selected),
        }
    }
    pub fn overwritten_inputs(&self) -> &'static [usize] {
        match self {
            Self::Rotary { .. } | Self::IndexQuery => &[0],
            Self::RotaryStore { .. } => &[1],
            Self::Decode { .. } => &[4, 5, 6, 7],
            _ => &[],
        }
    }
    pub fn validate(&self) -> bool {
        let frequencies = |f: &[u32; 32]| {
            f.iter()
                .all(|&x| f32::from_bits(x).is_finite() && f32::from_bits(x) > 0.)
        };
        match self {
            Self::RotaryTable {
                position,
                frequencies: f,
            } => *position < 1048576 && f.iter().all(frequencies),
            Self::Rotary {
                position,
                stride,
                frequencies: f,
                ..
            } => *position < 1048576 && *stride > 0 && frequencies(f),
            Self::RotaryStore {
                position,
                frequencies: f,
                ..
            } => *position < 1048576 && frequencies(f),
            Self::TopK { count } => *count > 0 && *count <= 1048576,
            Self::CandidateSelect { rows } => *rows > 16384 && *rows <= 1048576,
            Self::Decode {
                raw_count,
                raw_start,
                ..
            } => *raw_count > 0 && *raw_count <= 128 && *raw_start < 128,
            _ => true,
        }
    }
}
