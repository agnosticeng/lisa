//! Original affine Q4/group-32 expert and routing primitive laws.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AffineExperts {
    RouterVector,
    RouterMatrix {
        rows: i32,
    },
    Route {
        top_k: i32,
        experts: i32,
        hidden: i32,
        rows: i32,
    },
    GateUp {
        top_k: i32,
        hidden: i32,
        intermediate: i32,
        rows: i32,
        wide: bool,
    },
    DownCombine {
        top_k: i32,
        hidden: i32,
        intermediate: i32,
        rows: i32,
        wide: bool,
    },
    CountingSort {
        experts: i32,
        top_k: i32,
    },
    Tiles {
        rows: i32,
        experts: i32,
    },
    TiledGateUp {
        rows: i32,
        experts: i32,
        hidden: i32,
        intermediate: i32,
    },
    TiledDown {
        rows: i32,
        experts: i32,
        hidden: i32,
        intermediate: i32,
    },
    Gather,
    SortedCombine {
        top_k: i32,
        hidden: i32,
        rows: i32,
    },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AffineExpertsKind {
    RouterVector,
    RouterMatrix,
    Route,
    GateUp(bool),
    DownCombine(bool),
    CountingSort,
    Tiles,
    TiledGateUp,
    TiledDown,
    Gather,
    SortedCombine,
}
impl AffineExperts {
    pub fn kind(&self) -> AffineExpertsKind {
        use AffineExpertsKind as K;
        match self {
            Self::RouterVector => K::RouterVector,
            Self::RouterMatrix { .. } => K::RouterMatrix,
            Self::Route { .. } => K::Route,
            Self::GateUp { wide, .. } => K::GateUp(*wide),
            Self::DownCombine { wide, .. } => K::DownCombine(*wide),
            Self::CountingSort { .. } => K::CountingSort,
            Self::Tiles { .. } => K::Tiles,
            Self::TiledGateUp { .. } => K::TiledGateUp,
            Self::TiledDown { .. } => K::TiledDown,
            Self::Gather => K::Gather,
            Self::SortedCombine { .. } => K::SortedCombine,
        }
    }
    pub fn arity(&self) -> usize {
        match self {
            Self::RouterVector | Self::RouterMatrix { .. } => 2,
            Self::Route { .. } | Self::Gather | Self::SortedCombine { .. } => 5,
            Self::GateUp { .. } => 12,
            Self::DownCombine { .. } | Self::TiledGateUp { .. } => 10,
            Self::CountingSort { .. } | Self::Tiles { .. } => 1,
            Self::TiledDown { .. } => 6,
        }
    }
    pub fn outputs(&self) -> usize {
        if matches!(self, Self::Route { .. } | Self::CountingSort { .. }) {
            3
        } else {
            1
        }
    }
    pub fn validate(&self) -> bool {
        let assignments = |rows: i32, k: i32| {
            rows > 0 && k > 0 && k.checked_add(1).and_then(|n| rows.checked_mul(n)).is_some()
        };
        let widths = |h: i32, n: i32| {
            h > 0 && n > 0 && h % 128 == 0 && n % 32 == 0 && n.checked_mul(2).is_some()
        };
        match self {
            Self::RouterVector | Self::Gather => true,
            Self::RouterMatrix { rows } => (1..=1024).contains(rows),
            Self::Route {
                top_k,
                experts,
                hidden,
                rows,
            } => {
                assignments(*rows, *top_k)
                    && *experts > 0
                    && experts % 128 == 0
                    && *top_k <= *experts
                    && *top_k <= 32
                    && *hidden > 0
                    && hidden % 32 == 0
            }
            Self::GateUp {
                top_k,
                hidden,
                intermediate,
                rows,
                ..
            }
            | Self::DownCombine {
                top_k,
                hidden,
                intermediate,
                rows,
                ..
            } => assignments(*rows, *top_k) && widths(*hidden, *intermediate),
            Self::CountingSort { experts, top_k } => {
                *experts > 0 && *experts <= 4096 && experts % 256 == 0 && *top_k > 0
            }
            Self::Tiles { rows, experts } => {
                *rows > 0
                    && *experts > 0
                    && rows
                        .checked_add(31)
                        .and_then(|n| (n / 32).checked_add(*experts))
                        .is_some_and(|n| n <= i32::MAX / 2)
            }
            Self::TiledGateUp {
                rows,
                experts,
                hidden,
                intermediate,
            }
            | Self::TiledDown {
                rows,
                experts,
                hidden,
                intermediate,
            } => {
                *rows > 0
                    && *rows <= i32::MAX - 4096
                    && *experts > 0
                    && *experts <= 4096
                    && widths(*hidden, *intermediate)
                    && intermediate % 64 == 0
            }
            Self::SortedCombine {
                top_k,
                hidden,
                rows,
            } => assignments(*rows, *top_k) && *hidden > 0,
        }
    }
}
