//! The pinned reference's kernel calibration (`KernelCalibration::of`): the
//! measured constants its kernel selection reads, one row per Apple GPU class
//! and, where its measurements say so, per core range. A pure function of
//! the device's Apple GPU family and GPU cores; the rules that read the rows
//! are in [`super::dense`] (projections), [`super::target`] (attention, MoE
//! and float tiles) and the draft step.
//!
//! Rows of tiles the Inkling step never runs (the int8 decode tiles of
//! rotated PQ2_0 units, the bf16 dense tiles of other models' vision
//! towers) are left out.

/// A "from N" threshold nothing reaches, and a register boundary every grid
/// is below (`kNever`, `kAlways`).
pub(crate) const NEVER: u32 = u32::MAX;
pub(crate) const ALWAYS: u32 = u32::MAX;
/// The core count kernel policy plans for when the device reports none
/// (`DeviceCapabilities::kAssumedGpuCores`).
pub(crate) const ASSUMED_CORES: u32 = 32;

/// The reference's GPU classes (`GpuClass`): Apple9 (M3, M4), Apple10 (M5),
/// Apple11 and newer. The reference has no class below Apple9; M1 and M2
/// (Apple7/8) take Apple9's rows, by core count as an M3 or M4 would.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GpuClass {
    Apple9,
    Apple10,
    Apple11,
}

impl GpuClass {
    /// The class of a device of Apple GPU family `family` (7 or newer).
    pub(crate) fn of(family: u32) -> Self {
        match family {
            11.. => Self::Apple11,
            10 => Self::Apple10,
            _ => Self::Apple9,
        }
    }
}

/// One tier of a tier law (`SplitTier`): asks for more K partitions, up to
/// `splits`, while the grid holds fewer than `per_core` threadgroups a core
/// and each partition keeps `inputs` inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SplitTier {
    pub per_core: u32,
    pub inputs: u32,
    pub splits: u32,
}

/// The register tile's tiers (`kRegisterTiers`).
pub(crate) const REGISTER_TIERS: [SplitTier; 3] = [
    SplitTier {
        per_core: 4,
        inputs: 256,
        splits: 8,
    },
    SplitTier {
        per_core: 32,
        inputs: 1024,
        splits: 8,
    },
    SplitTier {
        per_core: 3,
        inputs: 512,
        splits: 16,
    },
];

/// A staged tile's occupancy (`StagedOccupancy`): threadgroups resident on a
/// core, and the last wave's floor when it is the grid's only one and when
/// it follows others.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Occupancy {
    pub resident: u64,
    pub saturating: u64,
    pub following: u64,
}

/// The staged decode tiles' wave law (`WaveLaw`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WaveLaw {
    /// The single tensor tile and the one-pass gate/up pair's.
    pub tile: Occupancy,
    pub pair: Occupancy,
    /// From these bits per weight (sixteenths) the tile's first wave
    /// saturates from `wide_saturating` threadgroups a core.
    pub wide_bits16: u64,
    pub wide_saturating: u64,
    /// The tile's residency at 16 rows.
    pub sixteen_row_resident: u64,
    /// Partitions keep this many inputs, `maximum_splits` at most.
    pub partition_inputs: u32,
    pub maximum_splits: u32,
}

/// A K-split law (`SplitLaw`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SplitLaw {
    Tiers(&'static [SplitTier]),
    Waves(WaveLaw),
}

/// A unit's format class (`FormatClass`), the row of its register boundary.
/// Every class names its row of the reference's tables; the Inkling step's
/// units (Q8_0, af4g64) store four bits per weight or more.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum FormatClass {
    GridLookup,
    GridLookupOrQ2K,
    UnderFourBits,
    FourBitsOrMore,
}

/// The register tile's boundary (`RegisterBoundary`): a plan of `tiles`
/// table tiles (1 to 4) over `columns` column tiles of a class registers
/// where `4 columns tiles < below_quarters[class][tiles - 1] cores`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RegisterBoundary {
    pub below_quarters: [[u32; 4]; 4],
}

/// The projections' row (`ProjectionCalibration`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ProjectionCalibration {
    pub register_splits: SplitLaw,
    pub staged_splits: SplitLaw,
    pub decode_register: RegisterBoundary,
    /// From this many cores the staged walks spread.
    pub spread_from_cores: u32,
    /// From this many cores a gate/up pair of one format may run in one pass.
    pub one_pass_from_cores: u32,
    /// One unsplit table tile of fewer bits per weight (sixteenths) runs the
    /// staged 16-row tile; 0 never.
    pub sixteen_row_below_bits16: u64,
    /// The most waves of the 16-row tile such a plan runs in from there
    /// under a wave law; 0 never.
    pub sixteen_row_waves: u64,
    /// Prefill chunks past the decode tiles run the int8 prefill tile where
    /// it takes the unit.
    pub prefill_int8: bool,
}

/// The float projections' tile rule (`FloatTileCalibration`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FloatTileCalibration {
    pub neural_accelerator: bool,
    pub minimum_rows: u32,
    pub bf16_halves: u32,
    pub f32_halves: u32,
    pub quantized_rows_halves: u32,
}

/// The relative attention's row (`AttentionCalibration`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AttentionCalibration {
    /// A prompt chunk's tiles take a history split per this many pages.
    pub pages_per_split: u32,
    /// Splits reading at least this many pages score INT8 queries.
    pub int8_queries_from_pages: u32,
}

/// The MoE expert passes' row (`MoeCalibration`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct MoeCalibration {
    /// A prefill chunk whose routes average at least this many an expert
    /// runs 64-row expert tiles; [`NEVER`] never.
    pub wide_tile_routes: u32,
}

/// Every measured constant of kernel selection for one device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Calibration {
    pub class: GpuClass,
    /// The GPU cores kernel policy plans for.
    pub cores: u32,
    pub projection: ProjectionCalibration,
    pub float_tile: FloatTileCalibration,
    pub attention: AttentionCalibration,
    pub moe: MoeCalibration,
}

/// The Apple10 staged tiles' wave law (`kApple10StagedLaw`).
const APPLE10_STAGED_LAW: WaveLaw = WaveLaw {
    tile: Occupancy {
        resident: 10,
        saturating: 6,
        following: 2,
    },
    pair: Occupancy {
        resident: 16,
        saturating: 10,
        following: 3,
    },
    wide_bits16: 100,
    wide_saturating: 6,
    sixteen_row_resident: 9,
    partition_inputs: 512,
    maximum_splits: 8,
};

const APPLE9_DECODE_REGISTER: RegisterBoundary = RegisterBoundary {
    below_quarters: [
        [0, 0, ALWAYS, 0],
        [ALWAYS, 0, ALWAYS, 0],
        [ALWAYS; 4],
        [ALWAYS; 4],
    ],
};
const APPLE10_DECODE_REGISTER: RegisterBoundary = RegisterBoundary {
    below_quarters: [[3; 4], [3; 4], [3; 4], [6; 4]],
};

const APPLE9_PROJECTION: ProjectionCalibration = ProjectionCalibration {
    register_splits: SplitLaw::Tiers(&REGISTER_TIERS),
    // The register tile's first two tiers (`kApple9StagedTiers`).
    staged_splits: SplitLaw::Tiers(REGISTER_TIERS.split_at(2).0),
    decode_register: APPLE9_DECODE_REGISTER,
    spread_from_cores: NEVER,
    one_pass_from_cores: NEVER,
    sixteen_row_below_bits16: 0,
    sixteen_row_waves: 0,
    prefill_int8: false,
};
const APPLE10_PROJECTION: ProjectionCalibration = ProjectionCalibration {
    register_splits: SplitLaw::Tiers(&REGISTER_TIERS),
    staged_splits: SplitLaw::Waves(APPLE10_STAGED_LAW),
    decode_register: APPLE10_DECODE_REGISTER,
    spread_from_cores: 16,
    one_pass_from_cores: 16,
    sixteen_row_below_bits16: 72,
    sixteen_row_waves: 2,
    prefill_int8: true,
};
/// From these cores Apple10's wide units saturate from four threadgroups a
/// core (`kApple10LargeGpuCores`, `kApple10LargeProjection`).
const APPLE10_LARGE_CORES: u32 = 32;
const APPLE10_LARGE_PROJECTION: ProjectionCalibration = ProjectionCalibration {
    staged_splits: SplitLaw::Waves(WaveLaw {
        wide_saturating: 4,
        ..APPLE10_STAGED_LAW
    }),
    ..APPLE10_PROJECTION
};
/// Apple11 keeps 8-row lanes from 4.5 bits (`kApple11Projection`).
const APPLE11_PROJECTION: ProjectionCalibration = ProjectionCalibration {
    sixteen_row_waves: 0,
    ..APPLE10_PROJECTION
};

const APPLE9_FLOAT_TILE: FloatTileCalibration = FloatTileCalibration {
    neural_accelerator: false,
    minimum_rows: 16,
    bf16_halves: 3,
    f32_halves: 4,
    quantized_rows_halves: 6,
};
const APPLE10_FLOAT_TILE: FloatTileCalibration = FloatTileCalibration {
    neural_accelerator: true,
    ..APPLE9_FLOAT_TILE
};

const APPLE10_ATTENTION: AttentionCalibration = AttentionCalibration {
    pages_per_split: 32,
    int8_queries_from_pages: 8,
};
const APPLE9_ATTENTION: AttentionCalibration = AttentionCalibration {
    int8_queries_from_pages: NEVER,
    ..APPLE10_ATTENTION
};
const APPLE9_SMALL_ATTENTION: AttentionCalibration = AttentionCalibration {
    pages_per_split: 64,
    ..APPLE9_ATTENTION
};
/// Apple9 GPUs of up to this many cores take the 10-core M4's rows.
const APPLE9_SMALL_CORES: u32 = 10;

/// The M5 Pro's core range runs 64-row expert tiles: from 20 cores past 64
/// routes an expert, the 16-core past 128.
const APPLE10_PRO_CORES: u32 = 16;
const APPLE10_PRO_MOE_FROM_CORES: u32 = 20;

impl Calibration {
    /// The row of a device of Apple GPU family `family` with `cores` GPU
    /// cores (0: unreported, planned as [`ASSUMED_CORES`]).
    pub(crate) fn of(family: u32, cores: u32) -> Self {
        let cores = if cores == 0 { ASSUMED_CORES } else { cores };
        let class = GpuClass::of(family);
        let attention = match class {
            GpuClass::Apple9 if cores <= APPLE9_SMALL_CORES => APPLE9_SMALL_ATTENTION,
            GpuClass::Apple9 => APPLE9_ATTENTION,
            _ => APPLE10_ATTENTION,
        };
        let pro =
            class == GpuClass::Apple10 && (APPLE10_PRO_CORES..APPLE10_LARGE_CORES).contains(&cores);
        let moe = MoeCalibration {
            wide_tile_routes: if !pro {
                NEVER
            } else if cores < APPLE10_PRO_MOE_FROM_CORES {
                128
            } else {
                64
            },
        };
        let (projection, float_tile) = match class {
            GpuClass::Apple9 => (APPLE9_PROJECTION, APPLE9_FLOAT_TILE),
            GpuClass::Apple10 if cores >= APPLE10_LARGE_CORES => {
                (APPLE10_LARGE_PROJECTION, APPLE10_FLOAT_TILE)
            }
            GpuClass::Apple10 => (APPLE10_PROJECTION, APPLE10_FLOAT_TILE),
            GpuClass::Apple11 => (APPLE11_PROJECTION, APPLE10_FLOAT_TILE),
        };
        Self {
            class,
            cores,
            projection,
            float_tile,
            attention,
            moe,
        }
    }
}

/// A float projection's tile (`FloatTile`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FloatTile {
    Simdgroup,
    NeuralAccelerator,
}

/// The stored type of a float projection's rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum FloatRows {
    F32,
    Bf16,
}

/// The tile of a float projection of `rows` rows into `outputs` columns
/// (`selectFloatTile`).
pub(crate) fn float_tile(c: &Calibration, rows: u32, outputs: u32, stored: FloatRows) -> FloatTile {
    let f = &c.float_tile;
    let tiles = u64::from(rows.div_ceil(64)) * u64::from(outputs.div_ceil(32));
    let halves = match stored {
        FloatRows::Bf16 => f.bf16_halves,
        FloatRows::F32 => f.f32_halves,
    };
    if f.neural_accelerator
        && rows >= f.minimum_rows
        && 2 * tiles >= u64::from(halves) * u64::from(c.cores)
    {
        FloatTile::NeuralAccelerator
    } else {
        FloatTile::Simdgroup
    }
}

/// The expert tile rows of a step of `rows` rows routing `top_k` of
/// `experts` experts (`selectMoe`, no int8 passes: the Inkling experts'
/// formats have none): 8-row tiles while the routes average at most one
/// row an expert, 64-row tiles for a prefill chunk whose routes average the
/// calibrated count, else 32.
pub(crate) fn expert_tile_rows(
    c: &Calibration,
    prefill: bool,
    rows: u32,
    top_k: u32,
    experts: u32,
) -> u32 {
    let routes = u64::from(rows) * u64::from(top_k);
    if routes <= u64::from(experts) {
        8
    } else if prefill && routes >= u64::from(c.moe.wide_tile_routes) * u64::from(experts) {
        64
    } else {
        32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_follow_the_device() {
        // A 40-core M3 Max and a 10-core M4: tier laws, no INT8 queries,
        // 32-row expert tiles; the small GPU splits per 64 pages.
        let m3 = Calibration::of(9, 40);
        assert_eq!(m3.class, GpuClass::Apple9);
        assert_eq!(m3.projection, APPLE9_PROJECTION);
        assert_eq!(m3.attention.pages_per_split, 32);
        assert_eq!(m3.attention.int8_queries_from_pages, NEVER);
        assert_eq!(m3.moe.wide_tile_routes, NEVER);
        assert!(!m3.float_tile.neural_accelerator);
        let m4 = Calibration::of(9, 10);
        assert_eq!(m4.attention.pages_per_split, 64);
        // An 8-core M1 and a 38-core M2 Max take Apple9's rows by cores.
        assert_eq!(
            Calibration {
                cores: 10,
                ..Calibration::of(7, 8)
            },
            m4
        );
        assert_eq!(Calibration::of(8, 38).class, GpuClass::Apple9);
        assert_eq!(Calibration::of(8, 38).attention, m3.attention);
        // A 40-core M5 Max: the large row (wide units saturate from four).
        let max = Calibration::of(10, 40);
        assert_eq!(max.class, GpuClass::Apple10);
        let SplitLaw::Waves(law) = max.projection.staged_splits else {
            panic!("Apple10 stages by its wave law")
        };
        assert_eq!(law.wide_saturating, 4);
        assert_eq!(max.attention.int8_queries_from_pages, 8);
        assert_eq!(max.moe.wide_tile_routes, NEVER);
        // A 20-core M5 Pro: six, and 64-row expert tiles from 64 routes an
        // expert; the 16-core from 128.
        let pro = Calibration::of(10, 20);
        let SplitLaw::Waves(law) = pro.projection.staged_splits else {
            panic!("Apple10 stages by its wave law")
        };
        assert_eq!(law.wide_saturating, 6);
        assert_eq!(pro.moe.wide_tile_routes, 64);
        assert_eq!(Calibration::of(10, 16).moe.wide_tile_routes, 128);
        // A 12-core Apple11: Apple10's row without the 16-row lane from
        // 4.5 bits, at any core count.
        let m6 = Calibration::of(11, 12);
        assert_eq!(m6.projection.sixteen_row_waves, 0);
        assert_eq!(m6.projection.sixteen_row_below_bits16, 72);
        assert_eq!(Calibration::of(11, 40).projection, APPLE11_PROJECTION);
        assert_eq!(Calibration::of(12, 12).class, GpuClass::Apple11);
        // An unreported core count plans for 32.
        assert_eq!(Calibration::of(10, 0).cores, ASSUMED_CORES);
    }

    #[test]
    fn float_tiles_need_rows_and_a_wide_grid() {
        // The router (128 F32 outputs) and the sinks (24) of a 2048-row
        // chunk: the accelerator tile from 64 and 32 tiles' halves.
        let max = Calibration::of(10, 40);
        assert_eq!(
            float_tile(&max, 2048, 128, FloatRows::F32),
            FloatTile::NeuralAccelerator
        );
        assert_eq!(
            float_tile(&max, 2048, 24, FloatRows::F32),
            FloatTile::Simdgroup
        );
        assert_eq!(
            float_tile(&max, 16, 128, FloatRows::F32),
            FloatTile::Simdgroup
        );
        let m6 = Calibration::of(11, 12);
        assert_eq!(
            float_tile(&m6, 2048, 24, FloatRows::Bf16),
            FloatTile::NeuralAccelerator
        );
        assert_eq!(
            float_tile(&Calibration::of(9, 40), 2048, 128, FloatRows::F32),
            FloatTile::Simdgroup
        );
    }

    #[test]
    fn expert_tiles_follow_the_routes() {
        let max = Calibration::of(10, 40);
        assert_eq!(expert_tile_rows(&max, false, 16, 8, 128), 8);
        assert_eq!(expert_tile_rows(&max, true, 17, 8, 128), 32);
        assert_eq!(expert_tile_rows(&max, true, 2048, 8, 128), 32);
        let pro = Calibration::of(10, 20);
        assert_eq!(expert_tile_rows(&pro, true, 1023, 8, 128), 32);
        assert_eq!(expert_tile_rows(&pro, true, 1024, 8, 128), 64);
        assert_eq!(expert_tile_rows(&pro, false, 16, 8, 128), 8);
        let small = Calibration::of(10, 16);
        assert_eq!(expert_tile_rows(&small, true, 1024, 8, 128), 32);
        assert_eq!(expert_tile_rows(&small, true, 2048, 8, 128), 64);
    }
}
