//! The pinned reference's tile selection for a dense projection
//! (`selectProjection` and its `bf16Projection` law) on a device's
//! [`Calibration`] row: past the decode tiles' rows (a decode step's 128, a
//! prompt chunk's 64) the prefill tile, the int8 one where the row, the
//! activations and the unit take it; otherwise the column tile at one or two
//! rows, the register tile where the class's register boundary holds, else
//! the staged tile of 8 to 64 rows (16 for one lane where its sixteen-row
//! rule takes it), K split by the row's tier or wave laws. A [`Unit`]
//! carries what the law reads of a projection's format; [`Selection`] binds
//! the row, the step kind and the activations. Pure functions. Also the
//! split scratch the step's projections need.
use super::{
    calibration::{Calibration, FormatClass, Occupancy, SplitLaw, SplitTier, WaveLaw},
    schedule::{
        ATTENTION_WIDTH, HIDDEN, KV_WIDTH, PACKED_WIDTH, QUERY_COLUMNS, RELATIVE_COLUMNS,
        VOCABULARY,
    },
};

const TILE_COLUMNS: u32 = 64;
/// Most rows a decode step's verify block takes.
pub(crate) const DENSE_ROWS: u32 = 16;
/// Most rows of one staged tile, which a prompt chunk's decode tiles take
/// (`kMaximumDecodeTileRows`), and a decode step's two row groups
/// (`kMaximumStagedRows`).
pub(crate) const ROW_GROUP_ROWS: u32 = 64;
const STAGED_ROWS: u32 = 2 * ROW_GROUP_ROWS;
/// Rows of a prefill tile (`GGUF_PREFILL_ROWS`).
pub(crate) const PREFILL_TILE_ROWS: u32 = 128;
/// Rows a one-pass gate/up pair holds (`kMaximumPairRows`).
const PAIR_ROWS: u32 = 32;

/// Whether a plan is a decode step's or a prompt chunk's (`StepKind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StepKind {
    Decode,
    Prefill,
}

/// What the selection reads of a projection's format: its bits per weight
/// in sixteenths, its class, and whether the register and int8 prefill
/// tiles take its tensors at their inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Unit {
    pub bits16: u64,
    pub class: FormatClass,
    pub register_takes: bool,
    pub prefill_int8_takes: bool,
}

impl Unit {
    /// Q8_0 (34 bytes per 32 weights), which the register tile takes at any
    /// whole 64-input span and the int8 prefill tile takes.
    pub(crate) const Q80: Self = Self {
        bits16: 34 * 128 / 32,
        class: FormatClass::FourBitsOrMore,
        register_takes: true,
        prefill_int8_takes: true,
    };
    /// MLX affine 4-bit in groups of 64 (36 bytes per 64 weights) of `k`
    /// inputs: the register tile takes whole 256-input coefficient units.
    pub(crate) fn af4g64(k: u32) -> Self {
        Self {
            bits16: 36 * 128 / 64,
            class: FormatClass::FourBitsOrMore,
            register_takes: k.is_multiple_of(256),
            prefill_int8_takes: true,
        }
    }
}

/// A tile, its K splits and, for the staged tile, its height and whether
/// its walks spread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DenseTile {
    Column,
    Register {
        splits: u32,
    },
    Staged {
        rows: u32,
        splits: u32,
        spread: bool,
    },
    /// The 128-row bf16 prefill tile.
    Prefill,
    /// The int8 prefill tile over the input's int8 codes.
    PrefillInt8,
}

impl DenseTile {
    /// Rows the tile reads and writes for `rows` live rows.
    pub(crate) fn storage_rows(self, rows: u32) -> u32 {
        match self {
            Self::Column => rows,
            Self::Register { .. } => rows.div_ceil(8) * 8,
            Self::Staged { rows: r, .. } => r,
            Self::Prefill | Self::PrefillInt8 => {
                rows.div_ceil(PREFILL_TILE_ROWS) * PREFILL_TILE_ROWS
            }
        }
    }
    pub(crate) fn splits(self) -> u32 {
        match self {
            Self::Column | Self::Prefill | Self::PrefillInt8 => 1,
            Self::Register { splits } | Self::Staged { splits, .. } => splits,
        }
    }
}

/// A gate/up pair's plan: both tensors on the column tile in one pass, on
/// the register tile as a gate pass and an up pass with the gate, or on the
/// staged tile in one pass or two.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PairTile {
    Column,
    Register {
        splits: u32,
    },
    Staged {
        rows: u32,
        splits: u32,
        one_pass: bool,
        spread: bool,
    },
}

/// The staged tile height of one row group of `rows` rows
/// (`stagedGroupRows`).
fn group_rows(rows: u32) -> u32 {
    match rows {
        0..=8 => 8,
        9..=16 => 16,
        17..=32 => 32,
        33..=48 => 48,
        _ => 64,
    }
}

/// The staged tile's row groups of `rows` rows (`stagedRowGroups`): one up
/// to 64 rows, past them two of the tile of half of them.
fn row_groups(rows: u32) -> (u32, u32) {
    if rows <= ROW_GROUP_ROWS {
        (group_rows(rows), 1)
    } else {
        (group_rows(rows.div_ceil(2)), 2)
    }
}

/// Rows of the staged tile or its two row groups (`stagedTileRows`).
fn staged_tile_rows(rows: u32) -> u32 {
    let (height, groups) = row_groups(rows);
    height * groups
}

fn tier_splits(n: u32, k: u32, cores: u64, tiers: &[SplitTier]) -> u32 {
    let grid = u64::from(n / TILE_COLUMNS);
    let mut splits = 1u32;
    let asks = |splits: u32| {
        tiers.iter().any(|t| {
            splits < t.splits
                && grid * u64::from(splits) < u64::from(t.per_core) * cores
                && k / (2 * splits) >= t.inputs
        })
    };
    while (k / 32).is_multiple_of(2 * splits) && asks(splits) {
        splits *= 2;
    }
    splits
}

fn waves(groups: u64, cores: u64, resident: u64) -> u64 {
    groups.div_ceil(cores).div_ceil(resident)
}

/// The wave law's time of a grid (`stagedTime`).
fn staged_time(
    groups: u64,
    steps: u64,
    splits: u32,
    rows: u64,
    bits16: u64,
    cores: u64,
    tile: Occupancy,
) -> u64 {
    let count = waves(groups, cores, tile.resident);
    let full = (count - 1) * tile.resident * cores;
    let floor = if count > 1 {
        tile.following
    } else {
        tile.saturating
    } * cores;
    let step_bits = u64::from(TILE_COLUMNS) * 32 * bits16 / 16;
    let partial_bits = if splits > 1 {
        2 * rows * u64::from(TILE_COLUMNS) * 32
    } else {
        0
    };
    steps * step_bits * (full + groups.saturating_sub(full).max(floor)) + groups * partial_bits
}

/// A wave law's split of `rows` rows and, for a pair that may run in one
/// pass (`pair`), whether it does (`waveSplits`): least time, ties keeping
/// fewer partitions, then one pass.
#[allow(clippy::too_many_arguments)]
fn wave_splits(
    law: &WaveLaw,
    n: u32,
    k: u32,
    rows: u32,
    bits16: u64,
    cores: u64,
    pair: bool,
) -> (u32, bool) {
    let tile_rows = u64::from(staged_tile_rows(rows));
    let mut tile = law.tile;
    if bits16 >= law.wide_bits16 {
        tile.saturating = law.wide_saturating;
    }
    let (mut best, mut least) = ((1u32, false), u64::MAX);
    let mut splits = 1u32;
    while splits <= law.maximum_splits
        && (k / 32).is_multiple_of(splits)
        && (splits == 1 || k / splits >= law.partition_inputs)
    {
        let steps = u64::from(k / 32 / splits);
        let tiles = u64::from(n / TILE_COLUMNS) * u64::from(splits);
        if pair {
            let time = staged_time(2 * tiles, steps, splits, tile_rows, bits16, cores, law.pair);
            if time < least {
                (best, least) = ((splits, true), time);
            }
        }
        let passes = if pair { 2 } else { 1 };
        let time = passes * staged_time(tiles, steps, splits, tile_rows, bits16, cores, tile);
        if time < least {
            (best, least) = ((splits, false), time);
        }
        splits *= 2;
    }
    best
}

/// A split law's split (`splitsFor`).
fn splits_for(
    law: &SplitLaw,
    n: u32,
    k: u32,
    rows: u32,
    bits16: u64,
    cores: u64,
    pair: bool,
) -> (u32, bool) {
    match law {
        SplitLaw::Tiers(tiers) => (tier_splits(n, k, cores, tiers), false),
        SplitLaw::Waves(waves) => wave_splits(waves, n, k, rows, bits16, cores, pair),
    }
}

/// A projection selection: a device's row, a step kind, and whether the
/// projections take int8 activations (the target's do, the draft's not).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Selection<'a> {
    pub calibration: &'a Calibration,
    pub kind: StepKind,
    pub int8: bool,
}

impl Selection<'_> {
    fn cores(&self) -> u64 {
        u64::from(self.calibration.cores)
    }

    /// Rows the decode tiles take (`decodeTileRows`).
    fn decode_tile_rows(&self) -> u32 {
        match self.kind {
            StepKind::Decode => STAGED_ROWS,
            StepKind::Prefill => ROW_GROUP_ROWS,
        }
    }

    /// Whether the register tile runs `tiles` table tiles over `columns`
    /// column tiles of `unit` (`registers`).
    fn registers(&self, unit: Unit, tiles: u32, columns: u64) -> bool {
        let boundary = &self.calibration.projection.decode_register;
        tiles as usize <= boundary.below_quarters[0].len()
            && 4 * columns * u64::from(tiles)
                < u64::from(boundary.below_quarters[unit.class as usize][tiles as usize - 1])
                    * self.cores()
    }

    /// Whether one unsplit lane of `tiles` column tiles takes the 16-row
    /// tile (`sixteenRowLane`).
    fn sixteen_row_lane(&self, unit: Unit, tiles: u64) -> bool {
        let p = &self.calibration.projection;
        if unit.bits16 == 0 {
            return false;
        }
        if unit.bits16 < p.sixteen_row_below_bits16 {
            return true;
        }
        let SplitLaw::Waves(law) = &p.staged_splits else {
            return false;
        };
        let sixteen = waves(tiles, self.cores(), law.sixteen_row_resident);
        sixteen <= p.sixteen_row_waves && sixteen == waves(tiles, self.cores(), law.tile.resident)
    }

    /// The staged tile of `rows` rows (`staged`) and, for a pair, whether
    /// it runs in one pass.
    fn staged(&self, unit: Unit, n: u32, k: u32, rows: u32, pair: bool) -> (u32, bool) {
        let (height, groups) = row_groups(rows);
        splits_for(
            &self.calibration.projection.staged_splits,
            groups * n,
            k,
            if groups > 1 { height } else { rows },
            unit.bits16,
            self.cores(),
            pair,
        )
    }

    /// The tile of a single tensor of `unit` whose selection sees `n`
    /// outputs and `k` inputs, at `rows` rows (`selectProjection`).
    pub(crate) fn tile(&self, unit: Unit, n: u32, k: u32, rows: u32) -> DenseTile {
        let p = &self.calibration.projection;
        if rows > self.decode_tile_rows() {
            return if self.int8
                && p.prefill_int8
                && self.kind == StepKind::Prefill
                && unit.prefill_int8_takes
            {
                DenseTile::PrefillInt8
            } else {
                DenseTile::Prefill
            };
        }
        if rows <= 2 {
            return DenseTile::Column;
        }
        let columns = u64::from(n / TILE_COLUMNS);
        let tiles = rows.div_ceil(8);
        if unit.register_takes && self.registers(unit, tiles, columns) {
            let (splits, _) = splits_for(
                &p.register_splits,
                n,
                k,
                rows,
                unit.bits16,
                self.cores(),
                false,
            );
            return DenseTile::Register { splits };
        }
        let (splits, _) = self.staged(unit, n, k, rows, false);
        let sixteen = tiles == 1 && splits == 1 && self.sixteen_row_lane(unit, columns);
        DenseTile::Staged {
            rows: if sixteen {
                staged_tile_rows(rows).max(16)
            } else {
                staged_tile_rows(rows)
            },
            splits,
            spread: self.calibration.cores >= p.spread_from_cores,
        }
    }

    /// The plan of a gate/up pair of one `unit` format, `n` outputs each, `k`
    /// inputs, at `rows` rows, at most one row group.
    pub(crate) fn pair(&self, unit: Unit, n: u32, k: u32, rows: u32) -> PairTile {
        let p = &self.calibration.projection;
        debug_assert!(
            rows <= ROW_GROUP_ROWS,
            "a pair past one row group runs two plans"
        );
        if rows <= 2 {
            return PairTile::Column;
        }
        let one_pass = self.calibration.cores >= p.one_pass_from_cores && rows <= PAIR_ROWS;
        let columns = u64::from(n / TILE_COLUMNS);
        let tiles = rows.div_ceil(8);
        let pair_columns = if one_pass { 2 * columns } else { columns };
        if unit.register_takes && self.registers(unit, tiles, pair_columns) {
            let (splits, _) = splits_for(
                &p.register_splits,
                n,
                k,
                rows,
                unit.bits16,
                self.cores(),
                false,
            );
            return PairTile::Register { splits };
        }
        let (splits, one) = self.staged(unit, n, k, rows, one_pass);
        // The 16-row tile where each tensor's own plan takes it.
        let sixteen = tiles == 1
            && splits == 1
            && matches!(
                self.tile(unit, n, k, rows),
                DenseTile::Staged {
                    rows: 16,
                    splits: 1,
                    ..
                }
            );
        PairTile::Staged {
            rows: if sixteen { 16 } else { staged_tile_rows(rows) },
            splits,
            one_pass: one,
            spread: self.calibration.cores >= p.spread_from_cores,
        }
    }
}

/// A projection of the step as its selection sees it: `selection` output
/// columns, `k` inputs, rows `out_stride` wide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct DenseShape {
    pub selection: u32,
    pub k: u32,
    pub out_stride: u32,
}

/// The step's dense projections: the packed q (selected over its padded
/// width), K|V side by side (selected as one tensor), K or V alone, the
/// attention output and the head.
pub(crate) const STEP_PROJECTIONS: [DenseShape; 5] = [
    DenseShape {
        selection: PACKED_WIDTH,
        k: HIDDEN,
        out_stride: PACKED_WIDTH,
    },
    DenseShape {
        selection: KV_WIDTH,
        k: HIDDEN,
        out_stride: 2 * KV_WIDTH,
    },
    DenseShape {
        selection: KV_WIDTH,
        k: HIDDEN,
        out_stride: KV_WIDTH,
    },
    DenseShape {
        selection: HIDDEN,
        k: ATTENTION_WIDTH,
        out_stride: HIDDEN,
    },
    DenseShape {
        selection: VOCABULARY,
        k: HIDDEN,
        out_stride: VOCABULARY,
    },
];

/// Split partials (F32 elements) and counters the step's Q8_0 projections
/// need at 1 to `rows` rows of `selection`'s plans: each split dispatch's
/// partials `[splits][storage rows][out_stride]`, a counter per 64 output
/// columns.
pub(crate) fn split_scratch(selection: &Selection<'_>, rows: u32) -> (u64, u64) {
    let (mut partials, mut counters) = (0u64, 0u64);
    for shape in STEP_PROJECTIONS {
        for r in 1..=rows {
            let tile = selection.tile(Unit::Q80, shape.selection, shape.k, r);
            if tile.splits() > 1 {
                partials = partials.max(
                    u64::from(tile.splits())
                        * u64::from(tile.storage_rows(r))
                        * u64::from(shape.out_stride),
                );
                counters = counters.max(u64::from(shape.out_stride / TILE_COLUMNS));
            }
        }
    }
    (partials.max(1), counters.max(1))
}

/// The packed q projection's Q8_0 segments: queries, gates, relative states.
pub(crate) const PACKED_SEGMENTS: [(u32, u32); 3] = [
    (QUERY_COLUMNS, 0),
    (QUERY_COLUMNS, QUERY_COLUMNS),
    (RELATIVE_COLUMNS, 2 * QUERY_COLUMNS),
];

#[cfg(test)]
#[path = "dense_tests.rs"]
mod tests;
