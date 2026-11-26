use super::*;
use DenseTile::*;
const D: StepKind = StepKind::Decode;
const P: StepKind = StepKind::Prefill;

fn select(calibration: &Calibration, kind: StepKind, int8: bool) -> Selection<'_> {
    Selection {
        calibration,
        kind,
        int8,
    }
}

/// A projection's selection: outputs, inputs, unit, int8 activations.
type Shape = (u32, u32, Unit, bool);
const PACKED: Shape = (PACKED_WIDTH, HIDDEN, Unit::Q80, true);
const KV: Shape = (KV_WIDTH, HIDDEN, Unit::Q80, true);
const OUT: Shape = (HIDDEN, ATTENTION_WIDTH, Unit::Q80, true);
const HEAD: Shape = (VOCABULARY, HIDDEN, Unit::Q80, true);
/// The draft's context projection and its key/value rows (af4g64, bf16
/// activations).
const CONTEXT: Shape = (
    HIDDEN,
    4 * HIDDEN,
    Unit {
        bits16: 72,
        class: FormatClass::FourBitsOrMore,
        register_takes: true,
        prefill_int8_takes: true,
    },
    false,
);
const CONTEXT_KV: Shape = (
    2048,
    HIDDEN,
    Unit {
        bits16: 72,
        class: FormatClass::FourBitsOrMore,
        register_takes: true,
        prefill_int8_takes: true,
    },
    false,
);

/// The reference's own choices (its `KernelCalibration::of` and
/// `selectProjection` over these devices, shapes and rows): Apple GPU
/// family, cores, shape, step kind, rows, tile.
#[rustfmt::skip]
const REFERENCE: &[(u32, u32, Shape, StepKind, u32, DenseTile)] = &[
    (9, 40, PACKED, D, 8, Register { splits: 2 }),
    (9, 40, PACKED, D, 16, Register { splits: 2 }),
    (9, 40, PACKED, P, 2, Column),
    (9, 40, PACKED, P, 3, Register { splits: 2 }),
    (9, 40, PACKED, P, 8, Register { splits: 2 }),
    (9, 40, PACKED, P, 9, Register { splits: 2 }),
    (9, 40, PACKED, P, 16, Register { splits: 2 }),
    (9, 40, PACKED, P, 17, Register { splits: 2 }),
    (9, 40, PACKED, P, 24, Register { splits: 2 }),
    (9, 40, PACKED, P, 33, Staged { rows: 48, splits: 2, spread: false }),
    (9, 40, PACKED, P, 42, Staged { rows: 48, splits: 2, spread: false }),
    (9, 40, PACKED, P, 64, Staged { rows: 64, splits: 2, spread: false }),
    (9, 40, PACKED, P, 65, Prefill),
    (9, 40, PACKED, P, 2048, Prefill),
    (9, 40, KV, D, 8, Register { splits: 8 }),
    (9, 40, KV, D, 16, Register { splits: 8 }),
    (9, 40, KV, P, 2, Column),
    (9, 40, KV, P, 3, Register { splits: 8 }),
    (9, 40, KV, P, 8, Register { splits: 8 }),
    (9, 40, KV, P, 9, Register { splits: 8 }),
    (9, 40, KV, P, 16, Register { splits: 8 }),
    (9, 40, KV, P, 17, Register { splits: 8 }),
    (9, 40, KV, P, 24, Register { splits: 8 }),
    (9, 40, KV, P, 33, Staged { rows: 48, splits: 8, spread: false }),
    (9, 40, KV, P, 42, Staged { rows: 48, splits: 8, spread: false }),
    (9, 40, KV, P, 64, Staged { rows: 64, splits: 8, spread: false }),
    (9, 40, KV, P, 65, Prefill),
    (9, 40, KV, P, 2048, Prefill),
    (9, 40, OUT, D, 8, Register { splits: 8 }),
    (9, 40, OUT, D, 16, Register { splits: 8 }),
    (9, 40, OUT, P, 2, Column),
    (9, 40, OUT, P, 3, Register { splits: 8 }),
    (9, 40, OUT, P, 8, Register { splits: 8 }),
    (9, 40, OUT, P, 9, Register { splits: 8 }),
    (9, 40, OUT, P, 16, Register { splits: 8 }),
    (9, 40, OUT, P, 17, Register { splits: 8 }),
    (9, 40, OUT, P, 24, Register { splits: 8 }),
    (9, 40, OUT, P, 33, Staged { rows: 48, splits: 8, spread: false }),
    (9, 40, OUT, P, 42, Staged { rows: 48, splits: 8, spread: false }),
    (9, 40, OUT, P, 64, Staged { rows: 64, splits: 8, spread: false }),
    (9, 40, OUT, P, 65, Prefill),
    (9, 40, OUT, P, 2048, Prefill),
    (9, 40, HEAD, D, 8, Register { splits: 1 }),
    (9, 40, HEAD, D, 16, Register { splits: 1 }),
    (9, 40, HEAD, P, 2, Column),
    (9, 40, HEAD, P, 3, Register { splits: 1 }),
    (9, 40, HEAD, P, 8, Register { splits: 1 }),
    (9, 40, HEAD, P, 9, Register { splits: 1 }),
    (9, 40, HEAD, P, 16, Register { splits: 1 }),
    (9, 40, HEAD, P, 17, Register { splits: 1 }),
    (9, 40, HEAD, P, 24, Register { splits: 1 }),
    (9, 40, HEAD, P, 33, Staged { rows: 48, splits: 1, spread: false }),
    (9, 40, HEAD, P, 42, Staged { rows: 48, splits: 1, spread: false }),
    (9, 40, HEAD, P, 64, Staged { rows: 64, splits: 1, spread: false }),
    (9, 40, HEAD, P, 65, Prefill),
    (9, 40, HEAD, P, 2048, Prefill),
    (9, 40, CONTEXT, D, 8, Register { splits: 8 }),
    (9, 40, CONTEXT, D, 16, Register { splits: 8 }),
    (9, 40, CONTEXT, P, 2, Column),
    (9, 40, CONTEXT, P, 3, Register { splits: 8 }),
    (9, 40, CONTEXT, P, 8, Register { splits: 8 }),
    (9, 40, CONTEXT, P, 9, Register { splits: 8 }),
    (9, 40, CONTEXT, P, 16, Register { splits: 8 }),
    (9, 40, CONTEXT, P, 17, Register { splits: 8 }),
    (9, 40, CONTEXT, P, 24, Register { splits: 8 }),
    (9, 40, CONTEXT, P, 33, Staged { rows: 48, splits: 8, spread: false }),
    (9, 40, CONTEXT, P, 42, Staged { rows: 48, splits: 8, spread: false }),
    (9, 40, CONTEXT, P, 64, Staged { rows: 64, splits: 8, spread: false }),
    (9, 40, CONTEXT, P, 65, Prefill),
    (9, 40, CONTEXT, P, 2048, Prefill),
    (9, 40, CONTEXT_KV, D, 8, Register { splits: 8 }),
    (9, 40, CONTEXT_KV, D, 16, Register { splits: 8 }),
    (9, 40, CONTEXT_KV, P, 2, Column),
    (9, 40, CONTEXT_KV, P, 3, Register { splits: 8 }),
    (9, 40, CONTEXT_KV, P, 8, Register { splits: 8 }),
    (9, 40, CONTEXT_KV, P, 9, Register { splits: 8 }),
    (9, 40, CONTEXT_KV, P, 16, Register { splits: 8 }),
    (9, 40, CONTEXT_KV, P, 17, Register { splits: 8 }),
    (9, 40, CONTEXT_KV, P, 24, Register { splits: 8 }),
    (9, 40, CONTEXT_KV, P, 33, Staged { rows: 48, splits: 8, spread: false }),
    (9, 40, CONTEXT_KV, P, 42, Staged { rows: 48, splits: 8, spread: false }),
    (9, 40, CONTEXT_KV, P, 64, Staged { rows: 64, splits: 8, spread: false }),
    (9, 40, CONTEXT_KV, P, 65, Prefill),
    (9, 40, CONTEXT_KV, P, 2048, Prefill),
    (9, 10, PACKED, D, 8, Register { splits: 2 }),
    (9, 10, PACKED, D, 16, Register { splits: 2 }),
    (9, 10, PACKED, P, 2, Column),
    (9, 10, PACKED, P, 3, Register { splits: 2 }),
    (9, 10, PACKED, P, 8, Register { splits: 2 }),
    (9, 10, PACKED, P, 9, Register { splits: 2 }),
    (9, 10, PACKED, P, 16, Register { splits: 2 }),
    (9, 10, PACKED, P, 17, Register { splits: 2 }),
    (9, 10, PACKED, P, 24, Register { splits: 2 }),
    (9, 10, PACKED, P, 33, Staged { rows: 48, splits: 2, spread: false }),
    (9, 10, PACKED, P, 42, Staged { rows: 48, splits: 2, spread: false }),
    (9, 10, PACKED, P, 64, Staged { rows: 64, splits: 2, spread: false }),
    (9, 10, PACKED, P, 65, Prefill),
    (9, 10, PACKED, P, 2048, Prefill),
    (9, 10, KV, D, 8, Register { splits: 4 }),
    (9, 10, KV, D, 16, Register { splits: 4 }),
    (9, 10, KV, P, 2, Column),
    (9, 10, KV, P, 3, Register { splits: 4 }),
    (9, 10, KV, P, 8, Register { splits: 4 }),
    (9, 10, KV, P, 9, Register { splits: 4 }),
    (9, 10, KV, P, 16, Register { splits: 4 }),
    (9, 10, KV, P, 17, Register { splits: 4 }),
    (9, 10, KV, P, 24, Register { splits: 4 }),
    (9, 10, KV, P, 33, Staged { rows: 48, splits: 4, spread: false }),
    (9, 10, KV, P, 42, Staged { rows: 48, splits: 4, spread: false }),
    (9, 10, KV, P, 64, Staged { rows: 64, splits: 4, spread: false }),
    (9, 10, KV, P, 65, Prefill),
    (9, 10, KV, P, 2048, Prefill),
    (9, 10, OUT, D, 8, Register { splits: 2 }),
    (9, 10, OUT, D, 16, Register { splits: 2 }),
    (9, 10, OUT, P, 2, Column),
    (9, 10, OUT, P, 3, Register { splits: 2 }),
    (9, 10, OUT, P, 8, Register { splits: 2 }),
    (9, 10, OUT, P, 9, Register { splits: 2 }),
    (9, 10, OUT, P, 16, Register { splits: 2 }),
    (9, 10, OUT, P, 17, Register { splits: 2 }),
    (9, 10, OUT, P, 24, Register { splits: 2 }),
    (9, 10, OUT, P, 33, Staged { rows: 48, splits: 2, spread: false }),
    (9, 10, OUT, P, 42, Staged { rows: 48, splits: 2, spread: false }),
    (9, 10, OUT, P, 64, Staged { rows: 64, splits: 2, spread: false }),
    (9, 10, OUT, P, 65, Prefill),
    (9, 10, OUT, P, 2048, Prefill),
    (9, 10, HEAD, D, 8, Register { splits: 1 }),
    (9, 10, HEAD, D, 16, Register { splits: 1 }),
    (9, 10, HEAD, P, 2, Column),
    (9, 10, HEAD, P, 3, Register { splits: 1 }),
    (9, 10, HEAD, P, 8, Register { splits: 1 }),
    (9, 10, HEAD, P, 9, Register { splits: 1 }),
    (9, 10, HEAD, P, 16, Register { splits: 1 }),
    (9, 10, HEAD, P, 17, Register { splits: 1 }),
    (9, 10, HEAD, P, 24, Register { splits: 1 }),
    (9, 10, HEAD, P, 33, Staged { rows: 48, splits: 1, spread: false }),
    (9, 10, HEAD, P, 42, Staged { rows: 48, splits: 1, spread: false }),
    (9, 10, HEAD, P, 64, Staged { rows: 64, splits: 1, spread: false }),
    (9, 10, HEAD, P, 65, Prefill),
    (9, 10, HEAD, P, 2048, Prefill),
    (9, 10, CONTEXT, D, 8, Register { splits: 8 }),
    (9, 10, CONTEXT, D, 16, Register { splits: 8 }),
    (9, 10, CONTEXT, P, 2, Column),
    (9, 10, CONTEXT, P, 3, Register { splits: 8 }),
    (9, 10, CONTEXT, P, 8, Register { splits: 8 }),
    (9, 10, CONTEXT, P, 9, Register { splits: 8 }),
    (9, 10, CONTEXT, P, 16, Register { splits: 8 }),
    (9, 10, CONTEXT, P, 17, Register { splits: 8 }),
    (9, 10, CONTEXT, P, 24, Register { splits: 8 }),
    (9, 10, CONTEXT, P, 33, Staged { rows: 48, splits: 8, spread: false }),
    (9, 10, CONTEXT, P, 42, Staged { rows: 48, splits: 8, spread: false }),
    (9, 10, CONTEXT, P, 64, Staged { rows: 64, splits: 8, spread: false }),
    (9, 10, CONTEXT, P, 65, Prefill),
    (9, 10, CONTEXT, P, 2048, Prefill),
    (9, 10, CONTEXT_KV, D, 8, Register { splits: 2 }),
    (9, 10, CONTEXT_KV, D, 16, Register { splits: 2 }),
    (9, 10, CONTEXT_KV, P, 2, Column),
    (9, 10, CONTEXT_KV, P, 3, Register { splits: 2 }),
    (9, 10, CONTEXT_KV, P, 8, Register { splits: 2 }),
    (9, 10, CONTEXT_KV, P, 9, Register { splits: 2 }),
    (9, 10, CONTEXT_KV, P, 16, Register { splits: 2 }),
    (9, 10, CONTEXT_KV, P, 17, Register { splits: 2 }),
    (9, 10, CONTEXT_KV, P, 24, Register { splits: 2 }),
    (9, 10, CONTEXT_KV, P, 33, Staged { rows: 48, splits: 2, spread: false }),
    (9, 10, CONTEXT_KV, P, 42, Staged { rows: 48, splits: 2, spread: false }),
    (9, 10, CONTEXT_KV, P, 64, Staged { rows: 64, splits: 2, spread: false }),
    (9, 10, CONTEXT_KV, P, 65, Prefill),
    (9, 10, CONTEXT_KV, P, 2048, Prefill),
    (10, 40, PACKED, D, 8, Staged { rows: 8, splits: 2, spread: true }),
    (10, 40, PACKED, D, 16, Staged { rows: 16, splits: 2, spread: true }),
    (10, 40, PACKED, P, 2, Column),
    (10, 40, PACKED, P, 3, Staged { rows: 8, splits: 2, spread: true }),
    (10, 40, PACKED, P, 8, Staged { rows: 8, splits: 2, spread: true }),
    (10, 40, PACKED, P, 9, Staged { rows: 16, splits: 2, spread: true }),
    (10, 40, PACKED, P, 16, Staged { rows: 16, splits: 2, spread: true }),
    (10, 40, PACKED, P, 17, Staged { rows: 32, splits: 2, spread: true }),
    (10, 40, PACKED, P, 24, Staged { rows: 32, splits: 2, spread: true }),
    (10, 40, PACKED, P, 33, Staged { rows: 48, splits: 2, spread: true }),
    (10, 40, PACKED, P, 42, Staged { rows: 48, splits: 2, spread: true }),
    (10, 40, PACKED, P, 64, Staged { rows: 64, splits: 2, spread: true }),
    (10, 40, PACKED, P, 65, PrefillInt8),
    (10, 40, PACKED, P, 2048, PrefillInt8),
    (10, 40, KV, D, 8, Register { splits: 8 }),
    (10, 40, KV, D, 16, Register { splits: 8 }),
    (10, 40, KV, P, 2, Column),
    (10, 40, KV, P, 3, Register { splits: 8 }),
    (10, 40, KV, P, 8, Register { splits: 8 }),
    (10, 40, KV, P, 9, Register { splits: 8 }),
    (10, 40, KV, P, 16, Register { splits: 8 }),
    (10, 40, KV, P, 17, Register { splits: 8 }),
    (10, 40, KV, P, 24, Register { splits: 8 }),
    (10, 40, KV, P, 33, Staged { rows: 48, splits: 4, spread: true }),
    (10, 40, KV, P, 42, Staged { rows: 48, splits: 4, spread: true }),
    (10, 40, KV, P, 64, Staged { rows: 64, splits: 4, spread: true }),
    (10, 40, KV, P, 65, PrefillInt8),
    (10, 40, KV, P, 2048, PrefillInt8),
    (10, 40, OUT, D, 8, Register { splits: 8 }),
    (10, 40, OUT, D, 16, Staged { rows: 16, splits: 4, spread: true }),
    (10, 40, OUT, P, 2, Column),
    (10, 40, OUT, P, 3, Register { splits: 8 }),
    (10, 40, OUT, P, 8, Register { splits: 8 }),
    (10, 40, OUT, P, 9, Staged { rows: 16, splits: 4, spread: true }),
    (10, 40, OUT, P, 16, Staged { rows: 16, splits: 4, spread: true }),
    (10, 40, OUT, P, 17, Staged { rows: 32, splits: 4, spread: true }),
    (10, 40, OUT, P, 24, Staged { rows: 32, splits: 4, spread: true }),
    (10, 40, OUT, P, 33, Staged { rows: 48, splits: 4, spread: true }),
    (10, 40, OUT, P, 42, Staged { rows: 48, splits: 4, spread: true }),
    (10, 40, OUT, P, 64, Staged { rows: 64, splits: 4, spread: true }),
    (10, 40, OUT, P, 65, PrefillInt8),
    (10, 40, OUT, P, 2048, PrefillInt8),
    (10, 40, HEAD, D, 8, Staged { rows: 8, splits: 1, spread: true }),
    (10, 40, HEAD, D, 16, Staged { rows: 16, splits: 1, spread: true }),
    (10, 40, HEAD, P, 2, Column),
    (10, 40, HEAD, P, 3, Staged { rows: 8, splits: 1, spread: true }),
    (10, 40, HEAD, P, 8, Staged { rows: 8, splits: 1, spread: true }),
    (10, 40, HEAD, P, 9, Staged { rows: 16, splits: 1, spread: true }),
    (10, 40, HEAD, P, 16, Staged { rows: 16, splits: 1, spread: true }),
    (10, 40, HEAD, P, 17, Staged { rows: 32, splits: 1, spread: true }),
    (10, 40, HEAD, P, 24, Staged { rows: 32, splits: 1, spread: true }),
    (10, 40, HEAD, P, 33, Staged { rows: 48, splits: 1, spread: true }),
    (10, 40, HEAD, P, 42, Staged { rows: 48, splits: 1, spread: true }),
    (10, 40, HEAD, P, 64, Staged { rows: 64, splits: 1, spread: true }),
    (10, 40, HEAD, P, 65, PrefillInt8),
    (10, 40, HEAD, P, 2048, PrefillInt8),
    (10, 40, CONTEXT, D, 8, Register { splits: 8 }),
    (10, 40, CONTEXT, D, 16, Staged { rows: 16, splits: 8, spread: true }),
    (10, 40, CONTEXT, P, 2, Column),
    (10, 40, CONTEXT, P, 3, Register { splits: 8 }),
    (10, 40, CONTEXT, P, 8, Register { splits: 8 }),
    (10, 40, CONTEXT, P, 9, Staged { rows: 16, splits: 8, spread: true }),
    (10, 40, CONTEXT, P, 16, Staged { rows: 16, splits: 8, spread: true }),
    (10, 40, CONTEXT, P, 17, Staged { rows: 32, splits: 8, spread: true }),
    (10, 40, CONTEXT, P, 24, Staged { rows: 32, splits: 8, spread: true }),
    (10, 40, CONTEXT, P, 33, Staged { rows: 48, splits: 8, spread: true }),
    (10, 40, CONTEXT, P, 42, Staged { rows: 48, splits: 8, spread: true }),
    (10, 40, CONTEXT, P, 64, Staged { rows: 64, splits: 8, spread: true }),
    (10, 40, CONTEXT, P, 65, Prefill),
    (10, 40, CONTEXT, P, 2048, Prefill),
    (10, 40, CONTEXT_KV, D, 8, Register { splits: 8 }),
    (10, 40, CONTEXT_KV, D, 16, Staged { rows: 16, splits: 4, spread: true }),
    (10, 40, CONTEXT_KV, P, 2, Column),
    (10, 40, CONTEXT_KV, P, 3, Register { splits: 8 }),
    (10, 40, CONTEXT_KV, P, 8, Register { splits: 8 }),
    (10, 40, CONTEXT_KV, P, 9, Staged { rows: 16, splits: 4, spread: true }),
    (10, 40, CONTEXT_KV, P, 16, Staged { rows: 16, splits: 4, spread: true }),
    (10, 40, CONTEXT_KV, P, 17, Staged { rows: 32, splits: 4, spread: true }),
    (10, 40, CONTEXT_KV, P, 24, Staged { rows: 32, splits: 4, spread: true }),
    (10, 40, CONTEXT_KV, P, 33, Staged { rows: 48, splits: 4, spread: true }),
    (10, 40, CONTEXT_KV, P, 42, Staged { rows: 48, splits: 4, spread: true }),
    (10, 40, CONTEXT_KV, P, 64, Staged { rows: 64, splits: 4, spread: true }),
    (10, 40, CONTEXT_KV, P, 65, Prefill),
    (10, 40, CONTEXT_KV, P, 2048, Prefill),
    (10, 20, PACKED, D, 8, Staged { rows: 16, splits: 1, spread: true }),
    (10, 20, PACKED, D, 16, Staged { rows: 16, splits: 1, spread: true }),
    (10, 20, PACKED, P, 2, Column),
    (10, 20, PACKED, P, 3, Staged { rows: 16, splits: 1, spread: true }),
    (10, 20, PACKED, P, 8, Staged { rows: 16, splits: 1, spread: true }),
    (10, 20, PACKED, P, 9, Staged { rows: 16, splits: 1, spread: true }),
    (10, 20, PACKED, P, 16, Staged { rows: 16, splits: 1, spread: true }),
    (10, 20, PACKED, P, 17, Staged { rows: 32, splits: 1, spread: true }),
    (10, 20, PACKED, P, 24, Staged { rows: 32, splits: 1, spread: true }),
    (10, 20, PACKED, P, 33, Staged { rows: 48, splits: 1, spread: true }),
    (10, 20, PACKED, P, 42, Staged { rows: 48, splits: 1, spread: true }),
    (10, 20, PACKED, P, 64, Staged { rows: 64, splits: 1, spread: true }),
    (10, 20, PACKED, P, 65, PrefillInt8),
    (10, 20, PACKED, P, 2048, PrefillInt8),
    (10, 20, KV, D, 8, Register { splits: 8 }),
    (10, 20, KV, D, 16, Staged { rows: 16, splits: 4, spread: true }),
    (10, 20, KV, P, 2, Column),
    (10, 20, KV, P, 3, Register { splits: 8 }),
    (10, 20, KV, P, 8, Register { splits: 8 }),
    (10, 20, KV, P, 9, Staged { rows: 16, splits: 4, spread: true }),
    (10, 20, KV, P, 16, Staged { rows: 16, splits: 4, spread: true }),
    (10, 20, KV, P, 17, Staged { rows: 32, splits: 4, spread: true }),
    (10, 20, KV, P, 24, Staged { rows: 32, splits: 4, spread: true }),
    (10, 20, KV, P, 33, Staged { rows: 48, splits: 4, spread: true }),
    (10, 20, KV, P, 42, Staged { rows: 48, splits: 4, spread: true }),
    (10, 20, KV, P, 64, Staged { rows: 64, splits: 4, spread: true }),
    (10, 20, KV, P, 65, PrefillInt8),
    (10, 20, KV, P, 2048, PrefillInt8),
    (10, 20, OUT, D, 8, Staged { rows: 8, splits: 4, spread: true }),
    (10, 20, OUT, D, 16, Staged { rows: 16, splits: 4, spread: true }),
    (10, 20, OUT, P, 2, Column),
    (10, 20, OUT, P, 3, Staged { rows: 8, splits: 4, spread: true }),
    (10, 20, OUT, P, 8, Staged { rows: 8, splits: 4, spread: true }),
    (10, 20, OUT, P, 9, Staged { rows: 16, splits: 4, spread: true }),
    (10, 20, OUT, P, 16, Staged { rows: 16, splits: 4, spread: true }),
    (10, 20, OUT, P, 17, Staged { rows: 32, splits: 4, spread: true }),
    (10, 20, OUT, P, 24, Staged { rows: 32, splits: 4, spread: true }),
    (10, 20, OUT, P, 33, Staged { rows: 48, splits: 4, spread: true }),
    (10, 20, OUT, P, 42, Staged { rows: 48, splits: 4, spread: true }),
    (10, 20, OUT, P, 64, Staged { rows: 64, splits: 4, spread: true }),
    (10, 20, OUT, P, 65, PrefillInt8),
    (10, 20, OUT, P, 2048, PrefillInt8),
    (10, 20, HEAD, D, 8, Staged { rows: 8, splits: 1, spread: true }),
    (10, 20, HEAD, D, 16, Staged { rows: 16, splits: 1, spread: true }),
    (10, 20, HEAD, P, 2, Column),
    (10, 20, HEAD, P, 3, Staged { rows: 8, splits: 1, spread: true }),
    (10, 20, HEAD, P, 8, Staged { rows: 8, splits: 1, spread: true }),
    (10, 20, HEAD, P, 9, Staged { rows: 16, splits: 1, spread: true }),
    (10, 20, HEAD, P, 16, Staged { rows: 16, splits: 1, spread: true }),
    (10, 20, HEAD, P, 17, Staged { rows: 32, splits: 1, spread: true }),
    (10, 20, HEAD, P, 24, Staged { rows: 32, splits: 1, spread: true }),
    (10, 20, HEAD, P, 33, Staged { rows: 48, splits: 1, spread: true }),
    (10, 20, HEAD, P, 42, Staged { rows: 48, splits: 1, spread: true }),
    (10, 20, HEAD, P, 64, Staged { rows: 64, splits: 1, spread: true }),
    (10, 20, HEAD, P, 65, PrefillInt8),
    (10, 20, HEAD, P, 2048, PrefillInt8),
    (10, 20, CONTEXT, D, 8, Staged { rows: 8, splits: 4, spread: true }),
    (10, 20, CONTEXT, D, 16, Staged { rows: 16, splits: 4, spread: true }),
    (10, 20, CONTEXT, P, 2, Column),
    (10, 20, CONTEXT, P, 3, Staged { rows: 8, splits: 4, spread: true }),
    (10, 20, CONTEXT, P, 8, Staged { rows: 8, splits: 4, spread: true }),
    (10, 20, CONTEXT, P, 9, Staged { rows: 16, splits: 4, spread: true }),
    (10, 20, CONTEXT, P, 16, Staged { rows: 16, splits: 4, spread: true }),
    (10, 20, CONTEXT, P, 17, Staged { rows: 32, splits: 4, spread: true }),
    (10, 20, CONTEXT, P, 24, Staged { rows: 32, splits: 4, spread: true }),
    (10, 20, CONTEXT, P, 33, Staged { rows: 48, splits: 4, spread: true }),
    (10, 20, CONTEXT, P, 42, Staged { rows: 48, splits: 4, spread: true }),
    (10, 20, CONTEXT, P, 64, Staged { rows: 64, splits: 4, spread: true }),
    (10, 20, CONTEXT, P, 65, Prefill),
    (10, 20, CONTEXT, P, 2048, Prefill),
    (10, 20, CONTEXT_KV, D, 8, Staged { rows: 8, splits: 4, spread: true }),
    (10, 20, CONTEXT_KV, D, 16, Staged { rows: 16, splits: 4, spread: true }),
    (10, 20, CONTEXT_KV, P, 2, Column),
    (10, 20, CONTEXT_KV, P, 3, Staged { rows: 8, splits: 4, spread: true }),
    (10, 20, CONTEXT_KV, P, 8, Staged { rows: 8, splits: 4, spread: true }),
    (10, 20, CONTEXT_KV, P, 9, Staged { rows: 16, splits: 4, spread: true }),
    (10, 20, CONTEXT_KV, P, 16, Staged { rows: 16, splits: 4, spread: true }),
    (10, 20, CONTEXT_KV, P, 17, Staged { rows: 32, splits: 4, spread: true }),
    (10, 20, CONTEXT_KV, P, 24, Staged { rows: 32, splits: 4, spread: true }),
    (10, 20, CONTEXT_KV, P, 33, Staged { rows: 48, splits: 4, spread: true }),
    (10, 20, CONTEXT_KV, P, 42, Staged { rows: 48, splits: 4, spread: true }),
    (10, 20, CONTEXT_KV, P, 64, Staged { rows: 64, splits: 4, spread: true }),
    (10, 20, CONTEXT_KV, P, 65, Prefill),
    (10, 20, CONTEXT_KV, P, 2048, Prefill),
    (11, 12, PACKED, D, 8, Staged { rows: 8, splits: 1, spread: false }),
    (11, 12, PACKED, D, 16, Staged { rows: 16, splits: 1, spread: false }),
    (11, 12, PACKED, P, 2, Column),
    (11, 12, PACKED, P, 3, Staged { rows: 8, splits: 1, spread: false }),
    (11, 12, PACKED, P, 8, Staged { rows: 8, splits: 1, spread: false }),
    (11, 12, PACKED, P, 9, Staged { rows: 16, splits: 1, spread: false }),
    (11, 12, PACKED, P, 16, Staged { rows: 16, splits: 1, spread: false }),
    (11, 12, PACKED, P, 17, Staged { rows: 32, splits: 1, spread: false }),
    (11, 12, PACKED, P, 24, Staged { rows: 32, splits: 1, spread: false }),
    (11, 12, PACKED, P, 33, Staged { rows: 48, splits: 1, spread: false }),
    (11, 12, PACKED, P, 42, Staged { rows: 48, splits: 1, spread: false }),
    (11, 12, PACKED, P, 64, Staged { rows: 64, splits: 1, spread: false }),
    (11, 12, PACKED, P, 65, PrefillInt8),
    (11, 12, PACKED, P, 2048, PrefillInt8),
    (11, 12, KV, D, 8, Register { splits: 4 }),
    (11, 12, KV, D, 16, Staged { rows: 16, splits: 4, spread: false }),
    (11, 12, KV, P, 2, Column),
    (11, 12, KV, P, 3, Register { splits: 4 }),
    (11, 12, KV, P, 8, Register { splits: 4 }),
    (11, 12, KV, P, 9, Staged { rows: 16, splits: 4, spread: false }),
    (11, 12, KV, P, 16, Staged { rows: 16, splits: 4, spread: false }),
    (11, 12, KV, P, 17, Staged { rows: 32, splits: 4, spread: false }),
    (11, 12, KV, P, 24, Staged { rows: 32, splits: 4, spread: false }),
    (11, 12, KV, P, 33, Staged { rows: 48, splits: 4, spread: false }),
    (11, 12, KV, P, 42, Staged { rows: 48, splits: 4, spread: false }),
    (11, 12, KV, P, 64, Staged { rows: 64, splits: 4, spread: false }),
    (11, 12, KV, P, 65, PrefillInt8),
    (11, 12, KV, P, 2048, PrefillInt8),
    (11, 12, OUT, D, 8, Staged { rows: 8, splits: 2, spread: false }),
    (11, 12, OUT, D, 16, Staged { rows: 16, splits: 2, spread: false }),
    (11, 12, OUT, P, 2, Column),
    (11, 12, OUT, P, 3, Staged { rows: 8, splits: 2, spread: false }),
    (11, 12, OUT, P, 8, Staged { rows: 8, splits: 2, spread: false }),
    (11, 12, OUT, P, 9, Staged { rows: 16, splits: 2, spread: false }),
    (11, 12, OUT, P, 16, Staged { rows: 16, splits: 2, spread: false }),
    (11, 12, OUT, P, 17, Staged { rows: 32, splits: 2, spread: false }),
    (11, 12, OUT, P, 24, Staged { rows: 32, splits: 2, spread: false }),
    (11, 12, OUT, P, 33, Staged { rows: 48, splits: 2, spread: false }),
    (11, 12, OUT, P, 42, Staged { rows: 48, splits: 2, spread: false }),
    (11, 12, OUT, P, 64, Staged { rows: 64, splits: 2, spread: false }),
    (11, 12, OUT, P, 65, PrefillInt8),
    (11, 12, OUT, P, 2048, PrefillInt8),
    (11, 12, HEAD, D, 8, Staged { rows: 8, splits: 1, spread: false }),
    (11, 12, HEAD, D, 16, Staged { rows: 16, splits: 1, spread: false }),
    (11, 12, HEAD, P, 2, Column),
    (11, 12, HEAD, P, 3, Staged { rows: 8, splits: 1, spread: false }),
    (11, 12, HEAD, P, 8, Staged { rows: 8, splits: 1, spread: false }),
    (11, 12, HEAD, P, 9, Staged { rows: 16, splits: 1, spread: false }),
    (11, 12, HEAD, P, 16, Staged { rows: 16, splits: 1, spread: false }),
    (11, 12, HEAD, P, 17, Staged { rows: 32, splits: 1, spread: false }),
    (11, 12, HEAD, P, 24, Staged { rows: 32, splits: 1, spread: false }),
    (11, 12, HEAD, P, 33, Staged { rows: 48, splits: 1, spread: false }),
    (11, 12, HEAD, P, 42, Staged { rows: 48, splits: 1, spread: false }),
    (11, 12, HEAD, P, 64, Staged { rows: 64, splits: 1, spread: false }),
    (11, 12, HEAD, P, 65, PrefillInt8),
    (11, 12, HEAD, P, 2048, PrefillInt8),
    (11, 12, CONTEXT, D, 8, Staged { rows: 8, splits: 2, spread: false }),
    (11, 12, CONTEXT, D, 16, Staged { rows: 16, splits: 2, spread: false }),
    (11, 12, CONTEXT, P, 2, Column),
    (11, 12, CONTEXT, P, 3, Staged { rows: 8, splits: 2, spread: false }),
    (11, 12, CONTEXT, P, 8, Staged { rows: 8, splits: 2, spread: false }),
    (11, 12, CONTEXT, P, 9, Staged { rows: 16, splits: 2, spread: false }),
    (11, 12, CONTEXT, P, 16, Staged { rows: 16, splits: 2, spread: false }),
    (11, 12, CONTEXT, P, 17, Staged { rows: 32, splits: 2, spread: false }),
    (11, 12, CONTEXT, P, 24, Staged { rows: 32, splits: 2, spread: false }),
    (11, 12, CONTEXT, P, 33, Staged { rows: 48, splits: 2, spread: false }),
    (11, 12, CONTEXT, P, 42, Staged { rows: 48, splits: 2, spread: false }),
    (11, 12, CONTEXT, P, 64, Staged { rows: 64, splits: 2, spread: false }),
    (11, 12, CONTEXT, P, 65, Prefill),
    (11, 12, CONTEXT, P, 2048, Prefill),
    (11, 12, CONTEXT_KV, D, 8, Staged { rows: 8, splits: 2, spread: false }),
    (11, 12, CONTEXT_KV, D, 16, Staged { rows: 16, splits: 2, spread: false }),
    (11, 12, CONTEXT_KV, P, 2, Column),
    (11, 12, CONTEXT_KV, P, 3, Staged { rows: 8, splits: 2, spread: false }),
    (11, 12, CONTEXT_KV, P, 8, Staged { rows: 8, splits: 2, spread: false }),
    (11, 12, CONTEXT_KV, P, 9, Staged { rows: 16, splits: 2, spread: false }),
    (11, 12, CONTEXT_KV, P, 16, Staged { rows: 16, splits: 2, spread: false }),
    (11, 12, CONTEXT_KV, P, 17, Staged { rows: 32, splits: 2, spread: false }),
    (11, 12, CONTEXT_KV, P, 24, Staged { rows: 32, splits: 2, spread: false }),
    (11, 12, CONTEXT_KV, P, 33, Staged { rows: 48, splits: 2, spread: false }),
    (11, 12, CONTEXT_KV, P, 42, Staged { rows: 48, splits: 2, spread: false }),
    (11, 12, CONTEXT_KV, P, 64, Staged { rows: 64, splits: 2, spread: false }),
    (11, 12, CONTEXT_KV, P, 65, Prefill),
    (11, 12, CONTEXT_KV, P, 2048, Prefill),
];

#[test]
fn tiles_match_the_reference_on_every_row() {
    for &(family, cores, (n, k, unit, int8), kind, rows, expected) in REFERENCE {
        let calibration = Calibration::of(family, cores);
        assert_eq!(
            select(&calibration, kind, int8).tile(unit, n, k, rows),
            expected,
            "Apple{family} {cores} cores, {n} x {k}, {kind:?} {rows} rows"
        );
    }
}

#[test]
fn forty_core_af4g64_plans_follow_the_wave_law() {
    let calibration = Calibration::of(10, 40);
    let s = select(&calibration, D, false);
    let af4 = Unit::af4g64;
    assert_eq!(af4(2304).bits16, 72);
    assert!(!af4(2304 + 64).register_takes);
    // Narrow grids register; q|k|v stages on four partitions, the
    // wave law's least time (76.3M against 85.9M on two, 159M on one).
    assert_eq!(s.tile(af4(2304), 576, 2304, 8), Register { splits: 8 });
    assert_eq!(s.tile(af4(4096), 2304, 4096, 8), Register { splits: 8 });
    assert_eq!(
        s.tile(af4(2304), 6144, 2304, 8),
        Staged {
            rows: 8,
            splits: 4,
            spread: true
        }
    );
    // The MLP pair in one pass on two partitions (157M against 173M in
    // two passes and 172M in one pass on four).
    assert_eq!(
        s.pair(af4(2304), 6912, 2304, 8),
        PairTile::Staged {
            rows: 8,
            splits: 2,
            one_pass: true,
            spread: true
        }
    );
    assert_eq!(s.pair(af4(2304), 6912, 2304, 2), PairTile::Column);
    // Below 16 cores a pair never runs in one pass.
    let small = Calibration::of(10, 10);
    assert!(matches!(
        select(&small, D, false).pair(af4(2304), 6912, 2304, 8),
        PairTile::Staged {
            one_pass: false,
            ..
        } | PairTile::Register { .. }
    ));
}

#[test]
fn storage_rows_and_split_scratch() {
    assert_eq!(Column.storage_rows(2), 2);
    assert_eq!(Register { splits: 8 }.storage_rows(9), 16);
    assert_eq!(
        Staged {
            rows: 8,
            splits: 2,
            spread: true
        }
        .storage_rows(3),
        8
    );
    assert_eq!(PrefillInt8.storage_rows(65), 128);
    assert_eq!(Prefill.storage_rows(2048), 2048);
    // 40 cores: K|V side by side on the register tile, 16 rows and 8
    // splits, is the largest; the packed q's 103 tiles the most counters.
    let max = Calibration::of(10, 40);
    assert_eq!(
        split_scratch(&select(&max, D, true), DENSE_ROWS),
        (8 * 16 * 2048, 103)
    );
    // Every tile of every row count fits the scratch on other GPUs too.
    for (family, cores) in [
        (9, 8),
        (9, 10),
        (9, 40),
        (10, 16),
        (10, 20),
        (10, 40),
        (11, 12),
    ] {
        let calibration = Calibration::of(family, cores);
        for (kind, most) in [(D, DENSE_ROWS), (P, ROW_GROUP_ROWS)] {
            let s = select(&calibration, kind, true);
            let (partials, counters) = split_scratch(&s, most);
            for shape in STEP_PROJECTIONS {
                for rows in 1..=most {
                    let tile = s.tile(Unit::Q80, shape.selection, shape.k, rows);
                    assert!(tile.storage_rows(rows) <= most.max(16));
                    if tile.splits() > 1 {
                        assert!(
                            u64::from(tile.splits() * tile.storage_rows(rows) * shape.out_stride)
                                <= partials
                        );
                        assert!(u64::from(shape.out_stride / 64) <= counters);
                    }
                }
            }
        }
    }
    let columns: u32 = PACKED_SEGMENTS.iter().map(|s| s.0).sum();
    assert_eq!(columns, super::super::schedule::SINK_COLUMN);
}
