use super::*;

#[test]
fn plane_views_start_on_their_rows() {
    // The q|k|v planes of 6144 x 2304 af4g64 rows: plane0 then meta, and
    // the key and value rows from row 4096.
    let [p0, meta, total] = plane_offsets(Format::Af4g64, 6144, 2304, 0);
    assert_eq!((p0, meta), (0, 6144 * 72 * 16));
    assert_eq!(total, 6144 * 2304 / 64 * 36);
    let [p0, meta, _] = plane_offsets(Format::Af4g64, 6144, 2304, 4096);
    assert_eq!(p0, 4096 * 72 * 16);
    assert_eq!(meta, 6144 * 72 * 16 + 4096 * 72 * 2);
    // Q8_0: 34 bytes per 32 weights, as the step's dense planes.
    let [_, meta, total] = plane_offsets(Format::Q80, 201024, 2304, 0);
    assert_eq!(
        (meta, total),
        super::super::target::dense_plane_bytes(201024, 2304)
    );
}

#[test]
fn tile_scratch_holds_every_plan_on_any_gpu() {
    for (family, cores) in [
        (9, 10),
        (9, 40),
        (10, 8),
        (10, 16),
        (10, 20),
        (10, 40),
        (11, 12),
    ] {
        let calibration = Calibration::of(family, cores);
        let [table, sums, partials, counters] = tile_scratch(&calibration);
        assert!(table >= 1 && sums >= 1 && partials >= 1 && counters >= 1);
        // The widest register input of the context projections, a prompt
        // chunk's 32 rows of the features, fits the table where that plan
        // registers.
        for rows in [16, 32] {
            let tile =
                selection(&calibration, rows).tile(Unit::af4g64(FEATURES), HIDDEN, FEATURES, rows);
            if let DenseTile::Register { .. } = tile {
                assert!(table >= k::table16_bytes(FEATURES, rows / 8).0 / 2);
            }
        }
    }
}

#[test]
fn forty_core_draft_plans() {
    let calibration = Calibration::of(10, 40);
    let plans = |rows| selection(&calibration, rows);
    let af4 = Unit::af4g64;
    // Eight rows: the narrow projections register, q|k|v stages.
    assert!(matches!(
        plans(8).tile(af4(HIDDEN), DYNAMIC, HIDDEN, 8),
        DenseTile::Register { .. }
    ));
    assert!(matches!(
        plans(8).tile(af4(HIDDEN), k::DRAFT_QKV_WIDTH, HIDDEN, 8),
        DenseTile::Staged { .. }
    ));
    assert!(matches!(
        plans(8).tile(
            af4(k::DRAFT_ATTENTION_WIDTH),
            HIDDEN,
            k::DRAFT_ATTENTION_WIDTH,
            8
        ),
        DenseTile::Register { .. }
    ));
    assert!(matches!(
        plans(8).pair(af4(HIDDEN), INTERMEDIATE, HIDDEN, 8),
        PairTile::Staged { .. }
    ));
    // One or two committed rows run the column tile; a prompt chunk's
    // context past 64 rows the bf16 prefill tile.
    assert_eq!(
        plans(2).tile(af4(FEATURES), HIDDEN, FEATURES, 2),
        DenseTile::Column
    );
    assert_eq!(
        plans(1).pair(af4(HIDDEN), INTERMEDIATE, HIDDEN, 1),
        PairTile::Column
    );
    assert_eq!(
        plans(65).tile(af4(FEATURES), HIDDEN, FEATURES, 65),
        DenseTile::Prefill
    );
}
