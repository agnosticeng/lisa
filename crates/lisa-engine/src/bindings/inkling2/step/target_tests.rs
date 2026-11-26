use super::*;

/// The 40-core M5 Max's row.
fn max() -> Calibration {
    Calibration::of(10, 40)
}

#[test]
fn attention_choice_follows_the_decode_plan() {
    // A fresh lane: one page, one split, bf16 queries.
    assert_eq!(
        attention_choice(0, 8, true, &max()),
        AttentionChoice {
            splits: 1,
            int8_queries: false
        }
    );
    // 4016 visible tokens, 126 pages on a full layer: 12 splits of up to 11
    // pages score INT8 queries; the window's 34 pages do not.
    assert_eq!(
        attention_choice(4000, 16, false, &max()),
        AttentionChoice {
            splits: 12,
            int8_queries: true
        }
    );
    assert_eq!(
        attention_choice(4000, 16, true, &max()),
        AttentionChoice {
            splits: 12,
            int8_queries: false
        }
    );
    // A small GPU's window splits read 9 pages: INT8 queries on sliding
    // layers too, as the reference plans both variants alike.
    assert_eq!(
        attention_choice(4000, 8, true, &Calibration::of(10, 8)),
        AttentionChoice {
            splits: 4,
            int8_queries: true
        }
    );
}

#[test]
fn expert_plane_offsets_match_their_sizes() {
    for (ggml, rows, columns) in [
        (40u32, 1152u32, 2304u32),
        (40, 2304, 1152),
        (13, 1152, 2304),
        (8, 2304, 1152),
        (8, 1152, 2304),
    ] {
        let format = plane_format(ggml, columns).unwrap();
        let (at, total) = plane_offsets(format, 128, u64::from(rows), u64::from(columns));
        assert_eq!(
            total as usize,
            planes::plane_bytes(ggml, 128, rows as usize, columns as usize)
        );
        assert_eq!(at[0], 0);
        assert!(at[1] <= at[2] && at[2] < total);
        // Formats without plane1 name meta in its entry.
        assert_eq!(at[1] == at[2], format.plane1 == 0);
    }
    assert_eq!(plane_format(40, 2304).unwrap().id, 37);
    assert_eq!(plane_format(40, 1152).unwrap().id, 44);
    assert_eq!(plane_format(13, 2304).unwrap().id, 3);
    assert_eq!(plane_format(8, 1152).unwrap().id, k::Q80);
    assert!(plane_format(12, 2304).is_err());
}

#[test]
fn dense_planes_and_scratch_sizes() {
    // The head's planes: 463,159,296 bytes of codes, then the scales.
    assert_eq!(
        dense_plane_bytes(u64::from(VOCABULARY), u64::from(HIDDEN)),
        (463_159_296, 492_106_752)
    );
    let sizes = scratch_sizes(&max(), STORAGE_ROWS, false);
    // 32 splits x 2 tiles x 2 slots, 8 rows x 24 heads x 128 dimensions.
    assert_eq!(sizes.attention_partials, 128 * 192 * 128);
    assert_eq!(sizes.attention_statistics, 128 * 192 * 2);
    assert_eq!(sizes.table_bytes, 3072 * 8 * 2 * 2);
    assert_eq!(sizes.sums, 2 * (3072 / 16 * 8 + 3072 / 32 * 8));
    assert_eq!(
        (sizes.split_partials, sizes.split_counters),
        (8 * 16 * 2048, 103)
    );
    assert_eq!(sizes.maximum_tiles, 128);
    assert_eq!(sizes.grouped_input_bytes, 135 * 2304 * 2);
    assert_eq!(sizes.intermediate, 135 * 1152);
    assert_eq!(sizes.expert_output, 128 * 2304);
    assert_eq!(sizes.gate_low, 3 * 16 * 64);
    assert_eq!(sizes.gate_values, 3 * 16 * 2304);
    assert_eq!(sizes.gate_partials, 3 * 2 * 9 * 8);
    const { assert!(GATE_COUNTER_BYTES >= 3 * 2 * 4) };
}

#[test]
fn convolution_constants() {
    assert_eq!(convolution_divisor(true), 3.0);
    assert_eq!(convolution_divisor(false), 5f64.sqrt() as f32);
    assert_eq!(column(0, KEY_COLUMN), 0);
    assert_eq!(column(1, VALUE_COLUMN), 6656 + 1024);
    assert_eq!(column(47, FFN_COLUMN), 47 * 6656 + 4352);
    assert_eq!(column(47, FFN_COLUMN) + HIDDEN, STATE_COLUMNS as u32);
    let p = mix_params(5, 16, 2, 3);
    assert_eq!((p.stack_plane, p.scale), (16 * 2304, 1.0 / 48.0));
    let copy = row_copy(7, 0);
    assert_eq!(
        (copy.rows, copy.width, copy.destination_stride),
        (7, 2304, 2304)
    );
}

#[test]
fn apple9_decodes_without_int8_queries() {
    // A 40-core M3 Max plans the same splits as the M5 Max but scores no
    // INT8 queries.
    let m3 = Calibration::of(9, 40);
    assert_eq!(
        attention_choice(4000, 16, false, &m3),
        AttentionChoice {
            splits: 12,
            int8_queries: false
        }
    );
}

#[test]
fn prompt_attention_follows_the_prefill_plan() {
    let max = max();
    // One tile: at least 32 splits, a split per 16 pages.
    assert_eq!(
        prompt_attention(0, 8, false, &max),
        AttentionChoice {
            splits: 32,
            int8_queries: false
        }
    );
    // The 42-row prompt: six tiles of five shared splits.
    assert_eq!(prompt_attention(0, 42, false, &max).splits, 5);
    assert_eq!(prompt_attention(0, 42, true, &max).splits, 5);
    // A 2048-row chunk after none: one split, INT8 queries over 64 pages
    // (the window's 34).
    assert_eq!(
        prompt_attention(0, 2048, false, &max),
        AttentionChoice {
            splits: 1,
            int8_queries: true
        }
    );
    assert_eq!(
        prompt_attention(0, 2048, true, &max),
        AttentionChoice {
            splits: 1,
            int8_queries: true
        }
    );
    // A 2048-row chunk after 30K tokens: a split per 32 pages, at most one
    // a tile; 900 tokens' tiles take seven of 1023 pages.
    assert_eq!(prompt_attention(30_000, 2048, false, &max).splits, 1);
    assert_eq!(prompt_attention(30_000, 900, false, &max).splits, 2);
    // Apple9 scores bf16 queries, and a 10-core M4 splits per 64 pages.
    let m4 = Calibration::of(9, 10);
    assert!(!prompt_attention(0, 2048, false, &m4).int8_queries);
    assert_eq!(prompt_attention(8000, 200, false, &max).splits, 8);
    assert_eq!(prompt_attention(8000, 200, false, &m4).splits, 4);
}

#[test]
fn prompt_scratch_holds_a_whole_chunk() {
    let sizes = scratch_sizes(&max(), PROMPT_ROWS, true);
    // The sequence workspace: 256 tile splits of 192 fused rows.
    assert_eq!(sizes.attention_partials, 256 * 192 * 128);
    // Int8 codes and scales of 2048 rows of the attention output's inputs.
    assert_eq!(sizes.table_bytes, 3072 * 2048);
    assert_eq!(sizes.sums, 3072 / 64 * 2048 * 2);
    // 32-row expert tiles of 16384 routes.
    assert_eq!(sizes.maximum_tiles, 16384 / 32 + 128);
    assert_eq!(sizes.expert_output, 16384 * 2304);
    // Fused gates up to 128 rows.
    assert_eq!(sizes.gate_values, 3 * 128 * 2304);
}
