//! Concrete application aliases and native conformance tests for engine caches.
pub use lisa_engine::cache::NativeState;
use lisa_metal::Array;
pub type NativeCache = lisa_engine::cache::NativeCache<Array>;
pub type IndexerTape = lisa_engine::cache::indexer::IndexerTape<lisa_metal::device::MetalDevice>;
pub type TapeSnapshot = lisa_engine::cache::indexer::TapeSnapshot<Array>;
pub type FullAttentionCache =
    lisa_engine::cache::attention::FullAttentionCache<lisa_metal::device::MetalDevice>;
pub type GdnCache = lisa_engine::cache::recurrent::GdnCache<Array>;
pub type LayerCache = lisa_engine::cache::LayerCache<lisa_metal::device::MetalDevice>;
pub type LayerState = lisa_engine::cache::LayerState<Array>;
pub type LayerRollbackState = lisa_engine::cache::LayerRollbackState<Array>;
#[cfg(test)]
use lisa_engine::cache::FullPrefixState;
#[cfg(test)]
use lisa_metal::{Dtype, ops, ops::indexing::IndexOp};
#[cfg(test)]
mod cache_state_tests {
    #[test]
    fn full_attention_ple_rollback_retains_only_accepted_inputs() -> anyhow::Result<()> {
        let context = crate::metal_context::execution_context();
        for batch in [1, 2] {
            for history in [3, 9] {
                for advance in [1, 2, 7] {
                    let rows = history + advance;
                    let data: Vec<f32> = (0..batch * rows * 4).map(|i| i as f32).collect();
                    let full = Array::from_slice(&data, &[batch as i32, rows as i32, 4]);
                    for keep in 0..=advance {
                        let mut full_cache = FullAttentionCache::new(context.clone(), 1, 4);
                        full_cache.offset = 20 + advance;
                        full_cache.capture_ple_full = Some(full.clone());
                        full_cache.ple_conv =
                            Some(full.index((.., advance as i32.., ..)).contiguous()?);
                        let mut cache = LayerCache::Full(full_cache);
                        cache.rollback_verified(&context, advance, keep, true)?;
                        let LayerCache::Full(cache) = cache else {
                            unreachable!()
                        };
                        assert_eq!(cache.offset, 20 + keep);
                        let expected: Vec<f32> = (0..batch)
                            .flat_map(|b| {
                                data[(b * rows + keep) * 4..(b * rows + keep + history) * 4]
                                    .iter()
                                    .copied()
                            })
                            .collect();
                        assert_eq!(cache.ple_conv.unwrap().to_vec1::<f32>()?, expected);
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn full_attention_ple_rollback_rejects_missing_capture_before_mutation() {
        let context = crate::metal_context::execution_context();
        let mut full = FullAttentionCache::new(context.clone(), 1, 4);
        full.offset = 23;
        full.ple_conv = Some(Array::from_slice(&[9f32], &[1, 1, 1]));
        let mut cache = LayerCache::Full(full);
        for captured in [false, true] {
            assert!(cache.rollback_verified(&context, 3, 1, captured).is_err());
            let LayerCache::Full(full) = &cache else {
                unreachable!()
            };
            assert_eq!(full.offset, 23);
            assert_eq!(
                full.ple_conv.as_ref().unwrap().to_vec1::<f32>().unwrap(),
                [9.]
            );
        }
    }

    #[test]
    fn failed_delta_rollback_keeps_all_committed_handles() {
        let mut c = GdnCache::new();
        c.ssm = Some(Array::from_slice(&[7f32], &[1, 1, 1]));
        c.conv = Some(Array::from_slice(&[8f32], &[1, 1, 1]));
        c.ple_conv = Some(Array::from_slice(&[9f32], &[1, 1, 1]));
        c.capture_ssm = Some(Array::from_slice(&[1f32, 2.], &[2, 1, 1, 1]));
        // State selection succeeds, but the convolution history is too short.
        c.capture_conv_input = Some(Array::from_slice(&[3f32], &[1, 1, 1]));
        assert!(
            c.rollback_to(&crate::metal_context::execution_context(), 1, 4)
                .is_err()
        );
        assert_eq!(c.ssm.unwrap().to_vec1::<f32>().unwrap(), vec![7.]);
        assert_eq!(c.conv.unwrap().to_vec1::<f32>().unwrap(), vec![8.]);
        assert_eq!(c.ple_conv.unwrap().to_vec1::<f32>().unwrap(), vec![9.]);
    }

    use super::*;

    #[test]
    fn indexer_tape_append_fields_matches_packed_append() -> lisa_metal::error::Result<()> {
        let key = Array::from_slice(&[0.25f32, -1.0, 2.0, 3.5], &[1, 2, 2]);
        let gate = Array::from_slice(&[-0.5f32, 0.75], &[1, 2, 1]);
        let packed = ops::concatenate(&[&key, &gate], 2)?;
        let mut expected = IndexerTape::new(crate::metal_context::execution_context());
        let mut actual = IndexerTape::new(crate::metal_context::execution_context());
        expected.append(&packed, 4)?;
        actual.append_fields(&key, &gate, 4)?;
        let want = expected.raw_slice(0, 2).expect("packed raw rows");
        let got = actual.raw_slice(0, 2).expect("field raw rows");
        let want = want.to_vec1::<f32>()?;
        let got = got.to_vec1::<f32>()?;
        assert_eq!(
            want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            got.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn glm_mtp_compact_cache_replaces_rejected_draft_at_same_position()
    -> lisa_metal::error::Result<()> {
        let mut tape = IndexerTape::new(crate::metal_context::execution_context());
        tape.write_compact_f16_at(&Array::from_slice(&[1.25f32], &[1, 1, 1]), 10)?;
        tape.write_compact_f16_at(&Array::from_slice(&[2.5f32], &[1, 1, 1]), 11)?;

        // A rejected proposal is retried at the same absolute target position
        // with the replacement token's latent. The later row must stay intact.
        tape.write_compact_f16_at(&Array::from_slice(&[9.75f32], &[1, 1, 1]), 10)?;
        let rows = tape.compact_slice(10, 12).expect("written MTP cache rows");
        rows.eval()?;
        assert_eq!(
            rows.reshape(&[2])?.to_vec1::<half::f16>()?,
            &[half::f16::from_f32(9.75), half::f16::from_f32(2.5)]
        );
        assert_eq!(tape.compact_total(), 12);
        Ok(())
    }

    #[test]
    fn direct_compact_append_reservation_preserves_rows_across_growth()
    -> lisa_metal::error::Result<()> {
        const WIDTH: usize = 512;
        let mut tape = IndexerTape::new(crate::metal_context::execution_context());
        let key0 = Array::from_slice(&vec![0.0f32; 257], &[1, 257, 1]);
        let gate0 = Array::from_slice(&vec![0.0f32; 257], &[1, 257, 1]);
        let (cache0, start0) = tape.reserve_compact_f16_append(1, 257, WIDTH as i32)?;
        assert_eq!(start0, 0);
        assert_eq!(cache0.dim(1), 513);
        let first = Array::from_slice(
            &vec![half::f16::from_f32(1.25); 257 * WIDTH],
            &[1, 257, WIDTH as i32],
        );
        let stream = lisa_metal::Stream::thread_local_or_default();
        assert!(lisa_metal::kernels::copy_rows_into(&first, &cache0, 0, &stream).is_some());
        tape.append_fields(&key0, &gate0, 1)?;
        tape.commit_compact_f16_append(start0, 257)?;

        let (cache1, start1) = tape.reserve_compact_f16_append(1, 257, WIDTH as i32)?;
        assert_eq!(start1, 257);
        assert_eq!(cache1.dim(1), 1026);
        let replacement = Array::from_slice(
            &vec![half::f16::from_f32(2.5); 257 * WIDTH],
            &[1, 257, WIDTH as i32],
        );
        assert!(
            lisa_metal::kernels::copy_rows_into(&replacement, &cache1, start1 as i32, &stream)
                .is_some()
        );
        let key1 = Array::from_slice(&vec![0.0f32; 257], &[1, 257, 1]);
        let gate1 = Array::from_slice(&vec![0.0f32; 257], &[1, 257, 1]);
        tape.append_fields(&key1, &gate1, 1)?;
        tape.commit_compact_f16_append(start1, 257)?;

        let rows = tape
            .compact_slice(0, 514)
            .expect("reserved compact rows after growth");
        let values = rows.to_vec1::<half::f16>()?;
        assert!(
            values[..257 * WIDTH]
                .iter()
                .all(|value| *value == half::f16::from_f32(1.25))
        );
        assert!(
            values[257 * WIDTH..514 * WIDTH]
                .iter()
                .all(|value| *value == half::f16::from_f32(2.5))
        );
        Ok(())
    }

    #[test]
    fn indexer_tape_packs_and_extracts_equal_position_rows() -> lisa_metal::error::Result<()> {
        let mut tapes = Vec::new();
        for stream in 0..2 {
            let raw: Vec<f32> = (0..8).map(|i| (stream * 100 + i) as f32).collect();
            let compact: Vec<f32> = (0..12).map(|i| (stream * 1000 + i) as f32).collect();
            let pooled = [stream as f32 + 0.5, stream as f32 + 1.5];
            let mut tape = IndexerTape::new(crate::metal_context::execution_context());
            tape.append(&Array::from_slice(&raw, &[1, 4, 2]), 4)?;
            tape.append_compact_f16(&Array::from_slice(&compact, &[1, 4, 3]))?;
            tape.push_pooled_f16(&Array::from_slice(&pooled, &[1, 1, 2]))?;
            tapes.push(tape);
        }

        let packed = IndexerTape::pack_equal(
            crate::metal_context::execution_context(),
            &[&tapes[0], &tapes[1]],
        )?;
        assert_eq!(packed.total, 4);
        assert_eq!(packed.raw_start(), 0);
        assert_eq!(packed.pooled_upto, 1);
        assert_eq!(packed.compact_total(), 4);
        assert_eq!(packed.raw_live_for_test().unwrap().dim(0), 2);
        assert_eq!(packed.pooled_view().unwrap().dim(0), 2);
        assert_eq!(packed.compact_live_for_test().unwrap().dim(0), 2);

        for stream in 0..2 {
            let row = packed.extract_row(stream)?;
            let raw = row.raw_slice(0, 4).expect("packed raw row");
            let compact = row.compact_slice(0, 4).expect("packed compact row");
            let pooled = row.pooled_view().expect("packed pooled row");
            assert_eq!(
                raw.to_vec1::<f32>()?,
                (0..8)
                    .map(|i| (stream * 100 + i) as f32)
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                compact.to_vec1::<half::f16>()?,
                (0..12)
                    .map(|i| half::f16::from_f32((stream * 1000 + i) as f32))
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                pooled.to_vec1::<half::f16>()?,
                [
                    half::f16::from_f32(stream as f32 + 0.5),
                    half::f16::from_f32(stream as f32 + 1.5),
                ]
            );
        }
        Ok(())
    }

    #[test]
    fn indexer_compaction_snapshot_and_rollback_preserve_absolute_rows()
    -> lisa_metal::error::Result<()> {
        let values: Vec<f32> = (0..600 * 2).map(|v| v as f32).collect();
        let rows = Array::from_slice(&values, &[1, 600, 2]);
        let mut tape = IndexerTape::new(crate::metal_context::execution_context());
        tape.append(&rows, 4)?;
        tape.pooled_upto = 150;
        tape.compact_raw_history(32)?;

        assert_eq!(tape.total, 600);
        assert_eq!(tape.raw_start(), 568);
        assert!(tape.raw_slice(0, 4).is_none());
        let tail = tape.raw_slice(568, 600).expect("retained rollback tail");
        tail.eval()?;
        let expected: Vec<f32> = values[568 * 2..].to_vec();
        assert_eq!(tail.as_slice::<f32>(), expected.as_slice());

        let snapshot = tape.snapshot();
        let mut restored = IndexerTape::new(crate::metal_context::execution_context());
        restored.restore(&snapshot);
        assert_eq!(restored.raw_start(), 568);
        assert_eq!(restored.total, 600);
        assert_eq!(restored.pooled_upto, 150);
        assert!(restored.raw_slice(567, 569).is_none());

        // Roll back by two complete pools, then append replacement rows. The
        // absolute origin stays fixed while the retained tail is overwritten.
        restored.trim(592);
        let replacement_values = [9000.0f32, 9001.0, 9002.0, 9003.0];
        let replacement = Array::from_slice(&replacement_values, &[1, 2, 2]);
        restored.append(&replacement, 4)?;
        let replaced = restored.raw_slice(592, 594).expect("replacement rows");
        replaced.eval()?;
        assert_eq!(replaced.as_slice::<f32>(), &replacement_values);
        assert_eq!(restored.total, 594);
        assert_eq!(restored.raw_start(), 568);
        Ok(())
    }

    #[test]
    fn glm_indexer_prefix_restore_replays_partial_pool_after_rollback()
    -> lisa_metal::error::Result<()> {
        let raw_values: Vec<f32> = (0..600).map(|v| v as f32).collect();
        let mut full = FullAttentionCache::new(crate::metal_context::execution_context(), 1, 1);
        full.offset = 600;
        full.indexer_tape
            .append(&Array::from_slice(&raw_values, &[1, 600, 1]), 4)?;

        let compact_values: Vec<f32> = (0..600).map(|v| (v as f32) * 0.5).collect();
        full.indexer_tape
            .append_compact_f16(&Array::from_slice(&compact_values, &[1, 600, 1]))?;
        let pooled_values: Vec<f32> = (0..150).map(|v| v as f32).collect();
        full.indexer_tape
            .push_pooled_f16(&Array::from_slice(&pooled_values, &[1, 150, 1]))?;
        full.indexer_tape.compact_raw_history(32)?;
        assert_eq!(full.indexer_tape.raw_start(), 568);

        let prefix = LayerCache::Full(full).snapshot_prefix();
        let mut restored = LayerCache::Full(FullAttentionCache::new(
            crate::metal_context::execution_context(),
            1,
            1,
        ));
        restored.restore_prefix(&prefix);
        let LayerCache::Full(restored) = &mut restored else {
            unreachable!();
        };

        // Roll back into the final pool, then replace the two speculative rows.
        // The pool is rebuilt from the preserved prefix rows plus the new rows.
        restored.trim(6);
        let replacement_raw = Array::from_slice(&[900.0f32, 901.0], &[1, 2, 1]);
        restored.indexer_tape.append(&replacement_raw, 4)?;
        let replacement_compact = Array::from_slice(&[450.0f32, 450.5], &[1, 2, 1]);
        restored
            .indexer_tape
            .append_compact_f16(&replacement_compact)?;

        let pool_raw = restored
            .indexer_tape
            .raw_slice(592, 596)
            .expect("partial-pool prefix remains available after restore");
        pool_raw.eval()?;
        assert_eq!(
            pool_raw.reshape(&[4])?.to_vec1::<f32>()?,
            &[592.0, 593.0, 900.0, 901.0]
        );
        let rebuilt_pool = Array::from_slice(&[846.5f32], &[1, 1, 1]);
        restored.indexer_tape.push_pooled_f16(&rebuilt_pool)?;
        let pooled_tail = restored
            .indexer_tape
            .pooled_view()
            .expect("restored pooled-key cache")
            .index((.., 148..149, ..))
            .reshape(&[1])?;
        pooled_tail.eval()?;
        assert_eq!(
            pooled_tail.to_vec1::<half::f16>()?,
            &[half::f16::from_f32(846.5)]
        );

        let live_compact = restored
            .indexer_tape
            .compact_slice(592, 596)
            .expect("compact cache was truncated and replayed with raw rows");
        live_compact.eval()?;
        assert_eq!(
            live_compact.reshape(&[4])?.to_vec1::<half::f16>()?,
            &[
                half::f16::from_f32(296.0),
                half::f16::from_f32(296.5),
                half::f16::from_f32(450.0),
                half::f16::from_f32(450.5),
            ]
        );
        assert_eq!(restored.offset, 594);
        assert_eq!(restored.indexer_tape.total, 596);
        assert_eq!(restored.indexer_tape.pooled_upto, 149);
        assert_eq!(restored.indexer_tape.raw_start(), 568);
        Ok(())
    }

    /// The spill codec round-trips a Full layer state: shapes, dtypes and
    /// offset/tape scalars survive; the restored arrays are host-backed and
    /// carry the same values.
    #[test]
    fn layer_state_codec_round_trips() -> anyhow::Result<()> {
        let keys: Option<Array> = Some(Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 2, 1, 2]));
        let values = Some(Array::from_slice(
            &[
                half::bf16::from_f32(0.5),
                half::bf16::from_f32(0.25),
                half::bf16::from_f32(1.0),
                half::bf16::from_f32(2.0),
            ],
            &[1, 1, 2, 2],
        ));
        let tape_raw = Some(Array::from_slice(&[7u32, 8, 9], &[3]));
        let st = LayerState::Full(FullPrefixState {
            keys,
            values,
            offset: 2,
            tape: TapeSnapshot {
                raw: tape_raw,
                total: 3,
                raw_start: 0,
                pooled: None,
                pooled_upto: 1,
                cr: 4,
                compact: None,
                compact_total: 0,
            },
            ple_conv: None,
        });
        let bytes = st.encode(&crate::metal_context::device())?;
        let mut cur: &[u8] = &bytes;
        let back = LayerState::decode(&crate::metal_context::device(), &mut cur)?;
        assert!(cur.is_empty(), "one record consumes the whole buffer");
        let LayerState::Full(b) = back else {
            anyhow::bail!("kind mismatch");
        };
        assert_eq!(b.offset, 2);
        assert_eq!(b.tape.total, 3);
        assert_eq!(b.tape.pooled_upto, 1);
        assert_eq!(b.tape.cr, 4);
        let k = b.keys.as_ref().expect("keys");
        assert_eq!(k.shape(), &[1, 2, 1, 2]);
        assert_eq!(k.as_slice::<f32>(), &[1.0, 2.0, 3.0, 4.0]);
        let t = b.tape.raw.as_ref().expect("tape");
        assert_eq!(t.as_slice::<u32>(), &[7, 8, 9]);
        let v = b.values.as_ref().expect("values");
        assert_eq!(v.dtype(), Dtype::Bfloat16);
        Ok(())
    }

    #[test]
    fn glm_compacted_tape_spill_restore_preserves_live_absolute_rows() -> anyhow::Result<()> {
        let raw_values: Vec<f32> = (0..600 * 2).map(|v| v as f32 * 0.25).collect();
        let compact_values: Vec<f32> = (0..600 * 2).map(|v| v as f32 * 0.5).collect();
        let pooled_values: Vec<f32> = (0..150 * 2).map(|v| v as f32 * 0.125).collect();

        let mut full = FullAttentionCache::new(crate::metal_context::execution_context(), 1, 2);
        full.offset = 600;
        full.indexer_tape
            .append(&Array::from_slice(&raw_values, &[1, 600, 2]), 4)?;
        full.indexer_tape
            .append_compact_f16(&Array::from_slice(&compact_values, &[1, 600, 2]))?;
        full.indexer_tape
            .push_pooled_f16(&Array::from_slice(&pooled_values, &[1, 150, 2]))?;
        full.indexer_tape.compact_raw_history(32)?;
        assert_eq!(full.indexer_tape.raw_start(), 568);

        // Exercise the exact layer-state codec used by SSD prefix spill, then
        // restore into a fresh GLM full-attention cache.
        let state = LayerCache::Full(full).snapshot_prefix();
        let bytes = state.encode(&crate::metal_context::device())?;
        let mut cursor: &[u8] = &bytes;
        let restored_state = LayerState::decode(&crate::metal_context::device(), &mut cursor)?;
        assert!(cursor.is_empty());
        let mut restored = LayerCache::Full(FullAttentionCache::new(
            crate::metal_context::execution_context(),
            1,
            2,
        ));
        restored.restore_prefix(&restored_state);
        let LayerCache::Full(restored) = &mut restored else {
            anyhow::bail!("decoded cache kind mismatch");
        };

        assert_eq!(restored.offset, 600);
        assert_eq!(restored.indexer_tape.raw_start(), 568);
        assert_eq!(restored.indexer_tape.total, 600);
        assert_eq!(restored.indexer_tape.pooled_upto, 150);
        let raw = restored
            .indexer_tape
            .raw_slice(568, 600)
            .expect("retained raw rollback window");
        raw.eval()?;
        assert_eq!(
            raw.reshape(&[64])?.to_vec1::<f32>()?,
            &raw_values[568 * 2..]
        );
        let compact = restored
            .indexer_tape
            .compact_slice(568, 600)
            .expect("compact latent history");
        compact.eval()?;
        let expected_compact: Vec<half::f16> = compact_values[568 * 2..]
            .iter()
            .map(|&value| half::f16::from_f32(value))
            .collect();
        assert_eq!(
            compact.reshape(&[64])?.to_vec1::<half::f16>()?,
            expected_compact
        );
        let pooled = restored
            .indexer_tape
            .pooled_view()
            .expect("pooled key history")
            .index((.., 149..150, ..));
        pooled.eval()?;
        let expected_pooled: Vec<half::f16> = pooled_values[149 * 2..150 * 2]
            .iter()
            .map(|&value| half::f16::from_f32(value))
            .collect();
        assert_eq!(
            pooled.reshape(&[2])?.to_vec1::<half::f16>()?,
            expected_pooled
        );

        // Roll back across the final pool and replay replacement rows after
        // spill restore; absolute positions and the retained pool prefix must
        // remain valid.
        restored.trim(6);
        let replacement_raw = Array::from_slice(&[9000.0f32, 9001.0, 9002.0, 9003.0], &[1, 2, 2]);
        restored.indexer_tape.append(&replacement_raw, 4)?;
        let replacement_compact =
            Array::from_slice(&[9100.0f32, 9101.0, 9102.0, 9103.0], &[1, 2, 2]);
        restored
            .indexer_tape
            .append_compact_f16(&replacement_compact)?;
        let replayed_raw = restored
            .indexer_tape
            .raw_slice(594, 596)
            .expect("replacement raw rows");
        replayed_raw.eval()?;
        assert_eq!(
            replayed_raw.reshape(&[4])?.to_vec1::<f32>()?,
            &[9000.0, 9001.0, 9002.0, 9003.0]
        );
        let replayed_compact = restored
            .indexer_tape
            .compact_slice(594, 596)
            .expect("replacement compact rows");
        replayed_compact.eval()?;
        assert_eq!(
            replayed_compact.reshape(&[4])?.to_vec1::<half::f16>()?,
            &[
                half::f16::from_f32(9100.0),
                half::f16::from_f32(9101.0),
                half::f16::from_f32(9102.0),
                half::f16::from_f32(9103.0),
            ]
        );
        assert_eq!(restored.indexer_tape.raw_start(), 568);
        assert_eq!(restored.indexer_tape.total, 596);
        assert_eq!(restored.indexer_tape.pooled_upto, 148);
        Ok(())
    }
}
