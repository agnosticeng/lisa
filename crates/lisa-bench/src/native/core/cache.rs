//! Concrete application aliases and native conformance tests for engine caches.
pub use lisa_engine::cache::NativeState;
use lisa_metal::Array;
pub type NativeCache = lisa_engine::cache::NativeCache<Array>;
pub type IndexerTape = lisa_engine::indexer_cache::IndexerTape<lisa_metal::device::MetalDevice>;
pub type TapeSnapshot = lisa_engine::indexer_cache::TapeSnapshot<Array>;
pub type FullAttentionCache =
    lisa_engine::attention_cache::FullAttentionCache<lisa_metal::device::MetalDevice>;
pub type GdnCache = lisa_engine::recurrent_cache::GdnCache<Array>;
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
    fn dspine_replay_commits_conv_and_survives_snapshot_restore() -> lisa_metal::error::Result<()> {
        let stream = lisa_metal::Stream::thread_local_or_default();
        let (t, hk, hv) = (16, 16, 48);
        let values = |shape: &[i32], dtype| {
            let n: i32 = shape.iter().product();
            let v: Vec<f32> = (0..n)
                .map(|i| (((i * 17 + 5) % 67) as f32 - 33.0) / 257.0)
                .collect();
            Array::from_slice(&v, shape).as_dtype(dtype)
        };
        let q = values(&[1, t, hk, 128], Dtype::Bfloat16)?;
        let v = values(&[1, t, hv, 128], Dtype::Bfloat16)?;
        let gate = values(&[1, t, hv], Dtype::Float32)?.add(&Array::from_f32(0.7))?;
        // A preceding one-row capture leaves a three-dimensional live SSM.
        let initial = values(&[hv, 128, 128], Dtype::Float32)?;
        let (_, snapshots) =
            lisa_metal::kernels::gdn_two_row(&q, &q, &v, &gate, &gate, &initial, true, &stream)
                .unwrap();
        let (_, _, replay) = crate::core::recurrent::capture(&q, &q, &v, &gate, &gate, &initial)
            .unwrap()
            .unwrap();
        let conv = values(&[1, t + 3, 8], Dtype::Bfloat16)?;
        let mut cache = GdnCache::new();
        cache.capture_replay = Some(replay);
        cache.capture_conv_input = Some(conv.clone());
        for n in 1..=t {
            cache.rollback_to(&crate::metal_context::execution_context(), n as usize, 4)?;
            assert_eq!(
                cache.ssm.as_ref().unwrap().to_vec1::<u32>()?,
                snapshots
                    .index((n - 1, .., .., ..))
                    .contiguous()?
                    .to_vec1::<u32>()?
            );
            assert_eq!(
                cache.conv.as_ref().unwrap().to_vec1::<u16>()?,
                conv.index((.., n..n + 3, ..))
                    .contiguous()?
                    .to_vec1::<u16>()?
            );
        }
        cache.rollback_to(&crate::metal_context::execution_context(), 3, 4)?;
        let saved = cache.snapshot_state();
        cache.rollback_to(&crate::metal_context::execution_context(), 12, 4)?;
        cache.restore_state(&saved);
        assert!(cache.capture_replay.is_none());
        assert!(cache.capture_conv_input.is_none());
        let expected = saved.1.unwrap().to_vec1::<u32>()?;
        assert_eq!(cache.ssm.as_ref().unwrap().to_vec1::<u32>()?, expected);
        cache.rollback_to(&crate::metal_context::execution_context(), 8, 4)?; // Restoring a snapshot invalidates the old transaction.
        assert_eq!(cache.ssm.as_ref().unwrap().to_vec1::<u32>()?, expected);
        Ok(())
    }

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

    #[cfg(feature = "glm53-fixtures")]
    #[test]
    fn glm_compact_cache_matches_ds4_metal_positional_f16_writes() -> lisa_metal::error::Result<()>
    {
        const ROWS: usize = 8;
        const DIM: usize = 512;
        const START: usize = 254;
        const REPLACED_ROW: usize = 3;
        let external_fixture: &[u8] =
            &crate::bindings::glm53_test_fixtures::read("dsa_compact_cache_ds4_metal.bin");
        assert_eq!(external_fixture.len(), ROWS * DIM * 2);

        fn row_values(row_index: usize) -> Vec<f32> {
            (0..DIM)
                .map(|i| {
                    let logical = row_index * DIM + i;
                    let centered = ((logical * 73 + 29) % 2001) as i32 - 1000;
                    centered as f32 / 997.0
                })
                .collect()
        }

        let mut tape = IndexerTape::new(crate::metal_context::execution_context());
        for row in 0..ROWS {
            tape.write_compact_f16_at(
                &Array::from_slice(&row_values(row), &[1, 1, DIM as i32]),
                START + row,
            )?;
        }
        tape.write_compact_f16_at(
            &Array::from_slice(&row_values(ROWS + REPLACED_ROW), &[1, 1, DIM as i32]),
            START + REPLACED_ROW,
        )?;

        let rows = tape
            .compact_slice(START, START + ROWS)
            .expect("DS4 comparison cache rows");
        rows.eval()?;
        let actual = rows
            .reshape(&[(ROWS * DIM) as i32])?
            .to_vec1::<half::f16>()?;
        let actual_bits: Vec<u16> = actual.iter().map(|value| value.to_bits()).collect();
        let expected_bits: Vec<u16> = external_fixture
            .chunks_exact(2)
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .collect();
        assert_eq!(actual_bits, expected_bits);
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

    #[cfg(feature = "glm53-fixtures")]
    #[test]
    fn glm_indexer_pool_rollback_and_replay_match_ds4_incremental_cache() -> anyhow::Result<()> {
        if !lisa_metal::device::metal_available() {
            eprintln!("[glm-indexer-pool-replay] skipping: no Metal device available");
            return Ok(());
        }
        const POOLS: usize = 2;
        const POOL_SIZE: usize = 4;
        const DIM: usize = 128;
        const ROW_VALUES: usize = POOLS * POOL_SIZE * DIM;
        const ROW_BYTES: usize = ROW_VALUES * std::mem::size_of::<f32>();
        const NORM_BYTES: usize = DIM * std::mem::size_of::<f32>();
        const APE_BYTES: usize = POOL_SIZE * DIM * std::mem::size_of::<u16>();
        const POOL_BYTES: usize = POOLS * DIM * std::mem::size_of::<u16>();
        let external_fixture: &[u8] =
            &crate::bindings::glm53_test_fixtures::read("indexer_pool_lifecycle_ds4.bin");
        assert_eq!(
            external_fixture.len(),
            ROW_BYTES + NORM_BYTES + ROW_BYTES + NORM_BYTES + APE_BYTES + POOL_BYTES
        );

        let decode_f32 = |bytes: &[u8]| {
            bytes
                .chunks_exact(std::mem::size_of::<f32>())
                .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("F32 fixture word")))
                .collect::<Vec<_>>()
        };
        let decode_bf16 = |bytes: &[u8]| {
            bytes
                .chunks_exact(std::mem::size_of::<u16>())
                .map(|chunk| half::bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
                .collect::<Vec<_>>()
        };
        let expected: Vec<half::f16> = external_fixture[external_fixture.len() - POOL_BYTES..]
            .chunks_exact(std::mem::size_of::<u16>())
            .map(|chunk| half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])))
            .collect();
        let raw = decode_f32(&external_fixture[..ROW_BYTES]);
        let norm_weight = decode_f32(&external_fixture[ROW_BYTES..ROW_BYTES + NORM_BYTES]);
        let gate_start = ROW_BYTES + NORM_BYTES;
        let gate = decode_f32(&external_fixture[gate_start..gate_start + ROW_BYTES]);
        let bias_start = gate_start + ROW_BYTES;
        let norm_bias = decode_f32(&external_fixture[bias_start..bias_start + NORM_BYTES]);
        let ape_start = bias_start + NORM_BYTES;
        let ape = decode_bf16(&external_fixture[ape_start..ape_start + APE_BYTES]);

        let append_rows = |tape: &mut IndexerTape,
                           start: usize,
                           count: usize,
                           raw_rows: &[f32],
                           gate_rows: &[f32]|
         -> anyhow::Result<()> {
            let start_at = start * DIM;
            let end_at = (start + count) * DIM;
            let raw =
                Array::from_slice(&raw_rows[start_at..end_at], &[1, count as i32, DIM as i32]);
            let gate =
                Array::from_slice(&gate_rows[start_at..end_at], &[1, count as i32, DIM as i32]);
            tape.append_fields(&raw, &gate, POOL_SIZE)?;
            Ok(())
        };
        let pool_rows = |raw_rows: &[f32], gate_rows: &[f32]| -> anyhow::Result<Array> {
            let raw = Array::from_slice(raw_rows, &[1, 1, POOL_SIZE as i32, DIM as i32]);
            let gate = Array::from_slice(gate_rows, &[1, 1, POOL_SIZE as i32, DIM as i32]);
            let norm_weight = Array::from_slice(&norm_weight, &[DIM as i32]);
            let norm_bias = Array::from_slice(&norm_bias, &[DIM as i32]);
            let ape = Array::from_slice(&ape, &[POOL_SIZE as i32, DIM as i32]);
            lisa_metal::kernels::sparse::kernels::indexer_pool_fused(
                &raw,
                &gate,
                &ape,
                &norm_weight,
                &norm_bias,
                1e-6,
                &lisa_metal::Stream::thread_local_or_default(),
            )
            .ok_or_else(|| anyhow::anyhow!("DS4-shaped DSA pool kernel declined"))
        };

        let mut tape = IndexerTape::new(crate::metal_context::execution_context());
        for (start, count) in [(0, 1), (1, 2), (3, 1)] {
            append_rows(&mut tape, start, count, &raw, &gate)?;
        }
        let first_pool = pool_rows(&raw[..POOL_SIZE * DIM], &gate[..POOL_SIZE * DIM])?;
        tape.push_pooled_f16(&first_pool)?;
        let checkpoint = tape.snapshot();

        let mut speculative_raw = vec![0.0f32; POOL_SIZE * DIM];
        let mut speculative_gate = vec![0.0f32; POOL_SIZE * DIM];
        for row in 0..POOL_SIZE {
            for channel in 0..DIM {
                let index = row * DIM + channel;
                let source = ((row + 1) % POOL_SIZE) * DIM + channel;
                speculative_raw[index] = raw[source] * -0.5 + (row + 2) as f32 * 0.0625;
                speculative_gate[index] = gate[source] * 0.25 + (row + 1) as f32 * 0.125;
            }
        }
        let mut speculative_full_raw = vec![0.0f32; ROW_VALUES];
        let mut speculative_full_gate = vec![0.0f32; ROW_VALUES];
        speculative_full_raw[POOL_SIZE * DIM..].copy_from_slice(&speculative_raw);
        speculative_full_gate[POOL_SIZE * DIM..].copy_from_slice(&speculative_gate);
        append_rows(
            &mut tape,
            4,
            2,
            &speculative_full_raw,
            &speculative_full_gate,
        )?;
        append_rows(
            &mut tape,
            6,
            2,
            &speculative_full_raw,
            &speculative_full_gate,
        )?;
        let speculative_pool = pool_rows(&speculative_raw, &speculative_gate)?;
        let speculative_bits: Vec<u16> = speculative_pool
            .to_vec1::<half::f16>()?
            .iter()
            .map(|value| value.to_bits())
            .collect();
        tape.push_pooled_f16(&speculative_pool)?;
        tape.trim(4);
        assert_eq!(tape.total, 4);
        assert_eq!(tape.pooled_upto, 1);

        for (start, count) in [(4, 1), (5, 2), (7, 1)] {
            append_rows(&mut tape, start, count, &raw, &gate)?;
        }
        let replayed_pool = pool_rows(&raw[POOL_SIZE * DIM..], &gate[POOL_SIZE * DIM..])?;
        tape.push_pooled_f16(&replayed_pool)?;
        let actual: Vec<u16> = tape
            .pooled_view()
            .expect("replayed pooled cache")
            .to_vec1::<half::f16>()?
            .iter()
            .map(|value| value.to_bits())
            .collect();
        let expected_bits: Vec<u16> = expected.iter().map(|value| value.to_bits()).collect();
        anyhow::ensure!(
            actual == expected_bits,
            "LISA pooled cache after rollback/replay differs from DS4 incremental Metal"
        );
        anyhow::ensure!(
            speculative_bits != expected_bits[DIM..],
            "speculative pool fixture unexpectedly equals the replacement replay"
        );
        anyhow::ensure!(
            tape.total == 8 && tape.pooled_upto == 2 && tape.raw_start() == 0,
            "unexpected live DSA tape state after replay"
        );
        anyhow::ensure!(checkpoint.total == 4 && checkpoint.pooled_upto == 1);
        eprintln!(
            "[glm-indexer-pool-replay] all {} F16 cache values match DS4 after speculative completion, rollback, and replacement replay",
            actual.len()
        );
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
