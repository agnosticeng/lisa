//! Batching: cohort, ragged, and continuous.
//!
//! [`Batch`] is the fixed-membership case: all streams are prefilled up front
//! (together for equal lengths, or alone then packed when ragged) and then
//! advanced in lockstep. [`ContinuousBatch`] is the serving case: streams are
//! admitted and retired as they arrive and finish, and the packed caches are
//! compacted on every membership change (the continuous-batch v2 idea).
//!
//! The model is written with `B` as a batch dimension throughout (attention,
//! GDN state, PLE history), so a batched forward is a single `Tower::forward`
//! over `[N, S]` tokens. The MoE falls back to MLX's generic (batched) path
//! because the engine's fused decode/prefill kernels are single-batch.

#[cfg(test)]
use lisa_metal::Array;
#[cfg(test)]
use lisa_metal::ops::indexing::IndexOp;

#[cfg(test)]
use crate::core::cache::LayerCache;
#[cfg(test)]
use crate::core::cache::{FullAttentionCache, IndexerTape};

#[cfg(test)]
use crate::core::sampler::Sampler;
#[cfg(test)]
use crate::language::LanguageModel;

pub use crate::batching::{
    BATCHED_PER_SLOT_ATTN_MIN_KV, MAX_PAD_WASTE, PREFILL_CHUNK, batched_kv_keep_count,
    company_prefill_chunk, group_keep_count,
};
pub type Batch = crate::batching::Batch<crate::metal_language::MetalLanguageRuntime>;
pub type ContinuousBatch =
    crate::batching::ContinuousBatch<crate::metal_language::MetalLanguageRuntime>;

#[cfg(test)]
pub(crate) fn pack_layer(
    per: &[Vec<LayerCache>],
    li: usize,
    lens: &[usize],
    lmax: usize,
    cap: usize,
) -> anyhow::Result<LayerCache> {
    lisa_engine::cache::batching::pack_layer(
        &crate::metal_context::execution_context(),
        per,
        li,
        lens,
        lmax,
        cap,
    )
}
#[cfg(test)]
pub(crate) fn extract_streams(
    packed: &[LayerCache],
    survivors: &[usize],
    pack_lens: &[usize],
    lmax: usize,
) -> anyhow::Result<Vec<Vec<LayerCache>>> {
    lisa_engine::cache::batching::extract_streams(packed, survivors, pack_lens, lmax)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EosBatchModel {
        runtime: crate::metal_language::MetalLanguageRuntime,
        forward_calls: usize,
    }

    impl LanguageModel<crate::metal_language::MetalLanguageRuntime> for EosBatchModel {
        fn runtime(&self) -> &crate::metal_language::MetalLanguageRuntime {
            &self.runtime
        }
        fn max_position_embeddings(&self) -> usize {
            64
        }

        fn eos_token_id(&self) -> i64 {
            9
        }

        fn new_caches(&self) -> Vec<LayerCache> {
            Vec::new()
        }

        fn forward(
            &mut self,
            ids: &Array,
            _caches: Option<&mut Vec<LayerCache>>,
        ) -> anyhow::Result<(Array, Array)> {
            self.forward_calls += 1;
            let tokens = ids.to_vec1::<i32>()?;
            let values: Vec<f32> = tokens.iter().map(|&token| token as f32).collect();
            let hidden = Array::from_slice(&values, &[ids.dim(0), ids.dim(1), 1]);
            Ok((hidden.clone(), hidden))
        }

        fn forward_capture(
            &mut self,
            ids: &Array,
            caches: Option<&mut Vec<LayerCache>>,
            _capture: bool,
        ) -> anyhow::Result<(Array, Array)> {
            self.forward(ids, caches)
        }

        fn prefill(
            &mut self,
            tokens: &[u32],
            caches: &mut Vec<LayerCache>,
        ) -> anyhow::Result<Array> {
            let ids: Vec<i32> = tokens.iter().map(|&token| token as i32).collect();
            let input = Array::from_slice(&ids, &[1, tokens.len() as i32]);
            let (hidden, _) = self.forward(&input, Some(caches))?;
            Ok(hidden.index((0, tokens.len() as i32 - 1, ..)))
        }

        fn prefill_multi(
            &mut self,
            tokens: &[u32],
            caches: &mut Vec<LayerCache>,
        ) -> anyhow::Result<(Array, Array)> {
            let ids: Vec<i32> = tokens.iter().map(|&token| token as i32).collect();
            let input = Array::from_slice(&ids, &[1, tokens.len() as i32]);
            let (hidden, multi) = self.forward(&input, Some(caches))?;
            let index = (0, tokens.len() as i32 - 1, ..);
            Ok((hidden.index(index), multi.index(index)))
        }

        fn head(&self, mixed: &Array) -> lisa_metal::error::Result<Array> {
            const VOCAB: usize = 11;
            let hidden = mixed.to_vec1::<f32>().map_err(|error| {
                lisa_metal::error::Error::msg(format!("test hidden read failed: {error}"))
            })?;
            let mut logits = vec![-1.0f32; hidden.len() * VOCAB];
            for (row, value) in hidden.into_iter().enumerate() {
                let next = match value as u32 {
                    1 => 9,
                    2 => 3,
                    3 => 4,
                    4 => 10,
                    _ => 9,
                };
                logits[row * VOCAB + next] = 1.0;
            }
            Ok(Array::from_slice(
                &logits,
                &[mixed.size() as i32, VOCAB as i32],
            ))
        }

        fn warmup(&mut self, _seed: &[u32]) -> anyhow::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn batch_generation_stops_rows_whose_first_token_is_eos() -> anyhow::Result<()> {
        struct ResetEos;
        impl Drop for ResetEos {
            fn drop(&mut self) {
                crate::core::generate::set_eos_ids(Vec::new());
            }
        }

        let _reset_eos = ResetEos;
        crate::core::generate::set_eos_ids(vec![9, 10]);

        let mut all_eos = EosBatchModel {
            runtime: crate::metal_language::runtime(),
            forward_calls: 0,
        };
        let mut all_eos_sampler = Sampler::default();
        let mut no_op = |_: usize, _: u32| Ok(());
        let (all_eos_batch, all_eos_tokens) = Batch::generate(
            &mut all_eos,
            &[vec![9], vec![9]],
            4,
            &mut all_eos_sampler,
            &mut no_op,
        )?;
        anyhow::ensure!(all_eos_tokens == [vec![9], vec![9]]);
        anyhow::ensure!(all_eos_batch.done == [true, true]);
        anyhow::ensure!(all_eos.forward_calls == 1);

        let mut mixed = EosBatchModel {
            runtime: crate::metal_language::runtime(),
            forward_calls: 0,
        };
        let mut mixed_sampler = Sampler::default();
        let mut no_op = |_: usize, _: u32| Ok(());
        let (mixed_batch, mixed_tokens) = Batch::generate(
            &mut mixed,
            &[vec![1], vec![2]],
            5,
            &mut mixed_sampler,
            &mut no_op,
        )?;
        anyhow::ensure!(mixed_tokens == [vec![9], vec![3, 4, 10]]);
        anyhow::ensure!(mixed_batch.done == [true, true]);

        Ok(())
    }

    // Expected keep-count vectors.
    #[test]
    fn keep_count_caps_pad_waste() {
        // 10 beside 900 pads to exactly 1.98x useful — the pathological pair.
        assert_eq!(batched_kv_keep_count(&[10, 900]), 0);
        // Equal lengths pad nothing.
        assert_eq!(batched_kv_keep_count(&[64, 64, 64]), 3);
        // A mild skew keeps the pair, a hard one splits it.
        assert_eq!(batched_kv_keep_count(&[100, 200]), 2); // 1.33x
        assert_eq!(batched_kv_keep_count(&[100, 400]), 0); // 1.6x > 1.5x
    }

    #[test]
    fn per_slot_floor_voids_the_cap() {
        let skew = [100usize, 850, 900, 1300];
        // Without the per-slot arm the cap trims to the 3 shortest.
        assert_eq!(group_keep_count(&skew, false), 3);
        // With it (max >= 1024) every slot batches.
        assert_eq!(group_keep_count(&skew, true), 4);
        // Below the floor the stacked cap applies even when capable.
        assert_eq!(group_keep_count(&[10, 900], true), 0);
    }

    #[test]
    fn company_chunk_narrows_only_while_decoding() {
        assert_eq!(company_prefill_chunk(8192, 0), 8192);
        assert_eq!(company_prefill_chunk(8192, 1), 2048);
        assert_eq!(company_prefill_chunk(1024, 3), 1024);
    }

    #[test]
    fn glm_indexer_cache_repack_preserves_each_stream_tape() -> anyhow::Result<()> {
        let mut per = Vec::new();
        for stream in 0..2 {
            let raw: Vec<f32> = (0..8).map(|i| (stream * 100 + i) as f32).collect();
            let compact: Vec<f32> = (0..12).map(|i| (stream * 1000 + i) as f32).collect();
            let mut tape = IndexerTape::new(crate::metal_context::execution_context());
            let pooled: Vec<f32> = (0..6).map(|i| (stream * 10000 + i) as f32).collect();
            if stream == 0 {
                tape.append(&Array::from_slice(&raw, &[1, 4, 2]), 4)?;
                tape.append_compact_f16(&Array::from_slice(&compact, &[1, 4, 3]))?;
                tape.push_pooled_f16(&Array::from_slice(&pooled, &[1, 2, 3]))?;
            } else {
                tape.append(&Array::from_slice(&raw[..4], &[1, 2, 2]), 4)?;
                tape.append_compact_f16(&Array::from_slice(&compact[..6], &[1, 2, 3]))?;
                tape.push_pooled_f16(&Array::from_slice(&pooled[..3], &[1, 1, 3]))?;
                tape.append(&Array::from_slice(&raw[4..], &[1, 2, 2]), 4)?;
                tape.append_compact_f16(&Array::from_slice(&compact[6..], &[1, 2, 3]))?;
                tape.push_pooled_f16(&Array::from_slice(&pooled[3..], &[1, 1, 3]))?;
            }
            let mut cache =
                FullAttentionCache::new(crate::metal_context::execution_context(), 1, 128);
            cache.offset = 4;
            cache.capacity = 4;
            cache.indexer_tape = tape;
            per.push(vec![LayerCache::Full(cache)]);
        }
        let packed = pack_layer(&per, 0, &[4, 4], 4, 8)?;
        let extracted = extract_streams(&[packed], &[1, 0], &[4, 4], 4)?;
        for (row, stream) in [1usize, 0].into_iter().enumerate() {
            let LayerCache::Full(cache) = &extracted[row][0] else {
                anyhow::bail!("expected full DSA cache");
            };
            anyhow::ensure!(cache.keys.is_none() && cache.values.is_none());
            anyhow::ensure!(cache.offset == 4 && cache.indexer_tape.total == 4);
            let raw = cache
                .indexer_tape
                .raw_slice(0, 4)
                .ok_or_else(|| anyhow::anyhow!("missing raw row"))?
                .to_vec1::<f32>()?;
            anyhow::ensure!(
                raw == (0..8)
                    .map(|i| (stream * 100 + i) as f32)
                    .collect::<Vec<_>>(),
                "raw indexer row order changed during pack/extract"
            );
            let compact = cache
                .indexer_tape
                .compact_slice(0, 4)
                .ok_or_else(|| anyhow::anyhow!("missing compact row"))?
                .to_vec1::<half::f16>()?;
            anyhow::ensure!(
                compact
                    == (0..12)
                        .map(|i| half::f16::from_f32((stream * 1000 + i) as f32))
                        .collect::<Vec<_>>(),
                "compact latent row order changed during pack/extract"
            );
            let pooled = cache
                .indexer_tape
                .pooled_view()
                .ok_or_else(|| anyhow::anyhow!("missing pooled row"))?
                .to_vec1::<half::f16>()?;
            anyhow::ensure!(
                pooled
                    == (0..6)
                        .map(|i| half::f16::from_f32((stream * 10000 + i) as f32))
                        .collect::<Vec<_>>(),
                "pooled key row order changed during pack/extract"
            );
        }
        Ok(())
    }
}
