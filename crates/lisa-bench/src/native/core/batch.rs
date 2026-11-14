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
use crate::core::config::ReadConfig;
use lisa_metal::Array;
#[cfg(test)]
use lisa_metal::ops::indexing::IndexOp;

use crate::core::cache::LayerCache;
#[cfg(test)]
use crate::core::cache::{FullAttentionCache, IndexerTape};

use crate::core::sampler::Sampler;
#[cfg(test)]
use crate::eos::is_eos;
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
use crate::batching::prefill_one;
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
        assert!(1.5 < (2.0 * 900.0) / (10.0 + 900.0));
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

    #[test]
    fn glm_continuous_batch_decode_matches_unpacked_rows_with_real_checkpoint() -> anyhow::Result<()>
    {
        fn write_batch_capture(
            prefix: &std::path::Path,
            suffix: &str,
            bytes: &[u8],
        ) -> anyhow::Result<()> {
            let mut path = prefix.as_os_str().to_os_string();
            path.push(format!(".{suffix}"));
            std::fs::write(std::path::PathBuf::from(path), bytes)?;
            Ok(())
        }

        fn encode_u32(values: &[u32]) -> Vec<u8> {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        }

        fn encode_f32(values: &[f32]) -> Vec<u8> {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect()
        }

        fn snapshot_cache_layers(caches: &[LayerCache]) -> anyhow::Result<Vec<Vec<u8>>> {
            caches
                .iter()
                .map(|cache| {
                    cache
                        .snapshot_prefix()
                        .encode(&crate::metal_context::device())
                })
                .collect()
        }

        fn capture_cache_layers(
            caches: &[LayerCache],
            path: &std::path::Path,
        ) -> anyhow::Result<()> {
            let capture = crate::core::capture::Capture {
                layers: (0..caches.len()).collect(),
                ..Default::default()
            };
            let mut metadata = Vec::with_capacity(caches.len() * 7);
            for (layer, cache) in caches.iter().enumerate() {
                let prefix = format!("glm53.cache.layer_{layer:02}");
                match cache {
                    LayerCache::Native(_) => anyhow::bail!("GLM capture received a native cache"),
                    LayerCache::Full(full) => {
                        let tape = &full.indexer_tape;
                        metadata.extend_from_slice(&[
                            0.0,
                            full.offset as f32,
                            tape.total as f32,
                            tape.raw_start() as f32,
                            tape.pooled_upto as f32,
                            tape.compact_total() as f32,
                            tape.compression_ratio() as f32,
                        ]);
                        if let Some(keys) = &full.keys {
                            capture.store(
                                &format!("{prefix}.keys"),
                                &keys.index((.., .., 0..full.offset as i32, ..)),
                            );
                        }
                        if let Some(values) = &full.values {
                            capture.store(
                                &format!("{prefix}.values"),
                                &values.index((.., .., 0..full.offset as i32, ..)),
                            );
                        }
                        if let Some(raw) = tape.raw_slice(tape.raw_start(), tape.total) {
                            capture.store(&format!("{prefix}.indexer_raw"), &raw);
                        }
                        if let Some(compact) = tape.compact_slice(0, tape.compact_total()) {
                            capture.store(&format!("{prefix}.compact_cache"), &compact);
                        }
                        if let Some(pooled) = tape.pooled_view() {
                            capture.store(&format!("{prefix}.pooled_cache"), &pooled);
                        }
                        if let Some(ple) = &full.ple_conv {
                            capture.store(&format!("{prefix}.ple_conv"), ple);
                        }
                    }
                    LayerCache::Linear(linear) => {
                        metadata.extend_from_slice(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
                        if let Some(conv) = &linear.conv {
                            capture.store(&format!("{prefix}.kda_conv"), conv);
                        }
                        if let Some(ssm) = &linear.ssm {
                            capture.store(&format!("{prefix}.kda_ssm"), ssm);
                        }
                        if let Some(ple) = &linear.ple_conv {
                            capture.store(&format!("{prefix}.ple_conv"), ple);
                        }
                    }
                }
            }
            capture.store(
                "glm53.cache_meta",
                &Array::from_slice(&metadata, &[caches.len() as i32, 7]),
            );
            capture.flush(
                path.to_str()
                    .ok_or_else(|| anyhow::anyhow!("cache capture path is not UTF-8"))?,
            )
        }

        fn assert_cache_layers_equal(
            actual: &[Vec<u8>],
            expected: &[Vec<u8>],
            label: &str,
        ) -> anyhow::Result<()> {
            anyhow::ensure!(
                actual.len() == expected.len(),
                "{label}: cache layer count differs: {} != {}",
                actual.len(),
                expected.len()
            );
            for (layer, (actual, expected)) in actual.iter().zip(expected).enumerate() {
                if actual != expected {
                    anyhow::bail!(
                        "{label}: encoded live cache state differs at layer {layer} kind={} bytes={}/{}",
                        actual.first().copied().unwrap_or_default(),
                        actual.len(),
                        expected.len()
                    );
                }
            }
            Ok(())
        }

        fn serial_forward_token(
            tower: &mut dyn LanguageModel<crate::metal_language::MetalLanguageRuntime>,
            caches: &mut Vec<LayerCache>,
            token: u32,
        ) -> anyhow::Result<Vec<f32>> {
            let input = Array::from_slice(&[token as i32], &[1, 1]);
            let (hidden, _) = tower.forward(&input, Some(caches))?;
            let last = hidden.index((.., hidden.dim(1) - 1, ..));
            Ok(tower
                .head(&last)?
                .as_dtype(lisa_metal::Dtype::Float32)?
                .index((0, ..))
                .to_vec1::<f32>()?)
        }

        fn serial_cache_after(
            tower: &mut dyn LanguageModel<crate::metal_language::MetalLanguageRuntime>,
            prompt: &[u32],
            decoded: &[u32],
        ) -> anyhow::Result<(Vec<LayerCache>, Vec<f32>, Vec<f32>)> {
            let prefilled = prefill_one(tower, prompt)?;
            let prompt_logits = tower
                .head(&prefilled.last_hidden)?
                .as_dtype(lisa_metal::Dtype::Float32)?
                .index((0, ..))
                .to_vec1::<f32>()?;
            let mut caches = prefilled.caches;
            let mut next_logits = prompt_logits.clone();
            for &token in decoded {
                next_logits = serial_forward_token(tower, &mut caches, token)?;
            }
            Ok((caches, prompt_logits, next_logits))
        }

        if !lisa_metal::device::metal_available() {
            eprintln!("[glm53-batch] skipping: no Metal device available");
            return Ok(());
        }
        let Some(model_dir) = std::env::var_os("LISA_GLM53_MODEL") else {
            eprintln!("[glm53-batch] skipping: LISA_GLM53_MODEL is unset");
            return Ok(());
        };
        let model_dir = std::path::Path::new(&model_dir);
        let config =
            crate::checkpoint::glm53::ModelConfig::from_json(&model_dir.join("config.json"))?;
        let vocab_size = config.vocab_size;
        let sparse_boundary = config
            .index_topk
            .saturating_add(config.index_kpool.saturating_sub(1));
        let no_expert_residency = std::env::var_os("LISA_GLM53_TEST_NO_EXPERT_RESIDENCY").is_some();
        let deferred_expert_residency =
            std::env::var_os("LISA_GLM53_TEST_DEFER_EXPERT_RESIDENCY").is_some();
        anyhow::ensure!(
            !(no_expert_residency && deferred_expert_residency),
            "select only one GLM batch residency diagnostic mode"
        );
        let mut tower = if no_expert_residency {
            eprintln!(
                "[glm53-batch] diagnostic test uses per-dispatch expert resource declarations"
            );
            crate::bindings::glm53_affine_tower::Tower::load_without_expert_residency(
                crate::metal_language::runtime(),
                model_dir,
                config,
            )?
        } else if deferred_expert_residency {
            eprintln!("[glm53-batch] diagnostic test defers model-wide residency preparation");
            crate::bindings::glm53_affine_tower::Tower::load_with_deferred_expert_residency(
                crate::metal_language::runtime(),
                model_dir,
                config,
            )?
        } else {
            crate::bindings::glm53_affine_tower::Tower::load(model_dir, config)?
        };
        anyhow::ensure!(tower.batch_decode_ok());
        anyhow::ensure!(tower.batch_decode_equal_length_only());

        // Use ordinary tokenizer output (rather than reserved low token ids)
        // while exercising every GLM layer, KDA recurrent state, DSA
        // raw/pooled/compact cache, MoE, and mHC in one lockstep decode.
        let tokenizer = crate::core::tokenizer::Tokenizer::load(model_dir)?;
        let mut prompt0 = tokenizer.encode("What makes the sky blue?", false)?;
        let mut prompt1 = tokenizer.encode("Why does the sun look bright?", false)?;
        let pad_token = *tokenizer
            .encode(" the", false)?
            .first()
            .ok_or_else(|| anyhow::anyhow!("GLM tokenizer returned no padding token"))?;
        let ragged_prompts = [prompt0.clone(), prompt1.clone()];
        let requested_context = std::env::var("LISA_GLM53_BATCH_CONTEXT")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(0);
        let target_context = requested_context.max(prompt0.len()).max(prompt1.len());
        let row_pad_tokens = if requested_context > 0 {
            [
                *prompt0.last().unwrap_or(&pad_token),
                *prompt1.last().unwrap_or(&pad_token),
            ]
        } else {
            [pad_token; 2]
        };
        prompt0.resize(target_context, row_pad_tokens[0]);
        prompt1.resize(target_context, row_pad_tokens[1]);
        let prompts = [prompt0, prompt1];
        let ds4_capture_prefix =
            std::env::var_os("LISA_GLM53_BATCH_DS4_CAPTURE_PREFIX").map(std::path::PathBuf::from);
        if let Some(prefix) = &ds4_capture_prefix {
            if let Some(parent) = prefix.parent().filter(|path| !path.as_os_str().is_empty()) {
                std::fs::create_dir_all(parent)?;
            }
            for (row, prompt) in prompts.iter().enumerate() {
                write_batch_capture(
                    prefix,
                    &format!("prompt-row-{row}.u32"),
                    &encode_u32(prompt),
                )?;
            }
            let metadata = format!(
                "rows=2\nvocab={vocab_size}\nprompt_len={}\nprompt_lens={},{}\n",
                prompts[0].len(),
                prompts[0].len(),
                prompts[1].len()
            );
            write_batch_capture(prefix, "meta.txt", metadata.as_bytes())?;
        }
        let mut batch = ContinuousBatch::empty(8);
        let mut samplers = vec![Sampler::default(), Sampler::default()];
        let first = [
            batch.admit(&mut tower, &prompts[0], &mut samplers[0])?,
            batch.admit(&mut tower, &prompts[1], &mut samplers[1])?,
        ];
        anyhow::ensure!(batch.fed[0].len() == batch.fed[1].len());

        let make_capture = |preserve_dispatch| crate::core::capture::Capture {
            layers: (0..45).collect(),
            state_only: false,
            kda_prefill_only: false,
            preserve_dispatch,
            entries: std::sync::Mutex::new(std::collections::HashMap::new()),
            shapes: std::sync::Mutex::new(std::collections::HashMap::new()),
            flushed: std::sync::atomic::AtomicBool::new(false),
            ..Default::default()
        };

        // Decode once from copies of the independent session caches, then
        // compare the row-local DS4-shaped batch path against serial rows.
        let per_caches = batch
            .unpacked_caches_for_test()
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("GLM continuous batch packed its session caches"))?;
        let pack_lens: Vec<usize> = batch.fed.iter().map(Vec::len).collect();
        let pack_lmax = *pack_lens.iter().max().unwrap_or(&0);
        let packed_for_serial: Vec<LayerCache> = (0..per_caches[0].len())
            .map(|layer| pack_layer(per_caches, layer, &pack_lens, pack_lmax, pack_lmax + 16))
            .collect::<anyhow::Result<_>>()?;
        let mut serial_caches =
            extract_streams(&packed_for_serial, &[0, 1], &pack_lens, pack_lmax)?;
        let mut serial_logits = Vec::with_capacity(2);
        let mut serial_states = Vec::with_capacity(2);
        for row in 0..2 {
            let input = Array::from_slice(&[first[row] as i32], &[1, 1]);
            tower.set_context_tails(vec![batch.tails[row].clone()]);
            tower.capture = Some(make_capture(false));
            let (hidden, _) = tower.forward(&input, Some(&mut serial_caches[row]))?;
            let last = hidden.index((.., hidden.dim(1) - 1, ..));
            let logits = tower.head(&last)?.as_dtype(lisa_metal::Dtype::Float32)?;
            serial_logits.push(logits.index((0, ..)).to_vec1::<f32>()?);
            serial_states.push(
                tower
                    .capture
                    .as_ref()
                    .expect("test capture is installed")
                    .entries
                    .lock()
                    .expect("capture lock")
                    .clone(),
            );
        }

        let input = Array::from_slice(&[first[0] as i32, first[1] as i32], &[2, 1]);
        tower.set_context_tails(batch.tails.clone());
        tower.capture = Some(make_capture(true));
        let batch_caches = batch
            .unpacked_caches_mut_for_test()
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("GLM continuous batch lost session caches"))?;
        let (hidden, _) = tower.forward_ragged(&input, batch_caches)?;
        let capture_drain_path = std::env::temp_dir().join(format!(
            "lisa-glm53-batch-capture-{}.npz",
            std::process::id()
        ));
        tower
            .capture
            .as_ref()
            .expect("test capture is installed")
            .flush(
                capture_drain_path
                    .to_str()
                    .ok_or_else(|| anyhow::anyhow!("batch capture path is not UTF-8"))?,
            )?;
        if let Some(path) = std::env::var_os("LISA_GLM53_KDA_BATCH_CAPTURE") {
            let path = std::path::Path::new(&path);
            let capture = tower.capture.as_ref().expect("test capture is installed");
            let capture_all_kda_layers =
                std::env::var_os("LISA_GLM53_KDA_BATCH_CAPTURE_ALL").is_some();
            let capture_o_proj_only =
                std::env::var_os("LISA_GLM53_KDA_BATCH_CAPTURE_O_PROJ_ONLY").is_some();
            let capture_layer = |name: &str| {
                name.starts_with("L")
                    && name.contains(".kda_")
                    && (!capture_o_proj_only
                        || name.ends_with(".kda_decode_out")
                        || name.ends_with(".kda_o_proj"))
                    && (capture_all_kda_layers || name.starts_with("L4.kda_"))
            };
            let entries = capture
                .entries
                .lock()
                .expect("capture lock")
                .iter()
                .filter(|(name, _)| capture_layer(name))
                .map(|(name, values)| (name.clone(), values.clone()))
                .collect();
            let shapes = capture
                .shapes
                .lock()
                .expect("capture lock")
                .iter()
                .filter(|(name, _)| capture_layer(name))
                .map(|(name, shape)| (name.clone(), shape.clone()))
                .collect();
            crate::core::capture::Capture {
                layers: if capture_all_kda_layers {
                    (0..45).collect()
                } else {
                    vec![4]
                },
                state_only: false,
                kda_prefill_only: false,
                entries: std::sync::Mutex::new(entries),
                shapes: std::sync::Mutex::new(shapes),
                flushed: std::sync::atomic::AtomicBool::new(false),
                ..Default::default()
            }
            .flush(
                path.to_str()
                    .ok_or_else(|| anyhow::anyhow!("KDA batch capture path is not UTF-8"))?,
            )?;
            eprintln!(
                "[glm53-batch] wrote native-batch KDA inputs to {}",
                path.display()
            );
        }
        if let Some(path) = std::env::var_os("LISA_GLM53_MHC_BATCH_CAPTURE") {
            let path = std::path::Path::new(&path);
            let capture = tower.capture.as_ref().expect("test capture is installed");
            let capture_layer = |name: &str| {
                name.starts_with("L")
                    && (name.ends_with(".mhc_attn_in")
                        || name.ends_with(".mhc_attn_mixes")
                        || name.ends_with(".mhc_attn_split")
                        || name.ends_with(".mhc_attn_collapsed")
                        || name.ends_with(".mhc_attn_normed")
                        || name.ends_with(".mhc_attn_post")
                        || name.ends_with(".mhc_attn_comb")
                        || name.contains(".dec_attn_out_s")
                        || name.ends_with(".state_after_attn")
                        || name.ends_with(".mhc_ffn_mixes")
                        || name.ends_with(".mhc_ffn_split")
                        || name.ends_with(".mhc_ffn_collapsed")
                        || name.ends_with(".mhc_ffn_normed")
                        || name.ends_with(".mhc_ffn_post")
                        || name.ends_with(".mhc_ffn_comb")
                        || name.contains(".dec_moe_out_s")
                        || name.ends_with(".state_after_ffn"))
            };
            let entries = capture
                .entries
                .lock()
                .expect("capture lock")
                .iter()
                .filter(|(name, _)| capture_layer(name))
                .map(|(name, values)| (name.clone(), values.clone()))
                .collect();
            let shapes = capture
                .shapes
                .lock()
                .expect("capture lock")
                .iter()
                .filter(|(name, _)| capture_layer(name))
                .map(|(name, shape)| (name.clone(), shape.clone()))
                .collect();
            crate::core::capture::Capture {
                layers: (0..45).collect(),
                state_only: false,
                kda_prefill_only: false,
                entries: std::sync::Mutex::new(entries),
                shapes: std::sync::Mutex::new(shapes),
                flushed: std::sync::atomic::AtomicBool::new(false),
                ..Default::default()
            }
            .flush(
                path.to_str()
                    .ok_or_else(|| anyhow::anyhow!("mHC batch capture path is not UTF-8"))?,
            )?;
            eprintln!(
                "[glm53-batch] wrote native-batch mHC expansion inputs to {}",
                path.display()
            );
        }
        let packed_states = tower
            .capture
            .as_ref()
            .expect("test capture is installed")
            .entries
            .lock()
            .expect("capture lock")
            .clone();
        let _ = std::fs::remove_file(&capture_drain_path);
        if let Some(path) = std::env::var_os("LISA_GLM53_BATCH_DS4_STAGE_CAPTURE") {
            let capture = tower.capture.as_ref().expect("test capture is installed");
            let entries: std::collections::HashMap<String, Vec<f32>> = capture
                .entries
                .lock()
                .expect("capture lock")
                .iter()
                .filter(|(name, _)| name.starts_with("L3."))
                .map(|(name, values)| (name.clone(), values.clone()))
                .collect();
            let shapes: std::collections::HashMap<String, Vec<i64>> = capture
                .shapes
                .lock()
                .expect("capture lock")
                .iter()
                .filter(|(name, _)| name.starts_with("L3."))
                .map(|(name, shape)| (name.clone(), shape.clone()))
                .collect();
            if !entries.contains_key("L3.state_after_ffn") {
                let mut names: Vec<_> = entries.keys().cloned().collect();
                names.sort();
                eprintln!("[glm53-batch] L3 capture keys: {}", names.join(", "));
            }
            crate::core::capture::Capture {
                entries: std::sync::Mutex::new(entries),
                shapes: std::sync::Mutex::new(shapes),
                ..Default::default()
            }
            .flush(
                std::path::Path::new(&path)
                    .to_str()
                    .ok_or_else(|| anyhow::anyhow!("batch stage capture path is not UTF-8"))?,
            )?;
        }
        if let Some(path) = std::env::var_os("LISA_GLM53_ROUTER_BATCH_INPUT_CAPTURE") {
            let values = packed_states
                .get("L3.moe_input")
                .ok_or_else(|| anyhow::anyhow!("captured GLM layer-3 MoE input is missing"))?;
            anyhow::ensure!(values.len() == 2 * 4096);
            let bytes: Vec<u8> = values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect();
            if let Some(parent) = std::path::Path::new(&path).parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, bytes)?;
            eprintln!("[glm53-batch] captured layer-3 two-row MoE input for DS4 replay");
        }
        let last = hidden.index((.., hidden.dim(1) - 1, ..));
        let batch_caches = batch
            .unpacked_caches_for_test()
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("GLM continuous batch lost session caches"))?;
        let batched_logits = tower
            .head_ragged(&last, batch_caches)?
            .as_dtype(lisa_metal::Dtype::Float32)?;
        if let Some(prefix) = &ds4_capture_prefix {
            let values = batched_logits.to_vec1::<f32>()?;
            anyhow::ensure!(values.len() == 2 * vocab_size);
            write_batch_capture(prefix, "logits-step-1.f32", &encode_f32(&values))?;
            write_batch_capture(prefix, "inputs-step-1.u32", &encode_u32(&first))?;
        }
        let mut max_abs = 0.0f32;
        let mut rel_l2 = 0.0f64;
        let mut argmax_mismatch = false;
        for row in 0..2 {
            let normalize = |name: &str| {
                if let Some((prefix, suffix)) = name.rsplit_once("_s") {
                    if !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit()) {
                        return prefix.to_string();
                    }
                }
                name.to_string()
            };
            let mut state_diffs = Vec::new();
            for layer in 0..45 {
                let prefix = format!("L{layer}.");
                let mut serial_by_stage = std::collections::HashMap::new();
                for (name, values) in &serial_states[row] {
                    let static_kda_parameter = matches!(
                        name.rsplit('.').next(),
                        Some(
                            "kda_q_conv_weight"
                                | "kda_k_conv_weight"
                                | "kda_v_conv_weight"
                                | "kda_a_log"
                                | "kda_dt_bias"
                                | "kda_output_norm_weight"
                                | "kda_meta"
                        )
                    );
                    if name.starts_with(&prefix) && !static_kda_parameter {
                        serial_by_stage.insert(normalize(name), values);
                    }
                }
                let mut stages: Vec<_> = serial_by_stage.into_iter().collect();
                stages.sort_by(|a, b| a.0.cmp(&b.0));
                let mut layer_diffs = Vec::new();
                for (stage, serial) in stages {
                    let Some(packed) = packed_states
                        .iter()
                        .find(|(name, _)| normalize(name) == stage)
                        .map(|(_, values)| values)
                    else {
                        continue;
                    };
                    let offset = if packed.len() == serial.len() * 2 {
                        row * serial.len()
                    } else if packed.len() == serial.len() && row == 1 {
                        0
                    } else {
                        continue;
                    };
                    anyhow::ensure!(
                        packed.len() == serial.len() * 2 || packed.len() == serial.len(),
                        "captured {stage} has unexpected dimensions: serial={}, packed={}",
                        serial.len(),
                        packed.len()
                    );
                    let packed_row = &packed[offset..offset + serial.len()];
                    let mut stage_max = 0.0f32;
                    let mut stage_nonfinite = 0usize;
                    for (&a, &b) in serial.iter().zip(packed_row) {
                        if !a.is_finite() || !b.is_finite() {
                            stage_nonfinite += 1;
                        } else {
                            stage_max = stage_max.max((a - b).abs());
                        }
                    }
                    if stage_max > 0.0 || stage_nonfinite > 0 {
                        layer_diffs.push((stage, stage_max, stage_nonfinite));
                    }
                }
                for (stage, stage_max, stage_nonfinite) in layer_diffs {
                    state_diffs.push((layer, stage, stage_max, stage_nonfinite));
                }
            }
            let graph_order_diffs: Vec<_> = state_diffs.iter().take(12).cloned().collect();
            let layer4_projection_diffs: Vec<_> = state_diffs
                .iter()
                .filter(|(layer, stage, _, _)| {
                    *layer == 4
                        && matches!(
                            stage.rsplit('.').next(),
                            Some(
                                "kda_q_proj"
                                    | "kda_k_proj"
                                    | "kda_v_proj"
                                    | "kda_gate_raw"
                                    | "kda_beta_logits"
                                    | "kda_gb_proj"
                            )
                        )
                })
                .cloned()
                .collect();
            state_diffs.sort_by(|a, b| b.2.total_cmp(&a.2));
            let largest_state_diffs: Vec<_> = state_diffs.iter().take(8).collect();
            let actual = batched_logits.index((row as i32, ..)).to_vec1::<f32>()?;
            anyhow::ensure!(actual.len() == serial_logits[row].len());
            let mut err2 = 0.0f64;
            let mut ref2 = 0.0f64;
            let mut actual_argmax = 0usize;
            let mut ref_argmax = 0usize;
            for i in 0..actual.len() {
                anyhow::ensure!(
                    actual[i].is_finite() && serial_logits[row][i].is_finite(),
                    "nonfinite GLM batch logit in row {row}, vocab index {i}: packed={}, serial={}",
                    actual[i],
                    serial_logits[row][i]
                );
                let delta = actual[i] - serial_logits[row][i];
                max_abs = max_abs.max(delta.abs());
                err2 += (delta as f64) * (delta as f64);
                ref2 += (serial_logits[row][i] as f64) * (serial_logits[row][i] as f64);
                if actual[i] > actual[actual_argmax] {
                    actual_argmax = i;
                }
                if serial_logits[row][i] > serial_logits[row][ref_argmax] {
                    ref_argmax = i;
                }
            }
            rel_l2 = rel_l2.max((err2 / ref2.max(f64::MIN_POSITIVE)).sqrt());
            let top = |values: &[f32]| {
                let mut ids: Vec<usize> = (0..values.len()).collect();
                ids.sort_unstable_by(|&a, &b| values[b].total_cmp(&values[a]));
                ids.truncate(5);
                ids.into_iter().map(|i| (i, values[i])).collect::<Vec<_>>()
            };
            let mismatch = actual_argmax != ref_argmax;
            argmax_mismatch |= mismatch;
            eprintln!(
                "[glm53-batch] row={row} fed_token={} argmax packed={actual_argmax} serial={ref_argmax} mismatch={mismatch}; layer4 KDA projection deltas={layer4_projection_diffs:?}; earliest captured stage differences={graph_order_diffs:?}; largest captured stage differences={largest_state_diffs:?}; top5 packed={:?} serial={:?}",
                first[row],
                top(&actual),
                top(&serial_logits[row]),
            );
        }
        eprintln!(
            "[glm53-batch] packed-vs-unpacked decode logits: max_abs={max_abs:.6e}, rel_l2={rel_l2:.6e}"
        );
        anyhow::ensure!(
            !argmax_mismatch,
            "GLM packed and unpacked decode argmax differs"
        );

        // Continue from the just-produced greedy token so the comparison also
        // covers repeated updates to KDA recurrent state and packed DSA tapes,
        // rather than proving only a single read from the prefill cache.
        let argmax = |values: &[f32]| {
            values
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i as u32)
                .unwrap_or(0)
        };
        let mut feed = serial_logits
            .iter()
            .map(|row| argmax(row))
            .collect::<Vec<_>>();
        for decode_step in 2..=3 {
            let batch_input = feed.clone();
            let mut next_serial_logits = Vec::with_capacity(2);
            for row in 0..2 {
                let input = Array::from_slice(&[feed[row] as i32], &[1, 1]);
                tower.set_context_tails(vec![batch.tails[row].clone()]);
                tower.capture = None;
                let (hidden, _) = tower.forward(&input, Some(&mut serial_caches[row]))?;
                let last = hidden.index((.., hidden.dim(1) - 1, ..));
                let logits = tower.head(&last)?.as_dtype(lisa_metal::Dtype::Float32)?;
                next_serial_logits.push(logits.index((0, ..)).to_vec1::<f32>()?);
            }

            let input = Array::from_slice(&[feed[0] as i32, feed[1] as i32], &[2, 1]);
            tower.set_context_tails(batch.tails.clone());
            tower.capture = None;
            let batch_caches = batch
                .unpacked_caches_mut_for_test()
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("GLM continuous batch lost session caches"))?;
            let (hidden, _) = tower.forward_ragged(&input, batch_caches)?;
            let last = hidden.index((.., hidden.dim(1) - 1, ..));
            let batch_caches = batch
                .unpacked_caches_for_test()
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("GLM continuous batch lost session caches"))?;
            let logits = tower
                .head_ragged(&last, batch_caches)?
                .as_dtype(lisa_metal::Dtype::Float32)?;
            if let Some(prefix) = &ds4_capture_prefix {
                let values = logits.to_vec1::<f32>()?;
                anyhow::ensure!(values.len() == 2 * vocab_size);
                write_batch_capture(
                    prefix,
                    &format!("logits-step-{decode_step}.f32"),
                    &encode_f32(&values),
                )?;
                write_batch_capture(
                    prefix,
                    &format!("inputs-step-{decode_step}.u32"),
                    &encode_u32(&batch_input),
                )?;
            }

            let mut step_max_abs = 0.0f32;
            let mut step_err2 = 0.0f64;
            let mut step_ref2 = 0.0f64;
            for row in 0..2 {
                let packed = logits.index((row as i32, ..)).to_vec1::<f32>()?;
                anyhow::ensure!(packed.len() == next_serial_logits[row].len());
                for (&actual, &reference) in packed.iter().zip(&next_serial_logits[row]) {
                    anyhow::ensure!(
                        actual.is_finite() && reference.is_finite(),
                        "nonfinite GLM batch logit at decode step {decode_step}, row {row}"
                    );
                    let delta = actual - reference;
                    step_max_abs = step_max_abs.max(delta.abs());
                    step_err2 += (delta as f64) * (delta as f64);
                    step_ref2 += (reference as f64) * (reference as f64);
                }
                let packed_next = argmax(&packed);
                let serial_next = argmax(&next_serial_logits[row]);
                anyhow::ensure!(
                    packed_next == serial_next,
                    "GLM packed and unpacked row {row} diverged at decode step {decode_step}: packed={packed_next}, serial={serial_next}"
                );
                feed[row] = serial_next;
            }
            let step_rel_l2 = (step_err2 / step_ref2.max(f64::MIN_POSITIVE)).sqrt();
            eprintln!(
                "[glm53-batch] decode_step={decode_step} max_abs={step_max_abs:.6e} rel_l2={step_rel_l2:.6e} greedy_rows_match=true"
            );
            anyhow::ensure!(
                step_rel_l2 < 2.0e-4,
                "GLM packed-vs-unpacked relative L2 drift {step_rel_l2:.6e} at decode step {decode_step}"
            );
        }

        if requested_context == 0 {
            // Exercise the public session-prefill API as well as the lower
            // level ContinuousBatch path above. Equal-length prompts take
            // independent serial prefills followed by packed decode.
            let mut equal_sampler = Sampler::default();
            let mut on_token = |_: usize, _: u32| Ok(());
            let (_, equal_tokens) =
                Batch::generate(&mut tower, &prompts, 3, &mut equal_sampler, &mut on_token)?;
            for (row, prompt) in prompts.iter().enumerate() {
                let mut serial_sampler = Sampler::default();
                let mut serial_on_token = |_: u32| Ok(());
                let (serial_tokens, _) = crate::core::generate::generate(
                    &mut tower,
                    prompt,
                    3,
                    &mut serial_sampler,
                    0,
                    &mut serial_on_token,
                )?;
                anyhow::ensure!(
                    equal_tokens[row] == serial_tokens,
                    "equal-position GLM batch row {row} generation differs from serial: batch={:?}, serial={serial_tokens:?}",
                    equal_tokens[row]
                );
                eprintln!(
                    "[glm53-batch] equal-position public row={row} generated={:?} matches serial",
                    equal_tokens[row]
                );
            }

            // DS4 permits sessions at different decode positions. LISA keeps
            // each DSA tape and KDA recurrent state independent while batching
            // their stateless projections, mHC, and MLP work.
            anyhow::ensure!(ragged_prompts[0].len() != ragged_prompts[1].len());
            let mut ragged_sampler = Sampler::default();
            let mut on_token = |_: usize, _: u32| Ok(());
            let (_, batched_tokens) = Batch::generate(
                &mut tower,
                &ragged_prompts,
                3,
                &mut ragged_sampler,
                &mut on_token,
            )?;
            anyhow::ensure!(batched_tokens.iter().all(|tokens| tokens.len() == 3));

            // Run the serial references after the complete batch generation.
            // Interleaving forward/readback calls between batch rounds can
            // perturb Metal command scheduling and hide ordering races.
            for (row, prompt) in ragged_prompts.iter().enumerate() {
                let mut serial_sampler = Sampler::default();
                let mut serial_on_token = |_: u32| Ok(());
                let (serial_tokens, _) = crate::core::generate::generate(
                    &mut tower,
                    prompt,
                    3,
                    &mut serial_sampler,
                    0,
                    &mut serial_on_token,
                )?;
                anyhow::ensure!(
                    batched_tokens[row] == serial_tokens,
                    "ragged GLM batch row {row} generation differs from serial: batch={:?}, serial={serial_tokens:?}",
                    batched_tokens[row]
                );
                eprintln!(
                    "[glm53-batch] unequal-position row={row} generated={:?} matches serial",
                    batched_tokens[row]
                );
            }

            // Optional direct DS4 oracle capture for row-local unequal
            // positions. Drive the same forward_ragged/head_ragged operations
            // as ContinuousBatch::step while retaining each full logit row.
            if let Some(prefix) = std::env::var_os("LISA_GLM53_RAGGED_DS4_CAPTURE_PREFIX") {
                let prefix = std::path::PathBuf::from(prefix);
                let mut capture_prompts = ragged_prompts.clone();
                for row in 0..capture_prompts.len() {
                    let variable = format!("LISA_GLM53_RAGGED_DS4_PROMPT_ROW_{row}");
                    if let Some(path) = std::env::var_os(&variable) {
                        let bytes = std::fs::read(path)?;
                        anyhow::ensure!(
                            bytes.len() % std::mem::size_of::<u32>() == 0,
                            "{variable} must contain little-endian u32 token IDs"
                        );
                        capture_prompts[row] = bytes
                            .chunks_exact(std::mem::size_of::<u32>())
                            .map(|chunk| {
                                u32::from_le_bytes(chunk.try_into().expect("four-byte token"))
                            })
                            .collect();
                        anyhow::ensure!(
                            !capture_prompts[row].is_empty(),
                            "{variable} contains no prompt tokens"
                        );
                    }
                }
                if let Ok(requested_context) = std::env::var("LISA_GLM53_RAGGED_DS4_CONTEXT") {
                    let context = requested_context.parse::<usize>()?;
                    for row in 0..capture_prompts.len() {
                        let target = context + row;
                        anyhow::ensure!(
                            target >= capture_prompts[row].len(),
                            "LISA_GLM53_RAGGED_DS4_CONTEXT must not shorten prompt row {row}"
                        );
                        let fill = *capture_prompts[row]
                            .last()
                            .ok_or_else(|| anyhow::anyhow!("empty ragged DS4 prompt row"))?;
                        capture_prompts[row].resize(target, fill);
                    }
                }
                if let Some(parent) = prefix.parent().filter(|path| !path.as_os_str().is_empty()) {
                    std::fs::create_dir_all(parent)?;
                }
                for (row, prompt) in capture_prompts.iter().enumerate() {
                    write_batch_capture(
                        &prefix,
                        &format!("prompt-row-{row}.u32"),
                        &encode_u32(prompt),
                    )?;
                }
                let metadata = format!(
                    "rows=2\nvocab={vocab_size}\nprompt_lens={},{}\n",
                    capture_prompts[0].len(),
                    capture_prompts[1].len()
                );
                write_batch_capture(&prefix, "meta.txt", metadata.as_bytes())?;

                let mut capture_batch = ContinuousBatch::empty(8);
                let mut capture_samplers = [Sampler::default(), Sampler::default()];
                let mut feed = [0u32; 2];
                if let Some(prefill_capture_prefix) =
                    std::env::var_os("LISA_GLM53_RAGGED_PREFILL_CAPTURE_PREFIX")
                {
                    let prefill_capture_layers =
                        std::env::var("LISA_GLM53_RAGGED_PREFILL_CAPTURE_LAYERS")
                            .ok()
                            .map(|value| {
                                value
                                    .split(',')
                                    .map(|item| item.trim().parse::<usize>())
                                    .collect::<Result<Vec<_>, _>>()
                            })
                            .transpose()?
                            .unwrap_or_else(|| vec![0]);
                    anyhow::ensure!(
                        !prefill_capture_layers.is_empty(),
                        "LISA_GLM53_RAGGED_PREFILL_CAPTURE_LAYERS must select at least one layer"
                    );
                    let capture_mhc =
                        std::env::var_os("LISA_GLM53_RAGGED_PREFILL_CAPTURE_MHC").is_some();
                    let prefill_capture_keys =
                        std::env::var("LISA_GLM53_RAGGED_PREFILL_CAPTURE_KEYS")
                            .ok()
                            .map(|value| {
                                value
                                    .split(',')
                                    .map(str::trim)
                                    .filter(|key| !key.is_empty())
                                    .map(str::to_string)
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default();
                    let prefill_capture_prefix = std::path::PathBuf::from(prefill_capture_prefix);
                    for row in 0..2 {
                        tower.capture = Some(crate::core::capture::Capture {
                            layers: prefill_capture_layers.clone(),
                            key_filters: if !prefill_capture_keys.is_empty() {
                                prefill_capture_keys.clone()
                            } else if capture_mhc {
                                vec!["mhc_attn".to_string()]
                            } else {
                                Vec::new()
                            },
                            kda_prefill_only: !capture_mhc && prefill_capture_keys.is_empty(),
                            preserve_dispatch: true,
                            ..Default::default()
                        });
                        feed[row] = capture_batch.admit(
                            &mut tower,
                            &capture_prompts[row],
                            &mut capture_samplers[row],
                        )?;
                        let mut capture_path = prefill_capture_prefix.as_os_str().to_os_string();
                        capture_path.push(format!(".row-{row}-kda-prefill.npz"));
                        tower
                            .capture
                            .as_ref()
                            .expect("KDA prefill capture is installed")
                            .flush(capture_path.to_str().ok_or_else(|| {
                                anyhow::anyhow!("ragged KDA capture path is not UTF-8")
                            })?)?;
                        tower.capture = None;
                    }
                } else {
                    feed = [
                        capture_batch.admit(
                            &mut tower,
                            &capture_prompts[0],
                            &mut capture_samplers[0],
                        )?,
                        capture_batch.admit(
                            &mut tower,
                            &capture_prompts[1],
                            &mut capture_samplers[1],
                        )?,
                    ];
                }
                let capture_steps = std::env::var("LISA_GLM53_RAGGED_DS4_CAPTURE_STEPS")
                    .ok()
                    .map(|value| value.parse::<usize>())
                    .transpose()?
                    .unwrap_or(3);
                anyhow::ensure!(
                    (1..=3).contains(&capture_steps),
                    "LISA_GLM53_RAGGED_DS4_CAPTURE_STEPS must be in 1..=3"
                );
                let stage_capture_path = std::env::var_os("LISA_GLM53_RAGGED_DS4_STAGE_CAPTURE")
                    .map(std::path::PathBuf::from);
                let cache_capture_dir = std::env::var_os("LISA_GLM53_RAGGED_CACHE_CAPTURE_DIR")
                    .map(std::path::PathBuf::from);
                if let Some(dir) = &cache_capture_dir {
                    std::fs::create_dir_all(dir)?;
                }
                for step in 1..=capture_steps {
                    tower.set_context_tails(capture_batch.tails.clone());
                    let input = Array::from_slice(&[feed[0] as i32, feed[1] as i32], &[2, 1]);
                    let per = capture_batch
                        .unpacked_caches_mut_for_test()
                        .as_mut()
                        .ok_or_else(|| anyhow::anyhow!("ragged DS4 capture packed row caches"))?;
                    // Keep optional all-layer tracing off the normal oracle
                    // path. It retains large arrays and forces readbacks; the
                    // logits comparison needs only the production result.
                    tower.capture = stage_capture_path.as_ref().map(|_| make_capture(true));
                    let (hidden, _) = tower.forward_ragged(&input, per)?;
                    if let Some(capture) = tower.capture.as_ref() {
                        let drain_path = std::env::temp_dir().join(format!(
                            "lisa-glm53-ragged-capture-{}.npz",
                            std::process::id()
                        ));
                        capture.flush(drain_path.to_str().ok_or_else(|| {
                            anyhow::anyhow!("ragged capture path is not UTF-8")
                        })?)?;
                        if let Some(path) = stage_capture_path.as_ref() {
                            let entries = capture
                                .entries
                                .lock()
                                .expect("capture lock")
                                .iter()
                                .map(|(name, values)| (name.clone(), values.clone()))
                                .collect();
                            let shapes = capture
                                .shapes
                                .lock()
                                .expect("capture lock")
                                .iter()
                                .map(|(name, shape)| (name.clone(), shape.clone()))
                                .collect();
                            crate::core::capture::Capture {
                                entries: std::sync::Mutex::new(entries),
                                shapes: std::sync::Mutex::new(shapes),
                                ..Default::default()
                            }
                            .flush(path.to_str().ok_or_else(
                                || anyhow::anyhow!("ragged stage capture path is not UTF-8"),
                            )?)?;
                        }
                        let _ = std::fs::remove_file(drain_path);
                    }
                    let updated_tails = tower.context_tails();
                    if updated_tails.len() == 2 {
                        capture_batch.tails = updated_tails;
                    }
                    for (row, token) in feed.iter().enumerate() {
                        capture_batch.fed[row].push(*token);
                    }
                    tower.set_context_tails(capture_batch.tails.clone());
                    let last = hidden.index((.., hidden.dim(1) - 1, ..));
                    if let Some(capture) = tower.capture.as_ref() {
                        capture.store("final_mixed", &last);
                    }
                    let per = capture_batch
                        .unpacked_caches_for_test()
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("ragged DS4 capture lost row caches"))?;
                    let logits = tower
                        .head_ragged(&last, per)?
                        .as_dtype(lisa_metal::Dtype::Float32)?;
                    if let (Some(path), Some(capture)) =
                        (stage_capture_path.as_ref(), tower.capture.as_ref())
                    {
                        capture.flush(path.to_str().ok_or_else(|| {
                            anyhow::anyhow!("ragged stage capture path is not UTF-8")
                        })?)?;
                    }
                    tower.capture = None;
                    let values = logits.to_vec1::<f32>()?;
                    anyhow::ensure!(values.len() == 2 * vocab_size);
                    write_batch_capture(
                        &prefix,
                        &format!("logits-step-{step}.f32"),
                        &encode_f32(&values),
                    )?;
                    write_batch_capture(
                        &prefix,
                        &format!("inputs-step-{step}.u32"),
                        &encode_u32(&feed),
                    )?;
                    if let Some(dir) = &cache_capture_dir {
                        let caches = capture_batch
                            .unpacked_caches_for_test()
                            .as_ref()
                            .ok_or_else(|| anyhow::anyhow!("cache capture lost row caches"))?;
                        for (row, row_caches) in caches.iter().enumerate() {
                            capture_cache_layers(
                                row_caches,
                                &dir.join(format!("lisa-ragged-cache-step-{step}-row-{row}.npz")),
                            )?;
                        }
                    }
                    for row in 0..2 {
                        feed[row] = argmax(&values[row * vocab_size..(row + 1) * vocab_size]);
                    }
                }
            }
            if std::env::var_os("LISA_GLM53_RAGGED_DS4_CAPTURE_ONLY").is_some() {
                anyhow::ensure!(
                    std::env::var_os("LISA_GLM53_RAGGED_DS4_CAPTURE_PREFIX").is_some(),
                    "capture-only mode requires LISA_GLM53_RAGGED_DS4_CAPTURE_PREFIX"
                );
                eprintln!("[glm53-batch] direct DS4 capture-only mode complete");
                return Ok(());
            }

            // Exercise the serving-side continuous admission/repack path with
            // the same mixed-position cohort. It must retain independent DSA
            // tapes and use the ragged tower forward rather than defer rows.
            let mut continuous_prompts = ragged_prompts.clone();
            for row in 0..continuous_prompts.len() {
                let variable = format!("LISA_GLM53_RAGGED_DS4_PROMPT_ROW_{row}");
                if let Some(path) = std::env::var_os(&variable) {
                    let bytes = std::fs::read(path)?;
                    anyhow::ensure!(
                        bytes.len() % std::mem::size_of::<u32>() == 0,
                        "{variable} must contain little-endian u32 token IDs"
                    );
                    continuous_prompts[row] = bytes
                        .chunks_exact(4)
                        .map(|chunk| u32::from_le_bytes(chunk.try_into().unwrap()))
                        .collect();
                }
            }
            let mut continuous = ContinuousBatch::empty(8);
            let mut batch_samplers = vec![Sampler::default(), Sampler::default()];
            let first = [
                continuous.admit(&mut tower, &continuous_prompts[0], &mut batch_samplers[0])?,
                continuous.admit(&mut tower, &continuous_prompts[1], &mut batch_samplers[1])?,
            ];
            anyhow::ensure!(
                continuous.unpacked_caches_for_test().is_some(),
                "mixed-position GLM admission unexpectedly packed DSA tape metadata"
            );
            let admitted_cache_layers = continuous
                .unpacked_caches_for_test()
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("continuous GLM batch lost row caches"))?
                .iter()
                .map(|caches| snapshot_cache_layers(caches))
                .collect::<anyhow::Result<Vec<_>>>()?;
            let next = continuous.step(&mut tower, &first, &mut batch_samplers)?;
            let mut serial_caches = Vec::with_capacity(2);
            let mut batch_cache_layers = Vec::with_capacity(2);
            // Batched and single-row projections use different matrix shapes;
            // DS4's native session batch has the same schedule boundary.
            // Serial cache bytes are therefore not the DS4 batch oracle. Keep
            // exact byte checks for cache preservation through row repacking.
            for row in 0..2 {
                tower.warmup(&continuous_prompts[row])?;
                let (mut row_caches, prompt_logits, _) =
                    serial_cache_after(&mut tower, &continuous_prompts[row], &[])?;
                assert_cache_layers_equal(
                    &admitted_cache_layers[row],
                    &snapshot_cache_layers(&row_caches)?,
                    &format!("row {row} cache after admission versus serial prefill"),
                )?;
                let next_logits = serial_forward_token(&mut tower, &mut row_caches, first[row])?;
                anyhow::ensure!(
                    first[row] == argmax(&prompt_logits) && next[row] == argmax(&next_logits),
                    "continuous mixed-position GLM row {row} differs from serial: batch={:?}, serial={:?}",
                    [first[row], next[row]],
                    [argmax(&prompt_logits), argmax(&next_logits)]
                );
                let batch_layers = snapshot_cache_layers(
                    &continuous
                        .unpacked_caches_for_test()
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("continuous GLM batch lost row caches"))?
                        [row],
                )?;
                batch_cache_layers.push(batch_layers);
                serial_caches.push(row_caches);
            }
            eprintln!("[glm53-batch] continuous mixed-position decode tokens match serial");

            // Retire one row, admit a replacement at a different absolute
            // position, and decode again. This covers extraction/repacking of
            // row-local DSA tapes as well as the mixed-position forward.
            let continuous_cache_capture_dir =
                std::env::var_os("LISA_GLM53_CONTINUOUS_CACHE_CAPTURE_DIR")
                    .map(std::path::PathBuf::from);
            if let Some(dir) = &continuous_cache_capture_dir {
                std::fs::create_dir_all(dir)?;
            }
            continuous.retire(&mut tower, &[true, false])?;
            anyhow::ensure!(continuous.len() == 1);
            let retained_layers = snapshot_cache_layers(
                &continuous
                    .unpacked_caches_for_test()
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("retirement packed GLM row caches"))?[0],
            )?;
            assert_cache_layers_equal(
                &retained_layers,
                &batch_cache_layers[0],
                "retained row state across retirement",
            )?;
            if let Some(dir) = &continuous_cache_capture_dir {
                capture_cache_layers(
                    &continuous
                        .unpacked_caches_for_test()
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("retirement packed GLM row caches"))?[0],
                    &dir.join("lisa-continuous-after-retire-row-0.npz"),
                )?;
            }
            let mut replacement_prompt = continuous_prompts[1].clone();
            if replacement_prompt.len() == continuous.fed[0].len() {
                replacement_prompt.push(pad_token);
            }
            anyhow::ensure!(replacement_prompt.len() != continuous.fed[0].len());
            let mut replacement_sampler = Sampler::default();
            let replacement_first =
                continuous.admit(&mut tower, &replacement_prompt, &mut replacement_sampler)?;
            anyhow::ensure!(
                continuous.unpacked_caches_for_test().is_some(),
                "replacement admission did not restore the mixed-position cache set"
            );
            tower.warmup(&replacement_prompt)?;
            let (mut replacement_caches, replacement_logits, _) =
                serial_cache_after(&mut tower, &replacement_prompt, &[])?;
            anyhow::ensure!(
                argmax(&replacement_logits) == replacement_first,
                "replacement admission first token differs from serial"
            );
            let continuous_layers = continuous.unpacked_caches_for_test().as_ref().unwrap();
            assert_cache_layers_equal(
                &snapshot_cache_layers(&continuous_layers[0])?,
                &batch_cache_layers[0],
                "retained row state across replacement admission",
            )?;
            let rebatch_next = continuous.step(
                &mut tower,
                &[next[0], replacement_first],
                &mut [Sampler::default(), Sampler::default()],
            )?;
            let first_redecode_layers = snapshot_cache_layers(
                &continuous
                    .unpacked_caches_for_test()
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("redecode lost GLM row caches"))?[0],
            )?;
            if let Some(dir) = &continuous_cache_capture_dir {
                capture_cache_layers(
                    &continuous
                        .unpacked_caches_for_test()
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("repacked GLM row caches"))?[0],
                    &dir.join("lisa-continuous-after-redecode-row-0.npz"),
                )?;
            }
            let serial_logits = serial_forward_token(&mut tower, &mut serial_caches[0], next[0])?;
            let replacement_logits =
                serial_forward_token(&mut tower, &mut replacement_caches, replacement_first)?;
            for (row, logits) in [serial_logits, replacement_logits].into_iter().enumerate() {
                anyhow::ensure!(
                    argmax(&logits) == rebatch_next[row],
                    "continuous row {row} output differs from its serial cache forward after repack"
                );
            }
            eprintln!(
                "[glm53-batch] continuous retirement/repack outputs match serial; retained cache bytes survive repacking exactly"
            );

            // Exercise a second retire/admit/redecode cycle while the longer
            // row remains live. When the direct DS4 capture is enabled, the
            // retained row must consume the same third-step token as DS4.
            continuous.retire(&mut tower, &[true, false])?;
            anyhow::ensure!(continuous.len() == 1);
            let second_retained_layers = snapshot_cache_layers(
                &continuous
                    .unpacked_caches_for_test()
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("second retirement packed GLM row caches"))?[0],
            )?;
            assert_cache_layers_equal(
                &second_retained_layers,
                &first_redecode_layers,
                "retained row state across second retirement",
            )?;
            if let Some(dir) = &continuous_cache_capture_dir {
                capture_cache_layers(
                    &continuous
                        .unpacked_caches_for_test()
                        .as_ref()
                        .ok_or_else(|| {
                            anyhow::anyhow!("second retirement packed GLM row caches")
                        })?[0],
                    &dir.join("lisa-continuous-after-retire-cycle-2-row-0.npz"),
                )?;
            }

            let mut second_replacement_prompt = continuous_prompts[1].clone();
            if second_replacement_prompt.len() == continuous.fed[0].len() {
                second_replacement_prompt.push(pad_token);
            }
            anyhow::ensure!(second_replacement_prompt.len() != continuous.fed[0].len());
            let mut second_replacement_sampler = Sampler::default();
            let second_replacement_first = continuous.admit(
                &mut tower,
                &second_replacement_prompt,
                &mut second_replacement_sampler,
            )?;
            anyhow::ensure!(continuous.len() == 2);
            tower.warmup(&second_replacement_prompt)?;
            let (mut second_replacement_caches, second_replacement_logits, _) =
                serial_cache_after(&mut tower, &second_replacement_prompt, &[])?;
            anyhow::ensure!(
                argmax(&second_replacement_logits) == second_replacement_first,
                "second replacement admission first token differs from serial"
            );

            let second_cycle_inputs = [rebatch_next[0], second_replacement_first];
            if let Some(prefix) = std::env::var_os("LISA_GLM53_RAGGED_DS4_CAPTURE_PREFIX") {
                let mut prompt_path = prefix.clone();
                prompt_path.push(".prompt-row-0.u32");
                let captured_prompt = std::fs::read(std::path::PathBuf::from(prompt_path))?;
                if captured_prompt == encode_u32(&continuous_prompts[0]) {
                    let mut input_path = prefix;
                    input_path.push(".inputs-step-3.u32");
                    let ds4_inputs = std::fs::read(std::path::PathBuf::from(input_path))?;
                    anyhow::ensure!(
                        ds4_inputs.len() == 2 * std::mem::size_of::<u32>(),
                        "DS4 third-step input capture must contain two token IDs"
                    );
                    let ds4_retained = u32::from_le_bytes(
                        ds4_inputs[..4].try_into().expect("four-byte DS4 token"),
                    );
                    anyhow::ensure!(
                        second_cycle_inputs[0] == ds4_retained,
                        "second continuous redecode feeds retained token {}, but DS4 step 3 feeds {ds4_retained}",
                        second_cycle_inputs[0]
                    );
                } else {
                    eprintln!(
                        "[glm53-batch] skipping DS4 third-step token cross-check because oracle and continuous prompts differ"
                    );
                }
            }

            let second_rebatch_next = continuous.step(
                &mut tower,
                &second_cycle_inputs,
                &mut [Sampler::default(), Sampler::default()],
            )?;
            let retained_serial_logits =
                serial_forward_token(&mut tower, &mut serial_caches[0], second_cycle_inputs[0])?;
            let replacement_serial_logits = serial_forward_token(
                &mut tower,
                &mut second_replacement_caches,
                second_cycle_inputs[1],
            )?;
            anyhow::ensure!(
                argmax(&retained_serial_logits) == second_rebatch_next[0]
                    && argmax(&replacement_serial_logits) == second_rebatch_next[1],
                "second continuous retire/repack decode differs from serial rows"
            );
            if let Some(dir) = &continuous_cache_capture_dir {
                capture_cache_layers(
                    &continuous
                        .unpacked_caches_for_test()
                        .as_ref()
                        .ok_or_else(|| anyhow::anyhow!("second redecode lost GLM row caches"))?[0],
                    &dir.join("lisa-continuous-after-redecode-cycle-2-row-0.npz"),
                )?;
                std::fs::write(
                    dir.join("lisa-continuous-cycle-2-inputs.u32"),
                    encode_u32(&second_cycle_inputs),
                )?;
            }
            eprintln!(
                "[glm53-batch] second continuous retirement/repack and retained-row decode match serial"
            );
        }
        if let Ok(requested_context) = std::env::var("LISA_GLM53_RAGGED_CONTEXT") {
            let context = requested_context.parse::<usize>()?;
            anyhow::ensure!(
                context >= sparse_boundary,
                "LISA_GLM53_RAGGED_CONTEXT must be at least the DS4 dense/sparse boundary {sparse_boundary}"
            );
            let mut long_prompt = ragged_prompts[0].clone();
            long_prompt.resize(context, pad_token);
            let prompts = [long_prompt, ragged_prompts[1].clone()];
            let mut mixed = ContinuousBatch::empty(2);
            let mut samplers = [Sampler::default(), Sampler::default()];
            let first = [
                mixed.admit(&mut tower, &prompts[0], &mut samplers[0])?,
                mixed.admit(&mut tower, &prompts[1], &mut samplers[1])?,
            ];
            anyhow::ensure!(mixed.unpacked_caches_for_test().is_some());
            let next = mixed.step(&mut tower, &first, &mut samplers)?;
            for row in 0..2 {
                let mut serial_sampler = Sampler::default();
                let mut serial_on_token = |_: u32| Ok(());
                let (serial_tokens, _) = crate::core::generate::generate(
                    &mut tower,
                    &prompts[row],
                    2,
                    &mut serial_sampler,
                    0,
                    &mut serial_on_token,
                )?;
                anyhow::ensure!(
                    [first[row], next[row]] == serial_tokens[..2],
                    "mixed sparse-boundary GLM row {row} differs from serial: batch={:?}, serial={serial_tokens:?}",
                    [first[row], next[row]]
                );
            }
            eprintln!(
                "[glm53-batch] mixed-position sparse-boundary fallback at context {context} matches serial"
            );
        }

        // Force the first actual token of row zero to be an EOS marker. This
        // exercises the loaded GLM tower's batch/session integration while
        // preserving its configured EOS IDs; row one must continue advancing.
        struct ResetBatchEos;
        impl Drop for ResetBatchEos {
            fn drop(&mut self) {
                crate::core::generate::set_eos_ids(Vec::new());
            }
        }
        let _reset_eos = ResetBatchEos;
        let forced_eos = argmax(&serial_logits[0]);
        let row_one_second = argmax(&serial_logits[1]);
        anyhow::ensure!(
            forced_eos != first[0] && forced_eos != row_one_second,
            "real GLM EOS marker collides with an earlier or row-one token"
        );
        anyhow::ensure!(
            !tower.eos_token_ids().iter().any(|&id| {
                u32::try_from(id).ok() == Some(first[0])
                    || u32::try_from(id).ok() == Some(first[1])
                    || u32::try_from(id).ok() == Some(row_one_second)
            }),
            "a pre-EOS row token is already a configured GLM EOS token"
        );
        crate::core::generate::set_eos_ids_for_model(vec![forced_eos], &tower);
        let mut eos_sampler = Sampler::default();
        let mut eos_on_token = |_: usize, _: u32| Ok(());
        let (eos_batch, eos_tokens) =
            Batch::generate(&mut tower, &prompts, 3, &mut eos_sampler, &mut eos_on_token)?;
        anyhow::ensure!(
            eos_tokens[0] == [first[0], forced_eos],
            "row zero did not stop on its second model-produced EOS token: {:?}",
            eos_tokens[0]
        );
        anyhow::ensure!(
            eos_tokens[1].len() > 2,
            "row one stopped with row zero despite a non-EOS first token: {:?}",
            eos_tokens[1]
        );
        anyhow::ensure!(eos_batch.done[0], "finished row zero was not marked done");
        anyhow::ensure!(
            eos_batch.done[1] == eos_tokens[1].last().is_some_and(|&token| is_eos(token)),
            "row one's done flag does not match its last emitted token"
        );
        anyhow::ensure!(
            eos_batch.fed[0].len() == prompts[0].len() + 1,
            "finished row zero advanced its target cache past the prompt"
        );
        anyhow::ensure!(
            eos_batch.fed[1].len() == prompts[1].len() + eos_tokens[1].len() - 1,
            "row one's cache boundary does not include exactly the prior emitted tokens"
        );
        for row in 0..2 {
            let mut serial_sampler = Sampler::default();
            let mut serial_on_token = |_: u32| Ok(());
            let (serial_tokens, _) = crate::core::generate::generate(
                &mut tower,
                &prompts[row],
                3,
                &mut serial_sampler,
                0,
                &mut serial_on_token,
            )?;
            anyhow::ensure!(
                eos_tokens[row] == serial_tokens,
                "real GLM mixed-EOS row {row} differs from serial: batch={:?}, serial={serial_tokens:?}",
                eos_tokens[row]
            );
        }
        // Exercise the serving scheduler as well as fixed Batch::generate:
        // it must retire the EOS row from ContinuousBatch and keep decoding
        // the survivor from its own cache.
        let scheduler_requests = prompts
            .iter()
            .enumerate()
            .map(|(id, prompt)| crate::core::sched::Request {
                id,
                prompt: prompt.clone(),
                max_tokens: 3,
                sampler: Sampler::default(),
                ignore_eos: false,
            })
            .collect();
        let mut scheduler_on_token = |_: usize, _: u32| Ok(());
        let scheduler_tokens = crate::core::sched::run(
            &mut tower,
            scheduler_requests,
            2,
            0,
            &mut scheduler_on_token,
        )?;
        anyhow::ensure!(
            scheduler_tokens[0] == [first[0], forced_eos],
            "continuous scheduler did not retire row zero at its forced EOS: {:?}",
            scheduler_tokens[0]
        );
        anyhow::ensure!(
            scheduler_tokens[1] == eos_tokens[1],
            "continuous scheduler survivor diverged after row-zero retirement: scheduler={:?}, batch={:?}",
            scheduler_tokens[1],
            eos_tokens[1]
        );
        eprintln!(
            "[glm53-batch] real-checkpoint mixed EOS matches serial output in fixed and continuous batches"
        );
        Ok(())
    }
}

/// Application composition for portable row sampling and cache packing.
#[cfg(test)]
pub(crate) fn draw_rows(
    logits: &Array,
    n: usize,
    sampler: &mut Sampler,
    history: &[Vec<u32>],
    ds4_sampling: bool,
) -> anyhow::Result<Vec<u32>> {
    let device = crate::metal_context::device();
    lisa_engine::sampling::draw_rows(
        &crate::core::sampler::context(&device),
        logits,
        n,
        &mut sampler.0,
        history,
        ds4_sampling,
    )
}
#[cfg(test)]
pub(crate) fn draw_rows_multi(
    logits: &Array,
    n: usize,
    samplers: &mut [Sampler],
    history: &[Vec<u32>],
    ds4_sampling: bool,
) -> anyhow::Result<Vec<u32>> {
    let device = crate::metal_context::device();
    lisa_engine::sampling::draw_rows_multi(
        &crate::core::sampler::context(&device),
        logits,
        n,
        samplers,
        history,
        ds4_sampling,
    )
}
#[cfg(test)]
pub(crate) fn pack_layer(
    per: &[Vec<LayerCache>],
    li: usize,
    lens: &[usize],
    lmax: usize,
    cap: usize,
) -> anyhow::Result<LayerCache> {
    lisa_engine::cache_batching::pack_layer(
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
    lisa_engine::cache_batching::extract_streams(packed, survivors, pack_lens, lmax)
}

use crate::bindings::glm53_affine_tower::TowerLoad as _;
