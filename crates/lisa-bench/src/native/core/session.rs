//! Metal storage aliases and model-dependent validation for language sessions.
#[cfg(test)]
use crate::core::cache::{IndexerTape, LayerCache};
#[cfg(test)]
use crate::core::sampler::Sampler;
#[cfg(test)]
use crate::eos::is_eos;
use crate::language::LanguageModel;
use lisa_metal::Array;
use lisa_metal::ops::indexing::IndexOp;
pub type Session = crate::conversation::Conversation<crate::metal_language::MetalLanguageRuntime>;
pub use crate::conversation::{TargetMtpEosPolicy, TargetMtpSamplingMode};
pub use crate::speculation::MtpDepth;
#[cfg(test)]
use crate::speculation::target_mtp_draw_reuses_argmax;
pub type LayerSnapshot = lisa_engine::cache::SessionSnapshot<Array>;

/// Verify that feeding a token sequence in chunks through one [`Session`] gives
/// the same logits as feeding it in a single prefill with fresh caches.
///
/// Three passes are run: pass 1 is discarded (the first forwards in a process
/// can differ while pages are lazily faulted/resident), pass 2 is the measured
/// incremental run, and pass 3 is the full single-prefill reference.
/// Returns `(argmax_equal, max_abs_logit_diff, max_abs_hidden_diff)`.
pub fn verify_incremental(
    tower: &mut dyn LanguageModel<crate::metal_language::MetalLanguageRuntime>,
    tokens: &[u32],
    chunk: usize,
) -> anyhow::Result<(bool, f32, f32)> {
    let saved = tower.context_tails();

    let run_incremental = |tower: &mut dyn LanguageModel<
        crate::metal_language::MetalLanguageRuntime,
    >|
     -> anyhow::Result<(Array, Array)> {
        tower.clear_context();
        let mut s = Session::new(tower);
        let mut logits = None;
        let mut mixed = None;
        for c in tokens.chunks(chunk.max(1)) {
            logits = Some(s.feed(tower, c)?);
            mixed = s.last_mixed.clone();
        }
        Ok((logits.expect("non-empty"), mixed.expect("non-empty")))
    };

    let run_full = |tower: &mut dyn LanguageModel<crate::metal_language::MetalLanguageRuntime>| -> anyhow::Result<(Array, Array)> {
        tower.clear_context();
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let arr = Array::from_slice(&ids, &[1i32, ids.len() as i32]);
        let mut caches = tower.new_caches();
        let (mixed, _) = tower.forward(&arr, Some(&mut caches))?;
        let last = mixed.index((.., mixed.dim(1) - 1, ..));
        let logits = tower.head(&last)?;
        Ok((logits, last))
    };

    // warm (discarded)
    let _ = run_incremental(tower)?;
    let _ = run_full(tower)?;

    let (inc, inc_m) = run_incremental(tower)?;
    let (full, full_m) = run_full(tower)?;

    let maxdiff = |a: &Array, b: &Array| -> anyhow::Result<f32> {
        let a = a.as_dtype(lisa_metal::Dtype::Float32)?;
        let b = b.as_dtype(lisa_metal::Dtype::Float32)?;
        let _ = (a.eval(), b.eval());
        let av = a.as_slice::<f32>();
        let bv = b.as_slice::<f32>();
        Ok(av
            .iter()
            .zip(bv.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max))
    };
    let d_log = maxdiff(&inc, &full)?;
    let d_hid = maxdiff(&inc_m, &full_m)?;
    let am_inc = lisa_metal::ops::indexing::argmax(&inc, None)?.item_cast::<i32>();
    let am_full = lisa_metal::ops::indexing::argmax(&full, None)?.item_cast::<i32>();
    tower.set_context_tails(saved);
    Ok((am_inc == am_full, d_log, d_hid))
}

#[cfg(test)]
mod target_mtp_tests {
    use super::*;

    // Retain the verifier origin when production calls the head on each row
    // separately; tensor row count alone cannot distinguish verify from replay.
    const VERIFY_LOGITS_MARKER: f32 = 1024.0;

    struct ToyTargetMtp {
        runtime: crate::metal_language::MetalLanguageRuntime,
        bad_first_draft: bool,
        eos_on_verify: bool,
        bias_verify_multi: bool,
        bias_verify_logits: bool,
        fail_verify_after_cache_append: bool,
        soft_logits: bool,
        context_limit: usize,
        mtp_calls: Vec<(u32, usize, usize, u32)>,
    }

    impl ToyTargetMtp {
        fn append_target_cache(
            caches: Option<&mut Vec<LayerCache>>,
            ids: &Array,
        ) -> anyhow::Result<()> {
            let Some(caches) = caches else {
                return Ok(());
            };
            let tokens = ids.to_vec1::<i32>()?;
            let values: Vec<f32> = tokens.iter().map(|&token| token as f32).collect();
            let rows = Array::from_slice(&values, &[1, values.len() as i32, 1]);
            for cache in caches {
                match cache {
                    LayerCache::Native(_) => {
                        anyhow::bail!("toy target-MTP model received a native cache")
                    }
                    LayerCache::Linear(state) => {
                        for slot in [&mut state.conv, &mut state.ssm] {
                            *slot = Some(match slot {
                                Some(prefix) => Array::concatenate(&[prefix, &rows], 1)?,
                                None => rows.clone(),
                            });
                        }
                    }
                    LayerCache::Full(state) => {
                        let keys = rows.as_dtype(lisa_metal::Dtype::Bfloat16)?.reshape(&[
                            1,
                            1,
                            values.len() as i32,
                            1,
                        ])?;
                        let _ = state.update(&keys, &keys)?;
                        state.indexer_tape.append(&rows, 1)?;
                    }
                }
            }
            Ok(())
        }

        fn assert_cache_matches_fed(session: &Session) -> anyhow::Result<()> {
            let [LayerCache::Linear(cache), LayerCache::Full(full)] = session.caches.as_slice()
            else {
                anyhow::bail!("toy MTP session should own linear and full caches");
            };
            let expected: Vec<f32> = session.fed.iter().map(|&token| token as f32).collect();
            for state in [&cache.conv, &cache.ssm] {
                let actual = state
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("toy target cache state is missing"))?
                    .to_vec1::<f32>()?;
                anyhow::ensure!(
                    actual == expected,
                    "target cache {:?} differs from fed tokens {:?}",
                    actual,
                    expected
                );
            }
            anyhow::ensure!(
                full.offset == session.fed.len() && full.indexer_tape.total == session.fed.len(),
                "full target cache boundary differs from fed tokens"
            );
            let actual_keys = full
                .keys
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("toy full-attention keys are missing"))?
                .index((0, 0, 0..full.offset as i32, 0))
                .as_dtype(lisa_metal::Dtype::Float32)?
                .reshape(&[full.offset as i32])?
                .to_vec1::<f32>()?;
            anyhow::ensure!(
                actual_keys == expected,
                "full target keys {:?} differ from fed tokens {:?}",
                actual_keys,
                expected
            );
            Ok(())
        }

        fn outputs(&mut self, ids: &Array) -> anyhow::Result<(Array, Array)> {
            let tokens = ids.to_vec1::<i32>()?;
            let seq = tokens.len();
            let mixed_values: Vec<f32> = tokens
                .iter()
                .enumerate()
                .map(|(row, &token)| {
                    token as f32
                        + if self.bias_verify_logits && seq == 2 && row == 0 {
                            VERIFY_LOGITS_MARKER
                        } else {
                            0.0
                        }
                })
                .collect();
            let verify_bias = if self.bias_verify_multi && seq == 2 {
                1000.0
            } else {
                0.0
            };
            let multi_values: Vec<f32> = tokens
                .iter()
                .flat_map(|&t| [t as f32 + verify_bias; 4])
                .collect();
            Ok((
                Array::from_slice(&mixed_values, &[1, seq as i32, 1]),
                Array::from_slice(&multi_values, &[1, seq as i32, 4]),
            ))
        }

        fn logits(next: u32) -> Array {
            let mut values = vec![-100.0f32; 32];
            values[(next % 32) as usize] = 100.0;
            Array::from_slice(&values, &[1, 1, 32])
        }

        fn target_logits(next: u32, soft: bool) -> Array {
            let mut values = vec![-100.0f32; 32];
            values[(next % 32) as usize] = if soft { 1.0 } else { 100.0 };
            if soft {
                values[((next + 1) % 32) as usize] = 0.9;
            }
            Array::from_slice(&values, &[1, 1, 32])
        }
    }

    impl LanguageModel<crate::metal_language::MetalLanguageRuntime> for ToyTargetMtp {
        fn runtime(&self) -> &crate::metal_language::MetalLanguageRuntime {
            &self.runtime
        }
        fn max_position_embeddings(&self) -> usize {
            self.context_limit
        }

        fn eos_token_id(&self) -> i64 {
            31
        }

        fn new_caches(&self) -> Vec<LayerCache> {
            vec![
                LayerCache::Linear(crate::core::cache::GdnCache::new()),
                LayerCache::Full(crate::core::cache::FullAttentionCache::new(
                    crate::metal_context::execution_context(),
                    1,
                    1,
                )),
            ]
        }

        fn forward(
            &mut self,
            ids: &Array,
            caches: Option<&mut Vec<LayerCache>>,
        ) -> anyhow::Result<(Array, Array)> {
            Self::append_target_cache(caches, ids)?;
            self.outputs(ids)
        }

        fn forward_capture(
            &mut self,
            ids: &Array,
            caches: Option<&mut Vec<LayerCache>>,
            _capture: bool,
        ) -> anyhow::Result<(Array, Array)> {
            Self::append_target_cache(caches, ids)?;
            if self.fail_verify_after_cache_append && ids.dim(1) == 2 {
                anyhow::bail!("injected target-MTP verify failure after cache append");
            }
            self.outputs(ids)
        }

        fn prefill(
            &mut self,
            tokens: &[u32],
            caches: &mut Vec<LayerCache>,
        ) -> anyhow::Result<Array> {
            let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            let input = Array::from_slice(&ids, &[1, ids.len() as i32]);
            Self::append_target_cache(Some(caches), &input)?;
            let (mixed, _) = self.outputs(&input)?;
            Ok(mixed
                .index((.., tokens.len() as i32 - 1, ..))
                .contiguous()?)
        }

        fn prefill_multi(
            &mut self,
            tokens: &[u32],
            caches: &mut Vec<LayerCache>,
        ) -> anyhow::Result<(Array, Array)> {
            let last = self.prefill(tokens, caches)?;
            let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            let (_, multi) = self.outputs(&Array::from_slice(&ids, &[1, ids.len() as i32]))?;
            Ok((last, multi))
        }

        fn head(&self, mixed: &Array) -> lisa_metal::error::Result<Array> {
            let tokens = mixed.to_vec1::<f32>()?;
            let verify_eos = self.eos_on_verify && tokens.len() == 2;
            let replay_eos = self.eos_on_verify && tokens.len() == 1 && tokens[0] == 28.0;
            let mut logits = Vec::with_capacity(tokens.len() * 32);
            for (row, token) in tokens.into_iter().enumerate() {
                let verify_logits = self.bias_verify_logits && token >= VERIFY_LOGITS_MARKER;
                let token = if verify_logits {
                    token - VERIFY_LOGITS_MARKER
                } else {
                    token
                };
                let next = if (verify_eos && row == 0) || replay_eos {
                    31
                } else if verify_logits {
                    (token as u32 + 3) % 32
                } else {
                    (token as u32 + 1) % 32
                };
                logits.extend(Self::target_logits(next, self.soft_logits).to_vec1::<f32>()?);
            }
            Ok(Array::from_slice(
                &logits,
                &[1, (logits.len() / 32) as i32, 32],
            ))
        }

        fn warmup(&mut self, _seed: &[u32]) -> anyhow::Result<()> {
            Ok(())
        }

        fn has_target_mtp(&self) -> bool {
            true
        }

        fn target_mtp_step(
            &mut self,
            cache: &mut IndexerTape,
            target_multi: &Array,
            next_token: u32,
            position: usize,
            min_position: usize,
        ) -> anyhow::Result<Array> {
            let latent = Array::from_slice(&[next_token as f32], &[1, 1, 1]);
            cache.write_compact_f16_at(&latent, position)?;
            let target_hidden_token = target_multi.to_vec1::<f32>()?[0] as u32;
            self.mtp_calls
                .push((target_hidden_token, position, min_position, next_token));
            let candidate = if self.bad_first_draft && self.mtp_calls.len() == 1 {
                next_token + 2
            } else {
                next_token + 1
            };
            Ok(Self::logits(candidate))
        }
    }

    struct ToyDecodeDispatch {
        target: ToyTargetMtp,
        prefill_calls: Vec<Vec<u32>>,
        decode_calls: Vec<u32>,
        law: u64,
        fail_decode: bool,
        fail_head: bool,
    }

    impl ToyDecodeDispatch {
        fn new(context_limit: usize) -> Self {
            Self {
                target: ToyTargetMtp {
                    runtime: crate::metal_language::runtime(),
                    bad_first_draft: false,
                    eos_on_verify: false,
                    bias_verify_multi: false,
                    bias_verify_logits: false,
                    fail_verify_after_cache_append: false,
                    soft_logits: false,
                    context_limit,
                    mtp_calls: Vec::new(),
                },
                prefill_calls: Vec::new(),
                decode_calls: Vec::new(),
                law: 1,
                fail_decode: false,
                fail_head: false,
            }
        }
    }

    impl LanguageModel<crate::metal_language::MetalLanguageRuntime> for ToyDecodeDispatch {
        fn runtime(&self) -> &crate::metal_language::MetalLanguageRuntime {
            self.target.runtime()
        }
        fn cache_numerical_law(&self) -> Option<u64> {
            Some(self.law)
        }

        fn max_position_embeddings(&self) -> usize {
            self.target.max_position_embeddings()
        }

        fn eos_token_id(&self) -> i64 {
            self.target.eos_token_id()
        }

        fn new_caches(&self) -> Vec<LayerCache> {
            self.target.new_caches()
        }

        fn forward(
            &mut self,
            ids: &Array,
            caches: Option<&mut Vec<LayerCache>>,
        ) -> anyhow::Result<(Array, Array)> {
            self.target.forward(ids, caches)
        }

        fn forward_capture(
            &mut self,
            ids: &Array,
            caches: Option<&mut Vec<LayerCache>>,
            capture: bool,
        ) -> anyhow::Result<(Array, Array)> {
            self.target.forward_capture(ids, caches, capture)
        }

        fn prefill(
            &mut self,
            tokens: &[u32],
            caches: &mut Vec<LayerCache>,
        ) -> anyhow::Result<Array> {
            self.prefill_calls.push(tokens.to_vec());
            self.target.prefill(tokens, caches)
        }

        fn decode_token(
            &mut self,
            token: u32,
            caches: &mut Vec<LayerCache>,
        ) -> anyhow::Result<Array> {
            self.decode_calls.push(token);
            let ids = Array::from_slice(&[token as i32], &[1, 1]);
            let (mixed, _) = self.target.forward(&ids, Some(caches))?;
            anyhow::ensure!(
                !self.fail_decode,
                "injected decode failure after cache append"
            );
            Ok(mixed.index((.., 0, ..)).contiguous()?)
        }

        fn prefill_multi(
            &mut self,
            tokens: &[u32],
            caches: &mut Vec<LayerCache>,
        ) -> anyhow::Result<(Array, Array)> {
            self.target.prefill_multi(tokens, caches)
        }

        fn head(&self, mixed: &Array) -> lisa_metal::error::Result<Array> {
            if self.fail_head {
                return Err(lisa_metal::error::Error::msg(
                    "injected decode head failure",
                ));
            }
            self.target.head(mixed)
        }

        fn warmup(&mut self, seed: &[u32]) -> anyhow::Result<()> {
            self.target.warmup(seed)
        }
    }

    #[test]
    fn session_decode_token_separates_generation_from_prompt_prefill() -> anyhow::Result<()> {
        for live_sampling_mode in [false, true] {
            let mut tower = ToyDecodeDispatch::new(8);
            let mut session = Session::new(&mut tower);
            let mut sampler = Sampler::default();
            let generated = if live_sampling_mode {
                session.generate_with_sampling_mode(
                    &mut tower,
                    &[1],
                    3,
                    &mut sampler,
                    None,
                    &mut |_| Ok(TargetMtpSamplingMode::Greedy),
                )?
            } else {
                session.generate(&mut tower, &[1], 3, &mut sampler, 0, None, &mut |_| Ok(()))?
            };
            assert_eq!(generated, vec![2, 3, 4]);
            assert_eq!(tower.prefill_calls, vec![vec![1]]);
            assert_eq!(tower.decode_calls, vec![2, 3]);
            assert_eq!(session.fed, vec![1, 2, 3]);
            ToyTargetMtp::assert_cache_matches_fed(&session)?;
            let _ = session.decode_token(&mut tower, 4)?;
            assert_eq!(
                session.last_mixed.as_ref().unwrap().to_vec1::<f32>()?,
                vec![4.0]
            );
            // A one-token appended prompt suffix must keep its prefill path.
            let _ = session.feed(&mut tower, &[7])?;
            assert_eq!(tower.prefill_calls, vec![vec![1], vec![7]]);
            assert_eq!(tower.decode_calls, vec![2, 3, 4]);
            assert_eq!(session.fed, vec![1, 2, 3, 4, 7]);
            ToyTargetMtp::assert_cache_matches_fed(&session)?;
        }
        Ok(())
    }

    #[test]
    fn session_decode_token_checks_bounds_and_law_before_mutation() -> anyhow::Result<()> {
        let mut tower = ToyDecodeDispatch::new(2);
        let mut session = Session::new(&mut tower);
        let _ = session.feed(&mut tower, &[1])?;
        tower.law = 2;
        let error = session
            .decode_token(&mut tower, 2)
            .err()
            .expect("law refusal");
        assert!(error.to_string().contains("numerical law changed"));
        assert!(tower.decode_calls.is_empty());
        assert_eq!(session.fed, vec![1]);
        assert!(session.is_valid());
        ToyTargetMtp::assert_cache_matches_fed(&session)?;
        tower.law = 1;
        let _ = session.decode_token(&mut tower, 2)?;
        assert_eq!(session.fed, vec![1, 2]);
        assert_eq!(tower.decode_calls, vec![2]);
        let error = session
            .decode_token(&mut tower, 3)
            .err()
            .expect("context refusal");
        assert!(error.to_string().contains("context overflow"));
        assert_eq!(tower.decode_calls, vec![2]);
        assert_eq!(session.fed, vec![1, 2]);
        assert_eq!(
            session.last_mixed.as_ref().unwrap().to_vec1::<f32>()?,
            vec![2.0]
        );
        assert!(session.is_valid());
        ToyTargetMtp::assert_cache_matches_fed(&session)?;
        Ok(())
    }

    #[test]
    fn session_decode_token_invalidates_after_model_or_head_failure() -> anyhow::Result<()> {
        for fail_head in [false, true] {
            let mut tower = ToyDecodeDispatch::new(8);
            let mut session = Session::new(&mut tower);
            let _ = session.feed(&mut tower, &[1])?;
            tower.fail_decode = !fail_head;
            tower.fail_head = fail_head;
            assert!(session.decode_token(&mut tower, 2).is_err());
            assert!(!session.is_valid());
            assert_eq!(session.fed, vec![1]);
            assert_eq!(
                session.last_mixed.as_ref().unwrap().to_vec1::<f32>()?,
                vec![1.0]
            );
            assert_eq!(tower.decode_calls, vec![2]);
            let error = session
                .decode_token(&mut tower, 3)
                .err()
                .expect("invalid session refusal");
            assert!(error.to_string().contains("session state is invalid"));
            assert_eq!(tower.decode_calls, vec![2]);
        }
        Ok(())
    }

    #[test]
    fn glm53_target_mtp_reuses_argmax_only_for_unmodified_greedy_draws() {
        assert!(target_mtp_draw_reuses_argmax::<Sampler>(
            None,
            TargetMtpSamplingMode::Configured,
        ));
        assert!(target_mtp_draw_reuses_argmax::<Sampler>(
            None,
            TargetMtpSamplingMode::Greedy,
        ));
        for mtp_exact_sampling in [false, true] {
            for (temperature, top_k, repetition_penalty, reusable) in [
                (0.0, 0, 1.0, true),
                (0.0, 1, 1.0, true),
                (-1.0, 0, 1.0, true),
                (0.0, 0, 1.1, false),
                (0.0, 0, 0.9, false),
                (0.7, 0, 1.0, false),
                (0.7, 1, 1.0, false),
                (1e-6, 1, 1.0, false),
            ] {
                let sampler = Sampler(lisa_engine::sampling::Sampler {
                    temperature,
                    top_k,
                    repetition_penalty,
                    mtp_exact_sampling,
                    ..Default::default()
                });
                assert_eq!(
                    target_mtp_draw_reuses_argmax::<Sampler>(
                        Some(&sampler),
                        TargetMtpSamplingMode::Configured,
                    ),
                    reusable,
                    "{sampler:?}",
                );
                assert!(target_mtp_draw_reuses_argmax::<Sampler>(
                    Some(&sampler),
                    TargetMtpSamplingMode::Greedy,
                ));
            }
        }
    }

    #[test]
    fn glm53_target_mtp_greedy_accept_and_reject_commit_target_tokens() -> anyhow::Result<()> {
        for bad_first_draft in [false, true] {
            let mut tower = ToyTargetMtp {
                runtime: crate::metal_language::runtime(),
                bad_first_draft,
                eos_on_verify: false,
                bias_verify_multi: bad_first_draft,
                bias_verify_logits: false,
                fail_verify_after_cache_append: false,
                soft_logits: false,
                context_limit: 128,
                mtp_calls: Vec::new(),
            };
            let mut session = Session::new(&mut tower);
            let mut emitted = Vec::new();
            let generated = session.generate_mtp(&mut tower, &[1], 4, 2, &mut |token| {
                emitted.push(token);
                Ok(())
            })?;

            assert_eq!(generated, vec![2, 3, 4, 5]);
            assert_eq!(emitted, generated);
            assert_eq!(session.fed, vec![1, 2, 3, 4, 5]);
            ToyTargetMtp::assert_cache_matches_fed(&session)?;
            assert_eq!(session.glm_mtp_cache_for_test().total, 0);
            assert_eq!(tower.mtp_calls.first(), Some(&(2, 1, 1, 3)));
            if bad_first_draft {
                assert_eq!(tower.mtp_calls.get(1), Some(&(3, 2, 1, 4)));
                assert_ne!(
                    tower.mtp_calls[1].0, 1002,
                    "rejection must seed MTP from the replayed target hidden row, not the biased two-row verify output"
                );
            } else {
                assert_eq!(tower.mtp_calls.len(), 4);
            }
        }
        Ok(())
    }

    #[test]
    fn glm53_mtp_scheduler_admits_late_requests_and_reuses_slots() -> anyhow::Result<()> {
        use crate::core::sched::{Request, run_speculative_streaming};
        use std::cell::Cell;
        for capacity in [1, 2] {
            let make_tower = || ToyTargetMtp {
                runtime: crate::metal_language::runtime(),
                bad_first_draft: false,
                eos_on_verify: false,
                bias_verify_multi: false,
                bias_verify_logits: false,
                fail_verify_after_cache_append: false,
                soft_logits: true,
                context_limit: 128,
                mtp_calls: Vec::new(),
            };
            let request = |id| Request {
                id,
                prompt: vec![1],
                max_tokens: 8,
                sampler: Sampler(lisa_engine::sampling::Sampler {
                    temperature: 0.8,
                    seed: 0x5eed + id as u64,
                    ..Default::default()
                }),
                ignore_eos: false,
            };
            let mut expected = Vec::new();
            for id in 0..3 {
                let mut tower = make_tower();
                let mut session = Session::new(&mut tower);
                let mut req = request(id);
                expected.push(session.generate_mtp_depth(
                    &mut tower,
                    &req.prompt,
                    req.max_tokens,
                    MtpDepth::Fixed(2),
                    None,
                    Some(&mut req.sampler),
                    &mut |_| Ok(()),
                )?);
            }
            let first_emitted = Cell::new(false);
            let next_id = Cell::new(1usize);
            let mut events = Vec::new();
            let actual = run_speculative_streaming(
                &mut make_tower(),
                vec![request(0)],
                capacity,
                2,
                &mut |id, _| {
                    first_emitted.set(true);
                    events.push(id);
                    Ok(())
                },
                &mut || {
                    if !first_emitted.get() || next_id.get() >= 3 {
                        return None;
                    }
                    let id = next_id.get();
                    next_id.set(id + 1);
                    Some(request(id))
                },
            )?;
            assert_eq!(actual, expected, "capacity={capacity}");
            let first_late = events.iter().position(|&id| id == 1).unwrap();
            let last_first = events.iter().rposition(|&id| id == 0).unwrap();
            assert_eq!(first_late < last_first, capacity > 1);
        }
        Ok(())
    }

    #[test]
    fn glm53_target_mtp_resumes_at_round_boundaries() -> anyhow::Result<()> {
        for bad_first_draft in [false, true] {
            // Greedy, opportunistic sampling, and exact point-mass sampling.
            for sampling in 0..3 {
                for budget in [1, 2, 3, 8] {
                    let make_tower = || ToyTargetMtp {
                        runtime: crate::metal_language::runtime(),
                        bad_first_draft,
                        eos_on_verify: false,
                        bias_verify_multi: bad_first_draft,
                        bias_verify_logits: false,
                        fail_verify_after_cache_append: false,
                        soft_logits: sampling != 0,
                        context_limit: 128,
                        mtp_calls: Vec::new(),
                    };
                    let make_sampler = || {
                        Sampler(lisa_engine::sampling::Sampler {
                            temperature: if sampling == 0 { 0.0 } else { 0.8 },
                            seed: 0x5eed,
                            mtp_exact_sampling: sampling == 2,
                            ..Default::default()
                        })
                    };
                    // Cross both live parser boundaries while generation is
                    // suspended: configured -> greedy -> configured.
                    let next_mode = |emitted: usize| {
                        if sampling == 2 && matches!(emitted % 4, 2 | 3) {
                            TargetMtpSamplingMode::Greedy
                        } else {
                            TargetMtpSamplingMode::Configured
                        }
                    };
                    let mut full_tower = make_tower();
                    let mut full_sampler = make_sampler();
                    let mut full_session = Session::new(&mut full_tower);
                    let mut full_emitted = Vec::new();
                    let full = full_session.generate_mtp_depth_with_sampling_mode(
                        &mut full_tower,
                        &[1],
                        budget,
                        MtpDepth::Fixed(2),
                        None,
                        Some(&mut full_sampler),
                        &mut |token| {
                            full_emitted.push(token);
                            Ok(next_mode(full_emitted.len()))
                        },
                    )?;

                    let mut chunked_tower = make_tower();
                    let mut chunked_sampler = make_sampler();
                    let mut chunked_session = Session::new(&mut chunked_tower);
                    let mut chunked = Vec::new();
                    let mut chunked_emitted = Vec::new();
                    let mut cycle_lengths = Vec::new();
                    while chunked.len() < budget {
                        let suffix = if chunked_session.fed.is_empty() {
                            &[1][..]
                        } else {
                            &[]
                        };
                        let next = chunked_session.target_mtp_cycle(
                            &mut chunked_tower,
                            suffix,
                            budget - chunked.len(),
                            Some(&mut chunked_sampler),
                            &mut |token| {
                                chunked_emitted.push(token);
                                Ok(next_mode(chunked_emitted.len()))
                            },
                        )?;
                        assert!(!next.is_empty() && next.len() <= 3);
                        cycle_lengths.push(next.len());
                        chunked.extend(next);
                        ToyTargetMtp::assert_cache_matches_fed(&chunked_session)?;
                    }

                    assert_eq!(
                        chunked, full,
                        "sampling={sampling}, bad_draft={bad_first_draft}, budget={budget}"
                    );
                    assert_eq!(chunked_emitted, full_emitted);
                    assert_eq!(
                        chunked_sampler.seed, full_sampler.seed,
                        "yield must not consume RNG again"
                    );
                    assert_eq!(chunked_session.fed, full_session.fed);
                    assert_eq!(
                        chunked_tower.mtp_calls, full_tower.mtp_calls,
                        "yield must not re-seed or replay MTP"
                    );
                    assert!(chunked_session.has_glm_mtp_pending_for_test() == false);
                    ToyTargetMtp::assert_cache_matches_fed(&full_session)?;
                    assert_eq!(
                        chunked_session.glm_mtp_cache_for_test().compact_total(),
                        full_session.glm_mtp_cache_for_test().compact_total()
                    );
                    assert_eq!(
                        chunked_session
                            .glm_mtp_cache_for_test()
                            .compact_live_for_test()
                            .unwrap()
                            .as_dtype(lisa_metal::Dtype::Float32)?
                            .to_vec1::<f32>()?,
                        full_session
                            .glm_mtp_cache_for_test()
                            .compact_live_for_test()
                            .unwrap()
                            .as_dtype(lisa_metal::Dtype::Float32)?
                            .to_vec1::<f32>()?,
                    );
                    if sampling == 0 && budget == 8 {
                        assert_eq!(cycle_lengths[0], 1, "first turn only seeds MTP");
                        assert_eq!(
                            cycle_lengths[1],
                            if bad_first_draft { 1 } else { 2 },
                            "a rejected draft yields after one committed target row"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn glm53_target_mtp_cycle_context_boundary_commits_then_refuses() -> anyhow::Result<()> {
        for context_limit in 2..=5 {
            let mut tower = ToyTargetMtp {
                runtime: crate::metal_language::runtime(),
                bad_first_draft: false,
                eos_on_verify: false,
                bias_verify_multi: false,
                bias_verify_logits: false,
                fail_verify_after_cache_append: false,
                soft_logits: true,
                context_limit,
                mtp_calls: Vec::new(),
            };
            let mut sampler = Sampler(lisa_engine::sampling::Sampler {
                temperature: 0.8,
                seed: 0x5eed,
                ..Default::default()
            });
            let mut session = Session::new(&mut tower);
            let mut emitted = Vec::new();
            while session.fed.len() < context_limit {
                let suffix = if session.fed.is_empty() {
                    &[1][..]
                } else {
                    &[]
                };
                session.target_mtp_cycle(
                    &mut tower,
                    suffix,
                    16,
                    Some(&mut sampler),
                    &mut |token| {
                        emitted.push(token);
                        Ok(TargetMtpSamplingMode::Configured)
                    },
                )?;
                ToyTargetMtp::assert_cache_matches_fed(&session)?;
            }
            assert_eq!(emitted.len(), context_limit - 1);
            assert_eq!(session.fed.len(), context_limit);
            let fed = session.fed.clone();
            let rng = sampler.seed;
            let calls = tower.mtp_calls.clone();
            let error = session
                .target_mtp_cycle(&mut tower, &[], 16, Some(&mut sampler), &mut |_| {
                    anyhow::bail!("callback ran after the context boundary")
                })
                .unwrap_err();
            assert!(error.to_string().contains("context overflow"));
            assert_eq!(session.fed, fed);
            assert_eq!(sampler.seed, rng);
            assert_eq!(tower.mtp_calls, calls);
            ToyTargetMtp::assert_cache_matches_fed(&session)?;
        }
        Ok(())
    }

    #[test]
    fn glm53_target_mtp_batch_scheduler_interleaves_sessions() -> anyhow::Result<()> {
        let mut tower = ToyTargetMtp {
            runtime: crate::metal_language::runtime(),
            bad_first_draft: false,
            eos_on_verify: false,
            bias_verify_multi: false,
            bias_verify_logits: false,
            fail_verify_after_cache_append: false,
            soft_logits: false,
            context_limit: 128,
            mtp_calls: Vec::new(),
        };
        let requests = (0..2)
            .map(|id| crate::core::sched::Request {
                id,
                prompt: vec![1],
                max_tokens: 5,
                sampler: crate::core::sampler::Sampler::default(),
                ignore_eos: false,
            })
            .collect();
        let mut order = Vec::new();
        let outputs = crate::core::sched::run(&mut tower, requests, 2, 2, &mut |id, _| {
            order.push(id);
            Ok(())
        })?;

        assert_eq!(outputs[0], vec![2, 3, 4, 5, 6]);
        assert_eq!(outputs[1], vec![2, 3, 4, 5, 6]);
        assert_eq!(order, vec![0, 1, 0, 0, 1, 1, 0, 0, 1, 1]);
        Ok(())
    }

    #[test]
    fn glm53_target_mtp_rejection_uses_replayed_target_logits() -> anyhow::Result<()> {
        let mut tower = ToyTargetMtp {
            runtime: crate::metal_language::runtime(),
            bad_first_draft: true,
            eos_on_verify: false,
            bias_verify_multi: false,
            bias_verify_logits: true,
            fail_verify_after_cache_append: false,
            soft_logits: false,
            context_limit: 128,
            mtp_calls: Vec::new(),
        };
        let mut session = Session::new(&mut tower);
        let generated = session.generate_mtp(&mut tower, &[1], 4, 2, &mut |_| Ok(()))?;

        assert_eq!(generated, vec![2, 3, 4, 5]);
        assert_eq!(
            tower.mtp_calls.get(1),
            Some(&(3, 2, 1, 6)),
            "rejection seeds the private proposal from the verifier argmax"
        );
        assert_eq!(
            tower.mtp_calls.get(2),
            Some(&(4, 3, 1, 5)),
            "the replayed logits sample carry 4, which invalidates verifier parent 6"
        );
        ToyTargetMtp::assert_cache_matches_fed(&session)?;
        Ok(())
    }

    #[test]
    fn glm53_target_mtp_verify_error_restores_preverify_cache_boundary() -> anyhow::Result<()> {
        for yielded in [false, true] {
            let mut tower = ToyTargetMtp {
                runtime: crate::metal_language::runtime(),
                bad_first_draft: false,
                eos_on_verify: false,
                bias_verify_multi: false,
                bias_verify_logits: false,
                fail_verify_after_cache_append: true,
                soft_logits: false,
                context_limit: 128,
                mtp_calls: Vec::new(),
            };
            let mut session = Session::new(&mut tower);
            let error = if yielded {
                let seed = session.target_mtp_cycle(&mut tower, &[1], 4, None, &mut |_| {
                    Ok(TargetMtpSamplingMode::Configured)
                })?;
                assert_eq!(seed, vec![2]);
                session
                    .target_mtp_cycle(&mut tower, &[], 3, None, &mut |_| {
                        Ok(TargetMtpSamplingMode::Configured)
                    })
                    .expect_err("injected two-row verify must fail after resume")
            } else {
                session
                    .generate_mtp(&mut tower, &[1], 4, 2, &mut |_| Ok(()))
                    .expect_err("injected two-row verify must fail")
            };
            assert!(
                error
                    .to_string()
                    .contains("injected target-MTP verify failure after cache append")
            );
            assert_eq!(session.fed, vec![1, 2]);
            assert!(session.has_glm_mtp_pending_for_test() == false);
            ToyTargetMtp::assert_cache_matches_fed(&session)?;
        }
        Ok(())
    }

    #[test]
    fn glm53_target_mtp_commit_boundary_survives_suffix_extension() -> anyhow::Result<()> {
        let mut tower = ToyTargetMtp {
            runtime: crate::metal_language::runtime(),
            bad_first_draft: false,
            eos_on_verify: false,
            bias_verify_multi: false,
            bias_verify_logits: false,
            fail_verify_after_cache_append: false,
            soft_logits: false,
            context_limit: 128,
            mtp_calls: Vec::new(),
        };
        let mut session = Session::new(&mut tower);
        let first = session.generate_mtp(&mut tower, &[1], 3, 2, &mut |_| Ok(()))?;
        assert_eq!(first, vec![2, 3, 4]);
        assert_eq!(session.fed, vec![1, 2, 3, 4]);

        // The final emitted token is already in the target cache, as in DS4.
        // The next request contributes a new suffix token and reuses the
        // private MTP tape across that prompt extension.
        let next = session.generate_mtp(&mut tower, &[5], 3, 2, &mut |_| Ok(()))?;
        assert_eq!(next, vec![6, 7, 8]);
        assert_eq!(
            tower.mtp_calls,
            vec![
                (2, 1, 1, 3),
                (3, 2, 1, 4),
                (4, 3, 1, 5),
                (6, 5, 1, 7),
                (7, 6, 1, 8),
                (8, 7, 1, 9),
            ],
            "accepted rows must advance the retained MTP cache at absolute positions"
        );
        let compact = session
            .glm_mtp_cache_for_test()
            .compact_slice(0, 8)
            .expect("retained absolute-position MTP cache")
            .as_dtype(lisa_metal::Dtype::Float32)?
            .to_vec1::<f32>()?;
        assert_eq!(compact, vec![0.0, 3.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]);
        Ok(())
    }

    #[test]
    fn glm53_target_mtp_cache_prefix_snapshot_restores_private_tape() -> anyhow::Result<()> {
        let mut tower = ToyTargetMtp {
            runtime: crate::metal_language::runtime(),
            bad_first_draft: false,
            eos_on_verify: false,
            bias_verify_multi: false,
            bias_verify_logits: false,
            fail_verify_after_cache_append: false,
            soft_logits: false,
            context_limit: 128,
            mtp_calls: Vec::new(),
        };
        let mut session = Session::new(&mut tower);
        session.enable_snapshots();
        let initial = Array::from_slice(&[3.0f32], &[1, 1, 1]);
        session
            .glm_mtp_cache_mut_for_test()
            .write_compact_f16_at(&initial, 1)?;
        *session.glm_mtp_min_position_for_test() = Some(1);
        session.fed = vec![10, 11];
        session.capture_snapshot(1, 0);

        let speculative = Array::from_slice(&[9.0f32], &[1, 1, 1]);
        session
            .glm_mtp_cache_mut_for_test()
            .write_compact_f16_at(&speculative, 4)?;
        *session.glm_mtp_min_position_for_test() = Some(4);
        session.fed.push(12);
        assert_eq!(session.restore_snapshot(&mut tower, 1), Some(2));
        assert_eq!(session.fed, vec![10, 11]);
        assert_eq!(*session.glm_mtp_min_position_for_test(), Some(1));
        assert_eq!(
            session
                .glm_mtp_cache_for_test()
                .compact_slice(0, 2)
                .expect("snapshot MTP rows")
                .as_dtype(lisa_metal::Dtype::Float32)?
                .to_vec1::<f32>()?,
            vec![0.0, 3.0]
        );
        assert!(
            session
                .glm_mtp_cache_for_test()
                .compact_slice(0, 5)
                .is_none()
        );
        Ok(())
    }

    #[test]
    fn glm53_target_mtp_opportunistic_and_exact_accept_reject() -> anyhow::Result<()> {
        for (bad_first_draft, exact_sampling) in
            [(false, false), (true, false), (false, true), (true, true)]
        {
            let mut tower = ToyTargetMtp {
                runtime: crate::metal_language::runtime(),
                bad_first_draft,
                eos_on_verify: false,
                bias_verify_multi: false,
                bias_verify_logits: false,
                fail_verify_after_cache_append: false,
                soft_logits: false,
                context_limit: 128,
                mtp_calls: Vec::new(),
            };
            let mut sampler = crate::core::sampler::Sampler(lisa_engine::sampling::Sampler {
                temperature: 0.8,
                seed: 0x5eed,
                mtp_exact_sampling: exact_sampling,
                ..Default::default()
            });
            let mut session = Session::new(&mut tower);
            let generated = session.generate_mtp_depth(
                &mut tower,
                &[1],
                4,
                MtpDepth::Fixed(2),
                None,
                Some(&mut sampler),
                &mut |_| Ok(()),
            )?;

            assert_eq!(generated, vec![2, 3, 4, 5]);
            assert_eq!(session.fed, vec![1, 2, 3, 4, 5]);
            ToyTargetMtp::assert_cache_matches_fed(&session)?;
            assert_eq!(tower.mtp_calls.first(), Some(&(2, 1, 1, 3)));
            if bad_first_draft {
                let expected_call = if exact_sampling {
                    (4, 3, 1, 5)
                } else {
                    (3, 2, 1, 4)
                };
                assert_eq!(
                    tower.mtp_calls.get(1).copied(),
                    Some(expected_call),
                    "exact rejection must prime MTP from the replayed replacement row"
                );
            } else {
                assert_eq!(tower.mtp_calls.len(), 4);
            }
        }
        Ok(())
    }

    #[test]
    fn glm53_target_mtp_exact_sampling_rewinds_on_live_mode_change() -> anyhow::Result<()> {
        let mut tower = ToyTargetMtp {
            runtime: crate::metal_language::runtime(),
            bad_first_draft: false,
            eos_on_verify: false,
            bias_verify_multi: false,
            bias_verify_logits: false,
            fail_verify_after_cache_append: false,
            soft_logits: false,
            context_limit: 128,
            mtp_calls: Vec::new(),
        };
        let mut sampler = crate::core::sampler::Sampler(lisa_engine::sampling::Sampler {
            temperature: 0.8,
            seed: 0x5eed,
            mtp_exact_sampling: true,
            ..Default::default()
        });
        let mut session = Session::new(&mut tower);
        let mut emitted = Vec::new();
        let generated = session.generate_mtp_depth_with_sampling_mode(
            &mut tower,
            &[1],
            4,
            MtpDepth::Fixed(2),
            None,
            Some(&mut sampler),
            &mut |token| {
                emitted.push(token);
                Ok(if token >= 3 {
                    TargetMtpSamplingMode::Greedy
                } else {
                    TargetMtpSamplingMode::Configured
                })
            },
        )?;

        assert_eq!(generated, vec![2, 3, 4, 5]);
        assert_eq!(emitted, generated);
        assert_eq!(session.fed, vec![1, 2, 3, 4, 5]);
        ToyTargetMtp::assert_cache_matches_fed(&session)?;
        assert_eq!(
            tower.mtp_calls,
            vec![
                (2, 1, 1, 3),
                (3, 2, 1, 4),
                (4, 3, 1, 5),
                (4, 3, 1, 5),
                (5, 4, 1, 6),
            ],
            "mode rewind preserves the completed cycle's private-cache writes, clears its proposal, and rearms on a plain step"
        );
        Ok(())
    }

    #[test]
    fn glm53_target_mtp_sampled_parent_mismatch_falls_back_in_both_modes() -> anyhow::Result<()> {
        let first_logits = ToyTargetMtp::target_logits(2, true);
        let next_logits = ToyTargetMtp::target_logits(3, true);
        let seed = (0..256).find(|&seed| {
            let mut sampler = crate::core::sampler::Sampler(lisa_engine::sampling::Sampler {
                temperature: 0.8,
                seed,
                ..Default::default()
            });
            let first = sampler
                .draw_ds4(&crate::metal_language::runtime(), &first_logits, &[1])
                .ok();
            let next = sampler
                .draw_ds4(&crate::metal_language::runtime(), &next_logits, &[1, 2])
                .ok();
            first == Some(2) && next.is_some_and(|token| token != 3)
        });
        let seed = seed.expect("find deterministic seed with a non-greedy sampled parent");

        for exact_sampling in [false, true] {
            let mut tower = ToyTargetMtp {
                runtime: crate::metal_language::runtime(),
                bad_first_draft: false,
                eos_on_verify: false,
                bias_verify_multi: false,
                bias_verify_logits: false,
                fail_verify_after_cache_append: false,
                soft_logits: true,
                context_limit: 128,
                mtp_calls: Vec::new(),
            };
            let mut sampler = crate::core::sampler::Sampler(lisa_engine::sampling::Sampler {
                temperature: 0.8,
                seed,
                mtp_exact_sampling: exact_sampling,
                ..Default::default()
            });
            let mut session = Session::new(&mut tower);
            let generated = session.generate_mtp_depth(
                &mut tower,
                &[1],
                3,
                MtpDepth::Fixed(2),
                None,
                Some(&mut sampler),
                &mut |_| Ok(()),
            )?;

            assert_eq!(generated[0], 2);
            assert_ne!(
                generated[1], 3,
                "the sampled parent must differ from argmax"
            );
            assert_eq!(generated.len(), 3);
            assert_eq!(session.fed, vec![1, 2, generated[1], generated[2]]);
            assert_eq!(
                tower.mtp_calls[0],
                (2, 1, 1, 3),
                "the committed prefix MTP row must be retained"
            );
            assert_eq!(
                tower.mtp_calls[1],
                (generated[1], 2, 1, generated[1] + 1),
                "mismatched stochastic parent must target-fallback before reprime"
            );
            assert_eq!(
                tower.mtp_calls[2],
                (generated[2], 3, 1, generated[2] + 1),
                "the budget tail is primed from its own target hidden row"
            );
            ToyTargetMtp::assert_cache_matches_fed(&session)?;
        }
        Ok(())
    }

    #[test]
    fn glm53_target_mtp_commits_non_eos_output_through_budget_boundary() -> anyhow::Result<()> {
        for (budget, expected, fed, mtp_calls) in [
            (1, vec![2], vec![1, 2], 1),
            (2, vec![2, 3], vec![1, 2, 3], 2),
            (3, vec![2, 3, 4], vec![1, 2, 3, 4], 3),
        ] {
            let mut tower = ToyTargetMtp {
                runtime: crate::metal_language::runtime(),
                bad_first_draft: false,
                eos_on_verify: false,
                bias_verify_multi: false,
                bias_verify_logits: false,
                fail_verify_after_cache_append: false,
                soft_logits: false,
                context_limit: 128,
                mtp_calls: Vec::new(),
            };
            let mut session = Session::new(&mut tower);
            let generated = session.generate_mtp(&mut tower, &[1], budget, 2, &mut |_| Ok(()))?;
            assert_eq!(generated, expected, "budget {budget}");
            assert_eq!(session.fed, fed, "target-cache boundary at budget {budget}");
            assert_eq!(
                tower.mtp_calls.len(),
                mtp_calls,
                "MTP calls at budget {budget}"
            );
            let expected_compact = match budget {
                1 => vec![0.0, 3.0],
                2 => vec![0.0, 3.0, 4.0],
                3 => vec![0.0, 3.0, 4.0, 5.0],
                _ => unreachable!(),
            };
            assert_eq!(
                session
                    .glm_mtp_cache_for_test()
                    .compact_slice(0, expected_compact.len())
                    .expect("MTP cache through generated target tokens")
                    .as_dtype(lisa_metal::Dtype::Float32)?
                    .to_vec1::<f32>()?,
                expected_compact,
                "private MTP cache boundary at output budget {budget}"
            );
        }
        Ok(())
    }

    #[test]
    fn glm53_target_mtp_stops_on_eos_at_each_speculative_boundary() -> anyhow::Result<()> {
        struct ClearEos;
        impl Drop for ClearEos {
            fn drop(&mut self) {
                crate::core::generate::set_eos_ids(Vec::new());
            }
        }

        crate::core::generate::set_eos_ids(vec![31]);
        let _clear_eos = ClearEos;

        let cases = [
            // EOS is the first target token: it must not be fed.
            (vec![30], false, false, vec![31], vec![30], 0),
            // EOS is the target token after the first output: MTP is not armed.
            (vec![29], false, false, vec![30, 31], vec![29, 30], 1),
            // The draft predicts EOS and the target accepts it; DS4 commits
            // both verified rows, including the accepted EOS row.
            (
                vec![28],
                false,
                false,
                vec![29, 30, 31],
                vec![28, 29, 30, 31],
                3,
            ),
            // The draft is rejected and the target replacement is EOS.
            (vec![26], true, true, vec![27, 28, 31], vec![26, 27, 28], 2),
        ];

        for (prompt, bad_first_draft, eos_on_verify, expected, fed, mtp_calls) in cases {
            for yielded in [false, true] {
                let mut tower = ToyTargetMtp {
                    runtime: crate::metal_language::runtime(),
                    bad_first_draft,
                    eos_on_verify,
                    bias_verify_multi: false,
                    bias_verify_logits: false,
                    fail_verify_after_cache_append: false,
                    soft_logits: false,
                    context_limit: 128,
                    mtp_calls: Vec::new(),
                };
                let mut session = Session::new(&mut tower);
                let generated = if yielded {
                    let mut generated = Vec::new();
                    while generated.len() < 8
                        && !generated.last().is_some_and(|&token| is_eos(token))
                    {
                        let suffix = if session.fed.is_empty() {
                            prompt.as_slice()
                        } else {
                            &[]
                        };
                        generated.extend(session.target_mtp_cycle(
                            &mut tower,
                            suffix,
                            8 - generated.len(),
                            None,
                            &mut |_| Ok(TargetMtpSamplingMode::Configured),
                        )?);
                    }
                    generated
                } else {
                    session.generate_mtp(&mut tower, &prompt, 8, 2, &mut |_| Ok(()))?
                };
                assert_eq!(generated, expected, "prompt {prompt:?}");
                assert_eq!(session.fed, fed, "target-cache boundary for {prompt:?}");
                assert_eq!(tower.mtp_calls.len(), mtp_calls, "MTP calls for {prompt:?}");
                assert!(session.has_glm_mtp_pending_for_test() == false);
            }
        }
        Ok(())
    }

    #[test]
    fn glm53_target_mtp_exact_replacement_eos_is_committed() -> anyhow::Result<()> {
        struct ClearEos;
        impl Drop for ClearEos {
            fn drop(&mut self) {
                crate::core::generate::set_eos_ids(Vec::new());
            }
        }

        crate::core::generate::set_eos_ids(vec![31]);
        let _clear_eos = ClearEos;
        let mut tower = ToyTargetMtp {
            runtime: crate::metal_language::runtime(),
            bad_first_draft: true,
            eos_on_verify: true,
            bias_verify_multi: false,
            bias_verify_logits: false,
            fail_verify_after_cache_append: false,
            soft_logits: false,
            context_limit: 128,
            mtp_calls: Vec::new(),
        };
        let mut sampler = crate::core::sampler::Sampler(lisa_engine::sampling::Sampler {
            temperature: 0.8,
            seed: 0x5eed,
            mtp_exact_sampling: true,
            ..Default::default()
        });
        let mut session = Session::new(&mut tower);
        let generated = session.generate_mtp_depth(
            &mut tower,
            &[26],
            8,
            MtpDepth::Fixed(2),
            None,
            Some(&mut sampler),
            &mut |_| Ok(()),
        )?;

        assert_eq!(generated, vec![27, 28, 31]);
        assert_eq!(session.fed, vec![26, 27, 28, 31]);
        ToyTargetMtp::assert_cache_matches_fed(&session)?;
        assert_eq!(tower.mtp_calls.len(), 1);
        Ok(())
    }
}

#[cfg(test)]
#[path = "session_block_tests.rs"]
mod block_draft_session_tests;
