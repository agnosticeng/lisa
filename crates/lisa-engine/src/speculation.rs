//! Portable speculative execution: PLD, independent drafts, target-conditioned
//! MTP and block drafts. The engine owns transaction order; runtime adapters
//! provide tensor and cache storage operations without selecting a model.
use crate::conversation::{Conversation, TargetMtpPending};
pub use crate::conversation::{TargetMtpEosPolicy, TargetMtpSamplingMode};
use crate::core::copy_draft::{COPY_K, CopyIndex, copy_lookup_guarded};
use crate::eos::is_eos;
use crate::language::LanguageModel;
use crate::speculative_runtime::{SpeculativeRuntime, SpeculativeSampler, SpeculativeTensor};
#[cfg(any(test, feature = "diagnostics"))]
fn capture_glm_mtp_verify_hyper_states<T: SpeculativeTensor>(
    prefix: &std::ffi::OsStr,
    rows: &T,
    first_fed_position: usize,
) -> anyhow::Result<usize> {
    static NEXT_CAPTURE: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    anyhow::ensure!(
        rows.rank() == 3 && rows.dim(0) == 1 && rows.dim(1) == 2,
        "GLM MTP verify hyper-state capture expects shape [1, 2, width], got {:?}",
        rows.shape()
    );
    let capture_index = NEXT_CAPTURE.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let prefix = std::path::PathBuf::from(prefix).display().to_string();
    for row in 0..2usize {
        let values = rows
            .select(1, row as i32)?
            .contiguous()?
            .reshape(&[rows.dim(2)])?
            .read_f32()?;
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let path = format!("{prefix}.verify-hc-fed-{:05}.f32", first_fed_position + row);
        std::fs::write(path, &bytes)?;
        let path = format!(
            "{prefix}.verify-hc-capture-{capture_index:03}-fed-{:05}.f32",
            first_fed_position + row
        );
        std::fs::write(path, bytes)?;
    }
    Ok(capture_index)
}

#[cfg(any(test, feature = "diagnostics"))]
fn capture_glm_mtp_verify_logits<T: SpeculativeTensor>(
    prefix: &std::ffi::OsStr,
    capture_index: usize,
    logits: &T,
    fed_position: usize,
) -> anyhow::Result<()> {
    let values = logits
        .contiguous()?
        .reshape(&[logits.size() as i32])?
        .read_f32()?;
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let prefix = std::path::PathBuf::from(prefix).display().to_string();
    std::fs::write(
        format!("{prefix}.verify-logits-capture-{capture_index:03}-fed-{fed_position:05}.f32"),
        bytes,
    )?;
    Ok(())
}

impl<R: SpeculativeRuntime> Conversation<R> {
    #[cfg(any(test, feature = "diagnostics"))]
    pub fn glm_mtp_cache_for_test(&self) -> &R::DraftCache {
        &self.glm_mtp_cache
    }

    #[cfg(any(test, feature = "diagnostics"))]
    fn record_glm_mtp_target_logits_for_test(
        &mut self,
        token_count: usize,
        logits: &R::Tensor,
    ) -> anyhow::Result<()> {
        let values = logits
            .contiguous()?
            .reshape(&[logits.size() as i32])?
            .read_f32()?;
        self.glm_mtp_target_logits.push((token_count, values));
        Ok(())
    }

    #[cfg(any(test, feature = "diagnostics"))]
    pub fn glm_mtp_target_logits_for_test(&self) -> &[(usize, Vec<f32>)] {
        &self.glm_mtp_target_logits
    }

    #[cfg(any(test, feature = "diagnostics"))]
    pub fn set_glm_mtp_state_capture_for_test<F>(&mut self, capture: F)
    where
        F: FnMut(usize, &[R::Cache], &R::DraftCache) -> anyhow::Result<()> + 'static,
    {
        self.glm_mtp_state_capture = Some(Box::new(capture));
    }

    #[cfg(any(test, feature = "diagnostics"))]
    fn capture_glm_mtp_state_for_test(&mut self) -> anyhow::Result<()> {
        let token_count = self.fed.len();
        let caches = &self.caches;
        let mtp_cache = &self.glm_mtp_cache;
        if let Some(capture) = self.glm_mtp_state_capture.as_mut() {
            capture(token_count, caches, mtp_cache)?;
        }
        Ok(())
    }

    /// Feed `tokens`, then sample until EOS or `max_tokens`.
    ///
    /// The final EOS token is emitted but not fed (matching `generate`), so the
    /// next turn appends it as part of its suffix.
    pub fn generate(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        tokens: &[u32],
        max_tokens: usize,
        sampler: &mut R::Sampler,
        pld: usize,
        mut after_prefill: Option<&mut dyn FnMut(&[R::Cache])>,
        on_token: &mut dyn FnMut(u32) -> anyhow::Result<()>,
    ) -> anyhow::Result<Vec<u32>> {
        let logits = self.feed(tower, tokens)?;
        // Clean commit boundary (prompt fed, nothing decoded yet): the
        // cross-request prefix cache snapshots here.
        if let Some(cb) = after_prefill.as_deref_mut() {
            cb(&self.caches);
        }
        let mut token = if tower.uses_ds4_sampling() {
            sampler.draw_ds4(tower.runtime(), &logits, &self.fed)?
        } else {
            sampler.draw(tower.runtime(), &logits, &self.fed)?
        };
        let mut out = vec![token];
        on_token(token)?;
        // PLD path: prompt-lookup drafts verified by a wide forward; the caches
        // carry on exactly as in the serial loop. At temperature > 0 the verify
        // samples each row through the keyed Gumbel draw, which is what makes
        // the packed checkpoints' own default (generation_config: temperature
        // 1.0) speculative at all — it used to get none.
        if pld > 0 {
            let mut committed: Vec<u32> = self.fed.clone();
            committed.push(token);
            let (g, stats) = crate::generation::pld_decode(
                tower,
                &mut self.caches,
                &mut committed,
                token,
                max_tokens,
                out.len(),
                pld,
                if sampler.is_greedy() {
                    None
                } else {
                    Some(&mut *sampler)
                },
                on_token,
            )?;
            out.extend(g);
            // Engagement counter: a silent draft source must be visible (the
            // lesson from the copy counter — without this line the stochastic
            // path can be off and the run looks merely "slower").
            eprintln!(
                "[pld.round] depth {pld} rounds {} hits {} accepted {} verified_tokens {} \
                 (stochastic={})",
                stats.rounds,
                stats.hits,
                stats.accepted,
                stats.verified_tokens,
                !sampler.is_greedy()
            );
            self.fed = committed;
            return Ok(out);
        }
        while out.len() < max_tokens && !is_eos(token) {
            let logits = self.decode_token(tower, token)?;
            token = if tower.uses_ds4_sampling() {
                sampler.draw_ds4(tower.runtime(), &logits, &self.fed)?
            } else {
                sampler.draw(tower.runtime(), &logits, &self.fed)?
            };
            out.push(token);
            on_token(token)?;
        }
        Ok(out)
    }
}

/// Depth selection policy for the speculative loop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MtpDepth {
    /// Fixed draft depth for the selected model (0 selects serial). External
    /// DSpine blocks allow 1..=15; independent Qwen MTP chains use 2..=6. GLM's
    /// target-conditioned MTP uses its native single-step verify cycle.
    Fixed(usize),
    /// EV auto for independent draft chains, measured SPS for DSpine blocks.
    /// Target-conditioned GLM MTP keeps
    /// its native one-step cycle because the Qwen EV table does not price it.
    Auto,
}

/// One greedy serial decode step over the live caches: the MTP loop's serial
/// fallback. Returns the argmax token and the last multi row (`[1, D]`), so the
/// caller can hand the pair back to the draft chain.
fn serial_step<R: SpeculativeRuntime>(
    tower: &mut dyn LanguageModel<R>,
    caches: &mut Vec<R::Cache>,
    tail: &[i64],
    carry_token: u32,
    stochastic: Option<&mut R::Sampler>,
    // Absolute position of the token this step predicts — the serial non-greedy
    // path keys its Gumbel draw on it.
    position: u64,
) -> anyhow::Result<(u32, R::Tensor)> {
    tower.set_context_tails(vec![tail.to_vec()]);
    let arr = tower.runtime().tokens(&[carry_token as i32], &[1i32, 1])?;
    // No capture: the verify-capture buffers are large fresh allocations per
    // step (the SSM capture alone is ~S x the recurrent state). The caller
    // snapshots the GDN state beforehand (cheap Arc clones) and restores it on
    // EOS, which is the only case that needs to un-feed the step.
    let (mixed, multi) = tower.forward_capture(&arr, Some(caches), false)?;
    let logits = tower.head(&mixed)?;
    // A serial step is an ORDINARY draw, so it must sample when the stream is
    // non-greedy. Leaving this on argmax emits greedy tokens for the whole
    // serial stretch — the exact trap `pld_decode`'s no-draft branch had
    // (identical text across seeds at temperature 1.0).
    let t = match stochastic.as_deref() {
        // GPU-side keyed draw, the same one the certified serial non-greedy path
        // uses. `draw_row` (host softmax) copies 248k f32 PER TOKEN here.
        Some(sm) => sm
            .draw_at(tower.runtime(), &logits, position)
            .unwrap_or(logits.argmax_id()?),
        None => logits.argmax_id()?,
    };
    let row = multi.select(1, 0)?.contiguous()?;
    Ok((t, row))
}

fn sampling_history_with_suffix(history: &[u32], window: usize, suffix: &[u32]) -> Vec<u32> {
    let keep = window.saturating_sub(suffix.len());
    let start = history.len().saturating_sub(keep);
    let mut out = history[start..].to_vec();
    out.extend_from_slice(suffix);
    out
}

fn target_mtp_uses_exact_sampling<S: SpeculativeSampler>(
    sampler: Option<&S>,
    mode: TargetMtpSamplingMode,
) -> bool {
    mode == TargetMtpSamplingMode::Configured
        && sampler.is_some_and(|sampler| sampler.exact_sampling() && sampler.temperature() > 0.0)
}

/// A configured positive-temperature draw must still advance DS4's RNG, even
/// with top-k=1. A repetition penalty can also change the unmodified argmax.
pub fn target_mtp_draw_reuses_argmax<S: SpeculativeSampler>(
    sampler: Option<&S>,
    mode: TargetMtpSamplingMode,
) -> bool {
    mode == TargetMtpSamplingMode::Greedy
        || sampler.is_none_or(|sampler| {
            sampler.temperature() <= 0.0 && sampler.repetition_penalty() == 1.0
        })
}

/// Block drafts use the target's ordinary Qwen sampling law, including the
/// absolute-position key and repetition history, for both serial and verify rows.
fn draw_block_target_token<S: SpeculativeSampler<Tensor: SpeculativeTensor>>(
    context: &S::Context,
    sampler: Option<&mut S>,
    logits: &S::Tensor,
    history: &[u32],
    mode: TargetMtpSamplingMode,
) -> anyhow::Result<u32> {
    if mode == TargetMtpSamplingMode::Greedy {
        return logits.argmax_id();
    }
    match sampler {
        Some(sampler) => sampler.draw(context, logits, history),
        None => logits.argmax_id(),
    }
}

fn draw_target_mtp_token<S: SpeculativeSampler<Tensor: SpeculativeTensor>>(
    context: &S::Context,
    sampler: Option<&mut S>,
    logits: &S::Tensor,
    history: &[u32],
    mode: TargetMtpSamplingMode,
) -> anyhow::Result<u32> {
    if mode == TargetMtpSamplingMode::Greedy {
        return logits.argmax_id();
    }
    match sampler {
        Some(sampler) => sampler.draw_ds4(context, logits, history),
        None => logits.argmax_id(),
    }
}

impl<R: SpeculativeRuntime> Conversation<R> {
    fn restore_target_verify_to_first(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        before: &[R::Rollback],
        first: u32,
    ) -> anyhow::Result<R::Tensor> {
        let _trace = tower.runtime().trace("glm53.mtp.rollback_replay");
        self.restore_target_verify_boundary(before)?;
        #[cfg(any(test, feature = "diagnostics"))]
        let replay_capture_path = std::env::var("LISA_GLM53_MTP_REPLAY_CAPTURE_PATH").ok();
        #[cfg(any(test, feature = "diagnostics"))]
        let replay_capture = if replay_capture_path.is_some()
            && std::env::var("LISA_GLM53_MTP_REPLAY_CAPTURE_FED")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                == Some(self.fed.len())
        {
            let layers = std::env::var("LISA_GLM53_MTP_REPLAY_CAPTURE_LAYERS")
                .unwrap_or_else(|_| "0-44".to_owned())
                .split(',')
                .flat_map(|part| {
                    if let Some((start, end)) = part.split_once('-') {
                        match (start.trim().parse::<usize>(), end.trim().parse::<usize>()) {
                            (Ok(start), Ok(end)) if start <= end => (start..=end).collect(),
                            _ => Vec::new(),
                        }
                    } else {
                        part.trim().parse::<usize>().ok().into_iter().collect()
                    }
                })
                .collect::<Vec<_>>();
            let key_filters = std::env::var("LISA_GLM53_MTP_REPLAY_CAPTURE_KEYS")
                .unwrap_or_else(|_| "state_after_attn,state_after_ffn".to_owned())
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
                .collect::<Vec<_>>();
            anyhow::ensure!(!layers.is_empty(), "MTP replay capture has no layers");
            anyhow::ensure!(
                !key_filters.is_empty(),
                "MTP replay capture has no key filters"
            );
            tower.set_debug_capture_filtered_for_test(layers, key_filters)?;
            true
        } else {
            false
        };
        let replay = tower.runtime().tokens(&[first as i32], &[1i32, 1])?;
        let (mixed, multi) = tower.forward_capture(&replay, Some(&mut self.caches), false)?;
        #[cfg(any(test, feature = "diagnostics"))]
        if replay_capture {
            tower.flush_debug_capture_for_test(
                replay_capture_path
                    .as_deref()
                    .expect("capture path checked above"),
            )?;
        }
        self.fed.push(first);
        self.last_mixed = Some(mixed.select(1, 0)?.contiguous()?);
        Ok(multi)
    }

    fn restore_target_verify_boundary(&mut self, before: &[R::Rollback]) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.caches.len() == before.len(),
            "target cache layout changed during GLM MTP verify"
        );
        for (cache, state) in self.caches.iter_mut().zip(before) {
            R::restore_rollback(cache, state)?;
        }
        Ok(())
    }

    fn advance_target_mtp_cache(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        row0_multi: &R::Tensor,
        row1_multi: &R::Tensor,
        draft: u32,
        next: u32,
        position: usize,
        min_position: usize,
    ) -> anyhow::Result<R::Tensor> {
        let _trace = tower.runtime().trace("glm53.mtp.cache_catchup");
        let _ = tower.target_mtp_step(
            &mut self.glm_mtp_cache,
            row0_multi,
            draft,
            position,
            min_position,
        )?;
        tower.target_mtp_step(
            &mut self.glm_mtp_cache,
            row1_multi,
            next,
            position + 1,
            min_position,
        )
    }

    /// Finish the native cycle's private-cache writes, then retain only its
    /// first target row. DS4 performs this rewind after emitting a parser/control
    /// boundary; it preserves private-cache writes and clears the proposal.
    fn rewind_target_mtp_cycle(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        before: &[R::Rollback],
        first: u32,
        draft: u32,
        accepted: bool,
        replacement: Option<u32>,
        row0_multi: &R::Tensor,
        row1_multi: &R::Tensor,
        row0_logits: &R::Tensor,
        row1_logits: &R::Tensor,
        position: usize,
        min_position: usize,
    ) -> anyhow::Result<R::Tensor> {
        if accepted {
            let next = row1_logits.argmax_id()?;
            let _ = self.advance_target_mtp_cache(
                tower,
                row0_multi,
                row1_multi,
                draft,
                next,
                position,
                min_position,
            )?;
        } else if let Some(replacement) = replacement.filter(|token| {
            !is_eos(*token) || self.target_mtp_eos_policy == TargetMtpEosPolicy::CommitVerified
        }) {
            self.restore_target_verify_boundary(before)?;
            let first_ids = tower.runtime().tokens(&[first as i32], &[1i32, 1])?;
            let _ = tower.forward_capture(&first_ids, Some(&mut self.caches), false)?;
            let replacement_ids = tower.runtime().tokens(&[replacement as i32], &[1i32, 1])?;
            let (mixed, multi) =
                tower.forward_capture(&replacement_ids, Some(&mut self.caches), false)?;
            if !is_eos(replacement) {
                let next = tower.head(&mixed)?.argmax_id()?;
                let _ = tower.target_mtp_step(
                    &mut self.glm_mtp_cache,
                    &multi,
                    next,
                    position + 1,
                    min_position,
                )?;
            }
        }
        let replay_multi = self.restore_target_verify_to_first(tower, before, first)?;
        if !accepted && replacement.is_none() {
            // A greedy rejection committed only the first row and seeds from
            // its verifier argmax before the caller emits that row.
            let next = row0_logits.argmax_id()?;
            let _ = tower.target_mtp_step(
                &mut self.glm_mtp_cache,
                &replay_multi,
                next,
                position,
                min_position,
            )?;
        }
        let replay_logits = tower.head(
            self.last_mixed
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("GLM MTP rewind lost its target hidden state"))?,
        )?;
        #[cfg(any(test, feature = "diagnostics"))]
        self.record_glm_mtp_target_logits_for_test(self.fed.len(), &replay_logits)?;
        Ok(replay_logits)
    }

    fn commit_target_mtp_tail(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        token: u32,
        position: usize,
        min_position: usize,
        prime_mtp: bool,
    ) -> anyhow::Result<()> {
        let _trace = tower.runtime().trace("glm53.mtp.commit_tail");
        anyhow::ensure!(
            position == self.fed.len(),
            "GLM MTP tail position {position} does not match fed boundary {}",
            self.fed.len()
        );
        anyhow::ensure!(
            position < tower.max_position_embeddings(),
            "context overflow before GLM MTP tail commit"
        );
        let before: Vec<_> = self.caches.iter().map(R::snapshot_rollback).collect();
        let mtp_before = R::snapshot_draft(&self.glm_mtp_cache);
        let result: anyhow::Result<R::Tensor> = (|| -> anyhow::Result<R::Tensor> {
            let ids = tower.runtime().tokens(&[token as i32], &[1i32, 1])?;
            let (mixed, multi) = tower.forward_capture(&ids, Some(&mut self.caches), false)?;
            if prime_mtp {
                let next = tower.head(&mixed)?.argmax_id()?;
                let _ = tower.target_mtp_step(
                    &mut self.glm_mtp_cache,
                    &multi,
                    next,
                    position,
                    min_position,
                )?;
            }
            Ok(mixed.select(1, 0)?.contiguous()?)
        })();
        match result {
            Ok(mixed) => {
                anyhow::ensure!(
                    self.caches.len() == before.len(),
                    "target cache layout changed while committing the final GLM MTP token"
                );
                self.fed.push(token);
                #[cfg(any(test, feature = "diagnostics"))]
                if prime_mtp {
                    let logits = tower.head(&mixed)?;
                    self.record_glm_mtp_target_logits_for_test(position + 1, &logits)?;
                }
                #[cfg(any(test, feature = "diagnostics"))]
                self.capture_glm_mtp_state_for_test()?;
                self.last_mixed = Some(mixed);
                Ok(())
            }
            Err(error) => {
                anyhow::ensure!(
                    self.caches.len() == before.len(),
                    "target cache layout changed while rolling back the final GLM MTP token"
                );
                for (cache, state) in self.caches.iter_mut().zip(&before) {
                    R::restore_rollback(cache, state)?;
                }
                R::restore_draft(&mut self.glm_mtp_cache, &mtp_before);
                Err(error)
            }
        }
    }

    /// Finish a terminal target draw before notifying the caller. A callback
    /// stop then leaves a reusable committed boundary; model failures do not.
    fn finish_target_mtp_tail(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        token: u32,
        min_position: usize,
        on_token: &mut dyn FnMut(u32) -> anyhow::Result<TargetMtpSamplingMode>,
        committed_callback_stop: &mut bool,
    ) -> anyhow::Result<()> {
        if !is_eos(token) {
            self.commit_target_mtp_tail(tower, token, self.fed.len(), min_position, true)?;
        }
        if let Err(error) = on_token(token) {
            *committed_callback_stop = true;
            return Err(error);
        }
        Ok(())
    }

    /// Advance one target-conditioned MTP cycle, then yield with the pending
    /// proposal and sampled carry intact. `remaining_tokens` is the request's
    /// entire remaining output budget, not a scheduler slice. Initialization
    /// emits one seed token; later calls emit at most one verify result and its
    /// terminal carry (zero to three tokens). Pass the prompt only on the first
    /// call, then an empty suffix and the same per-request sampler.
    pub fn target_mtp_cycle(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        tokens: &[u32],
        remaining_tokens: usize,
        stochastic: Option<&mut R::Sampler>,
        on_token: &mut dyn FnMut(u32) -> anyhow::Result<TargetMtpSamplingMode>,
    ) -> anyhow::Result<Vec<u32>> {
        anyhow::ensure!(
            tower.has_target_mtp(),
            "model has no target-conditioned MTP head"
        );
        let _trace = tower
            .runtime()
            .trace_detail("glm53.mtp.cycle", self.fed.len() as u64);
        self.generate_target_mtp_inner(
            tower,
            tokens,
            remaining_tokens,
            None,
            stochastic,
            on_token,
            true,
        )
    }

    fn generate_target_mtp(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        tokens: &[u32],
        max_tokens: usize,
        after_prefill: Option<&mut dyn FnMut(&[R::Cache])>,
        stochastic: Option<&mut R::Sampler>,
        on_token: &mut dyn FnMut(u32) -> anyhow::Result<TargetMtpSamplingMode>,
    ) -> anyhow::Result<Vec<u32>> {
        self.generate_target_mtp_inner(
            tower,
            tokens,
            max_tokens,
            after_prefill,
            stochastic,
            on_token,
            false,
        )
    }

    fn generate_target_mtp_inner(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        tokens: &[u32],
        max_tokens: usize,
        after_prefill: Option<&mut dyn FnMut(&[R::Cache])>,
        stochastic: Option<&mut R::Sampler>,
        on_token: &mut dyn FnMut(u32) -> anyhow::Result<TargetMtpSamplingMode>,
        single_cycle: bool,
    ) -> anyhow::Result<Vec<u32>> {
        self.ensure_cache_numerical_law(tower)?;

        if max_tokens == 0 {
            return Ok(Vec::new());
        }
        // Reserve capacity before drawing or emitting a token. A cycle may
        // finish the last legal row, but a later call at the limit must leave
        // its RNG, callbacks, trunk caches, and private tape unchanged.
        let prompt_end = self
            .fed
            .len()
            .checked_add(tokens.len())
            .ok_or_else(|| anyhow::anyhow!("context position overflow before GLM MTP cycle"))?;
        anyhow::ensure!(
            prompt_end < tower.max_position_embeddings(),
            "context overflow before GLM MTP cycle: position {prompt_end}, limit {}",
            tower.max_position_embeddings()
        );
        let max_tokens = max_tokens.min(tower.max_position_embeddings() - prompt_end);
        let mut committed_callback_stop = false;
        let result = self.target_mtp_transaction(
            tower,
            tokens,
            max_tokens,
            after_prefill,
            stochastic,
            on_token,
            single_cycle,
            &mut committed_callback_stop,
        );
        if result.is_err() && !committed_callback_stop {
            // DS4 invalidates a checkpoint after a failed speculative operation.
            // Local rollback handles the common verifier/tail errors, but a
            // seed, replay, private-cache step, or callback can fail after other
            // state changed. Never expose that partial transaction for reuse.
            self.invalidate();
        }
        result
    }

    fn target_mtp_transaction(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        tokens: &[u32],
        max_tokens: usize,
        mut after_prefill: Option<&mut dyn FnMut(&[R::Cache])>,
        mut stochastic: Option<&mut R::Sampler>,
        on_token: &mut dyn FnMut(u32) -> anyhow::Result<TargetMtpSamplingMode>,
        single_cycle: bool,
        committed_callback_stop: &mut bool,
    ) -> anyhow::Result<Vec<u32>> {
        let pending = self
            .glm_mtp_pending
            .take()
            .filter(|pending| tokens.is_empty() && pending.fed_len == self.fed.len());
        let resuming = pending.is_some();
        let mut generated = Vec::with_capacity(if single_cycle { 3 } else { max_tokens });
        let (
            mut carry,
            mut draft_parent,
            mut draft,
            mut draft_logits,
            mut sampling_mode,
            min_position,
        ) = if let Some(pending) = pending {
            (
                pending.carry,
                pending.draft_parent,
                pending.draft,
                pending.draft_logits,
                pending.sampling_mode,
                pending.min_position,
            )
        } else {
            // Keep the private MTP window with the session across matching-prefix
            // extensions, as DS4 does. Prefix rewinds restore the corresponding
            // tape snapshot in `restore_snapshot`.
            // GLM's target-conditioned MTP only needs trunk hidden rows for newly
            // generated tokens. Retaining every prompt mHC stream would add
            // 16,384 F32 values per prompt token for no use.
            let logits = if tokens.is_empty() {
                let last = self.last_mixed.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("cannot resume GLM target-MTP before feeding a prompt")
                })?;
                tower.head(last)?
            } else {
                let _trace = tower.runtime().trace("glm53.mtp.prefill");
                let logits = self.feed(tower, tokens)?;
                if let Some(cb) = after_prefill.as_deref_mut() {
                    cb(&self.caches);
                }
                logits
            };
            let mut sampling_mode = TargetMtpSamplingMode::Configured;
            let first = draw_target_mtp_token(
                tower.runtime(),
                stochastic.as_deref_mut(),
                &logits,
                &self.fed,
                sampling_mode,
            )?;
            #[cfg(any(test, feature = "diagnostics"))]
            let first_decode_capture_path =
                std::env::var("LISA_GLM53_MTP_FIRST_DECODE_CAPTURE").ok();
            #[cfg(any(test, feature = "diagnostics"))]
            if first_decode_capture_path.is_some() {
                let layers = std::env::var("LISA_GLM53_MTP_FIRST_DECODE_CAPTURE_LAYERS")
                    .unwrap_or_else(|_| "9,10,11".to_owned())
                    .split(',')
                    .map(|item| {
                        item.trim().parse::<usize>().map_err(|error| {
                            anyhow::anyhow!("invalid MTP decode capture layer {item:?}: {error}")
                        })
                    })
                    .collect::<anyhow::Result<Vec<_>>>()?;
                anyhow::ensure!(
                    !layers.is_empty(),
                    "MTP decode capture layer list must not be empty"
                );
                let key_filters = std::env::var("LISA_GLM53_MTP_FIRST_DECODE_CAPTURE_KEYS")
                    .ok()
                    .map(|value| {
                        value
                            .split(',')
                            .map(str::trim)
                            .filter(|value| !value.is_empty())
                            .map(str::to_owned)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                if key_filters.is_empty() {
                    tower.set_debug_capture_for_test(layers)?;
                } else {
                    tower.set_debug_capture_filtered_for_test(layers, key_filters)?;
                }
            }
            generated.push(first);
            sampling_mode = match on_token(first) {
                Ok(mode) => mode,
                Err(error) => {
                    if !is_eos(first) {
                        let position = self.fed.len();
                        let min_position = *self.glm_mtp_min_position.get_or_insert(position);
                        self.commit_target_mtp_tail(tower, first, position, min_position, true)?;
                    }
                    *committed_callback_stop = true;
                    return Err(error);
                }
            };
            if is_eos(first) {
                return Ok(generated);
            }

            // The first generated token is fed once to obtain the target hidden
            // streams from which GLM's MTP block proposes the following token.
            let first_position = self.fed.len();
            let min_position = *self.glm_mtp_min_position.get_or_insert(first_position);
            anyhow::ensure!(
                first_position < tower.max_position_embeddings(),
                "context overflow before GLM MTP target step"
            );
            if generated.len() >= max_tokens {
                self.commit_target_mtp_tail(tower, first, first_position, min_position, true)?;
                return Ok(generated);
            }
            let first_arr = tower.runtime().tokens(&[first as i32], &[1i32, 1])?;
            let (first_mixed, first_multi) = {
                let _trace = tower.runtime().trace("glm53.mtp.target_seed_forward");
                tower.forward_capture(&first_arr, Some(&mut self.caches), false)?
            };
            self.last_mixed = Some(first_mixed.select(1, 0)?.contiguous()?);
            self.fed.push(first);
            let first_logits = {
                let _trace = tower.runtime().trace("glm53.mtp.target_seed_head");
                tower.head(&first_mixed)?
            };
            #[cfg(any(test, feature = "diagnostics"))]
            self.record_glm_mtp_target_logits_for_test(self.fed.len(), &first_logits)?;
            #[cfg(any(test, feature = "diagnostics"))]
            if let Some(path) = first_decode_capture_path.as_deref() {
                tower.flush_debug_capture_for_test(path)?;
            }
            let carry = draw_target_mtp_token(
                tower.runtime(),
                stochastic.as_deref_mut(),
                &first_logits,
                &self.fed,
                sampling_mode,
            )?;
            // DS4 seeds GLM MTP from the target argmax in both opportunistic and
            // exact modes. A sampled boundary token that differs from this parent
            // invalidates the pending proposal and takes the plain target path.
            let draft_parent =
                if target_mtp_draw_reuses_argmax(stochastic.as_deref(), sampling_mode) {
                    carry
                } else {
                    first_logits.argmax_id()?
                };
            // This step also writes the compact-cache prefix needed to continue the
            // session. Keep it even when the sampled boundary invalidates its draft.
            let draft_logits = {
                let _trace = tower.runtime().trace("glm53.mtp.proposal_step");
                Some(tower.target_mtp_step(
                    &mut self.glm_mtp_cache,
                    &first_multi,
                    draft_parent,
                    first_position,
                    min_position,
                )?)
            };
            if is_eos(carry) || generated.len() + 1 >= max_tokens {
                generated.push(carry);
                self.finish_target_mtp_tail(
                    tower,
                    carry,
                    min_position,
                    on_token,
                    committed_callback_stop,
                )?;
                return Ok(generated);
            }
            let draft = Some(
                draft_logits
                    .as_ref()
                    .expect("initial MTP logits")
                    .argmax_id()?,
            );
            #[cfg(any(test, feature = "diagnostics"))]
            self.capture_glm_mtp_state_for_test()?;
            (
                carry,
                draft_parent,
                draft,
                draft_logits,
                sampling_mode,
                min_position,
            )
        };

        // Yield only at the same committed boundary as the uninterrupted
        // generator. Neither the trunk nor the private MTP tape is replayed,
        // and the already drawn carry consumes no additional RNG on resume.
        macro_rules! yield_cycle {
            () => {
                if single_cycle {
                    self.glm_mtp_pending = Some(TargetMtpPending {
                        fed_len: self.fed.len(),
                        carry,
                        draft_parent,
                        draft,
                        draft_logits,
                        sampling_mode,
                        min_position,
                    });
                    return Ok(generated);
                }
            };
        }
        if !resuming {
            yield_cycle!();
        }

        while generated.len() < max_tokens {
            let remaining = max_tokens - generated.len();
            if is_eos(carry) {
                generated.push(carry);
                self.finish_target_mtp_tail(
                    tower,
                    carry,
                    min_position,
                    on_token,
                    committed_callback_stop,
                )?;
                return Ok(generated);
            }
            if remaining < 2 {
                generated.push(carry);
                self.finish_target_mtp_tail(
                    tower,
                    carry,
                    min_position,
                    on_token,
                    committed_callback_stop,
                )?;
                return Ok(generated);
            }

            // Match DS4's plain cycle when the sampled parent changed or
            // only one context row remains for this request.
            if draft.is_none()
                || carry != draft_parent
                || self.fed.len().saturating_add(2) > tower.max_position_embeddings()
            {
                let position = self.fed.len();
                anyhow::ensure!(
                    position < tower.max_position_embeddings(),
                    "context overflow before GLM MTP fallback step"
                );
                let ids = tower.runtime().tokens(&[carry as i32], &[1i32, 1])?;
                let (mixed, multi) = {
                    let _trace = tower.runtime().trace("glm53.mtp.opportunistic_fallback");
                    tower.forward_capture(&ids, Some(&mut self.caches), false)?
                };
                let logits = tower.head(&mixed)?;
                self.fed.push(carry);
                self.last_mixed = Some(mixed.select(1, 0)?.contiguous()?);
                draft_parent = logits.argmax_id()?;
                draft_logits = Some(tower.target_mtp_step(
                    &mut self.glm_mtp_cache,
                    &multi,
                    draft_parent,
                    position,
                    min_position,
                )?);
                draft = Some(
                    draft_logits
                        .as_ref()
                        .expect("fallback MTP logits")
                        .argmax_id()?,
                );
                generated.push(carry);
                sampling_mode = match on_token(carry) {
                    Ok(mode) => mode,
                    Err(error) => {
                        *committed_callback_stop = true;
                        return Err(error);
                    }
                };
                if is_eos(carry) || generated.len() >= max_tokens {
                    return Ok(generated);
                }
                #[cfg(any(test, feature = "diagnostics"))]
                self.capture_glm_mtp_state_for_test()?;
                carry = draw_target_mtp_token(
                    tower.runtime(),
                    stochastic.as_deref_mut(),
                    &logits,
                    &self.fed,
                    sampling_mode,
                )?;
                if is_eos(carry) || generated.len() + 1 >= max_tokens {
                    if generated.len() < max_tokens {
                        generated.push(carry);
                        self.finish_target_mtp_tail(
                            tower,
                            carry,
                            min_position,
                            on_token,
                            committed_callback_stop,
                        )?;
                    }
                    return Ok(generated);
                }
                yield_cycle!();
                continue;
            }

            let draft_token = draft.ok_or_else(|| {
                anyhow::anyhow!("GLM target-MTP proposal missing after parent match")
            })?;
            let verify_pos = self.fed.len();
            anyhow::ensure!(
                verify_pos.saturating_add(2) <= tower.max_position_embeddings(),
                "context overflow before GLM MTP two-token verify"
            );
            let before: Vec<_> = self.caches.iter().map(R::snapshot_rollback).collect();
            let verify_ids = tower
                .runtime()
                .tokens(&[carry as i32, draft_token as i32], &[1i32, 2])?;
            let verify_result = (|| -> anyhow::Result<_> {
                let _trace = tower.runtime().trace("glm53.mtp.verify_forward");
                let (verify_mixed, verify_multi) =
                    tower.forward_capture(&verify_ids, Some(&mut self.caches), false)?;
                #[cfg(any(test, feature = "diagnostics"))]
                let verify_capture = if let Some(prefix) =
                    std::env::var_os("LISA_GLM53_MTP_VERIFY_HC_CAPTURE_PREFIX")
                {
                    let capture_index = capture_glm_mtp_verify_hyper_states(
                        &prefix,
                        &verify_multi,
                        verify_pos + 1,
                    )?;
                    Some((prefix, capture_index))
                } else {
                    None
                };
                let (row0_logits, row1_logits) = {
                    let _trace = tower.runtime().trace("glm53.mtp.verify_head");
                    let row0_mixed = verify_mixed.select(1, 0)?.expand_dims(1)?.contiguous()?;
                    let row1_mixed = verify_mixed.select(1, 1)?.expand_dims(1)?.contiguous()?;
                    let row0_logits = tower.head(&row0_mixed)?;
                    row0_logits.eval()?;
                    #[cfg(any(test, feature = "diagnostics"))]
                    if let Some((prefix, capture_index)) = verify_capture.as_ref() {
                        capture_glm_mtp_verify_logits(
                            prefix,
                            *capture_index,
                            &row0_logits,
                            verify_pos + 1,
                        )?;
                    }
                    #[cfg(any(test, feature = "diagnostics"))]
                    self.record_glm_mtp_target_logits_for_test(verify_pos + 1, &row0_logits)?;
                    let row1_logits = tower.head(&row1_mixed)?;
                    row1_logits.eval()?;
                    #[cfg(any(test, feature = "diagnostics"))]
                    if let Some((prefix, capture_index)) = verify_capture.as_ref() {
                        capture_glm_mtp_verify_logits(
                            prefix,
                            *capture_index,
                            &row1_logits,
                            verify_pos + 2,
                        )?;
                    }
                    #[cfg(any(test, feature = "diagnostics"))]
                    self.record_glm_mtp_target_logits_for_test(verify_pos + 2, &row1_logits)?;
                    (
                        row0_logits.select(1, 0)?.contiguous()?,
                        row1_logits.select(1, 0)?.contiguous()?,
                    )
                };
                let row0_multi = verify_multi.select(1, 0)?.expand_dims(1)?.contiguous()?;
                let row1_multi = verify_multi.select(1, 1)?.expand_dims(1)?.contiguous()?;
                let row1_mixed = verify_mixed.select(1, 1)?.contiguous()?;
                Ok((row0_logits, row1_logits, row0_multi, row1_multi, row1_mixed))
            })();
            let (row0_logits, row1_logits, row0_multi, row1_multi, row1_mixed) = match verify_result
            {
                Ok(result) => result,
                Err(error) => {
                    if let Err(rollback_error) = self.restore_target_verify_boundary(&before) {
                        return Err(error.context(format!(
                                "GLM target-MTP verify failed and cache rollback also failed: {rollback_error:#}"
                            )));
                    }
                    return Err(error);
                }
            };

            // DS4 finishes the cycle's decision before its caller emits tokens
            // or detects a parser-mode transition. In particular, keep every
            // acceptance/residual RNG draw even when the callback discards row 1.
            let exact_sampling =
                target_mtp_uses_exact_sampling(stochastic.as_deref(), sampling_mode);
            let (accepted, actual) = match stochastic.as_deref_mut() {
                Some(s) if exact_sampling => {
                    let row0_history =
                        sampling_history_with_suffix(&self.fed, s.penalty_window(), &[carry]);
                    s.verify_draft_with_history(
                        tower.runtime(),
                        &row0_logits,
                        draft_token,
                        &row0_history,
                    )
                    .ok_or_else(|| anyhow::anyhow!("GLM MTP verification sampling failed"))?
                }
                Some(_) | None => {
                    let target = row0_logits.argmax_id()?;
                    (target == draft_token, target)
                }
            };
            generated.push(carry);
            let next_sampling_mode = match on_token(carry) {
                Ok(mode) => mode,
                Err(error) => {
                    let _ = self.rewind_target_mtp_cycle(
                        tower,
                        &before,
                        carry,
                        draft_token,
                        accepted,
                        exact_sampling.then_some(actual),
                        &row0_multi,
                        &row1_multi,
                        &row0_logits,
                        &row1_logits,
                        verify_pos,
                        min_position,
                    )?;
                    *committed_callback_stop = true;
                    return Err(error);
                }
            };
            let exact_mode_boundary = next_sampling_mode != sampling_mode
                && (accepted || exact_sampling)
                && stochastic
                    .as_deref()
                    .is_some_and(|sampler| sampler.exact_sampling() && sampler.temperature() > 0.0);
            sampling_mode = next_sampling_mode;

            if exact_mode_boundary {
                // The native cycle has already consumed its acceptance RNG.
                // Preserve its private writes, but discard its pending proposal
                // and sample from the retained row's freshly replayed logits.
                let replay_logits = self.rewind_target_mtp_cycle(
                    tower,
                    &before,
                    carry,
                    draft_token,
                    accepted,
                    exact_sampling.then_some(actual),
                    &row0_multi,
                    &row1_multi,
                    &row0_logits,
                    &row1_logits,
                    verify_pos,
                    min_position,
                )?;
                draft_parent = replay_logits.argmax_id()?;
                draft = None;
                draft_logits = None;
                if generated.len() >= max_tokens {
                    return Ok(generated);
                }
                carry = draw_target_mtp_token(
                    tower.runtime(),
                    stochastic.as_deref_mut(),
                    &replay_logits,
                    &self.fed,
                    sampling_mode,
                )?;
                if is_eos(carry) || generated.len() + 1 >= max_tokens {
                    if generated.len() < max_tokens {
                        generated.push(carry);
                        self.finish_target_mtp_tail(
                            tower,
                            carry,
                            min_position,
                            on_token,
                            committed_callback_stop,
                        )?;
                    }
                    return Ok(generated);
                }
                yield_cycle!();
                continue;
            }

            // Keep the draft/target pair in the Chrome trace so an MTP
            // acceptance-rate mismatch can be compared against DS4 without
            // adding a per-decode logging flag or synchronizing the GPU.
            let decision = if accepted {
                "glm53.mtp.accept"
            } else {
                "glm53.mtp.reject"
            };
            let decision_detail = ((draft_token as u64) << 32) | actual as u64;
            let _decision_trace = tower.runtime().trace_detail(decision, decision_detail);
            if !accepted {
                // Restore all recurrent/indexer state to the pre-verify
                // boundary, then commit only the first target row (ds4 replays
                // this token after restoring its GLM 5.3 speculative snapshot).
                let replay_multi = self.restore_target_verify_to_first(tower, &before, carry)?;
                if exact_sampling {
                    // DS4 exact rejection commits the residual replacement
                    // through the target trunk, then seeds MTP from that
                    // replacement row and the following target argmax.
                    generated.push(actual);
                    if is_eos(actual)
                        && self.target_mtp_eos_policy == TargetMtpEosPolicy::StopBeforeVerified
                    {
                        // Native HTTP discards a replacement EOS after the cycle.
                        // The retained carry has already been replayed, and EOS
                        // contributes no private-cache priming to preserve.
                        self.glm_mtp_pending = None;
                        self.finish_target_mtp_tail(
                            tower,
                            actual,
                            min_position,
                            on_token,
                            committed_callback_stop,
                        )?;
                        #[cfg(any(test, feature = "diagnostics"))]
                        self.capture_glm_mtp_state_for_test()?;
                        return Ok(generated);
                    }
                    sampling_mode = match on_token(actual) {
                        Ok(mode) => mode,
                        Err(error) => {
                            self.commit_target_mtp_tail(
                                tower,
                                actual,
                                verify_pos + 1,
                                min_position,
                                !is_eos(actual),
                            )?;
                            *committed_callback_stop = true;
                            return Err(error);
                        }
                    };
                    if is_eos(actual) {
                        self.commit_target_mtp_tail(
                            tower,
                            actual,
                            verify_pos + 1,
                            min_position,
                            false,
                        )?;
                        return Ok(generated);
                    }
                    let replacement_position = verify_pos + 1;
                    if generated.len() >= max_tokens {
                        self.commit_target_mtp_tail(
                            tower,
                            actual,
                            replacement_position,
                            min_position,
                            true,
                        )?;
                        return Ok(generated);
                    }
                    anyhow::ensure!(
                        replacement_position < tower.max_position_embeddings(),
                        "context overflow before GLM MTP replacement replay"
                    );
                    let replacement_ids = tower.runtime().tokens(&[actual as i32], &[1i32, 1])?;
                    let (replacement_mixed, replacement_multi) = {
                        let _trace = tower.runtime().trace("glm53.mtp.replacement_forward");
                        tower.forward_capture(&replacement_ids, Some(&mut self.caches), false)?
                    };
                    let replacement_logits = tower.head(&replacement_mixed)?;
                    self.fed.push(actual);
                    #[cfg(any(test, feature = "diagnostics"))]
                    self.record_glm_mtp_target_logits_for_test(
                        self.fed.len(),
                        &replacement_logits,
                    )?;
                    self.last_mixed = Some(replacement_mixed.select(1, 0)?.contiguous()?);
                    draft_parent = replacement_logits.argmax_id()?;
                    draft_logits = Some(tower.target_mtp_step(
                        &mut self.glm_mtp_cache,
                        &replacement_multi,
                        draft_parent,
                        replacement_position,
                        min_position,
                    )?);
                    draft = Some(
                        draft_logits
                            .as_ref()
                            .expect("replacement MTP logits")
                            .argmax_id()?,
                    );
                    #[cfg(any(test, feature = "diagnostics"))]
                    self.capture_glm_mtp_state_for_test()?;
                    carry = draw_target_mtp_token(
                        tower.runtime(),
                        stochastic.as_deref_mut(),
                        &replacement_logits,
                        &self.fed,
                        sampling_mode,
                    )?;
                    if is_eos(carry) || generated.len() + 1 >= max_tokens {
                        if generated.len() < max_tokens {
                            generated.push(carry);
                            self.finish_target_mtp_tail(
                                tower,
                                carry,
                                min_position,
                                on_token,
                                committed_callback_stop,
                            )?;
                        }
                        return Ok(generated);
                    }
                } else {
                    // DS4 restores the pre-verify state, replays the committed
                    // carry token, and recomputes target logits from that
                    // replayed hidden state. Reusing the speculative row-0
                    // logits can preserve argmax while changing the full
                    // post-rejection distribution.
                    let replay_logits =
                        tower.head(self.last_mixed.as_ref().ok_or_else(|| {
                            anyhow::anyhow!("GLM MTP rejection replay lost its target hidden state")
                        })?)?;
                    #[cfg(any(test, feature = "diagnostics"))]
                    self.record_glm_mtp_target_logits_for_test(self.fed.len(), &replay_logits)?;
                    // Native GLM seeds from the verifier's row-0 argmax even
                    // when replay changes that row's distribution. The next
                    // sampled carry still comes from the replayed logits.
                    draft_parent = actual;
                    draft_logits = Some(tower.target_mtp_step(
                        &mut self.glm_mtp_cache,
                        &replay_multi,
                        draft_parent,
                        verify_pos,
                        min_position,
                    )?);
                    draft = Some(
                        draft_logits
                            .as_ref()
                            .expect("replay MTP logits")
                            .argmax_id()?,
                    );
                    #[cfg(any(test, feature = "diagnostics"))]
                    self.capture_glm_mtp_state_for_test()?;
                    carry = draw_target_mtp_token(
                        tower.runtime(),
                        stochastic.as_deref_mut(),
                        &replay_logits,
                        &self.fed,
                        sampling_mode,
                    )?;
                }
                if is_eos(carry) || generated.len() + 1 >= max_tokens {
                    if generated.len() < max_tokens {
                        generated.push(carry);
                        self.finish_target_mtp_tail(
                            tower,
                            carry,
                            min_position,
                            on_token,
                            committed_callback_stop,
                        )?;
                    }
                    return Ok(generated);
                }
                yield_cycle!();
                continue;
            }

            // The target accepted the point-mass MTP proposal. DS4 commits both
            // verified target rows even when EOS or the output budget ends the
            // cycle; retain the same trunk and private-cache boundary here.
            generated.push(draft_token);
            if is_eos(draft_token)
                && self.target_mtp_eos_policy == TargetMtpEosPolicy::StopBeforeVerified
            {
                // The native cycle already completed both private-cache writes.
                // HTTP then rewinds the target checkpoint to before the EOS.
                let _ = self.rewind_target_mtp_cycle(
                    tower,
                    &before,
                    carry,
                    draft_token,
                    true,
                    None,
                    &row0_multi,
                    &row1_multi,
                    &row0_logits,
                    &row1_logits,
                    verify_pos,
                    min_position,
                )?;
                self.glm_mtp_pending = None;
                self.finish_target_mtp_tail(
                    tower,
                    draft_token,
                    min_position,
                    on_token,
                    committed_callback_stop,
                )?;
                #[cfg(any(test, feature = "diagnostics"))]
                self.capture_glm_mtp_state_for_test()?;
                return Ok(generated);
            }
            sampling_mode = match on_token(draft_token) {
                Ok(mode) => mode,
                Err(error) => {
                    let next = row1_logits.argmax_id()?;
                    let _ = self.advance_target_mtp_cache(
                        tower,
                        &row0_multi,
                        &row1_multi,
                        draft_token,
                        next,
                        verify_pos,
                        min_position,
                    )?;
                    self.fed.push(carry);
                    self.fed.push(draft_token);
                    self.last_mixed = Some(row1_mixed);
                    *committed_callback_stop = true;
                    return Err(error);
                }
            };
            if is_eos(draft_token) || generated.len() >= max_tokens {
                // DS4 advances its private MTP compact cache through both
                // accepted verify rows before returning, even when the token
                // budget or EOS prevents another proposal cycle.
                let next = row1_logits.argmax_id()?;
                let _ = self.advance_target_mtp_cache(
                    tower,
                    &row0_multi,
                    &row1_multi,
                    draft_token,
                    next,
                    verify_pos,
                    min_position,
                )?;
                self.fed.push(carry);
                self.fed.push(draft_token);
                self.last_mixed = Some(row1_mixed);
                #[cfg(any(test, feature = "diagnostics"))]
                self.capture_glm_mtp_state_for_test()?;
                return Ok(generated);
            }

            self.fed.push(carry);
            self.fed.push(draft_token);
            self.last_mixed = Some(row1_mixed);
            let remaining = max_tokens - generated.len();
            let next_parent = row1_logits.argmax_id()?;
            let next = if target_mtp_draw_reuses_argmax(stochastic.as_deref(), sampling_mode) {
                next_parent
            } else {
                draw_target_mtp_token(
                    tower.runtime(),
                    stochastic.as_deref_mut(),
                    &row1_logits,
                    &self.fed,
                    sampling_mode,
                )?
            };
            // Bring the private MTP cache through the accepted first target
            // row, then seed the next proposal from target row one.
            draft_logits = Some(self.advance_target_mtp_cache(
                tower,
                &row0_multi,
                &row1_multi,
                draft_token,
                next_parent,
                verify_pos,
                min_position,
            )?);
            draft_parent = next_parent;
            #[cfg(any(test, feature = "diagnostics"))]
            self.capture_glm_mtp_state_for_test()?;
            if is_eos(next) || remaining == 1 {
                generated.push(next);
                self.finish_target_mtp_tail(
                    tower,
                    next,
                    min_position,
                    on_token,
                    committed_callback_stop,
                )?;
                return Ok(generated);
            }
            carry = next;
            draft = Some(
                draft_logits
                    .as_ref()
                    .expect("accepted MTP logits")
                    .argmax_id()?,
            );
            yield_cycle!();
        }
        Ok(generated)
    }

    /// Speculative generation. Independent Qwen MTP uses the requested chain
    /// depth; GLM uses its target-conditioned one-step cycle.
    pub fn generate_mtp(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        tokens: &[u32],
        max_tokens: usize,
        depth: usize,
        on_token: &mut dyn FnMut(u32) -> anyhow::Result<()>,
    ) -> anyhow::Result<Vec<u32>> {
        self.generate_mtp_depth(
            tower,
            tokens,
            max_tokens,
            MtpDepth::Fixed(depth),
            None,
            None,
            on_token,
        )
    }

    /// [`Session::generate_mtp`] with the depth selection policy:
    /// [`MtpDepth::Fixed`] is the phase-1 opt-in (behavior unchanged);
    /// [`MtpDepth::Auto`] runs the EV controller (round_cost.rs): per-index
    /// acceptance EMAs + the measured round-cost table, depth re-picked every
    /// round, serial fallback (depth 0) when no speculative depth beats the
    /// serial step rate.
    pub fn generate_mtp_depth(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        tokens: &[u32],
        max_tokens: usize,
        depth_sel: MtpDepth,
        after_prefill: Option<&mut dyn FnMut(&[R::Cache])>,
        // Non-greedy verification. `None` keeps the certified greedy path.
        // `Some(sm)` runs the same Leviathan-exact scheme as `pld_decode`: the
        // MTP proposal is a POINT MASS at the head's argmax (`q = delta`), so a
        // draft token is accepted with `p_target[draft]` and the first rejection
        // emits a draw from the residual. Serial steps sample too.
        stochastic: Option<&mut R::Sampler>,
        on_token: &mut dyn FnMut(u32) -> anyhow::Result<()>,
    ) -> anyhow::Result<Vec<u32>> {
        let mut on_token_with_mode = |token| {
            on_token(token)?;
            Ok(TargetMtpSamplingMode::Configured)
        };
        self.generate_mtp_depth_with_sampling_mode(
            tower,
            tokens,
            max_tokens,
            depth_sel,
            after_prefill,
            stochastic,
            &mut on_token_with_mode,
        )
    }

    fn generate_block_drafter(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        tokens: &[u32],
        max_tokens: usize,
        depth_sel: MtpDepth,
        mut after_prefill: Option<&mut dyn FnMut(&[R::Cache])>,
        mut stochastic: Option<&mut R::Sampler>,
        on_token: &mut dyn FnMut(u32) -> anyhow::Result<TargetMtpSamplingMode>,
    ) -> anyhow::Result<Vec<u32>> {
        self.ensure_cache_numerical_law(tower)?;
        if max_tokens == 0 {
            return Ok(Vec::new());
        }
        anyhow::ensure!(
            tower.has_block_drafter(),
            "the checkpoint has no block drafter"
        );
        let cap = tower.max_speculative_depth();
        let depth = match depth_sel {
            MtpDepth::Fixed(depth) => depth,
            MtpDepth::Auto => tower.block_drafter_verify_depth(),
        };
        anyhow::ensure!(
            (1..=cap).contains(&depth),
            "DSpine depth must be between 1 and {cap}, got {depth}"
        );
        let prompt_end = self
            .fed
            .len()
            .checked_add(tokens.len())
            .ok_or_else(|| anyhow::anyhow!("context position overflow before DSpine"))?;
        anyhow::ensure!(
            prompt_end <= tower.max_position_embeddings(),
            "context overflow before DSpine: position {prompt_end}, limit {}",
            tower.max_position_embeddings()
        );
        let max_tokens = max_tokens.min(
            tower
                .max_position_embeddings()
                .saturating_sub(prompt_end)
                .saturating_add(1),
        );
        if max_tokens == 0 {
            return Ok(Vec::new());
        }
        if max_tokens == 1 {
            let result = (|| -> anyhow::Result<Vec<u32>> {
                let logits = self.feed(tower, tokens)?;
                if let Some(cb) = after_prefill.as_deref_mut() {
                    cb(&self.caches);
                }
                let token = draw_block_target_token(
                    tower.runtime(),
                    stochastic.as_deref_mut(),
                    &logits,
                    &self.fed,
                    TargetMtpSamplingMode::Configured,
                )?;
                on_token(token)?;
                Ok(vec![token])
            })();
            if result.is_err() {
                self.invalidate();
            }
            return result;
        }

        let result = (|| -> anyhow::Result<Vec<u32>> {
            let prior_fed = self.fed.len();
            let drafter_matches = tower.block_drafter_context_matches(&self.fed);
            let can_append = drafter_matches || prior_fed == 0;
            if can_append {
                if !drafter_matches {
                    tower.drafter_reset();
                }
                tower.set_block_drafter_capture(true);
                tower.set_block_drafter_prefill_append(true);
            } else {
                tower.set_block_drafter_capture(false);
                tower.set_block_drafter_prefill_append(false);
            }
            let (logits, _) = self.feed_multi(tower, tokens)?;
            tower.set_block_drafter_prefill_append(false);
            if !can_append {
                tower.rebuild_block_drafter_context(&self.fed)?;
                tower.set_block_drafter_capture(true);
            }
            if let Some(cb) = after_prefill.as_deref_mut() {
                cb(&self.caches);
            }
            let history = self.fed.clone();
            let mut sampling_mode = TargetMtpSamplingMode::Configured;
            let first = draw_block_target_token(
                tower.runtime(),
                stochastic.as_deref_mut(),
                &logits,
                &history,
                sampling_mode,
            )?;
            let mut generated = vec![first];
            sampling_mode = on_token(first)?;
            if is_eos(first) || max_tokens == 1 {
                return Ok(generated);
            }

            let mut carry = first;
            let mut block_policy =
                crate::core::block_policy::BlockPolicy::new(matches!(depth_sel, MtpDepth::Auto));
            let mut tail: Vec<i64> = self.fed.iter().map(|&token| token as i64).collect();
            let ctx_len = tower.context_window();
            if ctx_len > 0 && tail.len() > ctx_len {
                tail.drain(0..tail.len() - ctx_len);
            }
            while generated.len() < max_tokens && !is_eos(carry) {
                let chosen_depth = block_policy.depth(depth);
                let width = chosen_depth.min(max_tokens - generated.len() - 1);
                let round_start = std::time::Instant::now();
                let draft_span = tower.runtime().trace("dspine.propose");
                let proposal_ids = if width == 0 {
                    Vec::new()
                } else {
                    let proposals = tower.draft_block(carry)?;
                    anyhow::ensure!(
                        proposals.rank() == 2
                            && proposals.dim(0) == 1
                            && proposals.dim(1) >= width as i32,
                        "DSpine returned an invalid block shape {:?}",
                        proposals.shape()
                    );
                    proposals
                        .slice(1, 0, Some(width as i32))?
                        .contiguous()?
                        .read_raw_ids()?
                };
                drop(draft_span);
                let verify_span = tower.runtime().trace("dspine.verify");
                let mut verify_ids = Vec::with_capacity(width + 1);
                verify_ids.push(carry as i32);
                verify_ids.extend(proposal_ids.iter().map(|&token| token as i32));
                let verify = tower
                    .runtime()
                    .tokens(&verify_ids, &[1, (width + 1) as i32])?;
                tower.set_context_tails(vec![tail.clone()]);
                let (verified, _) =
                    tower.forward_capture(&verify, Some(&mut self.caches), chosen_depth != 0)?;
                let target_hidden = tower
                    .take_drafter_hidden()
                    .ok_or_else(|| anyhow::anyhow!("Qwen verify did not capture DSpine taps"))?;
                let logits = tower.head(&verified)?;
                // Greedy Qwen draws do not depend on history or callbacks. Read
                // every row's ordinary argmax in one completion; callbacks
                // still run only for the committed prefix. A stochastic
                // sampler keeps the per-row path, including mode transitions.
                let greedy_tokens = if stochastic.as_deref().is_none_or(R::Sampler::is_greedy) {
                    Some(logits.argmax_axis(-1)?.read_raw_ids()?)
                } else {
                    None
                };
                let mut row_history = self.fed.clone();
                row_history.push(carry);
                let mut main_tokens = Vec::with_capacity(width + 1);
                let mut emitted = 0usize;
                let mut eos_seen = false;
                for index in 0..=width {
                    // Draw exactly as serial Qwen does at this position. A
                    // deterministic proposal is accepted only when it matches
                    // that target draw; rejected suffixes consume no samples.
                    let token = if let Some(tokens) = &greedy_tokens {
                        tokens[index]
                    } else {
                        let row = logits.select(1, index as i32)?.contiguous()?;
                        draw_block_target_token(
                            tower.runtime(),
                            stochastic.as_deref_mut(),
                            &row,
                            &row_history,
                            sampling_mode,
                        )?
                    };
                    row_history.push(token);
                    main_tokens.push(token);
                    generated.push(token);
                    emitted += 1;
                    sampling_mode = on_token(token)?;
                    if is_eos(token) {
                        eos_seen = true;
                        break;
                    }
                    if generated.len() >= max_tokens {
                        break;
                    }
                    if index < width && token != proposal_ids[index] {
                        break;
                    }
                }
                drop(verify_span);
                let _commit_span = tower.runtime().trace("dspine.commit");
                if tower.runtime().trace_enabled("dspine.round") {
                    eprintln!(
                        "[dspine.round] start={} width={width} emitted={emitted} capture={} proposals={proposal_ids:?} target={main_tokens:?}",
                        generated.len() - emitted,
                        chosen_depth != 0,
                    );
                }
                let advance = width + 1;
                let keep = emitted;
                for cache in &mut self.caches {
                    tower
                        .runtime()
                        .rollback(cache, advance, keep, chosen_depth != 0)?;
                }
                let committed_hidden =
                    target_hidden.slice(1, 0, Some(keep as i32))?.contiguous()?;
                let committed_inputs = &verify_ids[..keep];
                let committed_tokens: Vec<u32> =
                    committed_inputs.iter().map(|&token| token as u32).collect();
                tower.append_drafter_context(&committed_hidden, &committed_tokens)?;
                self.fed
                    .extend(committed_inputs.iter().map(|&token| token as u32));
                self.last_mixed = Some(verified.select(1, keep as i32 - 1)?.contiguous()?);
                carry = *main_tokens.last().expect("DSpine emitted a verified token");
                tail.extend(committed_inputs.iter().map(|&token| token as i64));
                if ctx_len > 0 && tail.len() > ctx_len {
                    tail.drain(0..tail.len() - ctx_len);
                }
                if width == chosen_depth && !eos_seen {
                    block_policy.observe(chosen_depth, emitted, round_start.elapsed());
                }
                if eos_seen || generated.len() >= max_tokens {
                    break;
                }
            }
            Ok(generated)
        })();
        tower.set_block_drafter_prefill_append(false);
        tower.set_block_drafter_capture(false);
        if result.is_err() {
            self.invalidate();
        }
        result
    }

    fn generate_block_drafter_serial(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        tokens: &[u32],
        max_tokens: usize,
        mut after_prefill: Option<&mut dyn FnMut(&[R::Cache])>,
        mut stochastic: Option<&mut R::Sampler>,
        on_token: &mut dyn FnMut(u32) -> anyhow::Result<TargetMtpSamplingMode>,
    ) -> anyhow::Result<Vec<u32>> {
        self.ensure_cache_numerical_law(tower)?;
        if max_tokens == 0 {
            return Ok(Vec::new());
        }
        let prompt_end = self
            .fed
            .len()
            .checked_add(tokens.len())
            .ok_or_else(|| anyhow::anyhow!("context position overflow before serial decode"))?;
        anyhow::ensure!(
            prompt_end <= tower.max_position_embeddings(),
            "context overflow before serial decode: position {prompt_end}, limit {}",
            tower.max_position_embeddings()
        );
        let max_tokens = max_tokens.min(
            tower
                .max_position_embeddings()
                .saturating_sub(prompt_end)
                .saturating_add(1),
        );
        let result = (|| -> anyhow::Result<Vec<u32>> {
            let mut logits = self.feed(tower, tokens)?;
            if let Some(cb) = after_prefill.as_deref_mut() {
                cb(&self.caches);
            }
            let mut mode = TargetMtpSamplingMode::Configured;
            let mut token = draw_block_target_token(
                tower.runtime(),
                stochastic.as_deref_mut(),
                &logits,
                &self.fed,
                mode,
            )?;
            let mut output = vec![token];
            mode = on_token(token)?;
            while output.len() < max_tokens && !is_eos(token) {
                logits = self.decode_token(tower, token)?;
                token = draw_block_target_token(
                    tower.runtime(),
                    stochastic.as_deref_mut(),
                    &logits,
                    &self.fed,
                    mode,
                )?;
                output.push(token);
                mode = on_token(token)?;
            }
            Ok(output)
        })();
        if result.is_err() {
            self.invalidate();
        }
        result
    }

    /// [`Session::generate_mtp_depth`] with a live policy for the next target
    /// sample. The callback returns the mode derived from the just-emitted
    /// token; GLM MTP uses it at exact-sampling boundaries.
    pub fn generate_mtp_depth_with_sampling_mode(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        tokens: &[u32],
        max_tokens: usize,
        depth_sel: MtpDepth,
        mut after_prefill: Option<&mut dyn FnMut(&[R::Cache])>,
        mut stochastic: Option<&mut R::Sampler>,
        on_token: &mut dyn FnMut(u32) -> anyhow::Result<TargetMtpSamplingMode>,
    ) -> anyhow::Result<Vec<u32>> {
        if tower.has_target_mtp() {
            return self.generate_target_mtp(
                tower,
                tokens,
                max_tokens,
                after_prefill,
                stochastic,
                on_token,
            );
        }
        if tower.has_block_drafter() {
            if matches!(depth_sel, MtpDepth::Fixed(0)) {
                return self.generate_block_drafter_serial(
                    tower,
                    tokens,
                    max_tokens,
                    after_prefill,
                    stochastic,
                    on_token,
                );
            }
            return self.generate_block_drafter(
                tower,
                tokens,
                max_tokens,
                depth_sel,
                after_prefill,
                stochastic,
                on_token,
            );
        }
        let mut on_token_plain = |token| on_token(token).map(|_| ());
        use crate::core::round_cost::{
            DepthController, ModelCost, PROBE_TICKS, PROBE_WARM, RoundCostTable, probe_warm_stats,
        };

        anyhow::ensure!(tower.has_drafter(), "the checkpoint has no MTP head");
        let model_depth_cap = tower.max_speculative_depth();
        anyhow::ensure!(
            model_depth_cap >= 2,
            "the checkpoint does not support MTP verification"
        );
        let is_auto = matches!(depth_sel, MtpDepth::Auto);
        let mut depth: usize = match depth_sel {
            // Chain depth follows the copy cap (`LISA_COPY_LEN_MAX`): a fixed
            // depth past 6 runs chain-only wide rounds (S = depth + 1 > 7, the
            // tiled verify lane — specs/08's wide-lane test). Default unchanged.
            MtpDepth::Fixed(d) => d.clamp(2, crate::core::copy_draft::copy_len_max()),
            MtpDepth::Auto => 2, // re-picked by the controller every round
        };
        anyhow::ensure!(
            depth <= model_depth_cap,
            "this checkpoint supports at most {model_depth_cap} MTP draft tokens"
        );

        // 1. Prefill the turn's suffix (incremental) and take the first token.
        let (logits, multi) = self.feed_multi(tower, tokens)?;
        // Clean commit boundary (suffix fed, nothing decoded yet): the
        // cross-request prefix cache snapshots here — the same contract as
        // `generate`. Without this hook the cache was NEVER populated on the
        // MTP path, so with speculation on (the serve default) no request could
        // ever resume, and llmprobe reported the cache as "not detected".
        if let Some(cb) = after_prefill.as_deref_mut() {
            cb(&self.caches);
        }
        let first = logits.argmax_id()?;
        let t_s = tokens.len() as i32;

        // 2. Prime the head with the committed history. When the head's cache
        // still matches the history it primed last turn, only the new pairs are
        // fed (the MTP head is a single layer whose K/V is a pure function of
        // the committed token/multi sequence); otherwise it is reset and fully
        // re-primed.
        let head_off = if tower.drafter_offset() == self.mtp_head_len {
            self.mtp_head_len
        } else {
            tower.drafter_reset();
            0
        };
        let mut history: Vec<u32> = self.mtp_tokens.clone();
        history.extend_from_slice(tokens);
        let mut all_multi: Vec<R::Tensor> = self.mtp_multi.clone();
        all_multi.push(multi.slice(1, 0, Some(t_s))?.contiguous()?);
        let init_multis = concat_chunks(&all_multi)?;
        let mut init_tokens: Vec<i32> = history[1..].iter().map(|&t| t as i32).collect();
        init_tokens.push(first as i32);
        // The head already holds `head_off` pairs; feed the rest.
        anyhow::ensure!(
            head_off < init_tokens.len(),
            "MTP head offset {head_off} exceeds history pairs {}",
            init_tokens.len()
        );
        let init_multis = init_multis.slice(1, head_off as i32, None)?.contiguous()?;
        let init_tokens: Vec<i32> = init_tokens[head_off..].to_vec();

        // Prime the head in windows. A single whole-history forward sits at
        // `offset == 0`, where the head's attention takes the DENSE fallback and
        // materialises `[S, kv]` masks — at 100K tokens that is tens of GB and
        // fails the buffer allocation. Windows keep the head's cache advancing,
        // so from the second window on the block-sparse QSA path serves it.
        const PRIME_CHUNK: usize = 2048;
        let _trace_prime = tower
            .runtime()
            .trace_detail("mtp.prime", init_tokens.len() as u64);
        let mut primed: Option<(R::Tensor, R::Tensor)> = None;
        {
            let toks = &init_tokens;
            let mul = &init_multis;
            let mut i = 0usize;
            while i < toks.len() {
                let n = (toks.len() - i).min(PRIME_CHUNK);
                let t = tower.runtime().tokens(&toks[i..i + n], &[1i32, n as i32])?;
                let m = mul.slice(1, i as i32, Some((i + n) as i32))?.contiguous()?;
                primed = Some(tower.draft_step(&t, &m)?);
                i += n;
                tower.runtime().trim_memory();
            }
        }

        let mut generated: Vec<u32> = vec![first];
        on_token_plain(first)?;
        let mut store_tokens: Vec<u32> = vec![first];
        let mut store_multi: Vec<R::Tensor> = vec![multi.select(1, t_s - 1)?.contiguous()?];

        let mut backlog: Vec<(u32, R::Tensor)> = Vec::new();
        // Consecutive serial fallback ticks form a block; its first
        // PROBE_WARM ticks are transitions (MTP_ADAPTIVE_PROBE_WARM) and never
        // fold.
        let mut serial_block: u32 = 0;
        let mut carry_token: u32 = first;
        let mut carry_multi: R::Tensor = multi.select(1, t_s - 1)?.contiguous()?;
        // Per-index draft acceptance over the round loop (logged at the end).
        let mut rounds: usize = 0;
        let mut accept_hist: Vec<usize> = Vec::new();
        // Chain-only per-index acceptance: rounds whose proposals came from
        // the head chain (no copy hit, no widened width). Copy rounds accept
        // 1.0 at every index by construction and would swamp this table —
        // the chain-quality question needs them out.
        let mut chain_rounds: usize = 0;
        let mut chain_hist: Vec<usize> = Vec::new();
        let mut chain_w_max: usize = 0;
        // Engagement counters (output-equality tests cannot see a draft source
        // that silently never fires).
        let mut copy_rounds: usize = 0;
        let mut copy_tokens: usize = 0;
        let mut width_sum: usize = 0;
        // Two-chunk telemetry: rounds that considered extension (synced),
        // rounds where the tau gate fired, and the sum of sync wall ms.
        let mut tc_considered: usize = 0;
        let mut tc_fired: usize = 0;
        let mut tc_sync_ms_sum: f64 = 0.0;
        // Oracle (specs/08): when the CLI enabled it, keep every round's
        // chain-proposal ranking so the end of the run can say where the
        // target's true token sat on the rounds verify rejected. Pure
        // diagnostics — nothing here feeds the proposal or the commit.
        let oracle_on = crate::core::oracle::enabled();
        let mut oracle_samples: Vec<crate::core::oracle::RoundSample> = Vec::new();
        // Prompt-lookup (context-copy) state: the committed token sequence for
        // this turn and a k-gram index over it.
        let mut committed: Vec<u32> = history.clone();
        committed.push(first);
        let mut copy_index = CopyIndex::new(COPY_K);
        let ctx_len = tower.context_window();
        let mut tail: Vec<i64> = history[history.len().saturating_sub(ctx_len)..]
            .iter()
            .map(|&t| t as i64)
            .collect();

        // EV controller (auto only): the serial price reads every table value
        // through `bucket_to_read` (the
        // request's own bucket, else the nearest TRUSTED one) at
        // n >= TABLE_MIN_SAMPLES (specs/09 M1). Only when no bucket carries
        // a measured serial cell does an in-run probe run, and with the same
        // shape (specs/09 M2): PROBE_TICKS real serial steps
        // of the LIVE greedy stream, PROBE_WARM discarded as cold, trust by
        // warm-fold COUNT — an EOS-truncated probe leaves the serial price
        // untrusted instead of arming the gate on its cold samples.
        let mut auto: Option<DepthController> = None;
        if is_auto {
            let key = tower.mtp_cost_key();
            let table = RoundCostTable::load(&crate::core::round_cost::default_path())
                .and_then(|mut t| t.models.remove(&key));
            let prior_bucket = table
                .as_ref()
                .and_then(|t| t.bucket_to_read(self.fed.len()));
            let mut serial_ms = table
                .as_ref()
                .and_then(|t| t.serial_prior_ms(self.fed.len()))
                .unwrap_or(0.0);
            let mut serial_samples = if serial_ms > 0.0 {
                ModelCost::TABLE_MIN_SAMPLES
            } else {
                0
            };
            let source = if serial_ms > 0.0 {
                format!(
                    "table prior, bucket {}",
                    crate::core::round_cost::BUCKET_NAMES[prior_bucket.unwrap_or(0)]
                )
            } else {
                // Probe: real serial steps on the live stream. They are
                // ordinary committed tokens (greedy, lossless), so the output
                // is unaffected; the timing only prices the serial step.
                // NOTE: each tick feeds the CARRIED token (the previous
                // argmax) — feeding `first` every tick forked the greedy
                // stream and could hit a fork-only EOS before round 1 ever
                // ran (the specs/08 empty-picks death, specs/09).
                let mut probe: Vec<f64> = Vec::new();
                for _ in 0..PROBE_TICKS {
                    let t0 = std::time::Instant::now();
                    let (t, row) = serial_step(
                        tower,
                        &mut self.caches,
                        &tail,
                        carry_token,
                        stochastic.as_deref_mut(),
                        committed.len() as u64,
                    )?;
                    let dt = t0.elapsed().as_secs_f64() * 1e3;
                    probe.push(dt);
                    on_token_plain(t)?;
                    generated.push(t);
                    store_tokens.push(t);
                    store_multi.push(row.clone());
                    committed.push(t);
                    carry_token = t;
                    carry_multi = row;
                    tail.push(t as i64);
                    if tail.len() > ctx_len {
                        tail.drain(0..tail.len() - ctx_len);
                    }
                    if is_eos(t) || generated.len() >= max_tokens {
                        break;
                    }
                }
                let (val, warm) = probe_warm_stats(&probe);
                serial_ms = val;
                serial_samples = warm;
                if warm >= ModelCost::TABLE_MIN_SAMPLES {
                    format!("probe warm mean, n={warm} of {}", PROBE_TICKS)
                } else {
                    format!(
                        "probe UNTRUSTED, warm n={} of {} (cold/EOS-truncated)",
                        warm, PROBE_TICKS
                    )
                }
            };
            eprintln!(
                "[mtp.auto] controller on: table row {} {}, serial step {:.2} ms ({source})",
                key,
                if table.is_some() {
                    "found"
                } else {
                    "absent (in-run pricing)"
                },
                serial_ms
            );
            // P5: per-silicon cap row — the cold-start cap the EV picks may
            // never exceed. Logged once per process; a bare `depth=N` in the
            // round logs is an EV pick or `--depth`, never this row.
            let (cap_row, cap_log) = crate::core::round_cost::adaptive_cap_for_machine();
            eprintln!("[mtp.cap] {cap_log}");
            auto = Some(DepthController::new_with_serial(
                table,
                serial_ms,
                serial_samples,
                cap_row.min(model_depth_cap),
            ));
            // P3: machine-stamp the restored row — another machine's row
            // keeps its trust counts but re-folds its first live sample at
            // RESEED_WEIGHT 0.5 ("one machine's cliff is never served to
            // another"); a sweep row (machine empty) is portable.
            auto.as_mut().expect("just built").stamp_machine(
                &crate::core::round_cost::host_chip(),
                &key,
                std::env::consts::OS,
            );
            // P6: cross-request EV seed (default ON; a serving process with
            // many short requests re-warms instead of paying warmup again;
            // one request per process = the seed is usually absent, a no-op).
            let seed_path =
                crate::core::round_cost::default_path().with_file_name("mtp_ev_seed.json");
            if let Ok(data) = std::fs::read_to_string(&seed_path) {
                if let Ok(serde_json::Value::Object(m)) = serde_json::from_str(&data) {
                    if let Some(s) = m.get(&key) {
                        if let (Some(acc), Some(m_lo)) = (
                            s.get("ev_acc")
                                .and_then(|v| serde_json::from_value::<[f64; 6]>(v.clone()).ok()),
                            s.get("m_lo").and_then(|v| v.as_u64()),
                        ) {
                            auto.as_mut()
                                .expect("just built")
                                .seed_ev(acc, m_lo as usize);
                            eprintln!("[mtp.ev-seed] seeded from a previous request");
                        }
                    }
                }
            }
        }
        // P2 (port spec §2.3): the two-chunk extension. Default OFF; the env
        // kill-switch arms it for A/B. The controller plans (m_lo, m_hi, tau)
        // every round; the round drafts m_lo, syncs the chunk-A confidences
        // once, and extends to m_hi only when the chain log-confidence
        // clears tau. Collapsed rounds (m_hi == m_lo) never sync and are
        // byte-identical in shape to the fixed-depth round.
        let two_chunk_on =
            is_auto && model_depth_cap > 2 && crate::core::round_cost::two_chunk_enabled();
        if two_chunk_on {
            tower.set_conf_capture(true);
            eprintln!(
                "[mtp.two-chunk] armed (LISA_MTP_TWO_CHUNK=1): plan (m_lo,m_hi,tau) per round"
            );
        }
        let mut plan: Option<crate::core::round_cost::MtpPlan> = None;
        // P4: inter-round wall clock + the fired-shape flag the regime gate
        // reads (charged per round; a round that extended is two-chunk).
        let mut prev_round_end: Option<std::time::Instant> = None;

        'outer: while generated.len() < max_tokens
            && !is_eos(*generated.last().expect("generated is non-empty"))
        {
            // Auto: re-pick the depth every round (0 = serial fallback).
            if let Some(c) = auto.as_mut() {
                depth = c
                    .pick(self.fed.len(), max_tokens - generated.len())
                    .min(model_depth_cap);
                // P4: the regime gate may force the single-chunk shape
                // (two-chunk measured worse, outside an explore block). A
                // forced-single round takes the plain shape below — plan stays
                // None — and a serial pick under it must still reach the
                // copy-first branch, or depth 0 falls through with w = 0 and
                // no proposal at all (the w=0 verify panic).
                let regime_single = two_chunk_on && c.regime_force_single();
                if depth == 0 && (!two_chunk_on || regime_single) {
                    // A serial pick still attempts the prompt-lookup draft
                    // first: on predictable content (llmprobe's repeat-the-prompt
                    // arms) the copy hit is verified-lossless and accepts the
                    // full match, while the plain serial step below never
                    // speculates at all. Measured baseline: the controller
                    // picked 0 on 44-88 % of rounds and copy drafting was
                    // structurally locked out of them (speculation 1.38-1.94
                    // tok/step vs the reference's 3.6-4.4). A miss falls
                    // through to the serial step unchanged; a hit runs the
                    // common round path with depth = 0 (w = copy_len, the
                    // full-copy shape — no chain steps, and the EV never sees
                    // the round: draft_parts is empty, so every observe gate
                    // excludes it). llmprobe --bench-only after the fix:
                    // speculation rose at every rung.
                    copy_index.extend(&committed, committed.len().saturating_sub(COPY_K));
                    let hit = crate::core::copy_draft::copy_lookup_guarded(
                        &committed,
                        &copy_index,
                        COPY_K,
                        crate::core::copy_draft::copy_len_max().min(model_depth_cap),
                    );
                    if hit.is_none() {
                        // Snapshot the GDN state so an EOS step can be un-fed
                        // (the EOS is emitted but never committed, matching the
                        // spec path and `Session::generate`).
                        let snaps: Vec<_> = self.caches.iter().map(R::serial_snapshot).collect();
                        let t0 = std::time::Instant::now();
                        let (t, row) = serial_step(
                            tower,
                            &mut self.caches,
                            &tail,
                            carry_token,
                            stochastic.as_deref_mut(),
                            committed.len() as u64,
                        )?;
                        let wall = t0.elapsed().as_secs_f64() * 1e3;
                        on_token_plain(t)?;
                        generated.push(t);
                        store_tokens.push(t);
                        store_multi.push(row.clone());
                        committed.push(t);
                        backlog.push((carry_token, carry_multi.clone()));
                        carry_token = t;
                        carry_multi = row;
                        tail.push(t as i64);
                        if tail.len() > ctx_len {
                            tail.drain(0..tail.len() - ctx_len);
                        }
                        c.observe_serial(wall, serial_block < PROBE_WARM as u32);
                        serial_block += 1;
                        if is_eos(t) {
                            for (c, s) in self.caches.iter_mut().zip(snaps.iter()) {
                                R::restore_serial_eos(c, s);
                            }
                            break 'outer;
                        }
                        if generated.len() >= max_tokens {
                            break 'outer;
                        }
                        continue;
                    }
                    // a hit: fall through with depth = 0 — w = copy_len below.
                }
                // P2: plan the round (m_lo, m_hi, tau) — the pick above
                // prices `pick()`'s depth; the plan re-bases on the
                // conditional EMAs and caps the base at last round's m_lo+1.
                // P4: the regime gate may force the single-chunk shape
                // (two-chunk measured worse, outside an explore block).
                plan = if two_chunk_on && !c.regime_force_single() {
                    let ft = c.from_table(self.fed.len());
                    let mut p = c.plan_src(depth, self.fed.len(), ft);
                    p.m_lo = p.m_lo.min(model_depth_cap);
                    p.m_hi = p.m_hi.min(model_depth_cap);
                    depth = p.m_lo;
                    Some(p)
                } else {
                    None
                };
            }
            let round_t0 = std::time::Instant::now();
            let mut round_two_chunk = false; // set when the round extends
            serial_block = 0; // a speculative round ends the serial block
            let _trace_round = tower.runtime().trace("mtp.round");
            // --- Prompt-lookup (context-copy) proposal ---
            // If the last COPY_K committed tokens recur earlier, the tokens
            // that followed that occurrence are a strong draft. The two draft
            // sources are MIXED: copy hits lead the proposal and the MTP chain
            // fills the remainder, with every chain step conditioned on the
            // proposal token actually placed (a copy token is fed onward, so
            // the chain tail predicts after the accepted prefix, not after a
            // token that was discarded). A full-depth copy hit skips the draft
            // chain entirely — the old flow paid the whole chain and then threw
            // it away whenever the copy fired. Lossless either way: the verify
            // takes the longest accepting prefix, so a wrong proposal costs the
            // round tail, never an emitted token.
            copy_index.extend(&committed, committed.len().saturating_sub(COPY_K));
            let copy_cap = crate::core::copy_draft::copy_len_max().min(model_depth_cap);
            let copy: Option<Vec<u32>> =
                copy_lookup_guarded(&committed, &copy_index, COPY_K, copy_cap);
            let copy_len = copy.as_ref().map_or(0, |c| c.len());
            // The copy-hit length rides the trace (detail of `mtp.copy`): a
            // post-hoc tally of the length distribution needs it per round.
            let _trace_copy = tower.runtime().trace_detail("mtp.copy", copy_len as u64);
            // P1: the round width is the chain depth, WIDENED by a copy that
            // reaches deeper than it (capped at COPY_LEN_MAX). The controller
            // still picks `depth`; only a copy hit lengthens the round, and a
            // copy hit is already at ceiling, so this converts a fully-accepted
            // 3-wide window into the rest of the match.
            //
            // P2: with the plan armed (LISA_MTP_TWO_CHUNK=1) the base is the
            // plan's m_lo and the width may extend to m_hi after the chunk-A
            // sync. Two-chunk requires copy_len == 0: copy proposals carry no
            // head confidence, so the tau gate has nothing to read (documented
            // deviation — the reference has no copy source). Verify width
            // stays S = w+1 <= 7 either way.
            let tc: Option<crate::core::round_cost::MtpPlan> =
                plan.filter(|p| p.m_hi > p.m_lo && copy_len == 0);
            let mut w = match &tc {
                Some(p) => p.m_lo,
                None => depth.max(copy_len).min(copy_cap),
            };
            if copy_len > 0 {
                copy_rounds += 1;
                copy_tokens += copy_len;
            }
            let plan_w_max = tc.as_ref().map_or(w, |p| p.m_hi).max(w);
            let primed_now = primed.take();
            let _trace_draft = tower.runtime().trace("mtp.draft");
            // Restart: feed the previously accepted pairs to the head. This
            // runs even on a full-copy round — the backlog is rebuilt every
            // round, so skipping the restart would leave that round's pairs
            // permanently unfed and the head's later drafts would attend over
            // a hole in the committed history (§9.3).
            let (d, m) = match primed_now {
                Some(p) => p,
                None => {
                    let mut ft: Vec<i32> = backlog.iter().map(|(t, _)| *t as i32).collect();
                    ft.push(carry_token as i32);
                    let mut fm: Vec<R::Tensor> = backlog.iter().map(|(_, m)| m.clone()).collect();
                    fm.push(carry_multi.clone());
                    let rows = ft.len();
                    let tok_arr = tower.runtime().tokens(&ft, &[1i32, rows as i32])?;
                    let mul_arr = concat_chunks(&fm)?;
                    tower.draft_step(&tok_arr, &mul_arr)?
                }
            };
            let _trace_draft2 = tower.runtime().trace("mtp.draft_parts");
            // The distribution behind proposal 0: the restart above, or the
            // tail of the priming window chain on round 1 (both are the call
            // whose prediction became `d`). One slot, so priming leftovers
            // cannot be mistaken for it.
            let mut chain_pred: Vec<Option<crate::core::oracle::DraftScores>> =
                (0..plan_w_max).map(|_| None).collect();
            if oracle_on {
                chain_pred[0] = crate::core::oracle::last();
            }
            // Chunk-A confidences (two-chunk rounds): proposal i's log p_head,
            // unevaluated until the sync. confs[j] tracks parts[j].
            let mut confs: Vec<R::Tensor> = Vec::new();
            if tc.is_some() {
                if let Some(c) = tower.take_draft_confidence() {
                    confs.push(c);
                }
            }
            let mut chain_tail: Option<(R::Tensor, R::Tensor)> = None;
            let mut draft_parts: Vec<R::Tensor> = if copy_len >= w {
                // Full copy hit: the proposals are the copy tokens; only the
                // chain steps are skipped (the restart above still ran).
                Vec::new()
            } else {
                // --- Draft (device chain) ---
                // The draft id stays on the GPU as a `[1,1]` tensor and is fed
                // straight into the next chain step, so the chain no longer
                // round-trips through the host between steps (one readback per
                // round instead).
                let (mut d, mut m) = (d, m);
                let mut parts: Vec<R::Tensor> = Vec::with_capacity(w);
                for i in 0..w {
                    // Proposal i: the copy hit's token when one covers this
                    // position, else the chain's own prediction.
                    let prop = match copy.as_ref() {
                        Some(c) if i < c.len() => {
                            tower.runtime().tokens(&[c[i] as i32], &[1i32, 1])?
                        }
                        _ => d.reshape(&[1, 1])?,
                    };
                    let mul = m.reshape(&[1, 1, m.dim(-1)])?;
                    let (d2, m2) = tower.draft_step(&prop, &mul)?;
                    parts.push(prop);
                    if tc.is_some() {
                        if let Some(c) = tower.take_draft_confidence() {
                            confs.push(c);
                        }
                    }
                    if oracle_on && i + 1 < plan_w_max {
                        // This call's prediction becomes proposal i+1.
                        chain_pred[i + 1] = crate::core::oracle::last();
                    }
                    d = d2;
                    m = m2;
                }
                chain_tail = Some((d, m));
                parts
            };
            // --- Chunk-A sync + extension (two-chunk rounds only) ---
            // One bounded readback of the m_lo chain confidences (existing
            // logits rows — no new buffer), then extend to m_hi only when the
            // chain's log-confidence clears tau.
            if let Some(p) = &tc {
                // P4 dry-spell gate: when the extension-considered streak ran
                // dry (cost-aware threshold), collapse consideration — no
                // sync, no readback, the round is the plain m_lo shape.
                let dry_ok = auto.as_mut().map(|c| c.ext_dry_allows()).unwrap_or(true);
                if dry_ok && draft_parts.len() >= p.m_lo && confs.len() >= p.m_lo {
                    tc_considered += 1;
                    let t_sync = std::time::Instant::now();
                    let mut vals: Vec<f64> = Vec::with_capacity(confs.len());
                    for c in &confs {
                        c.eval().map_err(|e| anyhow::anyhow!("{e}"))?;
                        vals.push(c.read_f32()?[0] as f64);
                    }
                    let sync_ms = t_sync.elapsed().as_secs_f64() * 1e3;
                    tc_sync_ms_sum += sync_ms;
                    if let Some(c) = auto.as_mut() {
                        c.observe_sync_ms(sync_ms);
                    }
                    let log_conf = crate::core::round_cost::DepthController::chain_log_conf(&vals);
                    let cleared = log_conf > p.tau_ln;
                    if let Some(c) = auto.as_mut() {
                        c.ext_dry_observe(cleared);
                    }
                    if cleared {
                        tc_fired += 1;
                        round_two_chunk = true;
                        let (mut d, mut m) = chain_tail
                            .take()
                            .expect("chain tail exists when parts drafted");
                        for i in p.m_lo..p.m_hi {
                            let prop = d.reshape(&[1, 1])?;
                            let mul = m.reshape(&[1, 1, m.dim(-1)])?;
                            let (d2, m2) = tower.draft_step(&prop, &mul)?;
                            draft_parts.push(prop);
                            if oracle_on && i + 1 < p.m_hi {
                                chain_pred[i + 1] = crate::core::oracle::last();
                            }
                            let _ = tower.take_draft_confidence();
                            d = d2;
                            m = m2;
                        }
                        w = p.m_hi;
                    }
                }
            }
            width_sum += w;

            // --- Verify over [carry, drafts...] ---
            let _trace_verify = tower.runtime().trace("mtp.verify_build");
            tower.set_context_tails(vec![tail.clone()]);
            let carry_arr = tower.runtime().tokens_u32(&[carry_token], &[1i32, 1])?;
            let v_arr = if copy_len >= w {
                let c = copy.as_ref().expect("copy_len >= w implies a copy hit");
                let mut vt: Vec<i32> = Vec::with_capacity(w + 1);
                vt.push(carry_token as i32);
                vt.extend(c.iter().take(w).map(|&t| t as i32));
                tower.runtime().tokens(&vt, &[1i32, (w + 1) as i32])?
            } else {
                let mut verify_parts: Vec<&R::Tensor> = Vec::with_capacity(w + 1);
                verify_parts.push(&carry_arr);
                for p in &draft_parts {
                    verify_parts.push(p);
                }
                R::Tensor::concatenate(&verify_parts, 1).map_err(|e| anyhow::anyhow!("{e}"))?
            };
            // Tail-ULP contract scope (AGENTS.md §9): the verify forward + head
            // enqueue on the split-K lane; the guard covers ONLY this enqueue
            // window — serial, draft, prefill, and batch paths are untouched.
            let _verify_splitk = tower.runtime().verify_scope();
            let (v_mixed, v_multi) = tower.forward_capture(&v_arr, Some(&mut self.caches), true)?;
            let v_logits = tower.head(&v_mixed)?;
            drop(_verify_splitk);
            let top = v_logits
                .argmax_axis(-1)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let _trace_readback = tower.runtime().trace("mtp.readback");
            top.eval().map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut main_tokens = top.read_ids()?;
            // The draft ids: the mixed proposal, read back together with the
            // verify sync (v_arr holds [carry, proposals...]).
            let drafts: Vec<u32> = v_arr.read_raw_ids()?.into_iter().skip(1).collect();

            // Stochastic verify (Leviathan; see the parameter's doc). The
            // proposals are a point mass at the draft token (the copy token, or
            // the head's argmax), so acceptance is `p_target[drafts[i]]` and the
            // first rejection emits a residual draw. `main_tokens` is rewritten
            // to what the round actually emits, so the accept loop below — and
            // the commit/trim path — stay untouched.
            if let Some(sm) = stochastic.as_deref_mut() {
                let mut acc = 0usize;
                while acc < w {
                    let row = v_logits
                        .select(1, acc as i32)?
                        .contiguous()
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    let p = sm
                        .draft_accept_prob(tower.runtime(), &row, drafts[acc])
                        .unwrap_or(0.0);
                    if sm.accept_draw(p) {
                        acc += 1;
                    } else {
                        if let Some(rt) = sm.draw_residual(tower.runtime(), &row, drafts[acc]) {
                            main_tokens[acc] = rt;
                        }
                        break;
                    }
                }
                for i in 0..acc {
                    main_tokens[i] = drafts[i];
                }
                if acc == w {
                    let row = v_logits
                        .select(1, w as i32)?
                        .contiguous()
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    if let Some(bt) = sm.draw_row(tower.runtime(), &row) {
                        main_tokens[w] = bt;
                    }
                }
            }
            let mut a = 0usize;
            while a < w && main_tokens[a] == drafts[a] {
                a += 1;
            }
            rounds += 1;
            accept_hist.push(a);
            if !draft_parts.is_empty() && copy_len == 0 {
                // Chain-proposed round (copy rounds excluded; a two-chunk
                // extension is still chain — every index 0..w is a head
                // proposal, which is exactly what this table measures).
                chain_rounds += 1;
                chain_hist.push(a);
                chain_w_max = chain_w_max.max(w);
            }
            if oracle_on {
                // Rank the target's true token in the head's own set for
                // every index actually proposed by the head (a copy hit has
                // no distribution). `main_tokens[i]` is the target's argmax
                // on a correct prefix for i <= a, so at the first failure it
                // IS the token a branch would have had to carry.
                let ranks = (0..w)
                    .map(|i| {
                        if i < copy_len {
                            crate::core::oracle::StepRank::Copy
                        } else {
                            match chain_pred.get(i).and_then(|p| p.as_ref()) {
                                Some(rec) => crate::core::oracle::StepRank::Chain(
                                    crate::core::oracle::rank(rec, main_tokens[i], drafts[i]),
                                ),
                                None => crate::core::oracle::StepRank::NoRecord,
                            }
                        }
                    })
                    .collect();
                oracle_samples.push(crate::core::oracle::RoundSample {
                    depth: w,
                    accepted: a,
                    ranks,
                });
            }
            let advance = w + 1;
            let n = a + 1;
            // Emission + post-accept host state (the GPU-drained window).
            let _trace_commit = tower.runtime().trace("mtp.commit");

            // Emit the committed tokens, stopping at EOS or the token budget.
            // Whatever we stop at, the caches keep the committed prefix.
            let mut emitted = 0usize;
            let mut eos_seen = false;
            for &t in &main_tokens[..=a] {
                generated.push(t);
                on_token_plain(t)?;
                emitted += 1;
                if is_eos(t) {
                    eos_seen = true;
                    break;
                }
                if generated.len() >= max_tokens {
                    break;
                }
            }
            if eos_seen || emitted < a + 1 {
                // Stopped early: keep the committed prefix. When we stopped on
                // EOS, exclude it from the caches (matching `Session::generate`,
                // which emits EOS but never feeds it); `carry` is never EOS, so
                // the kept prefix is at least the carry.
                let keep = if eos_seen {
                    (emitted - 1).max(1)
                } else {
                    emitted
                };
                for c in self.caches.iter_mut() {
                    tower.runtime().rollback(c, advance, keep, true)?;
                }
                store_tokens.extend_from_slice(&main_tokens[..emitted]);
                store_multi.push(v_multi.slice(1, 0, Some(emitted as i32))?.contiguous()?);
                committed.extend_from_slice(&main_tokens[..emitted]);
                break 'outer;
            }

            for c in self.caches.iter_mut() {
                tower.runtime().rollback(c, advance, n, true)?;
            }
            // Head-cache trim: after every
            // round the head holds EXACTLY the committed pairs — one (token,
            // trunk-hidden) pair per committed token, at its absolute cache
            // position. The restart at the next round's start then re-feeds
            // only this round's committed pairs on top of an exact cache,
            // which is byte-equivalent to a truncate-to-off0 + appendHistory(stash)
            // sequence. The old `drafter_trim(depth-1)`
            // assumed chain row 0's pair was accepted; when it was not (or
            // was re-fed by the restart) the head's history accumulated
            // duplicate/stale pairs and position drift, and the deep chain
            // steps — which attend that history — degraded (the index-2
            // acceptance cliff).
            let target = committed.len() + a; // + n - 1: pairs = committed tokens - 1
            let off = tower.drafter_offset();
            if off > target {
                tower.drafter_trim(off - target);
            }
            backlog.clear();
            for i in 0..a {
                let mi = v_multi.select(1, i as i32)?.contiguous()?;
                backlog.push((main_tokens[i], mi));
            }
            carry_token = main_tokens[a];
            carry_multi = v_multi.select(1, a as i32)?.contiguous()?;
            tail.push(carry_token as i64);
            for i in 0..a {
                tail.push(main_tokens[i] as i64);
            }
            if tail.len() > ctx_len {
                tail.drain(0..tail.len() - ctx_len);
            }
            store_tokens.extend_from_slice(&main_tokens[..=a]);
            store_multi.push(v_multi.slice(1, 0, Some((a + 1) as i32))?.contiguous()?);
            committed.extend_from_slice(&main_tokens[..=a]);
            if let Some(c) = auto.as_mut() {
                // Copy-hit rounds are priced by the copy path, not the depth's
                // own chain+verify cost; sampling the (much cheaper) walls
                // into the per-depth cost EMA inflates that depth's EV and
                // makes the controller over-pick it. They are excluded from
                // both the cost samples and the acceptance EMAs (which then
                // profile the chain, consistently with the priced cost).
                let wall = round_t0.elapsed().as_secs_f64() * 1e3;
                if !draft_parts.is_empty() && w == depth {
                    c.observe_round(depth, a, wall);
                    // P3: fold the whole-round sample into the table's
                    // serving cells (single-chunk shape, solo stream; the
                    // transition rule lives in `observe_table_round`).
                    // Persist at request end when something folded.
                    c.observe_table_round(self.fed.len(), (w + 1) as u32, wall, (a + 1) as f64);
                }
                // Conditional EMAs (P2): every non-copy round feeds the
                // per-index conditional acceptance profile the extension
                // horizon prices (copy proposals carry no head confidence —
                // same exclusion shape as the cost rule above).
                if !draft_parts.is_empty() && copy_len == 0 {
                    // P7: the EV-mode per-round update — the conditional EMAs
                    // plus the sticky-disable floor (port spec §2.8); m_lo is
                    // this round's base depth, exactly what the reference
                    // passes as `mtp_ev_m_lo_prev`.
                    let m_lo = tc.as_ref().map_or(depth, |p| p.m_lo);
                    c.ev_round_update(w, a, m_lo);
                }
                // P4: regime gate + live-cost EMAs. Copy rounds are a third
                // shape entirely (no chain) — excluded like the cost rule.
                let m_lo = tc.as_ref().map_or(depth, |p| p.m_lo);
                let round_end = std::time::Instant::now();
                let inter_ms = prev_round_end.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e3);
                prev_round_end = Some(round_end);
                if copy_len == 0 {
                    c.observe_live_round_ms(wall);
                    c.regime_observe(round_two_chunk, m_lo, wall / (a + 1) as f64, inter_ms);
                    c.regime_tick();
                }
            }
            tower.runtime().trim_memory();
            tower.runtime().trim_memory();
        }

        // 3. Persist the turn's tokens/multi for the next turn's priming.
        // Rebuild the head's committed tail first: the round loop feeds a
        // round's committed tokens only at the start of the *next* round, so
        // after the last round the head is one round behind and may hold
        // rejected-draft rows. Trim back to the post-priming offset and feed the
        // generated tokens explicitly, so the persisted head is exactly the
        // committed prefix (rows 0..len-2 for the committed history).
        let after_prime = history.len();
        let off = tower.drafter_offset();
        let tail_pairs = store_tokens.len().saturating_sub(1);
        self.mtp_head_len = if off >= after_prime {
            tower.drafter_trim(off - after_prime);
            if tail_pairs > 0 {
                let tail: Vec<i32> = store_tokens[1..].iter().map(|&t| t as i32).collect();
                let tok = tower.runtime().tokens(&tail, &[1i32, tail_pairs as i32])?;
                let mul_all = concat_chunks(&store_multi)?;
                let mul = mul_all.slice(1, 0, Some(tail_pairs as i32))?.contiguous()?;
                let _ = tower.draft_step(&tok, &mul)?;
            }
            after_prime + tail_pairs
        } else {
            usize::MAX
        };
        self.mtp_tokens.extend_from_slice(tokens);
        self.mtp_tokens.extend_from_slice(&store_tokens);
        self.mtp_multi
            .push(multi.slice(1, 0, Some(t_s))?.contiguous()?);
        self.mtp_multi.extend(store_multi);
        if rounds > 0 {
            let total: usize = accept_hist.iter().sum();
            let log_d = if is_auto { 6 } else { depth };
            let per_index: Vec<String> = (0..log_d)
                .map(|i| {
                    let hits = accept_hist.iter().filter(|&&a| a > i).count();
                    format!("{:.3}", hits as f64 / rounds as f64)
                })
                .collect();
            eprintln!(
                "[mtp.round] depth {depth} rounds {rounds} accepted {total} \
                 (mean {:.2}/round) w_mean {:.2} copy_hits {copy_rounds} ({copy_tokens} tok, \
                 mean {:.2}) per-index acc [{}] two_chunk considered {tc_considered} fired {tc_fired} sync_mean {:.2} ms",
                total as f64 / rounds as f64,
                width_sum as f64 / rounds as f64,
                copy_tokens as f64 / copy_rounds.max(1) as f64,
                per_index.join(" "),
                tc_sync_ms_sum / tc_considered.max(1) as f64
            );
            if chain_rounds > 0 {
                let chain_total: usize = chain_hist.iter().sum();
                let chain_per_index: Vec<String> = (0..chain_w_max)
                    .map(|i| {
                        let hits = chain_hist.iter().filter(|&&a| a > i).count();
                        format!("{:.3}", hits as f64 / chain_rounds as f64)
                    })
                    .collect();
                eprintln!(
                    "[mtp.chain] chain rounds {chain_rounds} accepted {chain_total} \
                     (mean {:.2}/round) per-index [{}]",
                    chain_total as f64 / chain_rounds as f64,
                    chain_per_index.join(" ")
                );
            }
        }
        if let Some(c) = &auto {
            eprintln!("{}", c.log_summary());
        }
        // P3: persist folded in-run samples at request end (the reference
        // `persistRoundCost`). Serving folds keep the cells warm between
        // cold-start sweeps; `lisa round-cost` stays the component-cell
        // writer of truth.
        if let Some(c) = auto.as_mut() {
            if let Some(t) = c.take_folded_table() {
                let key = tower.mtp_cost_key();
                let path = crate::core::round_cost::default_path();
                let mut table = RoundCostTable::load(&path).unwrap_or_default();
                table.models.insert(key.clone(), t);
                match table.save(&path) {
                    Ok(()) => eprintln!("[mtp.auto] round-cost serving folds persisted ({key})"),
                    Err(e) => eprintln!("[mtp.auto] round-cost persist FAILED: {e}"),
                }
            }
        }
        // P6: persist the EV seed for the next request (the conditional
        // acceptance EMAs + last m_lo, next to the round-cost file).
        if let Some(c) = auto.as_ref() {
            let key = tower.mtp_cost_key();
            let (acc, m_lo) = c.ev_seed_state();
            let seed_path =
                crate::core::round_cost::default_path().with_file_name("mtp_ev_seed.json");
            if let Some(dir) = seed_path.parent().map(|p| p.to_path_buf()) {
                let _ = std::fs::create_dir_all(dir);
            }
            let mut store: serde_json::Map<String, serde_json::Value> =
                std::fs::read_to_string(&seed_path)
                    .ok()
                    .and_then(|d| serde_json::from_str(&d).ok())
                    .unwrap_or_default();
            store.insert(key, serde_json::json!({ "ev_acc": acc, "m_lo": m_lo }));
            if let Err(e) = std::fs::write(
                &seed_path,
                serde_json::to_string_pretty(&store).unwrap_or_default(),
            ) {
                eprintln!("[mtp.ev-seed] persist failed: {e}");
            }
        }
        if oracle_on {
            crate::core::oracle::log_summary(&oracle_samples, &accept_hist);
        }
        Ok(generated)
    }
}

/// Concatenate `multi` chunks along the sequence axis. Chunks may be rank 2
/// (`[1, D]`) or rank 3 (`[1, S, D]`); normalize to `[1, S, D]` first.
fn concat_chunks<T: SpeculativeTensor>(chunks: &[T]) -> anyhow::Result<T> {
    anyhow::ensure!(!chunks.is_empty(), "empty multi store");
    let mut rows: Vec<T> = Vec::with_capacity(chunks.len());
    for m in chunks {
        let d = m.dim(-1);
        let n: i32 = m.shape()[..m.shape().len() - 1].iter().product();
        rows.push(m.reshape(&[1, n, d])?);
    }
    let refs: Vec<&T> = rows.iter().collect();
    T::concatenate(&refs, 1).map_err(|e| anyhow::anyhow!("{e}"))
}

#[cfg(any(test, feature = "diagnostics"))]
impl<R: SpeculativeRuntime> Conversation<R> {
    pub fn glm_mtp_cache_mut_for_test(&mut self) -> &mut R::DraftCache {
        &mut self.glm_mtp_cache
    }
    pub fn glm_mtp_min_position_for_test(&mut self) -> &mut Option<usize> {
        &mut self.glm_mtp_min_position
    }
    pub fn has_glm_mtp_pending_for_test(&self) -> bool {
        self.glm_mtp_pending.is_some()
    }
}
