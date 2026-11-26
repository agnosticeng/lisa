//! Target-conditioned MTP: the draft head reads the target's hidden state, and
//! each cycle verifies, commits or rewinds target and draft state together.
use super::*;

pub(super) fn target_mtp_uses_exact_sampling<S: SpeculativeSampler>(
    sampler: Option<&S>,
    mode: TargetMtpSamplingMode,
) -> bool {
    mode == TargetMtpSamplingMode::Configured
        && sampler.is_some_and(|sampler| sampler.exact_sampling() && sampler.temperature() > 0.0)
}
/// A configured positive-temperature draw must still advance the reference's RNG, even
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
pub(super) fn draw_target_mtp_token<R: SpeculativeRuntime>(
    context: &R,
    sampler: Option<&mut R::Sampler>,
    logits: &R::Tensor,
    history: &[u32],
    mode: TargetMtpSamplingMode,
) -> anyhow::Result<u32> {
    if mode == TargetMtpSamplingMode::Greedy {
        return context.sampling_logits(logits)?.argmax_id();
    }
    match sampler {
        Some(sampler) => sampler.draw_host(context, logits, history),
        None => context.sampling_logits(logits)?.argmax_id(),
    }
}

impl<R: SpeculativeRuntime> Conversation<R> {
    pub(super) fn restore_target_verify_to_first(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        before: &[R::Rollback],
        first: u32,
    ) -> anyhow::Result<R::Tensor> {
        let _trace = tower.runtime().trace("glm53.mtp.rollback_replay");
        self.restore_target_verify_boundary(before)?;
        let replay = tower.runtime().tokens(&[first as i32], &[1i32, 1])?;
        let (mixed, multi) = tower.forward_capture(&replay, Some(&mut self.caches), false)?;
        self.fed.push(first);
        self.last_mixed = Some(mixed.select(1, 0)?.contiguous()?);
        Ok(multi)
    }

    pub(super) fn restore_target_verify_boundary(
        &mut self,
        before: &[R::Rollback],
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.caches.len() == before.len(),
            "target cache layout changed during GLM MTP verify"
        );
        for (cache, state) in self.caches.iter_mut().zip(before) {
            R::restore_rollback(cache, state)?;
        }
        Ok(())
    }

    pub(super) fn advance_target_mtp_cache(
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
            &mut self.target_mtp_cache,
            row0_multi,
            draft,
            position,
            min_position,
        )?;
        tower.target_mtp_step(
            &mut self.target_mtp_cache,
            row1_multi,
            next,
            position + 1,
            min_position,
        )
    }

    /// Finish the native cycle's private-cache writes, then retain only its
    /// first target row. The reference performs this rewind after emitting a parser/control
    /// boundary; it preserves private-cache writes and clears the proposal.
    pub(super) fn rewind_target_mtp_cycle(
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
                    &mut self.target_mtp_cache,
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
                &mut self.target_mtp_cache,
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
        Ok(replay_logits)
    }

    pub(super) fn commit_target_mtp_tail(
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
        let mtp_before = R::snapshot_draft(&self.target_mtp_cache);
        let result: anyhow::Result<R::Tensor> = (|| -> anyhow::Result<R::Tensor> {
            let ids = tower.runtime().tokens(&[token as i32], &[1i32, 1])?;
            let (mixed, multi) = tower.forward_capture(&ids, Some(&mut self.caches), false)?;
            if prime_mtp {
                let next = tower.head(&mixed)?.argmax_id()?;
                let _ = tower.target_mtp_step(
                    &mut self.target_mtp_cache,
                    &multi,
                    next,
                    position,
                    min_position,
                )?;
            }
            mixed.select(1, 0)?.contiguous()
        })();
        match result {
            Ok(mixed) => {
                anyhow::ensure!(
                    self.caches.len() == before.len(),
                    "target cache layout changed while committing the final GLM MTP token"
                );
                self.fed.push(token);
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
                R::restore_draft(&mut self.target_mtp_cache, &mtp_before);
                Err(error)
            }
        }
    }

    /// Finish a terminal target draw before notifying the caller. A callback
    /// stop then leaves a reusable committed boundary; model failures do not.
    pub(super) fn finish_target_mtp_tail(
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

    pub(super) fn generate_target_mtp(
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

    pub(super) fn generate_target_mtp_inner(
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
            // The reference invalidates a checkpoint after a failed speculative operation.
            // Local rollback handles the common verifier/tail errors, but a
            // seed, replay, private-cache step, or callback can fail after other
            // state changed. Never expose that partial transaction for reuse.
            self.invalidate();
        }
        result
    }

    pub(super) fn target_mtp_transaction(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        tokens: &[u32],
        max_tokens: usize,
        after_prefill: Option<&mut dyn FnMut(&[R::Cache])>,
        mut stochastic: Option<&mut R::Sampler>,
        on_token: &mut dyn FnMut(u32) -> anyhow::Result<TargetMtpSamplingMode>,
        single_cycle: bool,
        committed_callback_stop: &mut bool,
    ) -> anyhow::Result<Vec<u32>> {
        let pending = self
            .target_mtp_pending
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
            // extensions, as the reference does. Prefix rewinds restore the corresponding
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
                let logits = self.feed_for_generation(tower, tokens)?;
                if let Some(cb) = after_prefill {
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
            generated.push(first);
            sampling_mode = match on_token(first) {
                Ok(mode) => mode,
                Err(error) => {
                    if !is_eos(first) {
                        let position = self.fed.len();
                        let min_position = *self.target_mtp_min_position.get_or_insert(position);
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
            let min_position = *self.target_mtp_min_position.get_or_insert(first_position);
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
            let carry = draw_target_mtp_token(
                tower.runtime(),
                stochastic.as_deref_mut(),
                &first_logits,
                &self.fed,
                sampling_mode,
            )?;
            // The reference seeds GLM MTP from the target argmax in both opportunistic and
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
                    &mut self.target_mtp_cache,
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
                    self.target_mtp_pending = Some(TargetMtpPending {
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

            // Match the reference's plain cycle when the sampled parent changed or
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
                    &mut self.target_mtp_cache,
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
                let (row0_logits, row1_logits) = {
                    let _trace = tower.runtime().trace("glm53.mtp.verify_head");
                    let row0_mixed = verify_mixed.select(1, 0)?.expand_dims(1)?.contiguous()?;
                    let row1_mixed = verify_mixed.select(1, 1)?.expand_dims(1)?.contiguous()?;
                    let row0_logits = tower.head(&row0_mixed)?;
                    row0_logits.eval()?;
                    let row1_logits = tower.head(&row1_mixed)?;
                    row1_logits.eval()?;
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

            // The reference finishes the cycle's decision before its caller emits tokens
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
            // acceptance-rate mismatch can be compared against the reference without
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
                // boundary, then commit only the first target row (the reference replays
                // this token after restoring its GLM 5.3 speculative snapshot).
                let replay_multi = self.restore_target_verify_to_first(tower, &before, carry)?;
                if exact_sampling {
                    // The reference exact rejection commits the residual replacement
                    // through the target trunk, then seeds MTP from that
                    // replacement row and the following target argmax.
                    generated.push(actual);
                    if is_eos(actual)
                        && self.target_mtp_eos_policy == TargetMtpEosPolicy::StopBeforeVerified
                    {
                        // Native HTTP discards a replacement EOS after the cycle.
                        // The retained carry has already been replayed, and EOS
                        // contributes no private-cache priming to preserve.
                        self.target_mtp_pending = None;
                        self.finish_target_mtp_tail(
                            tower,
                            actual,
                            min_position,
                            on_token,
                            committed_callback_stop,
                        )?;
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
                    self.last_mixed = Some(replacement_mixed.select(1, 0)?.contiguous()?);
                    draft_parent = replacement_logits.argmax_id()?;
                    draft_logits = Some(tower.target_mtp_step(
                        &mut self.target_mtp_cache,
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
                    // The reference restores the pre-verify state, replays the committed
                    // carry token, and recomputes target logits from that
                    // replayed hidden state. Reusing the speculative row-0
                    // logits can preserve argmax while changing the full
                    // post-rejection distribution.
                    let replay_logits =
                        tower.head(self.last_mixed.as_ref().ok_or_else(|| {
                            anyhow::anyhow!("GLM MTP rejection replay lost its target hidden state")
                        })?)?;
                    // Native GLM seeds from the verifier's row-0 argmax even
                    // when replay changes that row's distribution. The next
                    // sampled carry still comes from the replayed logits.
                    draft_parent = actual;
                    draft_logits = Some(tower.target_mtp_step(
                        &mut self.target_mtp_cache,
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

            // The target accepted the point-mass MTP proposal. The reference commits both
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
                self.target_mtp_pending = None;
                self.finish_target_mtp_tail(
                    tower,
                    draft_token,
                    min_position,
                    on_token,
                    committed_callback_stop,
                )?;
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
                // The reference advances its private MTP compact cache through both
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
}
