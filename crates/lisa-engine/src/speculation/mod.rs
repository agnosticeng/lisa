//! Portable speculative execution: independent drafts with context-copy hits,
//! target-conditioned MTP and block drafters' tree rounds. The engine owns transaction order; runtime adapters
//! provide tensor and cache storage operations without selecting a model.
mod block;
mod independent;
pub mod runtime;
mod target_mtp;
pub mod tree;

use crate::conversation::{Conversation, TargetMtpPending};
pub use crate::conversation::{TargetMtpEosPolicy, TargetMtpSamplingMode};
use crate::core::copy_draft::{COPY_K, CopyIndex, copy_lookup_guarded};
use crate::eos::is_eos;
use crate::language::LanguageModel;
use crate::speculation::runtime::{SpeculativeRuntime, SpeculativeSampler, SpeculativeTensor};
use block::*;
pub use target_mtp::*;

impl<R: SpeculativeRuntime> Conversation<R> {
    pub fn target_mtp_cache_for_test(&self) -> &R::DraftCache {
        &self.target_mtp_cache
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
        mut after_prefill: Option<&mut dyn FnMut(&[R::Cache])>,
        on_token: &mut dyn FnMut(u32) -> anyhow::Result<()>,
    ) -> anyhow::Result<Vec<u32>> {
        self.ensure_cache_numerical_law(tower)?;
        if max_tokens == 0 {
            return Ok(Vec::new());
        }
        let prompt_end = self
            .fed
            .len()
            .checked_add(tokens.len())
            .ok_or_else(|| anyhow::anyhow!("context position overflow"))?;
        anyhow::ensure!(
            prompt_end <= tower.max_position_embeddings(),
            "context overflow before generation"
        );
        let max_tokens = max_tokens.min(tower.max_position_embeddings() - prompt_end + 1);
        let result = (|| -> anyhow::Result<Vec<u32>> {
            let logits = self.feed_for_generation(tower, tokens)?;
            // Clean commit boundary (prompt fed, nothing decoded yet): the
            // cross-request prefix cache snapshots here.
            if let Some(cb) = after_prefill.as_deref_mut() {
                cb(&self.caches);
            }
            let mut token = if tower.uses_host_sampling() {
                sampler.draw_host(tower.runtime(), &logits, &self.fed)?
            } else {
                sampler.draw(tower.runtime(), &logits, &self.fed)?
            };
            let mut out = vec![token];
            on_token(token)?;
            if self.can_queue_serial(tower, sampler) {
                self.generate_serial_lookahead(tower, &mut out, max_tokens, on_token)?;
                return Ok(out);
            }
            while out.len() < max_tokens && !is_eos(token) {
                let logits = self.decode_token(tower, token)?;
                token = if tower.uses_host_sampling() {
                    sampler.draw_host(tower.runtime(), &logits, &self.fed)?
                } else {
                    sampler.draw(tower.runtime(), &logits, &self.fed)?
                };
                out.push(token);
                on_token(token)?;
            }
            Ok(out)
        })();
        if result.is_err() {
            self.invalidate();
        }
        result
    }
}

/// Depth selection policy for the speculative loop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MtpDepth {
    /// Fixed draft depth for the selected model (0 selects serial). Block
    /// drafters accept 1 up to their declared maximum; independent Qwen MTP
    /// chains use 2..=6. GLM's target-conditioned MTP uses its native
    /// single-step verify cycle.
    Fixed(usize),
    /// EV auto for independent draft chains, measured SPS for block drafters.
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
    history: &[u32],
    mode: TargetMtpSamplingMode,
) -> anyhow::Result<(u32, R::Tensor, R::Tensor)> {
    tower.set_context_tails(vec![tail.to_vec()]);
    let arr = tower.runtime().tokens(&[carry_token as i32], &[1, 1])?;
    let (mixed, multi) = tower.forward_capture(&arr, Some(caches), false)?;
    let logits = tower.head(&mixed)?;
    let token = draw_block_target_token(tower.runtime(), stochastic, &logits, history, mode)?;
    Ok((
        token,
        multi.select(1, 0)?.contiguous()?,
        mixed.select(1, 0)?.contiguous()?,
    ))
}

fn sampling_history_with_suffix(history: &[u32], window: usize, suffix: &[u32]) -> Vec<u32> {
    let keep = window.saturating_sub(suffix.len());
    let start = history.len().saturating_sub(keep);
    let mut out = history[start..].to_vec();
    out.extend_from_slice(suffix);
    out
}

fn draw_serial_target_token<R: SpeculativeRuntime>(
    tower: &dyn LanguageModel<R>,
    sampler: Option<&mut R::Sampler>,
    logits: &R::Tensor,
    history: &[u32],
    mode: TargetMtpSamplingMode,
) -> anyhow::Result<u32> {
    if mode == TargetMtpSamplingMode::Greedy {
        return tower.runtime().sampling_logits(logits)?.argmax_id();
    }
    match sampler {
        Some(sampler) if tower.uses_host_sampling() => {
            sampler.draw_host(tower.runtime(), logits, history)
        }
        Some(sampler) => sampler.draw(tower.runtime(), logits, history),
        None => tower.runtime().sampling_logits(logits)?.argmax_id(),
    }
}

impl<R: SpeculativeRuntime> Conversation<R> {
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

    /// `[`Session::generate_mtp`]` with the depth selection policy:
    /// [`MtpDepth::Fixed`] is the phase-1 opt-in (behavior unchanged);
    /// [`MtpDepth::Auto`] runs the EV controller ([`crate::core::round_cost`]): per-index
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
        // `Some(sm)` checks deterministic draft proposals against ordinary
        // target draws, preserving the serial history, RNG and policy boundary.
        // `None` explicitly selects greedy target verification.
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

    /// `[`Session::generate_mtp_depth`]` with a live policy for the next target
    /// sample. The callback returns the mode derived from the just-emitted
    /// token; GLM MTP uses it at exact-sampling boundaries.
    pub fn generate_mtp_depth_with_sampling_mode(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        tokens: &[u32],
        max_tokens: usize,
        depth_sel: MtpDepth,
        after_prefill: Option<&mut dyn FnMut(&[R::Cache])>,
        stochastic: Option<&mut R::Sampler>,
        on_token: &mut dyn FnMut(u32) -> anyhow::Result<TargetMtpSamplingMode>,
    ) -> anyhow::Result<Vec<u32>> {
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
        let result = self.generate_independent_mtp(
            tower,
            tokens,
            max_tokens,
            depth_sel,
            after_prefill,
            stochastic,
            on_token,
        );
        if result.is_err() {
            self.invalidate();
            tower.drafter_reset();
        }
        result
    }
}

impl<R: SpeculativeRuntime> Conversation<R> {
    pub fn target_mtp_cache_mut_for_test(&mut self) -> &mut R::DraftCache {
        &mut self.target_mtp_cache
    }
    pub fn target_mtp_min_position_for_test(&mut self) -> &mut Option<usize> {
        &mut self.target_mtp_min_position
    }
    pub fn has_target_mtp_pending_for_test(&self) -> bool {
        self.target_mtp_pending.is_some()
    }
}
