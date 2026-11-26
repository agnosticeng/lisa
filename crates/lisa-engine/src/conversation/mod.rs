//! Device-independent conversation state and commit/failure transactions.
//! Specialized speculative kernels and cache layouts are bound by engine
//! runtime adapters. These common operations never inspect model topology.
pub mod cache;
mod lookahead;
mod policy;

pub use policy::GenerationPolicy;

use crate::language::{LanguageModel, LanguageRuntime};

pub trait SessionRuntime: LanguageRuntime {
    type Snapshot: Clone;
    type DraftSnapshot;
    type PrefixState;
    fn snapshot_cache(cache: &Self::Cache) -> Self::Snapshot;
    fn restore_cache(cache: &mut Self::Cache, snapshot: &Self::Snapshot);
    fn snapshot_draft(cache: &Self::DraftCache) -> Self::DraftSnapshot;
    fn restore_draft(cache: &mut Self::DraftCache, snapshot: &Self::DraftSnapshot);
    fn restore_prefix(cache: &mut Self::Cache, prefix: &Self::PrefixState);
    fn hidden_prefix(
        tensor: &Self::Tensor,
        rows: usize,
        keep: usize,
    ) -> anyhow::Result<Self::Tensor>;
    fn new_draft_cache(&self) -> Self::DraftCache;
}

pub struct Conversation<R: SessionRuntime> {
    /// False after a failed stateful transaction; only a fresh session can recover.
    pub(crate) valid: bool,
    /// Explicit model arithmetic identity; None preserves legacy model behavior.
    pub(crate) cache_numerical_law: Option<u64>,
    pub caches: Vec<R::Cache>,
    /// Every token fed into the caches, in order (the committed conversation).
    pub fed: Vec<u32>,
    /// Last-position hidden state of the most recent feed (diagnostics).
    pub last_mixed: Option<R::Tensor>,
    /// Successfully queued future steps before reading the current token.
    /// Diagnostics only; no production policy or timing depends on this count.
    /// MTP head priming store: every committed token and its hyper stream
    /// `multi` (paired as `(token[t+1], multi[t])`, matching the head's own
    /// convention). The head's caches are reset per turn and primed from this,
    /// which keeps the per-turn bookkeeping trivial.
    pub mtp_tokens: Vec<u32>,
    pub mtp_multi: Vec<R::Tensor>,
    /// The MTP head cache offset that corresponds to `mtp_tokens` at the end of
    /// the last turn (the number of committed history pairs it holds). Used to
    /// persist the head across turns instead of re-priming the whole history.
    pub mtp_head_len: usize,
    /// GLM 5.3's target-conditioned MTP private compact-attention cache.
    pub(crate) target_mtp_cache: R::DraftCache,
    /// First absolute row visible to that private cache for this session.
    pub(crate) target_mtp_min_position: Option<usize>,
    /// Pending proposal and already sampled boundary token between scheduler cycles.
    pub(crate) target_mtp_pending: Option<TargetMtpPending<R::Tensor>>,
    /// Whether a verified EOS is retained in the target checkpoint.
    pub(crate) target_mtp_eos_policy: TargetMtpEosPolicy,
    /// GLM MTP snapshots aligned by index with `snapshots`.
    pub(crate) target_mtp_snapshots: Vec<(R::DraftSnapshot, Option<usize>)>,
    /// Last target hidden state aligned with each saved cache boundary.
    snapshot_last_mixed: Vec<Option<R::Tensor>>,
    /// Candidate next-token distributions at committed and verified GLM MTP
    /// token boundaries. Test-only so full-vocabulary the reference comparisons do not
    /// add work to normal inference.
    /// Optional test-only callback for full cache snapshots at committed MTP
    /// boundaries. This lets long-run comparisons flush one state at a time.
    /// Prefix snapshots `(boundary, msgs_len, head_offset, per-layer state)`
    /// taken at the start of each turn. On a divergent request the server can
    /// rewind to the largest boundary at or below the common prefix and prefill
    /// only the tail.
    pub snapshots: Vec<(usize, usize, usize, Vec<R::Snapshot>)>,
    pub snapshots_enabled: bool,
}

/// Sampling policy for the next token after a token is emitted during
/// target-conditioned GLM MTP. Serving can switch this at live DSML boundaries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TargetMtpSamplingMode {
    #[default]
    Configured,
    Greedy,
}

/// Target-checkpoint behavior for an EOS already evaluated by a GLM MTP cycle.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TargetMtpEosPolicy {
    /// Native cycle API: retain accepted or replacement EOS in the checkpoint.
    #[default]
    CommitVerified,
    /// HTTP generation: report EOS, but retain only the preceding target rows.
    StopBeforeVerified,
}

/// A scheduler yield preserves the sampled carry and its proposal. Rebuilding
/// these from `last_mixed` would consume RNG again and replace a verify cycle
/// with an extra target seed step.
pub(crate) struct TargetMtpPending<T> {
    pub(crate) fed_len: usize,
    pub(crate) carry: u32,
    pub(crate) draft_parent: u32,
    pub(crate) draft: Option<u32>,
    pub(crate) draft_logits: Option<T>,
    pub(crate) sampling_mode: TargetMtpSamplingMode,
    pub(crate) min_position: usize,
}

fn ensure_matching_cache_law(expected: Option<u64>, actual: Option<u64>) -> anyhow::Result<()> {
    anyhow::ensure!(
        expected == actual,
        "session numerical law changed; start a fresh session instead of reusing cached state"
    );
    Ok(())
}

#[cfg(test)]
mod numerical_law_tests {
    #[test]
    fn session_cache_law_rejects_changed_model_arithmetic() {
        use super::ensure_matching_cache_law;
        assert!(ensure_matching_cache_law(None, None).is_ok());
        assert!(ensure_matching_cache_law(Some(17), Some(17)).is_ok());
        assert!(ensure_matching_cache_law(Some(17), Some(18)).is_err());
        assert!(ensure_matching_cache_law(Some(17), None).is_err());
        assert!(ensure_matching_cache_law(None, Some(17)).is_err());
    }
}

const SNAPSHOT_KEEP: usize = 4;

impl<R: SessionRuntime> Conversation<R> {
    /// Choose whether a verified GLM MTP EOS belongs to the target checkpoint.
    /// This does not change EOS reporting or the completed cycle's private writes.
    pub fn set_target_mtp_eos_policy(&mut self, policy: TargetMtpEosPolicy) {
        self.target_mtp_eos_policy = policy;
    }

    /// Whether this session may be resumed or exported to a prefix cache.
    pub fn is_valid(&self) -> bool {
        self.valid
    }

    /// Reject state whose target/private caches may have diverged during a failure.
    pub fn ensure_valid(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.valid,
            "session state is invalid after a failed transaction; start a fresh session"
        );
        Ok(())
    }

    /// Prevent reuse or snapshot export after a failed stateful transaction.
    /// Existing arrays remain available for diagnostics, but are not reusable.
    pub fn invalidate(&mut self) {
        self.valid = false;
        self.target_mtp_pending = None;
        self.snapshots.clear();
        self.target_mtp_snapshots.clear();
        self.snapshot_last_mixed.clear();
    }

    pub(crate) fn ensure_cache_numerical_law(
        &self,
        tower: &dyn LanguageModel<R>,
    ) -> anyhow::Result<()> {
        self.ensure_valid()?;
        ensure_matching_cache_law(self.cache_numerical_law, tower.cache_numerical_law())
    }

    /// A fresh conversation. Resets the PLE n-gram context so the first suffix
    /// is hashed from the default (EOS-filled) context, not a leftover one.
    pub fn new(tower: &mut dyn LanguageModel<R>) -> Self {
        tower.clear_context();
        Conversation {
            valid: true,
            cache_numerical_law: tower.cache_numerical_law(),
            caches: tower.new_caches(),
            fed: Vec::new(),
            last_mixed: None,
            mtp_tokens: Vec::new(),
            mtp_multi: Vec::new(),
            mtp_head_len: 0,
            target_mtp_cache: tower.runtime().new_draft_cache(),
            target_mtp_min_position: None,
            target_mtp_pending: None,
            target_mtp_eos_policy: TargetMtpEosPolicy::default(),
            target_mtp_snapshots: Vec::new(),
            snapshot_last_mixed: Vec::new(),
            snapshots: Vec::new(),
            snapshots_enabled: false,
        }
    }

    /// Prefill `tokens` (the appended suffix) and return the logits at the last
    /// position. Long suffixes are fed in windows (bounded activations).
    pub fn feed(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        tokens: &[u32],
    ) -> anyhow::Result<R::Tensor> {
        self.feed_impl(tower, tokens, false)
    }

    /// Feed the final prompt suffix before drawing a token. Aligned cache
    /// staging and history replay use `feed` so their chunk boundaries remain
    /// unchanged.
    pub fn feed_for_generation(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        tokens: &[u32],
    ) -> anyhow::Result<R::Tensor> {
        self.feed_impl(tower, tokens, true)
    }

    fn feed_impl(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        tokens: &[u32],
        sampling_boundary: bool,
    ) -> anyhow::Result<R::Tensor> {
        self.ensure_cache_numerical_law(tower)?;
        anyhow::ensure!(!tokens.is_empty(), "empty feed");
        anyhow::ensure!(
            self.fed
                .len()
                .checked_add(tokens.len())
                .is_some_and(|end| end <= tower.max_position_embeddings()),
            "context overflow: {} + {} > max_position_embeddings {}",
            self.fed.len(),
            tokens.len(),
            tower.max_position_embeddings()
        );
        self.target_mtp_pending = None;
        let result = (|| -> anyhow::Result<R::Tensor> {
            let last = if sampling_boundary {
                tower.prefill_for_generation(tokens, &mut self.caches)?
            } else {
                tower.prefill(tokens, &mut self.caches)?
            };
            let logits = tower.head(&last)?;
            self.last_mixed = Some(last);
            self.fed.extend_from_slice(tokens);
            Ok(logits)
        })();
        if result.is_err() {
            self.invalidate();
        }
        result
    }

    /// Commit one generated token through the model's decode graph and return
    /// its next-token logits. Prompt and suffix appends remain [`Self::feed`].
    pub fn decode_token(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        token: u32,
    ) -> anyhow::Result<R::Tensor> {
        self.ensure_cache_numerical_law(tower)?;
        anyhow::ensure!(
            self.fed
                .len()
                .checked_add(1)
                .is_some_and(|end| end <= tower.max_position_embeddings()),
            "context overflow: {} + 1 > max_position_embeddings {}",
            self.fed.len(),
            tower.max_position_embeddings()
        );
        self.target_mtp_pending = None;
        let result = (|| -> anyhow::Result<R::Tensor> {
            let last = tower.decode_token(token, &mut self.caches)?;
            let logits = tower.head(&last)?;
            self.last_mixed = Some(last);
            self.fed.push(token);
            Ok(logits)
        })();
        if result.is_err() {
            self.invalidate();
        }
        result
    }

    /// [`Conversation::feed`] but also returning the hyper stream `multi`
    /// `[1, len(tokens), hc*H]` for the fed tokens (needed by the MTP head).
    pub fn feed_multi(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        tokens: &[u32],
    ) -> anyhow::Result<(R::Tensor, R::Tensor)> {
        self.feed_multi_impl(tower, tokens, false)
    }

    /// Sampling-boundary feed with the model's auxiliary hidden stream.
    pub fn feed_multi_for_generation(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        tokens: &[u32],
    ) -> anyhow::Result<(R::Tensor, R::Tensor)> {
        self.feed_multi_impl(tower, tokens, true)
    }

    fn feed_multi_impl(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        tokens: &[u32],
        sampling_boundary: bool,
    ) -> anyhow::Result<(R::Tensor, R::Tensor)> {
        self.ensure_cache_numerical_law(tower)?;
        anyhow::ensure!(!tokens.is_empty(), "empty feed");
        anyhow::ensure!(
            self.fed
                .len()
                .checked_add(tokens.len())
                .is_some_and(|end| end <= tower.max_position_embeddings()),
            "context overflow: {} + {} > max_position_embeddings {}",
            self.fed.len(),
            tokens.len(),
            tower.max_position_embeddings()
        );
        self.target_mtp_pending = None;
        let result = (|| -> anyhow::Result<(R::Tensor, R::Tensor)> {
            let (last, multi) = if sampling_boundary {
                tower.prefill_multi_for_generation(tokens, &mut self.caches)?
            } else {
                tower.prefill_multi(tokens, &mut self.caches)?
            };
            let logits = tower.head(&last)?;
            self.last_mixed = Some(last);
            self.fed.extend_from_slice(tokens);
            Ok((logits, multi))
        })();
        if result.is_err() {
            self.invalidate();
        }
        result
    }

    /// A session resumed from a cross-request prefix-cache entry: the caches
    /// are seeded with the entry's state at `prefix_len` fed tokens, `fed`
    /// matches it, and the PLE host tail is set so the next suffix hashes
    /// from the right context. The caller prefills only `tokens[prefix_len..]`.
    pub fn from_prefix(
        tower: &mut dyn LanguageModel<R>,
        entry_layers: &[R::PrefixState],
        prompt: &[u32],
        prefix_len: usize,
    ) -> Self {
        let mut sess = Self::new(tower);
        for (c, st) in sess.caches.iter_mut().zip(entry_layers.iter()) {
            R::restore_prefix(c, st);
        }
        sess.fed = prompt[..prefix_len].to_vec();
        let ctx_len = tower.context_window();
        let tail: Vec<i64> = sess.fed[sess.fed.len().saturating_sub(ctx_len)..]
            .iter()
            .map(|&t| t as i64)
            .collect();
        tower.set_context_tails(vec![tail]);
        sess
    }

    /// Enable prefix snapshots (server sessions only; the Clone cost is per turn).
    pub fn enable_snapshots(&mut self) {
        self.snapshots_enabled = true;
    }

    /// Snapshot the committed state; call before feeding a turn's suffix.
    pub fn capture_snapshot(&mut self, msgs_len: usize, head_offset: usize) {
        if !self.valid || !self.snapshots_enabled {
            return;
        }
        let snaps: Vec<R::Snapshot> = self.caches.iter().map(R::snapshot_cache).collect();
        self.snapshots
            .push((self.fed.len(), msgs_len, head_offset, snaps));
        self.target_mtp_snapshots.push((
            R::snapshot_draft(&self.target_mtp_cache),
            self.target_mtp_min_position,
        ));
        self.snapshot_last_mixed.push(self.last_mixed.clone());
        if self.snapshots.len() > SNAPSHOT_KEEP {
            let drop = self.snapshots.len() - SNAPSHOT_KEEP;
            self.snapshots.drain(0..drop);
            self.target_mtp_snapshots.drain(0..drop);
            self.snapshot_last_mixed.drain(0..drop);
        }
    }

    /// Restore only an exact message boundary. A string suffix rendered after
    /// that boundary is invalid against an earlier snapshot.
    pub fn restore_snapshot_exact(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        msgs_len: usize,
    ) -> Option<usize> {
        if !self.snapshots.iter().any(|(_, m, _, _)| *m == msgs_len) {
            return None;
        }
        self.restore_snapshot(tower, msgs_len)
    }

    /// Rewind to the latest snapshot at or before `msgs_len` messages.
    /// Returns its token boundary; callers must replay any intervening history.
    pub fn restore_snapshot(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        msgs_len: usize,
    ) -> Option<usize> {
        self.ensure_cache_numerical_law(tower).ok()?;
        let idx = self
            .snapshots
            .iter()
            .rposition(|(_, m, _, _)| *m <= msgs_len)?;
        let boundary = self.snapshots[idx].0;
        let head = self.snapshots[idx].2;
        let snaps = self.snapshots[idx].3.clone();
        let target_mtp_min_position = self.target_mtp_snapshots[idx].1;
        for (c, s) in self.caches.iter_mut().zip(snaps.iter()) {
            R::restore_cache(c, s);
        }
        self.fed.truncate(boundary);
        self.snapshots.truncate(idx + 1);
        self.target_mtp_snapshots.truncate(idx + 1);
        self.snapshot_last_mixed.truncate(idx + 1);
        self.last_mixed = self.snapshot_last_mixed[idx].clone();
        let context = tower.context_window();
        tower.set_context_tails(vec![
            self.fed[self.fed.len().saturating_sub(context)..]
                .iter()
                .map(|&token| i64::from(token))
                .collect(),
        ]);
        // The priming history belongs to the target boundary, even if the
        // snapshot was taken before the head had consumed those rows.
        tower.drafter_restore_offset(head);
        self.truncate_head(boundary);
        self.mtp_head_len = head;
        if tower.has_drafter()
            && !tower.has_block_drafter()
            && !tower.has_target_mtp()
            && tower.drafter_offset() != head
        {
            // An intervening one-output/EOS turn may have cleared private KV.
            // An offset cannot restore missing storage. Retain the exact
            // priming rows and rebuild the head on its next use instead.
            tower.drafter_reset();
            self.mtp_head_len = 0;
        }
        // Restore the private GLM MTP window with the target prefix. Keeping
        // its tape in the same snapshot prevents a divergent turn from
        // reusing rows produced by the abandoned suffix.
        R::restore_draft(
            &mut self.target_mtp_cache,
            &self.target_mtp_snapshots[idx].0,
        );
        self.target_mtp_min_position = target_mtp_min_position;
        self.target_mtp_pending = None;
        Some(boundary)
    }

    /// Truncate the MTP head's stored token/multi history to `boundary` rows.
    pub(crate) fn truncate_head(&mut self, boundary: usize) {
        self.mtp_tokens.truncate(boundary);
        let mut out: Vec<R::Tensor> = Vec::new();
        let mut rem = boundary;
        for chunk in &self.mtp_multi {
            if rem == 0 {
                break;
            }
            let sh = R::shape(chunk);
            let rows: usize = sh[..sh.len() - 1].iter().map(|&d| d as usize).product();
            if rows <= rem {
                out.push(chunk.clone());
                rem -= rows;
            } else {
                if let Ok(sliced) = R::hidden_prefix(chunk, rows, rem) {
                    out.push(sliced);
                }
                rem = 0;
            }
        }
        self.mtp_multi = out;
    }
}

/// Sampling is engine policy; a device only executes the requested kernels.
/// Separate methods retain each reference's filtering, reduction and RNG law.
pub trait TokenSampler<R: LanguageRuntime> {
    fn greedy(&mut self, runtime: &R, logits: &R::Tensor) -> anyhow::Result<u32>;
    fn draw(&mut self, runtime: &R, logits: &R::Tensor, history: &[u32]) -> anyhow::Result<u32>;
    fn draw_host(
        &mut self,
        runtime: &R,
        logits: &R::Tensor,
        history: &[u32],
    ) -> anyhow::Result<u32>;
}

impl<R: SessionRuntime> Conversation<R> {
    /// Serial target decode with a live sampling policy for the token after
    /// each emitted token. GLM serving uses this for DSML structural syntax,
    /// which the reference samples greedily between parameter payloads.
    pub fn generate_with_sampling_mode(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        tokens: &[u32],
        max_tokens: usize,
        sampler: &mut impl TokenSampler<R>,
        mut after_prefill: Option<&mut dyn FnMut(&[R::Cache])>,
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
            .ok_or_else(|| anyhow::anyhow!("context position overflow"))?;
        anyhow::ensure!(
            prompt_end <= tower.max_position_embeddings(),
            "context overflow before generation"
        );
        let max_tokens =
            max_tokens.min((tower.max_position_embeddings() - prompt_end).saturating_add(1));
        let result = (|| -> anyhow::Result<Vec<u32>> {
            let mut logits = self.feed_for_generation(tower, tokens)?;
            if let Some(callback) = after_prefill.as_deref_mut() {
                callback(&self.caches);
            }
            let mut mode = TargetMtpSamplingMode::Configured;
            let mut token = if mode == TargetMtpSamplingMode::Greedy {
                sampler.greedy(tower.runtime(), &logits)?
            } else if tower.uses_host_sampling() {
                sampler.draw_host(tower.runtime(), &logits, &self.fed)?
            } else {
                sampler.draw(tower.runtime(), &logits, &self.fed)?
            };
            let mut generated = Vec::with_capacity(max_tokens);
            generated.push(token);
            mode = on_token(token)?;
            while generated.len() < max_tokens && !crate::eos::is_eos(token) {
                logits = self.decode_token(tower, token)?;
                token = if mode == TargetMtpSamplingMode::Greedy {
                    sampler.greedy(tower.runtime(), &logits)?
                } else if tower.uses_host_sampling() {
                    sampler.draw_host(tower.runtime(), &logits, &self.fed)?
                } else {
                    sampler.draw(tower.runtime(), &logits, &self.fed)?
                };
                generated.push(token);
                mode = on_token(token)?;
            }
            Ok(generated)
        })();
        if result.is_err() {
            self.invalidate();
        }
        result
    }
}

#[cfg(test)]
mod tests;
