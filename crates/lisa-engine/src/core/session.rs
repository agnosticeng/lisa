//! Multi-turn (and, later, batched) session state.
//!
//! A [`Session`] owns the caches for one conversation and the exact token ids
//! fed into them, so a new turn only prefills the appended suffix — the
//! attention cache continues from its offset, and the GDN recurrent state and
//! the PLE n-gram history are already carried by the caches and `Tower`.

use lisa_mlx::Array;
use lisa_mlx::ops::indexing::IndexOp;

use crate::core::cache::{IndexerTape, LayerCache, LayerRollbackState, TapeSnapshot};
use crate::core::copy_draft::{COPY_K, CopyIndex, copy_lookup_guarded};
use crate::core::generate::is_eos;
use crate::core::sampler::Sampler;
use crate::models::LanguageModel;

#[cfg(test)]
fn capture_glm_mtp_verify_hyper_states(
    prefix: &std::ffi::OsStr,
    rows: &Array,
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
            .index((.., row as i32, ..))
            .contiguous()?
            .as_dtype(lisa_mlx::Dtype::Float32)?
            .reshape(&[rows.dim(2)])?
            .to_vec1::<f32>()?;
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

#[cfg(test)]
fn capture_glm_mtp_verify_logits(
    prefix: &std::ffi::OsStr,
    capture_index: usize,
    logits: &Array,
    fed_position: usize,
) -> anyhow::Result<()> {
    let values = logits
        .as_dtype(lisa_mlx::Dtype::Float32)?
        .contiguous()?
        .reshape(&[logits.size() as i32])?
        .to_vec1::<f32>()?;
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

pub struct Session {
    /// False after a failed stateful transaction; only a fresh session can recover.
    valid: bool,
    /// Explicit model arithmetic identity; None preserves legacy model behavior.
    cache_numerical_law: Option<u64>,
    pub caches: Vec<LayerCache>,
    /// Every token fed into the caches, in order (the committed conversation).
    pub fed: Vec<u32>,
    /// Last-position hidden state of the most recent feed (diagnostics).
    pub last_mixed: Option<Array>,
    /// MTP head priming store: every committed token and its hyper stream
    /// `multi` (paired as `(token[t+1], multi[t])`, matching the head's own
    /// convention). The head's caches are reset per turn and primed from this,
    /// which keeps the per-turn bookkeeping trivial.
    pub mtp_tokens: Vec<u32>,
    pub mtp_multi: Vec<Array>,
    /// The MTP head cache offset that corresponds to `mtp_tokens` at the end of
    /// the last turn (the number of committed history pairs it holds). Used to
    /// persist the head across turns instead of re-priming the whole history.
    pub mtp_head_len: usize,
    /// GLM 5.3's target-conditioned MTP private compact-attention cache.
    glm_mtp_cache: IndexerTape,
    /// First absolute row visible to that private cache for this session.
    glm_mtp_min_position: Option<usize>,
    /// Pending proposal and already sampled boundary token between scheduler cycles.
    glm_mtp_pending: Option<TargetMtpPending>,
    /// Whether a verified EOS is retained in the target checkpoint.
    target_mtp_eos_policy: TargetMtpEosPolicy,
    /// GLM MTP snapshots aligned by index with `snapshots`.
    glm_mtp_snapshots: Vec<(TapeSnapshot, Option<usize>)>,
    /// Candidate next-token distributions at committed and verified GLM MTP
    /// token boundaries. Test-only so full-vocabulary DS4 comparisons do not
    /// add work to normal inference.
    #[cfg(test)]
    glm_mtp_target_logits: Vec<(usize, Vec<f32>)>,
    /// Optional test-only callback for full cache snapshots at committed MTP
    /// boundaries. This lets long-run comparisons flush one state at a time.
    #[cfg(test)]
    glm_mtp_state_capture:
        Option<Box<dyn FnMut(usize, &[LayerCache], &IndexerTape) -> anyhow::Result<()>>>,
    /// Prefix snapshots `(boundary, msgs_len, head_offset, per-layer state)`
    /// taken at the start of each turn. On a divergent request the server can
    /// rewind to the largest boundary at or below the common prefix and prefill
    /// only the tail.
    pub snapshots: Vec<(usize, usize, usize, Vec<LayerSnapshot>)>,
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
struct TargetMtpPending {
    fed_len: usize,
    carry: u32,
    draft_parent: u32,
    draft: Option<u32>,
    draft_logits: Option<Array>,
    sampling_mode: TargetMtpSamplingMode,
    min_position: usize,
}

/// A per-layer prefix snapshot: the full-attention live length, or the GDN
/// recurrent state (short-conv, fp32 SSM, PLE conv).
#[derive(Clone)]
pub struct LayerSnapshot {
    pub full_offset: Option<usize>,
    pub gdn: Option<(Option<Array>, Option<Array>, Option<Array>)>,
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

/// Keep at most this many prefix snapshots (the GDN state is ~113 MB each).
const SNAPSHOT_KEEP: usize = 4;

impl Session {
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
        self.glm_mtp_pending = None;
        self.snapshots.clear();
        self.glm_mtp_snapshots.clear();
    }

    fn ensure_cache_numerical_law(&self, tower: &dyn LanguageModel) -> anyhow::Result<()> {
        self.ensure_valid()?;
        ensure_matching_cache_law(self.cache_numerical_law, tower.cache_numerical_law())
    }

    /// A fresh conversation. Resets the PLE n-gram context so the first suffix
    /// is hashed from the default (EOS-filled) context, not a leftover one.
    pub fn new(tower: &mut dyn LanguageModel) -> Self {
        tower.clear_context();
        Session {
            valid: true,
            cache_numerical_law: tower.cache_numerical_law(),
            caches: tower.new_caches(),
            fed: Vec::new(),
            last_mixed: None,
            mtp_tokens: Vec::new(),
            mtp_multi: Vec::new(),
            mtp_head_len: 0,
            glm_mtp_cache: IndexerTape::new(),
            glm_mtp_min_position: None,
            glm_mtp_pending: None,
            target_mtp_eos_policy: TargetMtpEosPolicy::default(),
            glm_mtp_snapshots: Vec::new(),
            #[cfg(test)]
            glm_mtp_target_logits: Vec::new(),
            #[cfg(test)]
            glm_mtp_state_capture: None,
            snapshots: Vec::new(),
            snapshots_enabled: false,
        }
    }

    #[cfg(test)]
    pub(crate) fn glm_mtp_cache_for_test(&self) -> &IndexerTape {
        &self.glm_mtp_cache
    }

    #[cfg(test)]
    fn record_glm_mtp_target_logits_for_test(
        &mut self,
        token_count: usize,
        logits: &Array,
    ) -> anyhow::Result<()> {
        let values = logits
            .as_dtype(lisa_mlx::Dtype::Float32)?
            .contiguous()?
            .reshape(&[logits.size() as i32])?
            .to_vec1::<f32>()?;
        self.glm_mtp_target_logits.push((token_count, values));
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn glm_mtp_target_logits_for_test(&self) -> &[(usize, Vec<f32>)] {
        &self.glm_mtp_target_logits
    }

    #[cfg(test)]
    pub(crate) fn set_glm_mtp_state_capture_for_test<F>(&mut self, capture: F)
    where
        F: FnMut(usize, &[LayerCache], &IndexerTape) -> anyhow::Result<()> + 'static,
    {
        self.glm_mtp_state_capture = Some(Box::new(capture));
    }

    #[cfg(test)]
    fn capture_glm_mtp_state_for_test(&mut self) -> anyhow::Result<()> {
        let token_count = self.fed.len();
        let caches = &self.caches;
        let mtp_cache = &self.glm_mtp_cache;
        if let Some(capture) = self.glm_mtp_state_capture.as_mut() {
            capture(token_count, caches, mtp_cache)?;
        }
        Ok(())
    }

    /// A session resumed from a cross-request prefix-cache entry: the caches
    /// are seeded with the entry's state at `prefix_len` fed tokens, `fed`
    /// matches it, and the PLE host tail is set so the next suffix hashes
    /// from the right context. The caller prefills only `tokens[prefix_len..]`.
    pub fn from_prefix(
        tower: &mut dyn LanguageModel,
        entry_layers: &[crate::core::cache::LayerState],
        prompt: &[u32],
        prefix_len: usize,
    ) -> Self {
        let mut sess = Session::new(tower);
        for (c, st) in sess.caches.iter_mut().zip(entry_layers.iter()) {
            c.restore_prefix(st);
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
        let snaps: Vec<LayerSnapshot> = self
            .caches
            .iter()
            .map(|c| match c {
                LayerCache::Full(f) => LayerSnapshot {
                    full_offset: Some(f.offset),
                    gdn: None,
                },
                LayerCache::Linear(g) => LayerSnapshot {
                    full_offset: None,
                    gdn: Some(g.snapshot_state()),
                },
            })
            .collect();
        self.snapshots
            .push((self.fed.len(), msgs_len, head_offset, snaps));
        self.glm_mtp_snapshots
            .push((self.glm_mtp_cache.snapshot(), self.glm_mtp_min_position));
        if self.snapshots.len() > SNAPSHOT_KEEP {
            let drop = self.snapshots.len() - SNAPSHOT_KEEP;
            self.snapshots.drain(0..drop);
            self.glm_mtp_snapshots.drain(0..drop);
        }
    }

    /// Rewind to the largest snapshot boundary `<= cp` (a divergent request).
    /// Returns that boundary, or `None` when no snapshot is at or below `cp`.
    pub fn restore_snapshot(
        &mut self,
        tower: &mut dyn LanguageModel,
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
        let glm_mtp_min_position = self.glm_mtp_snapshots[idx].1;
        for (c, s) in self.caches.iter_mut().zip(snaps.iter()) {
            match (c, s) {
                (
                    LayerCache::Full(f),
                    LayerSnapshot {
                        full_offset: Some(o),
                        ..
                    },
                ) => f.restore_offset(*o),
                (LayerCache::Linear(g), LayerSnapshot { gdn: Some(st), .. }) => g.restore_state(st),
                _ => {}
            }
        }
        self.fed.truncate(boundary);
        self.snapshots.truncate(idx + 1);
        self.glm_mtp_snapshots.truncate(idx + 1);
        // Rewind the head's KV cache to the snapshot too, so no re-prime is
        // needed (the history pairs are its offset + 1).
        tower.drafter_restore_offset(head);
        self.truncate_head(head + 1);
        self.mtp_head_len = head;
        // Restore the private GLM MTP window with the target prefix. Keeping
        // its tape in the same snapshot prevents a divergent turn from
        // reusing rows produced by the abandoned suffix.
        self.glm_mtp_cache.restore(&self.glm_mtp_snapshots[idx].0);
        self.glm_mtp_min_position = glm_mtp_min_position;
        self.glm_mtp_pending = None;
        Some(boundary)
    }

    /// Truncate the MTP head's stored token/multi history to `boundary` rows.
    fn truncate_head(&mut self, boundary: usize) {
        self.mtp_tokens.truncate(boundary);
        let mut out: Vec<Array> = Vec::new();
        let mut rem = boundary;
        for chunk in &self.mtp_multi {
            if rem == 0 {
                break;
            }
            let sh = chunk.shape();
            let rows: usize = sh[..sh.len() - 1].iter().map(|&d| d as usize).product();
            let d = chunk.dim(-1);
            if rows <= rem {
                out.push(chunk.clone());
                rem -= rows;
            } else {
                if let Ok(r) = chunk.reshape(&[1, rows as i32, d]) {
                    if let Ok(sliced) = r.index((.., 0..rem as i32, ..)).contiguous() {
                        out.push(sliced);
                    }
                }
                rem = 0;
            }
        }
        self.mtp_multi = out;
    }

    /// Prefill `tokens` (the appended suffix) and return the logits at the last
    /// position. Long suffixes are fed in windows (bounded activations).
    pub fn feed(&mut self, tower: &mut dyn LanguageModel, tokens: &[u32]) -> anyhow::Result<Array> {
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
        self.glm_mtp_pending = None;
        let result = (|| -> anyhow::Result<Array> {
            let last = tower.prefill(tokens, &mut self.caches)?;
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
        tower: &mut dyn LanguageModel,
        token: u32,
    ) -> anyhow::Result<Array> {
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
        self.glm_mtp_pending = None;
        let result = (|| -> anyhow::Result<Array> {
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

    /// [`Session::feed`] but also returning the hyper stream `multi`
    /// `[1, len(tokens), hc*H]` for the fed tokens (needed by the MTP head).
    pub fn feed_multi(
        &mut self,
        tower: &mut dyn LanguageModel,
        tokens: &[u32],
    ) -> anyhow::Result<(Array, Array)> {
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
        self.glm_mtp_pending = None;
        let result = (|| -> anyhow::Result<(Array, Array)> {
            let (last, multi) = tower.prefill_multi(tokens, &mut self.caches)?;
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

    /// Feed `tokens`, then sample until EOS or `max_tokens`.
    ///
    /// The final EOS token is emitted but not fed (matching `generate`), so the
    /// next turn appends it as part of its suffix.
    pub fn generate(
        &mut self,
        tower: &mut dyn LanguageModel,
        tokens: &[u32],
        max_tokens: usize,
        sampler: &mut Sampler,
        pld: usize,
        mut after_prefill: Option<&mut dyn FnMut(&[LayerCache])>,
        on_token: &mut dyn FnMut(u32) -> anyhow::Result<()>,
    ) -> anyhow::Result<Vec<u32>> {
        let logits = self.feed(tower, tokens)?;
        // Clean commit boundary (prompt fed, nothing decoded yet): the
        // cross-request prefix cache snapshots here.
        if let Some(cb) = after_prefill.as_deref_mut() {
            cb(&self.caches);
        }
        let mut token = if tower.has_target_mtp() {
            sampler.draw_ds4(&logits, &self.fed)?
        } else {
            sampler.draw(&logits, &self.fed)?
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
            let (g, stats) = crate::core::generate::pld_decode(
                tower,
                &mut self.caches,
                &mut committed,
                token,
                max_tokens,
                out.len(),
                pld,
                if sampler.greedy() {
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
                !sampler.greedy()
            );
            self.fed = committed;
            return Ok(out);
        }
        while out.len() < max_tokens && !is_eos(token) {
            let logits = self.decode_token(tower, token)?;
            token = if tower.has_target_mtp() {
                sampler.draw_ds4(&logits, &self.fed)?
            } else {
                sampler.draw(&logits, &self.fed)?
            };
            out.push(token);
            on_token(token)?;
        }
        Ok(out)
    }

    /// Serial target decode with a live sampling policy for the token after
    /// each emitted token. GLM serving uses this for DSML structural syntax,
    /// which ds4 samples greedily between parameter payloads.
    pub fn generate_with_sampling_mode(
        &mut self,
        tower: &mut dyn LanguageModel,
        tokens: &[u32],
        max_tokens: usize,
        sampler: &mut Sampler,
        mut after_prefill: Option<&mut dyn FnMut(&[LayerCache])>,
        on_token: &mut dyn FnMut(u32) -> anyhow::Result<TargetMtpSamplingMode>,
    ) -> anyhow::Result<Vec<u32>> {
        self.ensure_cache_numerical_law(tower)?;
        if max_tokens == 0 {
            return Ok(Vec::new());
        }
        let mut logits = self.feed(tower, tokens)?;
        if let Some(callback) = after_prefill.as_deref_mut() {
            callback(&self.caches);
        }
        let mut mode = TargetMtpSamplingMode::Configured;
        let mut token = if mode == TargetMtpSamplingMode::Greedy {
            crate::models::speculate::argmax_id(&logits)?
        } else if tower.has_target_mtp() {
            sampler.draw_ds4(&logits, &self.fed)?
        } else {
            sampler.draw(&logits, &self.fed)?
        };
        let mut generated = Vec::with_capacity(max_tokens);
        generated.push(token);
        mode = on_token(token)?;
        while generated.len() < max_tokens && !is_eos(token) {
            logits = self.decode_token(tower, token)?;
            token = if mode == TargetMtpSamplingMode::Greedy {
                crate::models::speculate::argmax_id(&logits)?
            } else if tower.has_target_mtp() {
                sampler.draw_ds4(&logits, &self.fed)?
            } else {
                sampler.draw(&logits, &self.fed)?
            };
            generated.push(token);
            mode = on_token(token)?;
        }
        Ok(generated)
    }
}

/// Verify that feeding a token sequence in chunks through one [`Session`] gives
/// the same logits as feeding it in a single prefill with fresh caches.
///
/// Three passes are run: pass 1 is discarded (the first forwards in a process
/// can differ while pages are lazily faulted/resident), pass 2 is the measured
/// incremental run, and pass 3 is the full single-prefill reference.
/// Returns `(argmax_equal, max_abs_logit_diff, max_abs_hidden_diff)`.
pub fn verify_incremental(
    tower: &mut dyn LanguageModel,
    tokens: &[u32],
    chunk: usize,
) -> anyhow::Result<(bool, f32, f32)> {
    let saved = tower.context_tails();

    let run_incremental = |tower: &mut dyn LanguageModel| -> anyhow::Result<(Array, Array)> {
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

    let run_full = |tower: &mut dyn LanguageModel| -> anyhow::Result<(Array, Array)> {
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
        let a = a.as_dtype(lisa_mlx::Dtype::Float32)?;
        let b = b.as_dtype(lisa_mlx::Dtype::Float32)?;
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
    let am_inc = lisa_mlx::ops::indexing::argmax(&inc, None)?.item_cast::<i32>();
    let am_full = lisa_mlx::ops::indexing::argmax(&full, None)?.item_cast::<i32>();
    tower.set_context_tails(saved);
    Ok((am_inc == am_full, d_log, d_hid))
}

/// Depth selection policy for the speculative loop.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MtpDepth {
    /// Fixed draft depth 2..=6 for independent Qwen draft chains. GLM's
    /// target-conditioned MTP uses its native single-step verify cycle.
    Fixed(usize),
    /// EV auto for independent draft chains. Target-conditioned GLM MTP keeps
    /// its native one-step cycle because the Qwen EV table does not price it.
    Auto,
}

/// One greedy serial decode step over the live caches: the MTP loop's serial
/// fallback. Returns the argmax token and the last multi row (`[1, D]`), so the
/// caller can hand the pair back to the draft chain.
fn serial_step(
    tower: &mut dyn LanguageModel,
    caches: &mut Vec<LayerCache>,
    tail: &[i64],
    carry_token: u32,
    stochastic: Option<&mut crate::core::sampler::Sampler>,
    // Absolute position of the token this step predicts — the serial non-greedy
    // path keys its Gumbel draw on it.
    position: u64,
) -> anyhow::Result<(u32, Array)> {
    tower.set_context_tails(vec![tail.to_vec()]);
    let arr = Array::from_slice(&[carry_token as i32], &[1i32, 1]);
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
            .draw_at(&logits, position)
            .unwrap_or(crate::models::speculate::argmax_id(&logits)?),
        None => crate::models::speculate::argmax_id(&logits)?,
    };
    let row = multi.index((.., 0, ..)).contiguous()?;
    Ok((t, row))
}

fn sampling_history_with_suffix(history: &[u32], window: usize, suffix: &[u32]) -> Vec<u32> {
    let keep = window.saturating_sub(suffix.len());
    let start = history.len().saturating_sub(keep);
    let mut out = history[start..].to_vec();
    out.extend_from_slice(suffix);
    out
}

fn target_mtp_uses_exact_sampling(sampler: Option<&Sampler>, mode: TargetMtpSamplingMode) -> bool {
    mode == TargetMtpSamplingMode::Configured
        && sampler.is_some_and(|sampler| sampler.mtp_exact_sampling && sampler.temperature > 0.0)
}

/// A configured positive-temperature draw must still advance DS4's RNG, even
/// with top-k=1. A repetition penalty can also change the unmodified argmax.
fn target_mtp_draw_reuses_argmax(sampler: Option<&Sampler>, mode: TargetMtpSamplingMode) -> bool {
    mode == TargetMtpSamplingMode::Greedy
        || sampler.is_none_or(|sampler| {
            sampler.temperature <= 0.0 && sampler.repetition_penalty == 1.0
        })
}

fn draw_target_mtp_token(
    sampler: Option<&mut Sampler>,
    logits: &Array,
    history: &[u32],
    mode: TargetMtpSamplingMode,
) -> anyhow::Result<u32> {
    if mode == TargetMtpSamplingMode::Greedy {
        return crate::models::speculate::argmax_id(logits);
    }
    match sampler {
        Some(sampler) => sampler.draw_ds4(logits, history),
        None => crate::models::speculate::argmax_id(logits),
    }
}

impl Session {
    fn restore_target_verify_to_first(
        &mut self,
        tower: &mut dyn LanguageModel,
        before: &[LayerRollbackState],
        first: u32,
    ) -> anyhow::Result<Array> {
        let _trace = lisa_mlx::trace::span("glm53.mtp.rollback_replay");
        self.restore_target_verify_boundary(before)?;
        #[cfg(test)]
        let replay_capture_path = std::env::var("LISA_GLM53_MTP_REPLAY_CAPTURE_PATH").ok();
        #[cfg(test)]
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
        let replay = Array::from_slice(&[first as i32], &[1i32, 1]);
        let (mixed, multi) = tower.forward_capture(&replay, Some(&mut self.caches), false)?;
        #[cfg(test)]
        if replay_capture {
            tower.flush_debug_capture_for_test(
                replay_capture_path
                    .as_deref()
                    .expect("capture path checked above"),
            )?;
        }
        self.fed.push(first);
        self.last_mixed = Some(mixed.index((.., 0, ..)).contiguous()?);
        Ok(multi)
    }

    fn restore_target_verify_boundary(
        &mut self,
        before: &[LayerRollbackState],
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.caches.len() == before.len(),
            "target cache layout changed during GLM MTP verify"
        );
        for (cache, state) in self.caches.iter_mut().zip(before) {
            cache.restore_rollback(state)?;
        }
        Ok(())
    }

    fn advance_target_mtp_cache(
        &mut self,
        tower: &mut dyn LanguageModel,
        row0_multi: &Array,
        row1_multi: &Array,
        draft: u32,
        next: u32,
        position: usize,
        min_position: usize,
    ) -> anyhow::Result<Array> {
        let _trace = lisa_mlx::trace::span("glm53.mtp.cache_catchup");
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
        tower: &mut dyn LanguageModel,
        before: &[LayerRollbackState],
        first: u32,
        draft: u32,
        accepted: bool,
        replacement: Option<u32>,
        row0_multi: &Array,
        row1_multi: &Array,
        row0_logits: &Array,
        row1_logits: &Array,
        position: usize,
        min_position: usize,
    ) -> anyhow::Result<Array> {
        if accepted {
            let next = crate::models::speculate::argmax_id(row1_logits)?;
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
            let first_ids = Array::from_slice(&[first as i32], &[1i32, 1]);
            let _ = tower.forward_capture(&first_ids, Some(&mut self.caches), false)?;
            let replacement_ids = Array::from_slice(&[replacement as i32], &[1i32, 1]);
            let (mixed, multi) =
                tower.forward_capture(&replacement_ids, Some(&mut self.caches), false)?;
            if !is_eos(replacement) {
                let next = crate::models::speculate::argmax_id(&tower.head(&mixed)?)?;
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
            let next = crate::models::speculate::argmax_id(row0_logits)?;
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
        #[cfg(test)]
        self.record_glm_mtp_target_logits_for_test(self.fed.len(), &replay_logits)?;
        Ok(replay_logits)
    }

    fn commit_target_mtp_tail(
        &mut self,
        tower: &mut dyn LanguageModel,
        token: u32,
        position: usize,
        min_position: usize,
        prime_mtp: bool,
    ) -> anyhow::Result<()> {
        let _trace = lisa_mlx::trace::span("glm53.mtp.commit_tail");
        anyhow::ensure!(
            position == self.fed.len(),
            "GLM MTP tail position {position} does not match fed boundary {}",
            self.fed.len()
        );
        anyhow::ensure!(
            position < tower.max_position_embeddings(),
            "context overflow before GLM MTP tail commit"
        );
        let before: Vec<_> = self
            .caches
            .iter()
            .map(|cache| cache.snapshot_rollback())
            .collect();
        let mtp_before = self.glm_mtp_cache.snapshot();
        let result: anyhow::Result<Array> = (|| -> anyhow::Result<Array> {
            let ids = Array::from_slice(&[token as i32], &[1i32, 1]);
            let (mixed, multi) = tower.forward_capture(&ids, Some(&mut self.caches), false)?;
            if prime_mtp {
                let next = crate::models::speculate::argmax_id(&tower.head(&mixed)?)?;
                let _ = tower.target_mtp_step(
                    &mut self.glm_mtp_cache,
                    &multi,
                    next,
                    position,
                    min_position,
                )?;
            }
            Ok(mixed.index((.., 0, ..)).contiguous()?)
        })();
        match result {
            Ok(mixed) => {
                anyhow::ensure!(
                    self.caches.len() == before.len(),
                    "target cache layout changed while committing the final GLM MTP token"
                );
                self.fed.push(token);
                #[cfg(test)]
                if prime_mtp {
                    let logits = tower.head(&mixed)?;
                    self.record_glm_mtp_target_logits_for_test(position + 1, &logits)?;
                }
                #[cfg(test)]
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
                    cache.restore_rollback(state)?;
                }
                self.glm_mtp_cache.restore(&mtp_before);
                Err(error)
            }
        }
    }

    /// Finish a terminal target draw before notifying the caller. A callback
    /// stop then leaves a reusable committed boundary; model failures do not.
    fn finish_target_mtp_tail(
        &mut self,
        tower: &mut dyn LanguageModel,
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
        tower: &mut dyn LanguageModel,
        tokens: &[u32],
        remaining_tokens: usize,
        stochastic: Option<&mut Sampler>,
        on_token: &mut dyn FnMut(u32) -> anyhow::Result<TargetMtpSamplingMode>,
    ) -> anyhow::Result<Vec<u32>> {
        anyhow::ensure!(
            tower.has_target_mtp(),
            "model has no target-conditioned MTP head"
        );
        let _trace = lisa_mlx::trace::span_detail("glm53.mtp.cycle", self.fed.len() as u64);
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
        tower: &mut dyn LanguageModel,
        tokens: &[u32],
        max_tokens: usize,
        after_prefill: Option<&mut dyn FnMut(&[LayerCache])>,
        stochastic: Option<&mut Sampler>,
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
        tower: &mut dyn LanguageModel,
        tokens: &[u32],
        max_tokens: usize,
        after_prefill: Option<&mut dyn FnMut(&[LayerCache])>,
        stochastic: Option<&mut Sampler>,
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
        tower: &mut dyn LanguageModel,
        tokens: &[u32],
        max_tokens: usize,
        mut after_prefill: Option<&mut dyn FnMut(&[LayerCache])>,
        mut stochastic: Option<&mut Sampler>,
        on_token: &mut dyn FnMut(u32) -> anyhow::Result<TargetMtpSamplingMode>,
        single_cycle: bool,
        committed_callback_stop: &mut bool,
    ) -> anyhow::Result<Vec<u32>> {
        use lisa_mlx::ops::indexing::IndexOp;
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
                let _trace = lisa_mlx::trace::span("glm53.mtp.prefill");
                let logits = self.feed(tower, tokens)?;
                if let Some(cb) = after_prefill.as_deref_mut() {
                    cb(&self.caches);
                }
                logits
            };
            let mut sampling_mode = TargetMtpSamplingMode::Configured;
            let first = draw_target_mtp_token(
                stochastic.as_deref_mut(),
                &logits,
                &self.fed,
                sampling_mode,
            )?;
            #[cfg(test)]
            let first_decode_capture_path =
                std::env::var("LISA_GLM53_MTP_FIRST_DECODE_CAPTURE").ok();
            #[cfg(test)]
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
            let first_arr = Array::from_slice(&[first as i32], &[1i32, 1]);
            let (first_mixed, first_multi) = {
                let _trace = lisa_mlx::trace::span("glm53.mtp.target_seed_forward");
                tower.forward_capture(&first_arr, Some(&mut self.caches), false)?
            };
            self.last_mixed = Some(first_mixed.index((.., 0, ..)).contiguous()?);
            self.fed.push(first);
            let first_logits = {
                let _trace = lisa_mlx::trace::span("glm53.mtp.target_seed_head");
                tower.head(&first_mixed)?
            };
            #[cfg(test)]
            self.record_glm_mtp_target_logits_for_test(self.fed.len(), &first_logits)?;
            #[cfg(test)]
            if let Some(path) = first_decode_capture_path.as_deref() {
                tower.flush_debug_capture_for_test(path)?;
            }
            let carry = draw_target_mtp_token(
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
                    crate::models::speculate::argmax_id(&first_logits)?
                };
            // This step also writes the compact-cache prefix needed to continue the
            // session. Keep it even when the sampled boundary invalidates its draft.
            let draft_logits = {
                let _trace = lisa_mlx::trace::span("glm53.mtp.proposal_step");
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
            let draft = Some(crate::models::speculate::argmax_id(
                draft_logits.as_ref().expect("initial MTP logits"),
            )?);
            #[cfg(test)]
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
                let ids = Array::from_slice(&[carry as i32], &[1i32, 1]);
                let (mixed, multi) = {
                    let _trace = lisa_mlx::trace::span("glm53.mtp.opportunistic_fallback");
                    tower.forward_capture(&ids, Some(&mut self.caches), false)?
                };
                let logits = tower.head(&mixed)?;
                self.fed.push(carry);
                self.last_mixed = Some(mixed.index((.., 0, ..)).contiguous()?);
                draft_parent = crate::models::speculate::argmax_id(&logits)?;
                draft_logits = Some(tower.target_mtp_step(
                    &mut self.glm_mtp_cache,
                    &multi,
                    draft_parent,
                    position,
                    min_position,
                )?);
                draft = Some(crate::models::speculate::argmax_id(
                    draft_logits.as_ref().expect("fallback MTP logits"),
                )?);
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
                #[cfg(test)]
                self.capture_glm_mtp_state_for_test()?;
                carry = draw_target_mtp_token(
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
            let before: Vec<_> = self.caches.iter().map(|c| c.snapshot_rollback()).collect();
            let verify_ids = Array::from_slice(&[carry as i32, draft_token as i32], &[1i32, 2]);
            let verify_result = (|| -> anyhow::Result<_> {
                let _trace = lisa_mlx::trace::span("glm53.mtp.verify_forward");
                let (verify_mixed, verify_multi) =
                    tower.forward_capture(&verify_ids, Some(&mut self.caches), false)?;
                #[cfg(test)]
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
                    let _trace = lisa_mlx::trace::span("glm53.mtp.verify_head");
                    let row0_mixed = verify_mixed
                        .index((.., 0, ..))
                        .expand_dims(1)?
                        .contiguous()?;
                    let row1_mixed = verify_mixed
                        .index((.., 1, ..))
                        .expand_dims(1)?
                        .contiguous()?;
                    let row0_logits = tower.head(&row0_mixed)?;
                    row0_logits.eval()?;
                    #[cfg(test)]
                    if let Some((prefix, capture_index)) = verify_capture.as_ref() {
                        capture_glm_mtp_verify_logits(
                            prefix,
                            *capture_index,
                            &row0_logits,
                            verify_pos + 1,
                        )?;
                    }
                    #[cfg(test)]
                    self.record_glm_mtp_target_logits_for_test(verify_pos + 1, &row0_logits)?;
                    let row1_logits = tower.head(&row1_mixed)?;
                    row1_logits.eval()?;
                    #[cfg(test)]
                    if let Some((prefix, capture_index)) = verify_capture.as_ref() {
                        capture_glm_mtp_verify_logits(
                            prefix,
                            *capture_index,
                            &row1_logits,
                            verify_pos + 2,
                        )?;
                    }
                    #[cfg(test)]
                    self.record_glm_mtp_target_logits_for_test(verify_pos + 2, &row1_logits)?;
                    (
                        row0_logits.index((.., 0, ..)).contiguous()?,
                        row1_logits.index((.., 0, ..)).contiguous()?,
                    )
                };
                let row0_multi = verify_multi
                    .index((.., 0, ..))
                    .expand_dims(1)?
                    .contiguous()?;
                let row1_multi = verify_multi
                    .index((.., 1, ..))
                    .expand_dims(1)?
                    .contiguous()?;
                let row1_mixed = verify_mixed.index((.., 1, ..)).contiguous()?;
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
                        sampling_history_with_suffix(&self.fed, s.penalty_window, &[carry]);
                    s.verify_draft_with_history(&row0_logits, draft_token, &row0_history)
                        .ok_or_else(|| anyhow::anyhow!("GLM MTP verification sampling failed"))?
                }
                Some(_) | None => {
                    let target = crate::models::speculate::argmax_id(&row0_logits)?;
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
                    .is_some_and(|sampler| sampler.mtp_exact_sampling && sampler.temperature > 0.0);
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
                draft_parent = crate::models::speculate::argmax_id(&replay_logits)?;
                draft = None;
                draft_logits = None;
                if generated.len() >= max_tokens {
                    return Ok(generated);
                }
                carry = draw_target_mtp_token(
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
            let _decision_trace = lisa_mlx::trace::span_detail(decision, decision_detail);
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
                        #[cfg(test)]
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
                    let replacement_ids = Array::from_slice(&[actual as i32], &[1i32, 1]);
                    let (replacement_mixed, replacement_multi) = {
                        let _trace = lisa_mlx::trace::span("glm53.mtp.replacement_forward");
                        tower.forward_capture(&replacement_ids, Some(&mut self.caches), false)?
                    };
                    let replacement_logits = tower.head(&replacement_mixed)?;
                    self.fed.push(actual);
                    #[cfg(test)]
                    self.record_glm_mtp_target_logits_for_test(
                        self.fed.len(),
                        &replacement_logits,
                    )?;
                    self.last_mixed = Some(replacement_mixed.index((.., 0, ..)).contiguous()?);
                    draft_parent = crate::models::speculate::argmax_id(&replacement_logits)?;
                    draft_logits = Some(tower.target_mtp_step(
                        &mut self.glm_mtp_cache,
                        &replacement_multi,
                        draft_parent,
                        replacement_position,
                        min_position,
                    )?);
                    draft = Some(crate::models::speculate::argmax_id(
                        draft_logits.as_ref().expect("replacement MTP logits"),
                    )?);
                    #[cfg(test)]
                    self.capture_glm_mtp_state_for_test()?;
                    carry = draw_target_mtp_token(
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
                    #[cfg(test)]
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
                    draft = Some(crate::models::speculate::argmax_id(
                        draft_logits.as_ref().expect("replay MTP logits"),
                    )?);
                    #[cfg(test)]
                    self.capture_glm_mtp_state_for_test()?;
                    carry = draw_target_mtp_token(
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
                #[cfg(test)]
                self.capture_glm_mtp_state_for_test()?;
                return Ok(generated);
            }
            sampling_mode = match on_token(draft_token) {
                Ok(mode) => mode,
                Err(error) => {
                    let next = crate::models::speculate::argmax_id(&row1_logits)?;
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
                let next = crate::models::speculate::argmax_id(&row1_logits)?;
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
                #[cfg(test)]
                self.capture_glm_mtp_state_for_test()?;
                return Ok(generated);
            }

            self.fed.push(carry);
            self.fed.push(draft_token);
            self.last_mixed = Some(row1_mixed);
            let remaining = max_tokens - generated.len();
            let next_parent = crate::models::speculate::argmax_id(&row1_logits)?;
            let next = if target_mtp_draw_reuses_argmax(stochastic.as_deref(), sampling_mode) {
                next_parent
            } else {
                draw_target_mtp_token(
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
            #[cfg(test)]
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
            draft = Some(crate::models::speculate::argmax_id(
                draft_logits.as_ref().expect("accepted MTP logits"),
            )?);
            yield_cycle!();
        }
        Ok(generated)
    }

    /// Speculative generation. Independent Qwen MTP uses the requested chain
    /// depth; GLM uses its target-conditioned one-step cycle.
    pub fn generate_mtp(
        &mut self,
        tower: &mut dyn LanguageModel,
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
        tower: &mut dyn LanguageModel,
        tokens: &[u32],
        max_tokens: usize,
        depth_sel: MtpDepth,
        after_prefill: Option<&mut dyn FnMut(&[LayerCache])>,
        // Non-greedy verification. `None` keeps the certified greedy path.
        // `Some(sm)` runs the same Leviathan-exact scheme as `pld_decode`: the
        // MTP proposal is a POINT MASS at the head's argmax (`q = delta`), so a
        // draft token is accepted with `p_target[draft]` and the first rejection
        // emits a draw from the residual. Serial steps sample too.
        stochastic: Option<&mut crate::core::sampler::Sampler>,
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

    /// [`Session::generate_mtp_depth`] with a live policy for the next target
    /// sample. The callback returns the mode derived from the just-emitted
    /// token; GLM MTP uses it at exact-sampling boundaries.
    pub fn generate_mtp_depth_with_sampling_mode(
        &mut self,
        tower: &mut dyn LanguageModel,
        tokens: &[u32],
        max_tokens: usize,
        depth_sel: MtpDepth,
        mut after_prefill: Option<&mut dyn FnMut(&[LayerCache])>,
        mut stochastic: Option<&mut crate::core::sampler::Sampler>,
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
        let mut on_token_plain = |token| on_token(token).map(|_| ());
        use crate::core::round_cost::{
            DepthController, ModelCost, PROBE_TICKS, PROBE_WARM, RoundCostTable, probe_warm_stats,
        };
        use lisa_mlx::ops::indexing::{IndexOp, argmax_axis};
        anyhow::ensure!(tower.has_drafter(), "the checkpoint has no MTP head");
        let is_auto = matches!(depth_sel, MtpDepth::Auto);
        let mut depth: usize = match depth_sel {
            // Chain depth follows the copy cap (`LISA_COPY_LEN_MAX`): a fixed
            // depth past 6 runs chain-only wide rounds (S = depth + 1 > 7, the
            // tiled verify lane — specs/08's wide-lane test). Default unchanged.
            MtpDepth::Fixed(d) => d.clamp(2, crate::core::copy_draft::copy_len_max()),
            MtpDepth::Auto => 2, // re-picked by the controller every round
        };

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
        let first = crate::models::speculate::argmax_id(&logits)?;
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
        let mut all_multi: Vec<Array> = self.mtp_multi.clone();
        all_multi.push(multi.index((.., 0..t_s, ..)).contiguous()?);
        let init_multis = concat_chunks(&all_multi)?;
        let mut init_tokens: Vec<i32> = history[1..].iter().map(|&t| t as i32).collect();
        init_tokens.push(first as i32);
        // The head already holds `head_off` pairs; feed the rest.
        anyhow::ensure!(
            head_off < init_tokens.len(),
            "MTP head offset {head_off} exceeds history pairs {}",
            init_tokens.len()
        );
        let init_multis = init_multis
            .index((.., head_off as i32.., ..))
            .contiguous()?;
        let init_tokens: Vec<i32> = init_tokens[head_off..].to_vec();

        // Prime the head in windows. A single whole-history forward sits at
        // `offset == 0`, where the head's attention takes the DENSE fallback and
        // materialises `[S, kv]` masks — at 100K tokens that is tens of GB and
        // fails the buffer allocation. Windows keep the head's cache advancing,
        // so from the second window on the block-sparse QSA path serves it.
        const PRIME_CHUNK: usize = 2048;
        let _trace_prime = lisa_mlx::trace::span_detail("mtp.prime", init_tokens.len() as u64);
        let mut primed: Option<(Array, Array)> = None;
        {
            let toks = &init_tokens;
            let mul = &init_multis;
            let mut i = 0usize;
            while i < toks.len() {
                let n = (toks.len() - i).min(PRIME_CHUNK);
                let t = Array::from_slice(&toks[i..i + n], &[1i32, n as i32]);
                let m = mul.index((.., i as i32..(i + n) as i32, ..)).contiguous()?;
                primed = Some(tower.draft_step(&t, &m)?);
                i += n;
                lisa_mlx::memory::trim_cache();
            }
        }

        let mut generated: Vec<u32> = vec![first];
        on_token_plain(first)?;
        let mut store_tokens: Vec<u32> = vec![first];
        let mut store_multi: Vec<Array> = vec![multi.index((.., t_s - 1, ..)).contiguous()?];

        let mut backlog: Vec<(u32, Array)> = Vec::new();
        // Consecutive serial fallback ticks form a block; its first
        // PROBE_WARM ticks are transitions (MTP_ADAPTIVE_PROBE_WARM) and never
        // fold.
        let mut serial_block: u32 = 0;
        let mut carry_token: u32 = first;
        let mut carry_multi: Array = multi.index((.., t_s - 1, ..)).contiguous()?;
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
                cap_row,
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
        let two_chunk_on = is_auto && crate::core::round_cost::two_chunk_enabled();
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
                depth = c.pick(self.fed.len(), max_tokens - generated.len());
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
                        crate::core::copy_draft::copy_len_max(),
                    );
                    if hit.is_none() {
                        // Snapshot the GDN state so an EOS step can be un-fed
                        // (the EOS is emitted but never committed, matching the
                        // spec path and `Session::generate`).
                        let snaps: Vec<_> = self
                            .caches
                            .iter()
                            .map(|c| match c {
                                LayerCache::Linear(g) => Some(g.snapshot_state()),
                                _ => None,
                            })
                            .collect();
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
                                match (c, s) {
                                    (LayerCache::Full(f), _) => f.trim(1),
                                    (LayerCache::Linear(g), Some(st)) => g.restore_state(st),
                                    _ => {}
                                }
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
                    let p = c.plan_src(depth, self.fed.len(), ft);
                    depth = p.m_lo;
                    Some(p)
                } else {
                    None
                };
            }
            let round_t0 = std::time::Instant::now();
            let mut round_two_chunk = false; // set when the round extends
            serial_block = 0; // a speculative round ends the serial block
            let _trace_round = lisa_mlx::trace::span("mtp.round");
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
            let copy_cap = crate::core::copy_draft::copy_len_max();
            let copy: Option<Vec<u32>> =
                copy_lookup_guarded(&committed, &copy_index, COPY_K, copy_cap);
            let copy_len = copy.as_ref().map_or(0, |c| c.len());
            // The copy-hit length rides the trace (detail of `mtp.copy`): a
            // post-hoc tally of the length distribution needs it per round.
            let _trace_copy = lisa_mlx::trace::span_detail("mtp.copy", copy_len as u64);
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
            let _trace_draft = lisa_mlx::trace::span("mtp.draft");
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
                    let mut fm: Vec<Array> = backlog.iter().map(|(_, m)| m.clone()).collect();
                    fm.push(carry_multi.clone());
                    let rows = ft.len();
                    let tok_arr = Array::from_slice(&ft, &[1i32, rows as i32]);
                    let mul_arr = crate::models::speculate::concat_rows(&fm)?;
                    tower.draft_step(&tok_arr, &mul_arr)?
                }
            };
            let _trace_draft2 = lisa_mlx::trace::span("mtp.draft_parts");
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
            let mut confs: Vec<Array> = Vec::new();
            if tc.is_some() {
                if let Some(c) = tower.take_draft_confidence() {
                    confs.push(c);
                }
            }
            let mut chain_tail: Option<(Array, Array)> = None;
            let mut draft_parts: Vec<Array> = if copy_len >= w {
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
                let mut parts: Vec<Array> = Vec::with_capacity(w);
                for i in 0..w {
                    // Proposal i: the copy hit's token when one covers this
                    // position, else the chain's own prediction.
                    let prop = match copy.as_ref() {
                        Some(c) if i < c.len() => Array::from_slice(&[c[i] as i32], &[1i32, 1]),
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
                        vals.push(c.item_cast::<f32>() as f64);
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
            let _trace_verify = lisa_mlx::trace::span("mtp.verify_build");
            tower.set_context_tails(vec![tail.clone()]);
            let carry_arr = Array::from_slice(&[carry_token], &[1i32, 1]);
            let v_arr = if copy_len >= w {
                let c = copy.as_ref().expect("copy_len >= w implies a copy hit");
                let mut vt: Vec<i32> = Vec::with_capacity(w + 1);
                vt.push(carry_token as i32);
                vt.extend(c.iter().take(w).map(|&t| t as i32));
                Array::from_slice(&vt, &[1i32, (w + 1) as i32])
            } else {
                let mut verify_parts: Vec<&Array> = Vec::with_capacity(w + 1);
                verify_parts.push(&carry_arr);
                for p in &draft_parts {
                    verify_parts.push(p);
                }
                lisa_mlx::ops::concatenate(&verify_parts, 1).map_err(|e| anyhow::anyhow!("{e}"))?
            };
            // Tail-ULP contract scope (AGENTS.md §9): the verify forward + head
            // enqueue on the split-K lane; the guard covers ONLY this enqueue
            // window — serial, draft, prefill, and batch paths are untouched.
            let _verify_splitk = lisa_mlx::ops::enter_verify_splitk_scope();
            let (v_mixed, v_multi) = tower.forward_capture(&v_arr, Some(&mut self.caches), true)?;
            let v_logits = tower.head(&v_mixed)?;
            drop(_verify_splitk);
            let top = argmax_axis(&v_logits, -1, None).map_err(|e| anyhow::anyhow!("{e}"))?;
            let _trace_readback = lisa_mlx::trace::span("mtp.readback");
            top.eval().map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut main_tokens: Vec<u32> = top
                .as_dtype(lisa_mlx::Dtype::Int32)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .as_slice::<i32>()
                .iter()
                .map(|&t| t as u32)
                .collect();
            // The draft ids: the mixed proposal, read back together with the
            // verify sync (v_arr holds [carry, proposals...]).
            let drafts: Vec<u32> = v_arr
                .as_slice::<i32>()
                .iter()
                .skip(1)
                .map(|&t| t as u32)
                .collect();

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
                        .index((.., acc as i32, ..))
                        .contiguous()
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    let p = sm.draft_accept_prob(&row, drafts[acc]).unwrap_or(0.0);
                    if sm.accept_draw(p) {
                        acc += 1;
                    } else {
                        if let Some(rt) = sm.draw_residual(&row, drafts[acc]) {
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
                        .index((.., w as i32, ..))
                        .contiguous()
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    if let Some(bt) = sm.draw_row(&row) {
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
            let _trace_commit = lisa_mlx::trace::span("mtp.commit");

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
                    match c {
                        LayerCache::Full(f) => f.trim(advance - keep),
                        LayerCache::Linear(g) => g.rollback_to(keep, 4)?,
                    }
                }
                store_tokens.extend_from_slice(&main_tokens[..emitted]);
                store_multi.push(v_multi.index((.., 0..emitted as i32, ..)).contiguous()?);
                committed.extend_from_slice(&main_tokens[..emitted]);
                break 'outer;
            }

            for c in self.caches.iter_mut() {
                match c {
                    LayerCache::Full(f) => f.trim(advance - n),
                    LayerCache::Linear(g) => g.rollback_to(n, 4)?,
                }
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
                let mi = v_multi.index((.., i as i32, ..)).contiguous()?;
                backlog.push((main_tokens[i], mi));
            }
            carry_token = main_tokens[a];
            carry_multi = v_multi.index((.., a as i32, ..)).contiguous()?;
            tail.push(carry_token as i64);
            for i in 0..a {
                tail.push(main_tokens[i] as i64);
            }
            if tail.len() > ctx_len {
                tail.drain(0..tail.len() - ctx_len);
            }
            store_tokens.extend_from_slice(&main_tokens[..=a]);
            store_multi.push(v_multi.index((.., 0..(a + 1) as i32, ..)).contiguous()?);
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
            lisa_mlx::memory::trim_cache();
            lisa_mlx::memory::trim_cache();
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
                let tok = Array::from_slice(&tail, &[1i32, tail_pairs as i32]);
                let mul_all = concat_chunks(&store_multi)?;
                let mul = mul_all.index((.., 0..tail_pairs as i32, ..)).contiguous()?;
                let _ = tower.draft_step(&tok, &mul)?;
            }
            after_prime + tail_pairs
        } else {
            usize::MAX
        };
        self.mtp_tokens.extend_from_slice(tokens);
        self.mtp_tokens.extend_from_slice(&store_tokens);
        self.mtp_multi
            .push(multi.index((.., 0..t_s, ..)).contiguous()?);
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
fn concat_chunks(chunks: &[Array]) -> anyhow::Result<Array> {
    anyhow::ensure!(!chunks.is_empty(), "empty multi store");
    let mut rows: Vec<Array> = Vec::with_capacity(chunks.len());
    for m in chunks {
        let d = m.dim(-1);
        let n: i32 = m.shape()[..m.shape().len() - 1].iter().product();
        rows.push(m.reshape(&[1, n, d])?);
    }
    let refs: Vec<&Array> = rows.iter().collect();
    lisa_mlx::ops::concatenate(&refs, 1).map_err(|e| anyhow::anyhow!("{e}"))
}

#[cfg(test)]
mod target_mtp_tests {
    use super::*;

    // Retain the verifier origin when production calls the head on each row
    // separately; tensor row count alone cannot distinguish verify from replay.
    const VERIFY_LOGITS_MARKER: f32 = 1024.0;

    struct ToyTargetMtp {
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
                    LayerCache::Linear(state) => {
                        for slot in [&mut state.conv, &mut state.ssm] {
                            *slot = Some(match slot {
                                Some(prefix) => Array::concatenate(&[prefix, &rows], 1)?,
                                None => rows.clone(),
                            });
                        }
                    }
                    LayerCache::Full(state) => {
                        let keys = rows.as_dtype(lisa_mlx::Dtype::Bfloat16)?.reshape(&[
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
                .as_dtype(lisa_mlx::Dtype::Float32)?
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

    impl LanguageModel for ToyTargetMtp {
        fn max_position_embeddings(&self) -> usize {
            self.context_limit
        }

        fn eos_token_id(&self) -> i64 {
            31
        }

        fn new_caches(&self) -> Vec<LayerCache> {
            vec![
                LayerCache::Linear(crate::core::cache::GdnCache::new()),
                LayerCache::Full(crate::core::cache::FullAttentionCache::new(1, 1)),
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

        fn head(&self, mixed: &Array) -> lisa_mlx::error::Result<Array> {
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

    impl LanguageModel for ToyDecodeDispatch {
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

        fn prefill(&mut self, tokens: &[u32], caches: &mut Vec<LayerCache>) -> anyhow::Result<Array> {
            self.prefill_calls.push(tokens.to_vec());
            self.target.prefill(tokens, caches)
        }

        fn decode_token(&mut self, token: u32, caches: &mut Vec<LayerCache>) -> anyhow::Result<Array> {
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

        fn head(&self, mixed: &Array) -> lisa_mlx::error::Result<Array> {
            if self.fail_head {
                return Err(lisa_mlx::error::Error::msg("injected decode head failure"));
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
        let error = session.decode_token(&mut tower, 2).err().expect("law refusal");
        assert!(error.to_string().contains("numerical law changed"));
        assert!(tower.decode_calls.is_empty());
        assert_eq!(session.fed, vec![1]);
        assert!(session.is_valid());
        ToyTargetMtp::assert_cache_matches_fed(&session)?;
        tower.law = 1;
        let _ = session.decode_token(&mut tower, 2)?;
        assert_eq!(session.fed, vec![1, 2]);
        assert_eq!(tower.decode_calls, vec![2]);
        let error = session.decode_token(&mut tower, 3).err().expect("context refusal");
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
            let error = session.decode_token(&mut tower, 3).err().expect("invalid session refusal");
            assert!(error.to_string().contains("session state is invalid"));
            assert_eq!(tower.decode_calls, vec![2]);
        }
        Ok(())
    }

    #[test]
    fn glm53_target_mtp_reuses_argmax_only_for_unmodified_greedy_draws() {
        assert!(target_mtp_draw_reuses_argmax(
            None,
            TargetMtpSamplingMode::Configured,
        ));
        assert!(target_mtp_draw_reuses_argmax(
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
                let sampler = Sampler {
                    temperature,
                    top_k,
                    repetition_penalty,
                    mtp_exact_sampling,
                    ..Sampler::default()
                };
                assert_eq!(
                    target_mtp_draw_reuses_argmax(
                        Some(&sampler),
                        TargetMtpSamplingMode::Configured,
                    ),
                    reusable,
                    "{sampler:?}",
                );
                assert!(target_mtp_draw_reuses_argmax(
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
            assert_eq!(session.glm_mtp_cache.total, 0);
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
    fn glm53_target_mtp_resumes_at_round_boundaries() -> anyhow::Result<()> {
        for bad_first_draft in [false, true] {
            // Greedy, opportunistic sampling, and exact point-mass sampling.
            for sampling in 0..3 {
                for budget in [1, 2, 3, 8] {
                    let make_tower = || ToyTargetMtp {
                        bad_first_draft,
                        eos_on_verify: false,
                        bias_verify_multi: bad_first_draft,
                        bias_verify_logits: false,
                        fail_verify_after_cache_append: false,
                        soft_logits: sampling != 0,
                        context_limit: 128,
                        mtp_calls: Vec::new(),
                    };
                    let make_sampler = || Sampler {
                        temperature: if sampling == 0 { 0.0 } else { 0.8 },
                        seed: 0x5eed,
                        mtp_exact_sampling: sampling == 2,
                        ..Default::default()
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
                    assert!(chunked_session.glm_mtp_pending.is_none());
                    ToyTargetMtp::assert_cache_matches_fed(&full_session)?;
                    assert_eq!(
                        chunked_session.glm_mtp_cache.compact_total(),
                        full_session.glm_mtp_cache.compact_total()
                    );
                    assert_eq!(
                        chunked_session
                            .glm_mtp_cache
                            .compact_live_for_test()
                            .unwrap()
                            .as_dtype(lisa_mlx::Dtype::Float32)?
                            .to_vec1::<f32>()?,
                        full_session
                            .glm_mtp_cache
                            .compact_live_for_test()
                            .unwrap()
                            .as_dtype(lisa_mlx::Dtype::Float32)?
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
                bad_first_draft: false,
                eos_on_verify: false,
                bias_verify_multi: false,
                bias_verify_logits: false,
                fail_verify_after_cache_append: false,
                soft_logits: true,
                context_limit,
                mtp_calls: Vec::new(),
            };
            let mut sampler = Sampler {
                temperature: 0.8,
                seed: 0x5eed,
                ..Default::default()
            };
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
            assert!(session.glm_mtp_pending.is_none());
            ToyTargetMtp::assert_cache_matches_fed(&session)?;
        }
        Ok(())
    }

    #[test]
    fn glm53_target_mtp_commit_boundary_survives_suffix_extension() -> anyhow::Result<()> {
        let mut tower = ToyTargetMtp {
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
            .glm_mtp_cache
            .compact_slice(0, 8)
            .expect("retained absolute-position MTP cache")
            .as_dtype(lisa_mlx::Dtype::Float32)?
            .to_vec1::<f32>()?;
        assert_eq!(compact, vec![0.0, 3.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]);
        Ok(())
    }

    #[test]
    fn glm53_target_mtp_cache_prefix_snapshot_restores_private_tape() -> anyhow::Result<()> {
        let mut tower = ToyTargetMtp {
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
        session.glm_mtp_cache.write_compact_f16_at(&initial, 1)?;
        session.glm_mtp_min_position = Some(1);
        session.fed = vec![10, 11];
        session.capture_snapshot(1, 0);

        let speculative = Array::from_slice(&[9.0f32], &[1, 1, 1]);
        session
            .glm_mtp_cache
            .write_compact_f16_at(&speculative, 4)?;
        session.glm_mtp_min_position = Some(4);
        session.fed.push(12);
        assert_eq!(session.restore_snapshot(&mut tower, 1), Some(2));
        assert_eq!(session.fed, vec![10, 11]);
        assert_eq!(session.glm_mtp_min_position, Some(1));
        assert_eq!(
            session
                .glm_mtp_cache
                .compact_slice(0, 2)
                .expect("snapshot MTP rows")
                .as_dtype(lisa_mlx::Dtype::Float32)?
                .to_vec1::<f32>()?,
            vec![0.0, 3.0]
        );
        assert!(session.glm_mtp_cache.compact_slice(0, 5).is_none());
        Ok(())
    }

    #[test]
    fn glm53_target_mtp_opportunistic_and_exact_accept_reject() -> anyhow::Result<()> {
        for (bad_first_draft, exact_sampling) in
            [(false, false), (true, false), (false, true), (true, true)]
        {
            let mut tower = ToyTargetMtp {
                bad_first_draft,
                eos_on_verify: false,
                bias_verify_multi: false,
                bias_verify_logits: false,
                fail_verify_after_cache_append: false,
                soft_logits: false,
                context_limit: 128,
                mtp_calls: Vec::new(),
            };
            let mut sampler = crate::core::sampler::Sampler {
                temperature: 0.8,
                seed: 0x5eed,
                mtp_exact_sampling: exact_sampling,
                ..Default::default()
            };
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
            bad_first_draft: false,
            eos_on_verify: false,
            bias_verify_multi: false,
            bias_verify_logits: false,
            fail_verify_after_cache_append: false,
            soft_logits: false,
            context_limit: 128,
            mtp_calls: Vec::new(),
        };
        let mut sampler = crate::core::sampler::Sampler {
            temperature: 0.8,
            seed: 0x5eed,
            mtp_exact_sampling: true,
            ..Default::default()
        };
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
            let mut sampler = crate::core::sampler::Sampler {
                temperature: 0.8,
                seed,
                ..Default::default()
            };
            let first = sampler.draw_ds4(&first_logits, &[1]).ok();
            let next = sampler.draw_ds4(&next_logits, &[1, 2]).ok();
            first == Some(2) && next.is_some_and(|token| token != 3)
        });
        let seed = seed.expect("find deterministic seed with a non-greedy sampled parent");

        for exact_sampling in [false, true] {
            let mut tower = ToyTargetMtp {
                bad_first_draft: false,
                eos_on_verify: false,
                bias_verify_multi: false,
                bias_verify_logits: false,
                fail_verify_after_cache_append: false,
                soft_logits: true,
                context_limit: 128,
                mtp_calls: Vec::new(),
            };
            let mut sampler = crate::core::sampler::Sampler {
                temperature: 0.8,
                seed,
                mtp_exact_sampling: exact_sampling,
                ..Default::default()
            };
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
                    .glm_mtp_cache
                    .compact_slice(0, expected_compact.len())
                    .expect("MTP cache through generated target tokens")
                    .as_dtype(lisa_mlx::Dtype::Float32)?
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
                assert!(session.glm_mtp_pending.is_none());
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
            bad_first_draft: true,
            eos_on_verify: true,
            bias_verify_multi: false,
            bias_verify_logits: false,
            fail_verify_after_cache_append: false,
            soft_logits: false,
            context_limit: 128,
            mtp_calls: Vec::new(),
        };
        let mut sampler = crate::core::sampler::Sampler {
            temperature: 0.8,
            seed: 0x5eed,
            mtp_exact_sampling: true,
            ..Default::default()
        };
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
