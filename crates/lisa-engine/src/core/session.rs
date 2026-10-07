//! Multi-turn (and, later, batched) session state.
//!
//! A [`Session`] owns the caches for one conversation and the exact token ids
//! fed into them, so a new turn only prefills the appended suffix — the
//! attention cache continues from its offset, and the GDN recurrent state and
//! the PLE n-gram history are already carried by the caches and `Tower`.

use lisa_mlx::Array;
use lisa_mlx::ops::indexing::IndexOp;

use crate::core::cache::LayerCache;
use crate::core::copy_draft::{COPY_K, CopyIndex, copy_lookup_guarded};
use crate::core::generate::is_eos;
use crate::core::sampler::Sampler;
use crate::models::LanguageModel;

pub struct Session {
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
    /// Prefix snapshots `(boundary, per-layer state)` taken at the start of
    /// each turn. On a divergent request the server can rewind to the largest
    /// boundary at or below the common prefix and prefill only the tail.
    pub snapshots: Vec<(usize, usize, usize, Vec<LayerSnapshot>)>, // (boundary, msgs_len, head_offset, state)
    pub snapshots_enabled: bool,
}

/// A per-layer prefix snapshot: the full-attention live length, or the GDN
/// recurrent state (short-conv, fp32 SSM, PLE conv).
#[derive(Clone)]
pub struct LayerSnapshot {
    pub full_offset: Option<usize>,
    pub gdn: Option<(Option<Array>, Option<Array>, Option<Array>)>,
}

/// Keep at most this many prefix snapshots (the GDN state is ~113 MB each).
const SNAPSHOT_KEEP: usize = 4;

impl Session {
    /// A fresh conversation. Resets the PLE n-gram context so the first suffix
    /// is hashed from the default (EOS-filled) context, not a leftover one.
    pub fn new(tower: &mut dyn LanguageModel) -> Self {
        tower.clear_context();
        Session {
            caches: tower.new_caches(),
            fed: Vec::new(),
            last_mixed: None,
            mtp_tokens: Vec::new(),
            mtp_multi: Vec::new(),
            mtp_head_len: 0,
            snapshots: Vec::new(),
            snapshots_enabled: false,
        }
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
        if !self.snapshots_enabled {
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
        if self.snapshots.len() > SNAPSHOT_KEEP {
            let drop = self.snapshots.len() - SNAPSHOT_KEEP;
            self.snapshots.drain(0..drop);
        }
    }

    /// Rewind to the largest snapshot boundary `<= cp` (a divergent request).
    /// Returns that boundary, or `None` when no snapshot is at or below `cp`.
    pub fn restore_snapshot(
        &mut self,
        tower: &mut dyn LanguageModel,
        msgs_len: usize,
    ) -> Option<usize> {
        let idx = self
            .snapshots
            .iter()
            .rposition(|(_, m, _, _)| *m <= msgs_len)?;
        let boundary = self.snapshots[idx].0;
        let head = self.snapshots[idx].2;
        let snaps = self.snapshots[idx].3.clone();
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
        // Rewind the head's KV cache to the snapshot too, so no re-prime is
        // needed (the history pairs are its offset + 1).
        tower.drafter_restore_offset(head);
        self.truncate_head(head + 1);
        self.mtp_head_len = head;
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
        anyhow::ensure!(!tokens.is_empty(), "empty feed");
        anyhow::ensure!(
            self.fed.len() + tokens.len() <= tower.max_position_embeddings(),
            "context overflow: {} + {} > max_position_embeddings {}",
            self.fed.len(),
            tokens.len(),
            tower.max_position_embeddings()
        );
        let last = tower.prefill(tokens, &mut self.caches)?;
        let logits = tower.head(&last)?;
        self.last_mixed = Some(last);
        self.fed.extend_from_slice(tokens);
        Ok(logits)
    }

    /// [`Session::feed`] but also returning the hyper stream `multi`
    /// `[1, len(tokens), hc*H]` for the fed tokens (needed by the MTP head).
    pub fn feed_multi(
        &mut self,
        tower: &mut dyn LanguageModel,
        tokens: &[u32],
    ) -> anyhow::Result<(Array, Array)> {
        anyhow::ensure!(!tokens.is_empty(), "empty feed");
        anyhow::ensure!(
            self.fed.len() + tokens.len() <= tower.max_position_embeddings(),
            "context overflow: {} + {} > max_position_embeddings {}",
            self.fed.len(),
            tokens.len(),
            tower.max_position_embeddings()
        );
        let (last, multi) = tower.prefill_multi(tokens, &mut self.caches)?;
        let logits = tower.head(&last)?;
        self.last_mixed = Some(last);
        self.fed.extend_from_slice(tokens);
        Ok((logits, multi))
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
        let mut token = sampler.draw(&logits, &self.fed)?;
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
            let logits = self.feed(tower, &[token])?;
            token = sampler.draw(&logits, &self.fed)?;
            out.push(token);
            on_token(token)?;
        }
        Ok(out)
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
    /// Fixed draft depth 2..=6 (phase-1 opt-in; behavior unchanged).
    Fixed(usize),
    /// EV auto (phase 2, the default for drafter models): the controller in
    /// `round_cost.rs` re-picks the depth every round — or drops to serial
    /// (depth 0) when no speculative depth beats the serial step rate.
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

impl Session {
    /// Speculative (MTP) generation at a fixed depth: the target caches persist,
    /// the head is re-primed from the session's token/multi store, and only the
    /// turn's suffix is prefilled. Greedy only (drafts must match a
    /// deterministic target); call [`Session::generate`] otherwise.
    pub fn generate_mtp(
        &mut self,
        tower: &mut dyn LanguageModel,
        tokens: &[u32],
        max_tokens: usize,
        depth: usize,
        on_token: &mut dyn FnMut(u32) -> anyhow::Result<()>,
    ) -> anyhow::Result<Vec<u32>> {
        self.generate_mtp_depth(tower, tokens, max_tokens, MtpDepth::Fixed(depth), None, None, on_token)
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
        mut after_prefill: Option<&mut dyn FnMut(&[LayerCache])>,
        // Non-greedy verification. `None` keeps the certified greedy path.
        // `Some(sm)` runs the same Leviathan-exact scheme as `pld_decode`: the
        // MTP proposal is a POINT MASS at the head's argmax (`q = delta`), so a
        // draft token is accepted with `p_target[draft]` and the first rejection
        // emits a draw from the residual. Serial steps sample too.
        mut stochastic: Option<&mut crate::core::sampler::Sampler>,
        on_token: &mut dyn FnMut(u32) -> anyhow::Result<()>,
    ) -> anyhow::Result<Vec<u32>> {
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
        on_token(first)?;
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
                    let (t, row) = serial_step(tower, &mut self.caches, &tail, carry_token, stochastic.as_deref_mut(), committed.len() as u64)?;
                    let dt = t0.elapsed().as_secs_f64() * 1e3;
                    probe.push(dt);
                    on_token(t)?;
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
                        warm,
                        PROBE_TICKS
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
            auto.as_mut()
                .expect("just built")
                .stamp_machine(&crate::core::round_cost::host_chip(), &key, std::env::consts::OS);
            // P6: cross-request EV seed (default ON; a serving process with
            // many short requests re-warms instead of paying warmup again;
            // one request per process = the seed is usually absent, a no-op).
            let seed_path = crate::core::round_cost::default_path()
                .with_file_name("mtp_ev_seed.json");
            if let Ok(data) = std::fs::read_to_string(&seed_path) {
                if let Ok(serde_json::Value::Object(m)) = serde_json::from_str(&data) {
                    if let Some(s) = m.get(&key) {
                        if let (Some(acc), Some(m_lo)) = (
                            s.get("ev_acc").and_then(|v| {
                                serde_json::from_value::<[f64; 6]>(v.clone()).ok()
                            }),
                            s.get("m_lo").and_then(|v| v.as_u64()),
                        ) {
                            auto.as_mut().expect("just built").seed_ev(
                                acc,
                                m_lo as usize,
                            );
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
            eprintln!("[mtp.two-chunk] armed (LISA_MTP_TWO_CHUNK=1): plan (m_lo,m_hi,tau) per round");
        }
        let mut plan: Option<crate::core::round_cost::MtpPlan> = None;
        // P4: inter-round wall clock + the fired-shape flag the regime gate
        // reads (charged per round; a round that extended is two-chunk).
        let mut prev_round_end: Option<std::time::Instant> = None;
        let mut round_two_chunk = false;

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
                        let (t, row) = serial_step(tower, &mut self.caches, &tail, carry_token, stochastic.as_deref_mut(), committed.len() as u64)?;
                        let wall = t0.elapsed().as_secs_f64() * 1e3;
                        on_token(t)?;
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
            round_two_chunk = false; // reset per round; the sync sets it on a fired extension
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
                        Some(c) if i < c.len() => {
                            Array::from_slice(&[c[i] as i32], &[1i32, 1])
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
                        let (mut d, mut m) =
                            chain_tail.take().expect("chain tail exists when parts drafted");
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
                on_token(t)?;
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
                let inter_ms = prev_round_end
                    .map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e3);
                prev_round_end = Some(round_end);
                if copy_len == 0 {
                    c.observe_live_round_ms(wall);
                    c.regime_observe(
                        round_two_chunk,
                        m_lo,
                        wall / (a + 1) as f64,
                        inter_ms,
                    );
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
            let mut store: serde_json::Map<String, serde_json::Value> = std::fs::read_to_string(&seed_path)
                .ok()
                .and_then(|d| serde_json::from_str(&d).ok())
                .unwrap_or_default();
            store.insert(key, serde_json::json!({ "ev_acc": acc, "m_lo": m_lo }));
            if let Err(e) =
                std::fs::write(&seed_path, serde_json::to_string_pretty(&store).unwrap_or_default())
            {
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
