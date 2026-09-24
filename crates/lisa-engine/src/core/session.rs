//! Multi-turn (and, later, batched) session state.
//!
//! A [`Session`] owns the caches for one conversation and the exact token ids
//! fed into them, so a new turn only prefills the appended suffix — the
//! attention cache continues from its offset, and the GDN recurrent state and
//! the PLE n-gram history are already carried by the caches and `Tower`.

use lisa_mlx::ops::indexing::IndexOp;
use lisa_mlx::Array;

use crate::core::generate::is_eos;
use crate::core::sampler::Sampler;
use crate::core::cache::LayerCache;
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
        self.snapshots.push((self.fed.len(), msgs_len, head_offset, snaps));
        if self.snapshots.len() > SNAPSHOT_KEEP {
            let drop = self.snapshots.len() - SNAPSHOT_KEEP;
            self.snapshots.drain(0..drop);
        }
    }

    /// Rewind to the largest snapshot boundary `<= cp` (a divergent request).
    /// Returns that boundary, or `None` when no snapshot is at or below `cp`.
    pub fn restore_snapshot(&mut self, tower: &mut dyn LanguageModel, msgs_len: usize) -> Option<usize> {
        let idx = self.snapshots.iter().rposition(|(_, m, _, _)| *m <= msgs_len)?;
        let boundary = self.snapshots[idx].0;
        let head = self.snapshots[idx].2;
        let snaps = self.snapshots[idx].3.clone();
        for (c, s) in self.caches.iter_mut().zip(snaps.iter()) {
            match (c, s) {
                (LayerCache::Full(f), LayerSnapshot { full_offset: Some(o), .. }) => {
                    f.restore_offset(*o)
                }
                (LayerCache::Linear(g), LayerSnapshot { gdn: Some(st), .. }) => {
                    g.restore_state(st)
                }
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
        if std::env::var("LISA_DEBUG_SESSION").is_ok() {
            eprintln!(
                "[session] feed {} toks; ngram={:?}",
                tokens.len(),
                tower.context_tails().first().map(|t| t.len())
            );
        }
        let last = tower.prefill(tokens, &mut self.caches)?;
        let logits = tower.head(&last)?;
        self.last_mixed = Some(last);
        self.fed.extend_from_slice(tokens);
        Ok(logits)
    }

    /// [`Session::feed`] but also returning the hyper stream `multi`
    /// `[1, len(tokens), hc*H]` for the fed tokens (needed by the MTP head).
    pub fn feed_multi(&mut self, tower: &mut dyn LanguageModel, tokens: &[u32]) -> anyhow::Result<(Array, Array)> {
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
        on_token: &mut dyn FnMut(u32) -> anyhow::Result<()>,
    ) -> anyhow::Result<Vec<u32>> {
        let logits = self.feed(tower, tokens)?;
        let mut token = sampler.draw(&logits, &self.fed)?;
        if std::env::var("LISA_DEBUG_SESSION").is_ok() {
            eprintln!("[session] first token {token}");
        }
        let mut out = vec![token];
        on_token(token)?;
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

impl Session {
    /// Speculative (MTP) generation for this session: the target caches persist,
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
        use lisa_mlx::ops::indexing::{argmax_axis, IndexOp};
        anyhow::ensure!(tower.has_drafter(), "the checkpoint has no MTP head");
        let depth = depth.clamp(1, 6);

        // 1. Prefill the turn's suffix (incremental) and take the first token.
        let (logits, multi) = self.feed_multi(tower, tokens)?;
        let first = crate::models::qwen4::speculate::argmax_id(&logits)?;
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
        if std::env::var("LISA_DEBUG_SESSION").is_ok() {
            eprintln!(
                "[mtp-prime] head_off={head_off} pairs_fed={} history={}",
                init_tokens.len(),
                history.len()
            );
        }

        // Prime the head in windows. A single whole-history forward sits at
        // `offset == 0`, where the head's attention takes the DENSE fallback and
        // materialises `[S, kv]` masks — at 100K tokens that is tens of GB and
        // fails the buffer allocation. Windows keep the head's cache advancing,
        // so from the second window on the block-sparse QSA path serves it.
        const PRIME_CHUNK: usize = 2048;
        let t_prime = std::time::Instant::now();
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
        if std::env::var("LISA_DEBUG_MTP").is_ok() {
            eprintln!(
                "[mtp-prime-time] {:.1} ms pairs={}",
                t_prime.elapsed().as_secs_f64() * 1e3,
                init_tokens.len()
            );
        }

        let mut generated: Vec<u32> = vec![first];
        on_token(first)?;
        let mut store_tokens: Vec<u32> = vec![first];
        let mut store_multi: Vec<Array> = vec![multi.index((.., t_s - 1, ..)).contiguous()?];

        let mut backlog: Vec<(u32, Array)> = Vec::new();
        let mut carry_token: u32 = first;
        let mut carry_multi: Array = multi.index((.., t_s - 1, ..)).contiguous()?;
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

        'outer: while generated.len() < max_tokens
            && !is_eos(*generated.last().expect("generated is non-empty"))
        {
            let t_round = std::time::Instant::now();
            // --- Draft (device chain) ---
            // The draft id stays on the GPU as a `[1,1]` tensor and is fed
            // straight into the next chain step, so the chain no longer
            // round-trips through the host between steps (one readback per
            // round instead).
            let draft_parts: Vec<Array> = {
                let (mut d, mut m) = match primed.take() {
                    Some(p) => p,
                    None => {
                        let mut ft: Vec<i32> = backlog.iter().map(|(t, _)| *t as i32).collect();
                        ft.push(carry_token as i32);
                        let mut fm: Vec<Array> = backlog.iter().map(|(_, m)| m.clone()).collect();
                        fm.push(carry_multi.clone());
                        let rows = ft.len();
                        let tok_arr = Array::from_slice(&ft, &[1i32, rows as i32]);
                        let mul_arr = crate::models::qwen4::speculate::concat_rows(&fm)?;
                        tower.draft_step(&tok_arr, &mul_arr)?
                    }
                };
                let mut parts: Vec<Array> = Vec::with_capacity(depth);
                parts.push(d.reshape(&[1, 1])?);
                for _ in 1..depth {
                    let mul = m.reshape(&[1, 1, m.dim(-1)])?;
                    let (d2, m2) = tower.draft_step(&d, &mul)?;
                    d = d2;
                    m = m2;
                    parts.push(d.reshape(&[1, 1])?);
                }
                parts
            };
            let t_draft = t_round.elapsed();

            // --- Prompt-lookup (context-copy) proposal ---
            // If the last COPY_K committed tokens recur earlier, the tokens
            // that followed that occurrence are a strong draft. It is only a
            // proposal: the target verifies every token, so a miss costs one
            // rejected draft, never an emitted token.
            copy_index.extend(&committed, committed.len().saturating_sub(COPY_K));
            let copy: Option<Vec<u32>> = copy_lookup(&committed, &copy_index, COPY_K, depth);
            let used_copy = copy.is_some();

            // --- Verify over [carry, drafts...] ---
            tower.set_context_tails(vec![tail.clone()]);
            let carry_arr = Array::from_slice(&[carry_token], &[1i32, 1]);
            let v_arr = if let Some(c) = &copy {
                let mut vt: Vec<i32> = Vec::with_capacity(depth + 1);
                vt.push(carry_token as i32);
                vt.extend(c.iter().map(|&t| t as i32));
                Array::from_slice(&vt, &[1i32, (depth + 1) as i32])
            } else {
                let mut verify_parts: Vec<&Array> = Vec::with_capacity(depth + 1);
                verify_parts.push(&carry_arr);
                for p in &draft_parts {
                    verify_parts.push(p);
                }
                lisa_mlx::ops::concatenate(&verify_parts, 1)
                    .map_err(|e| anyhow::anyhow!("{e}"))?
            };
            let (v_mixed, v_multi) = tower.forward_capture(&v_arr, Some(&mut self.caches), true)?;
            let v_logits = tower.head(&v_mixed)?;
            let top = argmax_axis(&v_logits, -1, None).map_err(|e| anyhow::anyhow!("{e}"))?;
            top.eval().map_err(|e| anyhow::anyhow!("{e}"))?;
            let main_tokens: Vec<u32> = top
                .as_dtype(lisa_mlx::Dtype::Int32)
                .map_err(|e| anyhow::anyhow!("{e}"))?
                .as_slice::<i32>()
                .iter()
                .map(|&t| t as u32)
                .collect();
            // The draft ids: the copy proposal, or the device chain read back
            // together with the verify sync.
            let drafts: Vec<u32> = match &copy {
                Some(c) => c.clone(),
                None => v_arr
                    .as_slice::<i32>()
                    .iter()
                    .skip(1)
                    .map(|&t| t as u32)
                    .collect(),
            };
            if std::env::var("LISA_DUMP_LOGITS").is_ok() {
                let vl = v_logits.index((.., v_logits.dim(1) - 1, ..));
                let l = vl.as_dtype(lisa_mlx::Dtype::Float32).map_err(|e| anyhow::anyhow!("{e}"))?;
                let v: Vec<f32> = l.as_slice::<f32>().to_vec();
                let mut idx: Vec<usize> = (0..v.len()).collect();
                idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap());
                eprintln!(
                    "[mtp-logits] gen#{} top5: {:?}",
                    generated.len(),
                    idx[..5].iter().map(|&i| (i, v[i])).collect::<Vec<_>>()
                );
            }
            let t_verify = t_round.elapsed();

            if std::env::var("LISA_DEBUG_MTP").is_ok() {
                eprintln!(
                    "[mtp] round {} carry={} drafts={:?} main={:?}",
                    generated.len(),
                    carry_token,
                    &drafts,
                    &main_tokens[..main_tokens.len().min(6)]
                );
            }
            let mut a = 0usize;
            while a < depth && main_tokens[a] == drafts[a] {
                a += 1;
            }
            let advance = depth + 1;
            let n = a + 1;
            if std::env::var("LISA_DEBUG_MTP").is_ok() {
                eprintln!("[mtp]   a={a} n={n} advance={advance} copy={used_copy}");
            }

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
                let keep = if eos_seen { (emitted - 1).max(1) } else { emitted };
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
            tower.drafter_trim(depth - 1);
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
            if std::env::var("LISA_PROFILE_MTP").is_ok() {
                let total = t_round.elapsed();
                eprintln!(
                    "[mtp-prof] round {} | draft {:>7.2}ms verify {:>7.2}ms commit {:>7.2}ms | total {:>7.2}ms",
                    generated.len(),
                    t_draft.as_secs_f64() * 1e3,
                    (t_verify - t_draft).as_secs_f64() * 1e3,
                    (total - t_verify).as_secs_f64() * 1e3,
                    total.as_secs_f64() * 1e3,
                );
            }
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
        self.mtp_multi.push(multi.index((.., 0..t_s, ..)).contiguous()?);
        self.mtp_multi.extend(store_multi);
        if std::env::var("LISA_DEBUG_SESSION").is_ok() {
            eprintln!(
                "[mtp-off] head_off={} target={} mtp_tokens={} fed={}",
                tower.drafter_offset(),
                self.mtp_head_len,
                self.mtp_tokens.len(),
                self.fed.len()
            );
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

/// Prompt-lookup n-gram size for the context-copy drafter.
const COPY_K: usize = 4;

/// A rolling index from a k-gram to the latest position it started at, over
/// the committed token sequence. Used by the context-copy drafter: when the
/// trailing k-gram recurs, the tokens that followed that occurrence are a
/// strong (but verified) draft.
struct CopyIndex {
    k: usize,
    map: std::collections::HashMap<u64, Vec<u32>>,
    indexed: usize,
}

impl CopyIndex {
    fn new(k: usize) -> Self {
        Self {
            k,
            map: std::collections::HashMap::new(),
            indexed: 0,
        }
    }

    fn hash(&self, t: &[u32]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &x in t {
            h ^= x as u64;
            h = h.wrapping_mul(0x100_0000_01b3);
        }
        h
    }

    /// Index every k-gram whose start is `< upto`, keeping the two most recent
    /// positions (the newest is often a self-match with a short continuation,
    /// so the previous one is the useful occurrence).
    fn extend(&mut self, s: &[u32], upto: usize) {
        let end = upto.min(s.len().saturating_sub(self.k));
        while self.indexed < end {
            let h = self.hash(&s[self.indexed..self.indexed + self.k]);
            let e = self.map.entry(h).or_default();
            e.push(self.indexed as u32);
            if e.len() > 2 {
                e.remove(0);
            }
            self.indexed += 1;
        }
    }

    /// Indexed occurrences of the trailing k-gram of `s`, newest first.
    fn lookup(&self, s: &[u32]) -> Vec<usize> {
        if s.len() < self.k {
            return Vec::new();
        }
        self.map
            .get(&self.hash(&s[s.len() - self.k..]))
            .map(|v| v.iter().rev().map(|&p| p as usize).collect())
            .unwrap_or_default()
    }
}

/// A `depth`-token copy proposal for the token after the committed tail, or
/// `None` when the trailing k-gram has no earlier occurrence with enough
/// committed continuation.
fn copy_lookup(s: &[u32], idx: &CopyIndex, k: usize, depth: usize) -> Option<Vec<u32>> {
    if k != idx.k || s.len() < k + depth {
        return None;
    }
    for p in idx.lookup(s) {
        let start = p + k;
        if start + depth <= s.len() {
            return Some(s[start..start + depth].to_vec());
        }
    }
    None
}
