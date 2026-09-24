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
}

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
        }
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

        // 2. Re-prime the head from scratch with the whole committed history.
        tower.drafter_reset();
        let mut history: Vec<u32> = self.mtp_tokens.clone();
        history.extend_from_slice(tokens);
        let mut all_multi: Vec<Array> = self.mtp_multi.clone();
        all_multi.push(multi.index((.., 0..t_s, ..)).contiguous()?);
        let init_multis = concat_chunks(&all_multi)?;
        let mut init_tokens: Vec<i32> = history[1..].iter().map(|&t| t as i32).collect();
        init_tokens.push(first as i32);

        // Prime the head in windows. A single whole-history forward sits at
        // `offset == 0`, where the head's attention takes the DENSE fallback and
        // materialises `[S, kv]` masks — at 100K tokens that is tens of GB and
        // fails the buffer allocation. Windows keep the head's cache advancing,
        // so from the second window on the block-sparse QSA path serves it.
        const PRIME_CHUNK: usize = 2048;
        let mut primed: Option<(u32, Array)> = None;
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
        let mut carry_token: u32 = first;
        let mut carry_multi: Array = multi.index((.., t_s - 1, ..)).contiguous()?;
        let ctx_len = tower.context_window();
        let mut tail: Vec<i64> = history[history.len().saturating_sub(ctx_len)..]
            .iter()
            .map(|&t| t as i64)
            .collect();

        'outer: while generated.len() < max_tokens
            && !is_eos(*generated.last().expect("generated is non-empty"))
        {
            let t_round = std::time::Instant::now();
            // --- Draft ---
            let drafts = {
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
                let mut drafts: Vec<u32> = Vec::with_capacity(depth);
                drafts.push(d);
                for _ in 1..depth {
                    let tok = Array::from_slice(&[d as i32], &[1i32, 1]);
                    let mul = m.reshape(&[1, 1, m.dim(-1)])?;
                    let (d2, m2) = tower.draft_step(&tok, &mul)?;
                    d = d2;
                    m = m2;
                    drafts.push(d);
                }
                drafts
            };
            let t_draft = t_round.elapsed();

            // --- Verify over [carry, drafts...] ---
            tower.set_context_tails(vec![tail.clone()]);
            let mut verify_tokens: Vec<i32> = Vec::with_capacity(depth + 1);
            verify_tokens.push(carry_token as i32);
            verify_tokens.extend(drafts.iter().map(|&t| t as i32));
            let v_arr = Array::from_slice(&verify_tokens, &[1i32, (depth + 1) as i32]);
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
                eprintln!("[mtp]   a={a} n={n} advance={advance}");
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
        self.mtp_tokens.extend_from_slice(tokens);
        self.mtp_tokens.extend_from_slice(&store_tokens);
        self.mtp_multi.push(multi.index((.., 0..t_s, ..)).contiguous()?);
        self.mtp_multi.extend(store_multi);
        Ok(generated)
    }
}

/// Concatenate `multi` chunks along the sequence axis (each chunk is `[1,S,D]`).
fn concat_chunks(chunks: &[Array]) -> anyhow::Result<Array> {
    anyhow::ensure!(!chunks.is_empty(), "empty multi store");
    let refs: Vec<&Array> = chunks.iter().collect();
    lisa_mlx::ops::concatenate(&refs, 1).map_err(|e| anyhow::anyhow!("{e}"))
}
