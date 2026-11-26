//! Independent MTP: the draft head chains its own proposals, mixed with context-copy
//! hits, under the expected-value depth controller.
use super::*;

/// Concatenate `multi` chunks along the sequence axis. Chunks may be rank 2
/// (`[1, D]`) or rank 3 (`[1, S, D]`); normalize to `[1, S, D]` first.
pub(super) fn concat_chunks<T: SpeculativeTensor>(chunks: &[T]) -> anyhow::Result<T> {
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

impl<R: SpeculativeRuntime> Conversation<R> {
    pub(super) fn generate_independent_mtp(
        &mut self,
        tower: &mut dyn LanguageModel<R>,
        tokens: &[u32],
        max_tokens: usize,
        depth_sel: MtpDepth,
        after_prefill: Option<&mut dyn FnMut(&[R::Cache])>,
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
            .ok_or_else(|| anyhow::anyhow!("context position overflow before MTP"))?;
        anyhow::ensure!(
            prompt_end <= tower.max_position_embeddings(),
            "context overflow before MTP"
        );
        let max_tokens = max_tokens.min(tower.max_position_embeddings() - prompt_end + 1);
        // A serial/prefix-cache session may not retain trunk hidden rows. Rebuild
        // those rows once when entering MTP; never align an incomplete history
        // with a live target cache.
        let replay;
        let tokens = if self.mtp_tokens != self.fed {
            replay = self.fed.iter().chain(tokens).copied().collect::<Vec<_>>();
            let snapshots_enabled = self.snapshots_enabled;
            *self = Self::new(tower);
            self.snapshots_enabled = snapshots_enabled;
            tower.drafter_reset();
            replay.as_slice()
        } else {
            tokens
        };
        let mut mode = TargetMtpSamplingMode::Configured;
        use crate::core::round_cost::{
            DepthController, ModelCost, PROBE_TICKS, PROBE_WARM, RoundCostTable, probe_warm_stats,
        };

        anyhow::ensure!(tower.has_drafter(), "the checkpoint has no MTP head");
        let model_depth_cap = tower.max_speculative_depth().min(6);
        anyhow::ensure!(
            model_depth_cap >= 2,
            "the checkpoint does not support MTP verification"
        );
        let is_auto = matches!(depth_sel, MtpDepth::Auto);
        let mut depth: usize = match depth_sel {
            // Independent chains and copied proposals share the six-draft
            // bound (seven verification inputs), or the model's smaller cap.
            MtpDepth::Fixed(d) => {
                anyhow::ensure!(d >= 2, "independent MTP depth must be zero or at least two");
                anyhow::ensure!(
                    d <= model_depth_cap,
                    "this checkpoint supports at most {model_depth_cap} MTP draft tokens"
                );
                d
            }
            MtpDepth::Auto => 2, // re-picked by the controller every round
        };
        anyhow::ensure!(
            depth <= model_depth_cap,
            "this checkpoint supports at most {model_depth_cap} MTP draft tokens"
        );

        // Capture proposal zero during priming as well as later restarts.

        // 1. Prefill the turn's suffix (incremental) and take the first token.
        let (logits, multi) = self.feed_multi_for_generation(tower, tokens)?;
        // Clean commit boundary (suffix fed, nothing decoded yet): the
        // cross-request prefix cache snapshots here — the same contract as
        // `generate`. Without this hook the cache was NEVER populated on the
        // MTP path, so with speculation on (the serve default) no request could
        // ever resume, and llmprobe reported the cache as "not detected".
        if let Some(cb) = after_prefill {
            cb(&self.caches);
        }
        let first = draw_block_target_token(
            tower.runtime(),
            stochastic.as_deref_mut(),
            &logits,
            &self.fed,
            mode,
        )?;
        mode = on_token(first)?;
        if max_tokens == 1 || is_eos(first) {
            self.mtp_tokens = self.fed.clone();
            self.mtp_multi.push(multi);
            tower.drafter_reset();
            self.mtp_head_len = 0;
            return Ok(vec![first]);
        }
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
        // Oracle: when the CLI enabled it, keep every round's
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
        // n >= TABLE_MIN_SAMPLES. Only when no bucket carries
        // a measured serial cell does an in-run probe run, and with the same
        // shape: PROBE_TICKS real serial steps
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
                // ran (an empty-picks dead end).
                let mut probe: Vec<f64> = Vec::new();
                for _ in 0..PROBE_TICKS {
                    let t0 = std::time::Instant::now();
                    let (t, row, mixed) = serial_step(
                        tower,
                        &mut self.caches,
                        &tail,
                        carry_token,
                        stochastic.as_deref_mut(),
                        &committed,
                        mode,
                    )?;
                    let dt = t0.elapsed().as_secs_f64() * 1e3;
                    probe.push(dt);
                    mode = on_token(t)?;
                    generated.push(t);
                    store_tokens.push(t);
                    store_multi.push(row.clone());
                    committed.push(t);
                    self.fed.push(carry_token);
                    self.last_mixed = Some(mixed);
                    tail.push(carry_token as i64);
                    carry_token = t;
                    carry_multi = row;
                    if tail.len() > ctx_len {
                        tail.drain(0..tail.len() - ctx_len);
                    }
                    if is_eos(t) || generated.len() >= max_tokens {
                        break;
                    }
                }
                if !probe.is_empty() {
                    primed = None;
                    tower.drafter_trim(1);
                    let rows = concat_chunks(&store_multi)?;
                    backlog = store_tokens[..store_tokens.len() - 1]
                        .iter()
                        .enumerate()
                        .map(|(i, &token)| Ok((token, rows.select(1, i as i32)?.contiguous()?)))
                        .collect::<anyhow::Result<Vec<_>>>()?;
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
            if let Ok(data) = std::fs::read_to_string(&seed_path)
                && let Ok(serde_json::Value::Object(m)) = serde_json::from_str(&data)
                && let Some(s) = m.get(&key)
                && let (Some(acc), Some(m_lo)) = (
                    s.get("ev_acc")
                        .and_then(|v| serde_json::from_value::<[f64; 6]>(v.clone()).ok()),
                    s.get("m_lo").and_then(|v| v.as_u64()),
                )
            {
                auto.as_mut()
                    .expect("just built")
                    .seed_ev(acc, m_lo as usize);
                eprintln!("[mtp.ev-seed] seeded from a previous request");
            }
        }
        'outer: while generated.len() < max_tokens
            && !is_eos(*generated.last().expect("generated is non-empty"))
        {
            // Auto: re-pick the depth every round (0 = serial fallback).
            if let Some(c) = auto.as_mut() {
                depth = c
                    .pick(self.fed.len(), max_tokens - generated.len())
                    .min(model_depth_cap);
                if depth == 0 {
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
                        crate::core::copy_draft::COPY_LEN_MAX.min(model_depth_cap),
                    );
                    if hit.is_none() {
                        // A pre-primed proposal predicts after the old carry.
                        // Serial advancement makes it stale; replay that pair
                        // with the accumulated target-hidden backlog later.
                        if primed.take().is_some() {
                            tower.drafter_trim(1);
                        }
                        let t0 = std::time::Instant::now();
                        let (t, row, mixed) = serial_step(
                            tower,
                            &mut self.caches,
                            &tail,
                            carry_token,
                            stochastic.as_deref_mut(),
                            &committed,
                            mode,
                        )?;
                        let wall = t0.elapsed().as_secs_f64() * 1e3;
                        mode = on_token(t)?;
                        generated.push(t);
                        store_tokens.push(t);
                        store_multi.push(row.clone());
                        committed.push(t);
                        backlog.push((carry_token, carry_multi.clone()));
                        self.fed.push(carry_token);
                        self.last_mixed = Some(mixed);
                        tail.push(carry_token as i64);
                        carry_token = t;
                        carry_multi = row;
                        if tail.len() > ctx_len {
                            tail.drain(0..tail.len() - ctx_len);
                        }
                        c.observe_serial(wall, serial_block < PROBE_WARM as u32);
                        serial_block += 1;
                        if is_eos(t) {
                            break 'outer;
                        }
                        if generated.len() >= max_tokens {
                            break 'outer;
                        }
                        continue;
                    }
                    // a hit: fall through with depth = 0 — w = copy_len below.
                }
            }
            if max_tokens - generated.len() == 1 {
                let (token, row, mixed) = serial_step(
                    tower,
                    &mut self.caches,
                    &tail,
                    carry_token,
                    stochastic.as_deref_mut(),
                    &committed,
                    mode,
                )?;
                self.fed.push(carry_token);
                self.last_mixed = Some(mixed);
                generated.push(token);
                on_token(token)?;
                store_tokens.push(token);
                store_multi.push(row);
                committed.push(token);
                break;
            }
            let round_t0 = std::time::Instant::now();
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
            let copy_cap = crate::core::copy_draft::COPY_LEN_MAX.min(model_depth_cap);
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
            let mut w = depth.max(copy_len).min(copy_cap);
            if copy_len > 0 {
                copy_rounds += 1;
                copy_tokens += copy_len;
            }
            w = w.min(max_tokens - generated.len() - 1);
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
                (0..w).map(|_| None).collect();
            if oracle_on {
                chain_pred[0] = crate::core::oracle::last();
            }
            let draft_parts: Vec<R::Tensor> = if copy_len >= w {
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
                    if oracle_on && i + 1 < w {
                        // This call's prediction becomes proposal i+1.
                        chain_pred[i + 1] = crate::core::oracle::last();
                    }
                    d = d2;
                    m = m2;
                }
                parts
            };
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
            // The loaded target owns its verification arithmetic. An
            // independent draft source must not enable a different reduction
            // law for the target: the scoped split-K lane changes full-model
            // greedy decisions even when individual kernels meet its ULP bound.
            let (v_mixed, v_multi) = tower.forward_capture(&v_arr, Some(&mut self.caches), true)?;
            let v_logits = tower.head_verified(&v_mixed)?;
            let top = v_logits
                .argmax_axis(-1)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            let _trace_readback = tower.runtime().trace("mtp.readback");
            top.eval().map_err(|e| anyhow::anyhow!("{e}"))?;
            let mut main_tokens = top.read_ids()?;
            // The draft ids: the mixed proposal, read back together with the
            // verify sync (v_arr holds [carry, proposals...]).
            let drafts: Vec<u32> = v_arr.read_raw_ids()?.into_iter().skip(1).collect();

            let mut a = 0usize;
            let mut sampling_history = committed.clone();
            let mut emitted_tokens = Vec::new();
            for i in 0..=w {
                let token = if stochastic.is_some() && mode != TargetMtpSamplingMode::Greedy {
                    let row = v_logits.select(1, i as i32)?.contiguous()?;
                    draw_block_target_token(
                        tower.runtime(),
                        stochastic.as_deref_mut(),
                        &row,
                        &sampling_history,
                        mode,
                    )?
                } else {
                    main_tokens[i]
                };
                main_tokens[i] = token;
                generated.push(token);
                emitted_tokens.push(token);
                mode = on_token(token)?;
                sampling_history.push(token);
                let matches_draft = i < w && token == drafts[i];
                if matches_draft {
                    a += 1;
                }
                if is_eos(token) || generated.len() == max_tokens || !matches_draft {
                    break;
                }
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
            let emitted = emitted_tokens.len();
            let _trace_commit = tower.runtime().trace("mtp.commit");
            // Each emitted prediction consumes one verify input. The last
            // prediction (including EOS) stays un-fed; its input stays cached.
            for cache in &mut self.caches {
                tower.runtime().rollback(cache, advance, emitted, true)?;
            }
            self.last_mixed = Some(v_mixed.select(1, emitted as i32 - 1)?.contiguous()?);
            self.fed.push(carry_token);
            self.fed.extend_from_slice(&emitted_tokens[..emitted - 1]);
            if is_eos(*emitted_tokens.last().unwrap()) || generated.len() == max_tokens {
                store_tokens.extend_from_slice(&emitted_tokens);
                store_multi.push(v_multi.slice(1, 0, Some(emitted as i32))?.contiguous()?);
                committed.extend_from_slice(&emitted_tokens);
                break;
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
            let target = committed.len(); // Keep target-conditioned pairs only.
            let off = tower.drafter_offset();
            if off > target {
                tower.drafter_trim(off - target);
            }
            backlog.clear();
            for i in 0..a {
                let mi = v_multi.select(1, i as i32)?.contiguous()?;
                backlog.push((main_tokens[i], mi));
            }
            tail.push(carry_token as i64);
            carry_token = main_tokens[a];
            carry_multi = v_multi.select(1, a as i32)?.contiguous()?;
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
                    c.ev_round_update(w, a, depth);
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
        if tower.drafter_offset().checked_add(1) == Some(after_prime) {
            // A serial probe/fallback may end before replaying its backlog.
            let first_pair = tower.runtime().tokens_u32(&store_tokens[..1], &[1, 1])?;
            let first_hidden = store_multi[0].reshape(&[1, 1, store_multi[0].dim(-1)])?;
            let _ = tower.draft_step(&first_pair, &first_hidden)?;
        }
        let off = tower.drafter_offset();
        let tail_pairs = store_tokens.len().saturating_sub(1);
        // The final emitted carry has no target input row yet. Keep only pairs
        // formed by two fed tokens, so a subsequent suffix need not start with
        // the previous un-fed prediction.
        let head_tail_pairs = tail_pairs.saturating_sub(1);
        self.mtp_head_len = if off >= after_prime {
            tower.drafter_trim(off - after_prime);
            if head_tail_pairs > 0 {
                let tail: Vec<i32> = store_tokens[1..head_tail_pairs + 1]
                    .iter()
                    .map(|&t| t as i32)
                    .collect();
                let tok = tower
                    .runtime()
                    .tokens(&tail, &[1i32, head_tail_pairs as i32])?;
                let mul_all = concat_chunks(&store_multi)?;
                let mul = mul_all
                    .slice(1, 1, Some(head_tail_pairs as i32 + 1))?
                    .contiguous()?;
                let _ = tower.draft_step(&tok, &mul)?;
            }
            after_prime + head_tail_pairs
        } else {
            usize::MAX
        };
        self.mtp_tokens = self.fed.clone();
        self.mtp_multi = all_multi;
        if tail_pairs > 0 {
            self.mtp_multi.push(
                concat_chunks(&store_multi)?
                    .slice(1, 1, Some(tail_pairs as i32 + 1))?
                    .contiguous()?,
            );
        }
        tower.set_context_tails(vec![
            self.fed[self.fed.len().saturating_sub(ctx_len)..]
                .iter()
                .map(|&token| token as i64)
                .collect(),
        ]);
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
                 mean {:.2}) per-index acc [{}]",
                total as f64 / rounds as f64,
                width_sum as f64 / rounds as f64,
                copy_tokens as f64 / copy_rounds.max(1) as f64,
                per_index.join(" "),
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
        if let Some(c) = auto.as_mut()
            && let Some(t) = c.take_folded_table()
        {
            let key = tower.mtp_cost_key();
            let path = crate::core::round_cost::default_path();
            let mut table = RoundCostTable::load(&path).unwrap_or_default();
            table.models.insert(key.clone(), t);
            match table.save(&path) {
                Ok(()) => eprintln!("[mtp.auto] round-cost serving folds persisted ({key})"),
                Err(e) => eprintln!("[mtp.auto] round-cost persist FAILED: {e}"),
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
