//! Generation loop: prefill, decode, greedy and temperature sampling.
use crate::speculative_runtime::{SpeculativeRuntime, SpeculativeSampler, SpeculativeTensor};

use std::time::Instant;

use crate::language::LanguageModel;

pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
}

impl GenerationStats {
    pub fn prefill_tps(&self) -> f64 {
        self.prompt_tokens as f64 / self.prefill_seconds.max(1e-9)
    }
    pub fn decode_tps(&self) -> f64 {
        self.generated_tokens as f64 / self.decode_seconds.max(1e-9)
    }
}

pub use crate::eos::{
    IgnoreEosScope, ignore_eos_scope, is_eos, set_eos_ids, set_eos_ids_for_model,
};

use crate::core::copy_draft::{COPY_K, CopyIndex, copy_lookup_guarded};

/// PLD acceptance counters (logged by the callers).
#[derive(Default, Debug, Clone, Copy)]
pub struct PldStats {
    pub rounds: usize,
    /// Rounds where at least one draft was accepted.
    pub hits: usize,
    /// Sum of accepted drafts over all rounds.
    pub accepted: usize,
    /// Tokens committed by a multi-token verify round (the accepted drafts
    /// plus their verifying token, EOS excluded).
    pub verified_tokens: usize,
}

/// Prompt-lookup speculative decode (PLD) over an already-prefilled cache
/// state. Greedy only: every emitted token is the target's own argmax, drafts
/// come from a k-gram index over the committed sequence and are verified by a
/// wide `S = depth + 1` forward with rollback (same machinery as the MTP
/// verify, minus the head).
///
/// `committed` is the exact token sequence fed into the caches so far **plus**
/// the carry (it is fed by the first round's forward); EOS is never appended.
/// On return it is the fed sequence again, so a [`crate::conversation::Conversation`]
/// caller can hand it back as its `fed`.
#[allow(clippy::too_many_arguments)]
pub fn pld_decode<R: SpeculativeRuntime>(
    tower: &mut dyn LanguageModel<R>,
    caches: &mut Vec<R::Cache>,
    committed: &mut Vec<u32>,
    mut carry: u32,
    max_tokens: usize,
    mut generated_len: usize,
    depth: usize,
    // Non-greedy verification (stochastic speculative decoding). `None` keeps
    // the certified greedy path (argmax per row). `Some(sm)` runs lisa's
    // Leviathan-exact PLD: the draft is deterministic, so a draft token is
    // accepted with probability `p_target[draft]` and the first rejection emits
    // a draw from the residual `norm(max(p - one_hot, 0))`.
    mut stochastic: Option<&mut R::Sampler>,
    on_token: &mut dyn FnMut(u32) -> anyhow::Result<()>,
) -> anyhow::Result<(Vec<u32>, PldStats)> {
    let depth = depth.clamp(1, 6).min(tower.max_speculative_depth());
    let mut generated: Vec<u32> = Vec::new();
    let mut stats = PldStats::default();
    let ctx_len = tower.context_window();
    let mut tail: Vec<i64> = committed[committed.len().saturating_sub(ctx_len)..]
        .iter()
        .map(|&t| t as i64)
        .collect();
    let mut index = CopyIndex::new(COPY_K);

    while generated_len < max_tokens && !is_eos(carry) {
        index.extend(committed, committed.len().saturating_sub(COPY_K));
        let draft = copy_lookup_guarded(committed, &index, COPY_K, depth);
        // The verify/step forward must see the committed PLE tail, not the
        // rejected drafts a previous round fed (same guard as the MTP driver).
        tower.set_context_tails(vec![tail.clone()]);
        let next: Option<(u32, usize)> = match draft {
            None => {
                let step = tower.runtime().tokens(&[carry as i32], &[1i32, 1])?;
                let (mixed, _) = tower.forward(&step, Some(caches))?;
                let last = mixed.select(1, mixed.dim(1) - 1)?;
                let logits = tower.head(&last)?;
                // No draft to verify: this is an ordinary draw. Sampling from
                // the sampler is REQUIRED when the stream is non-greedy —
                // `argmax_id` here silently emitted greedy tokens for the whole
                // non-draft stretch, so at temperature 1.0 a no-copy prompt
                // produced deterministic text identical across seeds.
                let token = match stochastic.as_deref_mut() {
                    Some(sm) if tower.has_target_mtp() => {
                        sm.draw_ds4(tower.runtime(), &logits, committed)?
                    }
                    Some(sm) => match sm.draw_row(tower.runtime(), &logits) {
                        Some(t) => t,
                        None => logits.argmax_id()?,
                    },
                    None => logits.argmax_id()?,
                };
                Some((token, 1))
            }
            Some(d) => {
                let mut vt: Vec<i32> = Vec::with_capacity(depth + 1);
                vt.push(carry as i32);
                vt.extend(d.iter().map(|&t| t as i32));
                let v_arr = tower.runtime().tokens(&vt, &[1i32, (depth + 1) as i32])?;
                let (v_mixed, _) = tower.forward_capture(&v_arr, Some(caches), true)?;
                let v_logits = tower.head(&v_mixed)?;
                let top = v_logits
                    .argmax_axis(-1)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                top.eval().map_err(|e| anyhow::anyhow!("{e}"))?;
                let mut main_tokens = top.read_ids()?;
                // Stochastic verify (Leviathan-exact PLD). The
                // prompt-lookup draft is DETERMINISTIC (q = delta), so the exact
                // speculative-sampling test collapses to accepting `d[i]` with
                // probability `p_target[d[i]]`; the first rejection emits a draw
                // from the residual `norm(max(p - one_hot, 0))`. The emitted
                // prefix is the DRAFT itself, never the target's argmax, so
                // `main_tokens` is rewritten to the tokens this round emits and
                // the accept loop below reduces to the same equality test.
                if let Some(sm) = stochastic.as_deref_mut() {
                    let mut acc = 0usize;
                    while acc < depth {
                        let row = v_logits
                            .select(1, acc as i32)?
                            .contiguous()
                            .map_err(|e| anyhow::anyhow!("{e}"))?;
                        let p = sm
                            .draft_accept_prob(tower.runtime(), &row, d[acc])
                            .unwrap_or(0.0);
                        if sm.accept_draw(p) {
                            acc += 1;
                        } else {
                            // Residual: what the target wanted instead. Cannot be
                            // `d[acc]` (its mass is zeroed), so the round breaks.
                            if let Some(rt) = sm.draw_residual(tower.runtime(), &row, d[acc]) {
                                main_tokens[acc] = rt;
                            }
                            break;
                        }
                    }
                    for i in 0..acc {
                        main_tokens[i] = d[i];
                    }
                    // Full acceptance earns one free token from the row after
                    // the last draft.
                    if acc == depth {
                        let row = v_logits
                            .select(1, depth as i32)?
                            .contiguous()
                            .map_err(|e| anyhow::anyhow!("{e}"))?;
                        if let Some(bt) = sm.draw_row(tower.runtime(), &row) {
                            main_tokens[depth] = bt;
                        }
                    }
                }
                let mut a = 0usize;
                while a < depth && main_tokens[a] == d[a] {
                    a += 1;
                }
                stats.rounds += 1;
                if a > 0 {
                    stats.hits += 1;
                }
                stats.accepted += a;
                let advance = depth + 1;
                // Commit main_tokens[..=a]; stop early on EOS or the budget.
                let mut emitted = 0usize;
                let mut eos_seen = false;
                for &t in &main_tokens[..=a] {
                    generated.push(t);
                    on_token(t)?;
                    generated_len += 1;
                    emitted += 1;
                    if is_eos(t) {
                        eos_seen = true;
                        break;
                    }
                    committed.push(t);
                    tail.push(t as i64);
                    if tail.len() > ctx_len {
                        tail.remove(0);
                    }
                    if generated_len >= max_tokens {
                        break;
                    }
                }
                stats.verified_tokens += if eos_seen { emitted - 1 } else { emitted };
                // Keep the committed prefix in the caches: the verify window
                // re-fed the carry, so `keep` rows of it are the committed
                // continuation (EOS excluded; the rollback captures make the
                // GDN/PLE states bit-exact again).
                let keep = if eos_seen {
                    (emitted - 1).max(1)
                } else {
                    emitted
                };
                for c in caches.iter_mut() {
                    tower.runtime().rollback(c, advance, keep, true)?;
                }
                if eos_seen || emitted < a + 1 {
                    return Ok((generated, stats));
                }
                carry = main_tokens[a];
                None
            }
        };
        if let Some((token, _rows)) = next {
            generated.push(token);
            on_token(token)?;
            generated_len += 1;
            committed.push(token);
            tail.push(token as i64);
            if tail.len() > ctx_len {
                tail.remove(0);
            }
            carry = token;
        }
        if generated_len % 32 == 0 {
            tower.runtime().trim_memory();
        }
    }
    Ok((generated, stats))
}

/// Greedy (temperature 0) or temperature-scaled argmax generation.
///
/// Returns the generated ids (excluding the prompt).
#[allow(clippy::too_many_arguments)]
pub fn generate<R: SpeculativeRuntime>(
    tower: &mut dyn LanguageModel<R>,
    prompt: &[u32],
    max_tokens: usize,
    sampler: &mut R::Sampler,
    pld: usize,
    on_token: &mut dyn FnMut(u32) -> anyhow::Result<()>,
) -> anyhow::Result<(Vec<u32>, GenerationStats)> {
    // Compile/warm every kernel before the timed section (the engine warms at
    // init too; without this the first prefill pays ~0.45s of Metal JIT).
    anyhow::ensure!(
        prompt.len() + max_tokens <= tower.max_position_embeddings(),
        "context overflow: prompt {} + max_tokens {} > max_position_embeddings {}",
        prompt.len(),
        max_tokens,
        tower.max_position_embeddings()
    );
    tower.warmup(prompt)?;
    let mut caches: Vec<R::Cache> = tower.new_caches();

    // Prefill in windows (bounded activations for long prompts).
    let _trace_prefill = tower.runtime().trace("prefill");
    let t0 = Instant::now();
    let last = tower.prefill(prompt, &mut caches)?;
    let logits = tower.head(&last)?;
    let token = if tower.uses_ds4_sampling() {
        sampler.draw_ds4(tower.runtime(), &logits, prompt)?
    } else {
        sampler.draw(tower.runtime(), &logits, prompt)?
    };

    // Force eval of the whole prefill graph before timing decode.
    logits.eval().map_err(|e| anyhow::anyhow!("{e}"))?;
    let prefill_seconds = t0.elapsed().as_secs_f64();

    let mut generated = Vec::new();
    generated.push(token);
    on_token(token)?;

    // PLD path: prompt-lookup drafts verified by a wide forward. Greedy keeps
    // the certified argmax verify; at temperature > 0 the verify draws each row
    // through the keyed Gumbel sample.
    if pld > 0 {
        let start = Instant::now();
        let mut committed: Vec<u32> = prompt.to_vec();
        committed.push(token);
        let (g, stats) = pld_decode(
            tower,
            &mut caches,
            &mut committed,
            token,
            max_tokens,
            generated.len(),
            pld,
            if sampler.is_greedy() {
                None
            } else {
                Some(&mut *sampler)
            },
            on_token,
        )?;
        generated.extend(g);
        eprintln!(
            "[pld.round] depth {pld} rounds {} hits {} accepted {} verified_tokens {}",
            stats.rounds, stats.hits, stats.accepted, stats.verified_tokens
        );
        let decode_seconds = start.elapsed().as_secs_f64();
        let out_stats = GenerationStats {
            prompt_tokens: prompt.len(),
            generated_tokens: generated.len(),
            prefill_seconds,
            decode_seconds,
        };
        return Ok((generated, out_stats));
    }

    let mut decode_seconds = 0f64;
    while generated.len() < max_tokens && !is_eos(*generated.last().unwrap()) {
        let _trace_step = tower.runtime().trace("decode.step");
        let token = *generated.last().unwrap();
        let start = Instant::now();
        let step = tower.runtime().tokens(&[token as i32], &[1i32, 1])?;
        let (mixed, _) = tower.forward(&step, Some(&mut caches))?;
        let last = mixed.select(1, mixed.dim(1) - 1)?;
        let logits = tower.head(&last)?;
        let mut history: Vec<u32> = prompt.to_vec();
        history.extend_from_slice(&generated);
        let token = if tower.uses_ds4_sampling() {
            sampler.draw_ds4(tower.runtime(), &logits, &history)?
        } else {
            sampler.draw(tower.runtime(), &logits, &history)?
        };
        decode_seconds += start.elapsed().as_secs_f64();
        generated.push(token);
        on_token(token)?;
        // Release pooled temporaries periodically: the pool only sweeps
        // buffer pool from `wait_until_completed`, so a long decode would pin
        // every intermediate it ever allocated.
        if generated.len() % 32 == 0 {
            tower.runtime().trim_memory();
        }
    }

    let stats = GenerationStats {
        prompt_tokens: prompt.len(),
        generated_tokens: generated.len(),
        prefill_seconds,
        decode_seconds,
    };
    Ok((generated, stats))
}

#[cfg(test)]
mod eos_contract_tests {
    use super::*;

    /// Regression: set_eos_ids used OnceLock::set — the FIRST call stuck and
    /// any later call (e.g. `--ignore-eos` clearing im_end) was silently
    /// dropped, so the serve/run decode loop kept stopping at EOS at ~199/204
    /// tokens. The last call must win.
    #[test]
    fn eos_set_is_overwritable() {
        set_eos_ids(vec![1, 2, 3]);
        assert!(is_eos(2));
        set_eos_ids(Vec::new());
        assert!(!is_eos(2));
        set_eos_ids(vec![9]);
        assert!(is_eos(9) && !is_eos(1));
        set_eos_ids(Vec::new());
    }
}
