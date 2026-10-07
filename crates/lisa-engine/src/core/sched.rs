//! Continuous-batching scheduler.
//!
//! A queue of requests is admitted into a [`ContinuousBatch`] up to `max_batch`,
//! stepped in lockstep, and retired as each finishes; the freed slot is filled
//! from the queue on the next round. A continuous-batch v2 loop (without
//! paged KV: compaction is a copy).
//!
//! Two admission sources feed the loop: the initial request list, and — via
//! [`run_streaming`] — a callback polled between decode rounds, so a request
//! that arrives while others are decoding is admitted mid-wave instead of
//! waiting for the whole wave to drain. A long admission prefill is chunked and
//! yields decode ticks between chunks (`ContinuousBatch::admit_chunked`), the
//! prefill/decode interleave; `LISA_NO_INTERLEAVE=1`
//! restores the blocking admit.

use std::collections::{HashSet, VecDeque};
use std::time::Instant;

use crate::core::batch::{ContinuousBatch, MAX_PAD_WASTE, PREFILL_CHUNK};
use crate::core::generate::is_eos;
use crate::core::sampler::Sampler;
use crate::core::session::Session;
use crate::core::tokenizer::Tokenizer;
use crate::models::LanguageModel;

pub struct Request {
    pub id: usize,
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub sampler: Sampler,
    /// `ignore_eos`: EOS never ends the reply — exactly
    /// `max_tokens` forced out. Both arms of an A/B need this.
    pub ignore_eos: bool,
}

struct Active {
    id: usize,
    max_tokens: usize,
    sampler: Sampler,
    ignore_eos: bool,
    done: bool,
    /// True while this stream has never shared a decode step with another
    /// (it collapses to the exact single-stream path).
    lone: bool,
    /// Peak batch width this stream co-ran with.
    peak: usize,
}

/// Serve `requests` continuously with at most `max_batch` streams in flight.
/// `on_token(id, token)` is called as each token is produced. Returns the
/// generated ids per request (indexed by `request.id`).
///
/// `depth > 0` switches to **per-stream speculative decoding with batch size
/// 1** (the inline MTP drafter, `maximumSpeculativeBatch == 1`): each request
/// is driven on its own
/// `Session` with `generate_mtp`, one at a time. The continuous batch is used
/// for `depth == 0`. Depth 1 is rejected: a single draft never pays for a
/// speculative round.
pub fn run(
    tower: &mut dyn LanguageModel,
    requests: Vec<Request>,
    max_batch: usize,
    depth: usize,
    on_token: &mut dyn FnMut(usize, u32) -> anyhow::Result<()>,
) -> anyhow::Result<Vec<Vec<u32>>> {
    run_with_source(tower, requests, max_batch, depth, on_token, &mut || None)
}

/// Like [`run`], but `next` is polled for additional requests between decode
/// rounds (and while the batch is empty), so late arrivals join the live
/// batch instead of waiting for the wave to drain. Returned ids are indexed
/// by `request.id`; ids are assigned by the caller and may grow over time.
pub fn run_streaming(
    tower: &mut dyn LanguageModel,
    initial: Vec<Request>,
    max_batch: usize,
    on_token: &mut dyn FnMut(usize, u32) -> anyhow::Result<()>,
    next: &mut dyn FnMut() -> Option<Request>,
) -> anyhow::Result<Vec<Vec<u32>>> {
    run_with_source(tower, initial, max_batch, 0, on_token, next)
}

fn run_with_source(
    tower: &mut dyn LanguageModel,
    requests: Vec<Request>,
    max_batch: usize,
    depth: usize,
    on_token: &mut dyn FnMut(usize, u32) -> anyhow::Result<()>,
    next: &mut dyn FnMut() -> Option<Request>,
) -> anyhow::Result<Vec<Vec<u32>>> {
    anyhow::ensure!(
        depth != 1,
        "depth 1 is not supported; use 0 (serial) or 2..=6 (MTP)"
    );
    if depth > 0 {
        return run_speculative(tower, requests, depth, on_token);
    }
    run_batch(tower, requests, max_batch, on_token, next)
}

/// Per-stream speculative rounds (one stream at a time).
fn run_speculative(
    tower: &mut dyn LanguageModel,
    requests: Vec<Request>,
    depth: usize,
    on_token: &mut dyn FnMut(usize, u32) -> anyhow::Result<()>,
) -> anyhow::Result<Vec<Vec<u32>>> {
    let n = requests.iter().map(|r| r.id).max().map_or(0, |m| m + 1);
    let mut out: Vec<Vec<u32>> = vec![Vec::new(); n];
    for mut req in requests {
        let id = req.id;
        eprintln!("[batched] slot {id} serial: mtp depth={depth} (maximumSpeculativeBatch == 1)");
        let mut sess = Session::new(tower);
        let mut cb = |t: u32| -> anyhow::Result<()> {
            out[id].push(t);
            on_token(id, t)
        };
        sess.generate_mtp_depth(
            tower,
            &req.prompt,
            req.max_tokens,
            if depth == crate::core::round_cost::AUTO_DEPTH {
                crate::core::session::MtpDepth::Auto
            } else {
                crate::core::session::MtpDepth::Fixed(depth)
            },
            None,
            if req.sampler.greedy() {
                None
            } else {
                Some(&mut req.sampler)
            },
            &mut cb,
        )?;
    }
    Ok(out)
}

fn run_batch(
    tower: &mut dyn LanguageModel,
    requests: Vec<Request>,
    max_batch: usize,
    on_token: &mut dyn FnMut(usize, u32) -> anyhow::Result<()>,
    next: &mut dyn FnMut() -> Option<Request>,
) -> anyhow::Result<Vec<Vec<u32>>> {
    let mut queue: VecDeque<Request> = requests.into_iter().collect();
    // A tower whose B>1 forward is not safe is clamped to lone streams: the
    // continuous batch then always collapses to the exact single-stream path.
    let max_batch = if tower.batch_decode_ok() {
        max_batch
    } else {
        if max_batch > 1 {
            eprintln!(
                "[batched] model batch-unsafe: clamping max_batch {max_batch} -> 1 (lone-stream path)"
            );
        }
        1
    };
    let cap_hint = queue
        .iter()
        .map(|r| r.max_tokens)
        .max()
        .unwrap_or(0)
        .max(512);
    let mut out: Vec<Vec<u32>> = Vec::new();
    let mut batch = ContinuousBatch::empty(cap_hint);
    let mut active: Vec<Active> = Vec::new();
    let interleave = std::env::var("LISA_NO_INTERLEAVE").is_err();
    // One-shot `[batched] decode engaged` marker: output equality alone cannot
    // tell a batched run from N serial ones.
    static decode_engaged: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    // One lockstep decode round: feed each active stream its last emitted
    // token, emit the next. Used both by the main loop and by the interleave
    // ticks inside admission prefills.
    fn decode_round(
        tower: &mut dyn LanguageModel,
        batch: &mut ContinuousBatch,
        active: &mut [Active],
        out: &mut [Vec<u32>],
        on_token: &mut dyn FnMut(usize, u32) -> anyhow::Result<()>,
    ) -> anyhow::Result<()> {
        let tokens: Vec<u32> = active
            .iter()
            .map(|a| *out[a.id].last().expect("each active stream has a token"))
            .collect();
        let mut samplers: Vec<Sampler> = active.iter().map(|a| a.sampler.clone()).collect();
        let next = batch.step(tower, &tokens, &mut samplers)?;
        for (i, a) in active.iter_mut().enumerate() {
            a.sampler = samplers[i].clone();
            let t = next[i];
            out[a.id].push(t);
            on_token(a.id, t)?;
            // `ignore_eos` PREVENTS EOS from ending the reply — it does not end
            // it. The inverted conjunct (`a.ignore_eos ||`) marked every stream
            // that asked to ignore EOS as DONE after one round, so a request
            // with `ignore_eos: true` (exactly what llmprobe's decode benchmark
            // and every forced-length run sends) was truncated to ~2 tokens on
            // this path and reported `finish_reason: length`. Same shape as the
            // admission tick below, which had it right.
            if (!a.ignore_eos && is_eos(t)) || out[a.id].len() >= a.max_tokens {
                a.done = true;
            }
        }
        Ok(())
    }

    loop {
        // Admit from the queue (then from the live source) while there is room.
        // A request the pad-waste cap rejects is deferred (pushed back) and
        // retried after the next decode round — specs/13 §3b.
        let mut deferred: Vec<Request> = Vec::new();
        while batch.len() < max_batch {
            let Some(req) = queue.pop_front().or_else(|| next()) else {
                break;
            };
            if !batch.admits_within_pad_waste(req.prompt.len()) {
                eprintln!(
                    "[batched] slot {} deferred: padded attention would exceed {}x useful (kv skew {} vs {:?})",
                    req.id,
                    MAX_PAD_WASTE,
                    req.prompt.len(),
                    batch.kv_lens()
                );
                deferred.push(req);
                continue;
            }
            let id = req.id;
            if out.len() <= id {
                out.resize(id + 1, Vec::new());
            }
            let mut sampler = req.sampler;
            let lone = batch.is_empty();
            // Interleave: a long prefill for a new admit yields decode ticks
            // between chunks so already-decoding streams keep moving.
            let t_admit = Instant::now();
            let first = if interleave && !lone && req.prompt.len() > PREFILL_CHUNK {
                let mut last_chunk = Instant::now();
                let mut ticks = 0usize;
                let first =
                    batch.admit_chunked(tower, &req.prompt, &mut sampler, &mut |b, t| {
                        let chunk_wall = last_chunk.elapsed();
                        last_chunk = Instant::now();
                        let budget = chunk_wall / 4;
                        let start = Instant::now();
                        let mut n = 0usize;
                        while n < 8 && start.elapsed() < budget && active.iter().any(|a| !a.done) {
                            decode_round(t, b, &mut active, &mut out, on_token)?;
                            n += 1;
                        }
                        ticks += n;
                        Ok(())
                    })?;
                if ticks > 0 {
                    eprintln!(
                        "[batched] slot {id}: prefill interleave ({} chunks, {ticks} decode ticks, {:.2}s)",
                        req.prompt.len().div_ceil(PREFILL_CHUNK),
                        t_admit.elapsed().as_secs_f64()
                    );
                }
                first
            } else {
                batch.admit(tower, &req.prompt, &mut sampler)?
            };
            emit(&mut out, on_token, id, first)?;
            let done = (is_eos(first) && !req.ignore_eos) || out[id].len() >= req.max_tokens;
            active.push(Active {
                id,
                max_tokens: req.max_tokens,
                sampler,
                ignore_eos: req.ignore_eos,
                done,
                lone,
                peak: batch.len(),
            });
        }

        // Re-queue the cap rejects at the front, in their original order.
        for req in deferred.into_iter().rev() {
            queue.push_front(req);
        }

        // Retire the streams that are already finished (including ones admitted
        // above that hit EOS immediately); log each slot's batching verdict.
        retire_finished(tower, &mut batch, &mut active)?;
        if batch.is_empty() {
            if queue.is_empty() && next().is_none() {
                break;
            }
            continue;
        }

        decode_round(tower, &mut batch, &mut active, &mut out, on_token)?;
        if batch.len() > 1 && !decode_engaged.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!("[batched] decode engaged (slots={})", batch.len());
        }
        for a in active.iter_mut() {
            a.peak = a.peak.max(batch.len());
            if batch.len() > 1 {
                a.lone = false;
            }
        }
        retire_finished(tower, &mut batch, &mut active)?;
    }

    Ok(out)
}

fn emit(
    out: &mut [Vec<u32>],
    on_token: &mut dyn FnMut(usize, u32) -> anyhow::Result<()>,
    id: usize,
    t: u32,
) -> anyhow::Result<()> {
    out[id].push(t);
    on_token(id, t)
}

fn retire_finished(
    tower: &mut dyn LanguageModel,
    batch: &mut ContinuousBatch,
    active: &mut Vec<Active>,
) -> anyhow::Result<()> {
    if active.is_empty() || active.iter().all(|a| !a.done) {
        return Ok(());
    }
    // Verdicts for the streams leaving this round (every slot that ran serial
    // gets a logged reason).
    let mut logged: HashSet<usize> = HashSet::new();
    for a in active.iter().filter(|a| a.done) {
        if logged.insert(a.id) {
            if a.lone {
                eprintln!(
                    "[batched] slot {} serial: lone stream (single-stream collapse path)",
                    a.id
                );
            } else {
                eprintln!("[batched] slot {}: batched (peak {} streams)", a.id, a.peak);
            }
        }
    }
    let keep: Vec<bool> = active.iter().map(|a| !a.done).collect();
    batch.retire(tower, &keep)?;
    let mut it = keep.into_iter();
    active.retain(|_| it.next().unwrap_or(false));
    Ok(())
}

/// Convenience: decode a scheduler request's prompt from text and run it.
pub fn request_from_text(
    id: usize,
    tok: &Tokenizer,
    text: &str,
    raw: bool,
    max_tokens: usize,
    sampler: Sampler,
) -> anyhow::Result<Request> {
    let prompt = if raw {
        text.to_string()
    } else {
        crate::core::tokenizer::generation_prompt(&[("user".to_string(), text.to_string())])
    };
    Ok(Request {
        id,
        prompt: tok.encode(&prompt, false)?,
        max_tokens,
        sampler,
        ignore_eos: false,
    })
}
