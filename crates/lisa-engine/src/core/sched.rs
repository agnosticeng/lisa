//! Continuous-batching scheduler.
//!
//! A queue of requests is admitted into a [`ContinuousBatch`] up to `max_batch`,
//! stepped in lockstep, and retired as each finishes; the freed slot is filled
//! from the queue on the next round. This is the serving-equivalent of the
//! reference engine's CBv2 loop (without paged KV: compaction is a copy).

use std::collections::VecDeque;

use crate::core::batch::ContinuousBatch;
use crate::core::generate::is_eos;
use crate::models::LanguageModel;
use crate::core::sampler::Sampler;
use crate::core::session::Session;
use crate::core::tokenizer::Tokenizer;

pub struct Request {
    pub id: usize,
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub sampler: Sampler,
}

struct Active {
    id: usize,
    max_tokens: usize,
    sampler: Sampler,
    done: bool,
}

/// Serve `requests` continuously with at most `max_batch` streams in flight.
/// `on_token(id, token)` is called as each token is produced. Returns the
/// generated ids per request (indexed by `request.id`).
///
/// `depth > 0` switches to **per-stream speculative decoding with batch size
/// 1**, matching the reference engine's inline MTP drafter
/// (`maximumSpeculativeBatch == 1`): each request is driven on its own
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
    anyhow::ensure!(depth != 1, "depth 1 is not supported; use 0 (serial) or 2..=6 (MTP)");
    if depth > 0 {
        return run_speculative(tower, requests, depth, on_token);
    }
    run_batch(tower, requests, max_batch, on_token)
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
    for req in requests {
        anyhow::ensure!(req.sampler.greedy(), "MTP requires greedy sampling");
        let id = req.id;
        let mut sess = Session::new(tower);
        let mut cb = |t: u32| -> anyhow::Result<()> {
            out[id].push(t);
            on_token(id, t)
        };
        sess.generate_mtp(tower, &req.prompt, req.max_tokens, depth, &mut cb)?;
    }
    Ok(out)
}

fn run_batch(
    tower: &mut dyn LanguageModel,
    requests: Vec<Request>,
    max_batch: usize,
    on_token: &mut dyn FnMut(usize, u32) -> anyhow::Result<()>,
) -> anyhow::Result<Vec<Vec<u32>>> {
    let n = requests.len();
    let max_out = requests.iter().map(|r| r.max_tokens).max().unwrap_or(0);
    let mut out: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut pending: VecDeque<Request> = requests.into_iter().collect();
    let mut batch = ContinuousBatch::empty(max_out);
    let mut active: Vec<Active> = Vec::new();

    let emit = |out: &mut Vec<Vec<u32>>,
                on_token: &mut dyn FnMut(usize, u32) -> anyhow::Result<()>,
                id: usize,
                t: u32|
     -> anyhow::Result<()> {
        out[id].push(t);
        on_token(id, t)
    };

    loop {
        // Admit from the queue while there is room.
        while batch.len() < max_batch {
            let Some(mut req) = pending.pop_front() else { break };
            let first = batch.admit(tower, &req.prompt, &mut req.sampler)?;
            emit(&mut out, on_token, req.id, first)?;
            let done = is_eos(first) || out[req.id].len() >= req.max_tokens;
            active.push(Active {
                id: req.id,
                max_tokens: req.max_tokens,
                sampler: req.sampler,
                done,
            });
        }

        // Retire the streams that are already finished (including ones admitted
        // above that hit EOS immediately).
        retire_finished(tower, &mut batch, &mut active)?;
        if batch.is_empty() {
            if pending.is_empty() {
                break;
            }
            continue;
        }

        // One lockstep step: feed each stream its last emitted token.
        let tokens: Vec<u32> = active
            .iter()
            .map(|a| *out[a.id].last().expect("each active stream has a token"))
            .collect();
        let mut samplers: Vec<Sampler> = active.iter().map(|a| a.sampler.clone()).collect();
        let next = batch.step(tower, &tokens, &mut samplers)?;
        for (i, a) in active.iter_mut().enumerate() {
            a.sampler = samplers[i].clone();
            let t = next[i];
            emit(&mut out, on_token, a.id, t)?;
            if is_eos(t) || out[a.id].len() >= a.max_tokens {
                a.done = true;
            }
        }
        retire_finished(tower, &mut batch, &mut active)?;
    }

    Ok(out)
}

fn retire_finished(
    tower: &mut dyn LanguageModel,
    batch: &mut ContinuousBatch,
    active: &mut Vec<Active>,
) -> anyhow::Result<()> {
    if active.is_empty() || active.iter().all(|a| !a.done) {
        return Ok(());
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
    })
}
