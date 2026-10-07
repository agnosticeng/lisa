//! The draft-ranking oracle (specs/08 tree drafting).
//!
//! The MTP draft head proposes exactly ONE token per step and verify rejects
//! the round at the first mismatch, so a step where the head ranked the true
//! token #2 costs the whole round tail. This module records the head's own
//! re-scored candidate set for every draft step, letting the driver rank the
//! target's true token inside it *after* verify answers — "had the head
//! proposed its #2 there, would that round have extended?".
//!
//! Diagnostic only: nothing recorded here ever feeds a scored path (the
//! proposal, the verify forward, and the commit are untouched). Every hook is
//! a no-op unless [`enable`] was called, and [`enable`] is called from the
//! CLI (`lisa run --oracle`), not from an env var — the repo keeps its env
//! flag set closed.

use lisa_mlx::Array;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

static ON: AtomicBool = AtomicBool::new(false);
/// The distribution pushed by the most recent draft step on this thread.
static LAST: Mutex<Option<DraftScores>> = Mutex::new(None);

/// One draft step's candidate set as the head itself scored it: `cands` are
/// vocabulary ids, `scores` the exact re-scored logits (same order).
#[derive(Clone, Debug)]
pub struct DraftScores {
    pub cands: Vec<u32>,
    pub scores: Vec<f32>,
}

fn lock_last() -> std::sync::MutexGuard<'static, Option<DraftScores>> {
    LAST.lock().unwrap_or_else(|e| e.into_inner())
}

/// Turn the oracle on (clears any stale record). Call once, before
/// generation; the driver prints the summary at the end of the round loop.
pub fn enable() {
    ON.store(true, Ordering::Relaxed);
    *lock_last() = None;
}

/// Whether recording is on.
pub fn enabled() -> bool {
    ON.load(Ordering::Relaxed)
}

/// Record the most recent draft step's distribution (a no-op when off).
pub fn push(cands: Vec<u32>, scores: Vec<f32>) {
    if enabled() {
        *lock_last() = Some(DraftScores { cands, scores });
    }
}

/// The distribution pushed by the most recent draft step, if any.
pub fn last() -> Option<DraftScores> {
    lock_last().clone()
}

/// Record from device arrays: `scores` is the head's `[1, V]` (or `[1, K]`)
/// readout, `ids` the matching vocabulary ids (row -> id). `ids == None`
/// means the rows already ARE vocabulary ids (`0..V`).
pub fn push_arrays(ids: Option<&Array>, scores: &Array) -> anyhow::Result<()> {
    if !enabled() {
        return Ok(());
    }
    let scores = scores.as_dtype(lisa_mlx::Dtype::Float32)?;
    scores.eval()?;
    let scores = scores.as_slice::<f32>().to_vec();
    let cands = match ids {
        Some(a) => {
            let a = a.as_dtype(lisa_mlx::Dtype::Int32)?;
            a.eval()?;
            a.as_slice::<i32>().iter().map(|&x| x as u32).collect()
        }
        None => (0..scores.len() as u32).collect(),
    };
    push(cands, scores);
    Ok(())
}

/// 1-based rank of `token` in a recorded candidate set.
///
/// `proposal` is the token the head actually proposed (its argmax): a score
/// tie with the argmax loses, so a tied-but-not-proposed token ranks 2, never
/// 1. `None` means the token is outside the head's candidate set entirely (a
/// coarse shortlist miss) — unreachable for the top-`K` readouts only when
/// `K` covers the vocab.
pub fn rank(rec: &DraftScores, token: u32, proposal: u32) -> Option<usize> {
    let i = rec.cands.iter().position(|&c| c == token)?;
    let s = rec.scores[i];
    let mut better = rec.scores.iter().filter(|&&x| x > s).count();
    if better == 0 && token != proposal {
        better = 1; // tie lost to the argmax
    }
    Some(better + 1)
}

/// A ranked first-failure sample source: one round's chain proposals, in
/// index order, as ranked by the head's own distribution.
#[derive(Clone, Copy, Debug)]
pub enum StepRank {
    /// The proposal came from the draft head: its 1-based rank in the head's
    /// own re-scored set (`None` = the target's token is not in the head's
    /// candidate set at all — a coarse-shortlist miss, unrankable).
    Chain(Option<usize>),
    /// The proposal came from the prompt-lookup copy — there is no draft
    /// distribution to rank, so the round says nothing about the head.
    Copy,
    /// The proposal came from the head but no distribution was recorded —
    /// an instrumentation gap, reported so it can never hide in the numbers.
    NoRecord,
}

/// One round, kept for the end-of-run oracle summary.
#[derive(Clone, Debug)]
pub struct RoundSample {
    /// Draft depth used by this round (the auto controller re-picks it).
    pub depth: usize,
    /// The linear chain's accepted-prefix length (what verify returned).
    pub accepted: usize,
    /// Per chain index: where the target's true token ranked in the head.
    pub ranks: Vec<StepRank>,
}

const KS: [usize; 3] = [2, 4, 8];

/// End-of-run oracle summary (specs/08): where the target's true token sat
/// in the draft head's own ranking on the rounds the linear chain rejected,
/// and what a top-`K` branching draft would have accepted instead.
///
/// The reported gain is a FLOOR, not the tree's yield: repairing index `a`
/// is worth `+1` token for certain, and everything the tree drafts beyond
/// the repair is unmeasured here (it needs re-drafting conditioned on the
/// corrected token — phase 2).
pub fn log_summary(samples: &[RoundSample], accept_hist: &[usize]) {
    if samples.is_empty() || accept_hist.is_empty() {
        return;
    }
    let rounds = accept_hist.len();
    let d = samples.iter().map(|s| s.depth).max().unwrap_or(0);
    if d == 0 {
        return;
    }
    let total: usize = accept_hist.iter().sum();
    let mut base = vec![0usize; d];
    let mut hits = vec![[0usize; 3]; d];
    let mut cnt = vec![0usize; d];
    let mut first_fail = 0usize;
    let mut repair = [0usize; 3];
    let mut ranked: Vec<usize> = Vec::new();
    let mut copy_fail = 0usize;
    let mut outside = 0usize;
    let mut no_rec = 0usize;
    let mut down = 0usize;
    let mut down_le2 = 0usize;
    for (s, &a) in samples.iter().zip(accept_hist) {
        let mut rep = [false; 3];
        if a < s.depth {
            first_fail += 1;
            match s.ranks.get(a) {
                Some(StepRank::Chain(Some(r))) => {
                    ranked.push(*r);
                    for (ki, &k) in KS.iter().enumerate() {
                        if *r <= k {
                            rep[ki] = true;
                            repair[ki] += 1;
                        }
                    }
                }
                // The true token is not in the head's candidate set: no
                // top-k branch of THIS head can ever carry it.
                Some(StepRank::Chain(None)) => outside += 1,
                Some(StepRank::NoRecord) | None => no_rec += 1,
                _ => copy_fail += 1,
            }
            for i in (a + 1)..s.depth {
                if let Some(StepRank::Chain(Some(r))) = s.ranks.get(i) {
                    down += 1;
                    if *r <= 2 {
                        down_le2 += 1;
                    }
                }
            }
        }
        for i in 0..d {
            if i >= s.depth {
                continue;
            }
            cnt[i] += 1;
            if i < a {
                base[i] += 1;
                for h in hits[i].iter_mut() {
                    *h += 1;
                }
            } else if i == a {
                for (ki, on) in rep.iter().enumerate() {
                    if *on {
                        hits[i][ki] += 1;
                    }
                }
            }
        }
    }
    let pct = |n: usize, den: usize| -> f64 {
        if den == 0 {
            0.0
        } else {
            100.0 * n as f64 / den as f64
        }
    };
    let fmt = |v: &[usize], c: &[usize]| -> String {
        v.iter()
            .zip(c)
            .map(|(&x, &n)| format!("{:.3}", if n == 0 { 0.0 } else { x as f64 / n as f64 }))
            .collect::<Vec<_>>()
            .join(" ")
    };
    eprintln!(
        "[mtp.oracle] rounds {rounds} accepted {total} (mean {:.2}/round); \
         first failures {first_fail}: head-ranked {}, outside-the-head {outside}, \
         copy-proposal {copy_fail}, no-record {no_rec}",
        total as f64 / rounds as f64,
        ranked.len()
    );
    // The denominator every repair rate uses: every first failure where the
    // head itself made the proposal (a copy hit has no distribution; a
    // no-record is an instrumentation gap). `outside` stays in it — a top-k
    // branch cannot carry a token the head never ranked.
    let chain_fail = ranked.len() + outside;
    if chain_fail > 0 {
        let mut b = [0usize; 6];
        for &r in &ranked {
            b[match r {
                1 => 0,
                2 => 1,
                3..=4 => 2,
                5..=8 => 3,
                9..=32 => 4,
                _ => 5,
            }] += 1;
        }
        eprintln!(
            "[mtp.oracle] first-failure rank of the true token: 1:{} 2:{} 3-4:{} \
             5-8:{} 9-32:{} >32:{} outside:{} (n={chain_fail})",
            b[0],
            b[1],
            b[2],
            b[3],
            b[4],
            b[5],
            outside
        );
        for (ki, &k) in KS.iter().enumerate() {
            let gain = repair[ki] as f64 / rounds as f64;
            let base_mean = total as f64 / rounds as f64;
            eprintln!(
                "[mtp.oracle] top-{k} repair floor: {repair}/{} ({:.1}% of the \
                 {chain_fail} head first failures) -> +{gain:.3} tok/round \
                 ({:.1}% of the linear {base_mean:.2})",
                chain_fail,
                pct(repair[ki], chain_fail),
                100.0 * gain / base_mean.max(1e-9),
                repair = repair[ki]
            );
        }
        eprintln!(
            "[mtp.oracle] floor = the repaired index alone (+1 token: the repaired \
             row's own next token, which the chain can only emit next round). The \
             repaired branch's CHILDREN are unmeasured here — the chain never drafts \
             from the corrected token, so what they add is phase 2."
        );
    }
    if down > 0 {
        eprintln!(
            "[mtp.oracle] downstream-of-failure chain ranks <=2: {}/{} ({:.1}%) — \
             conditioned on a WRONG prefix, reported for context only, NOT in the gain",
            down_le2,
            down,
            pct(down_le2, down)
        );
    }
    eprintln!("[mtp.oracle] per-index acceptance, linear chain: [{}]", fmt(&base, &cnt));
    for (ki, &k) in KS.iter().enumerate() {
        let col: Vec<usize> = hits.iter().map(|h| h[ki]).collect();
        eprintln!(
            "[mtp.oracle] per-index acceptance, oracle top-{k}:  [{}]",
            fmt(&col, &cnt)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec() -> DraftScores {
        DraftScores {
            cands: vec![7, 9, 11],
            scores: vec![0.5, 0.3, 0.1],
        }
    }

    #[test]
    fn rank_orders_by_score_and_counts_the_proposal_first() {
        let r = rec();
        assert_eq!(rank(&r, 7, 7), Some(1));
        assert_eq!(rank(&r, 9, 7), Some(2));
        assert_eq!(rank(&r, 11, 7), Some(3));
        assert_eq!(rank(&r, 42, 7), None);
    }

    #[test]
    fn a_tie_the_head_did_not_pick_ranks_two() {
        let r = DraftScores {
            cands: vec![7, 9],
            scores: vec![0.5, 0.5],
        };
        assert_eq!(rank(&r, 7, 7), Some(1));
        assert_eq!(rank(&r, 9, 7), Some(2));
    }
}
