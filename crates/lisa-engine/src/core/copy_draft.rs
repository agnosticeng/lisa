//! Prompt-lookup (context-copy) drafting shared by the MTP rounds and the
//! plain serial PLD path: a rolling k-gram index over the committed token
//! sequence; when the trailing k-gram recurs, the tokens that followed the
//! earlier occurrence are a strong (but always verified) draft.

/// Prompt-lookup n-gram size for the context-copy drafter.
pub const COPY_K: usize = 3;

/// Longest copy draft a round may propose.
///
/// The draft used to be capped at the MTP `depth` (3 with the shipped
/// controller), and measured copy rounds were already at ceiling — 39/55 rounds
/// firing (71 %) at a mean 3.05 tokens, i.e. FULLY accepting a 3-wide window.
/// Capping there threw away the rest of the match. lisa drafts up to 7, which
/// keeps the verify at S = w + 1 <= 8 — exactly the split-K fast lane's
/// coverage since its MROWS range extended to M=8 (specs/08: in-situ S=8
/// verify-audit 72.4 -> 57.5 ms, -20.6 %, bit-exact). S = 9 still rides the
/// tiled lane's penalty; it unlocks only via the MPP-linkage follow-up.
/// The controller-picked chain depth stays capped at 6 (the head's acceptance
/// per index collapses past ~5); only copy hits — whose per-index acceptance
/// is 1.000 across 6-12 — reach 7.
pub const COPY_LEN_MAX: usize = 7;

/// Effective copy-length cap: `COPY_LEN_MAX`, or `LISA_COPY_LEN_MAX` when it
/// raises it. A value of 0, an unset variable, or a parse error all fall back
/// to the const default — the default path is byte-identical.
///
/// Measured test knob for the S >= 8 verify lane (specs/08): a copy hit wider
/// than 6 widens the round past the split-K fast lane (S = w + 1 > 7) onto the
/// tiled verify lane, whose wide end prices near parity with the fast lane.
pub fn copy_len_max() -> usize {
    match std::env::var("LISA_COPY_LEN_MAX") {
        Ok(s) => {
            let v = s.trim().parse::<usize>().unwrap_or(COPY_LEN_MAX);
            if v == 0 {
                COPY_LEN_MAX
            } else {
                v
            }
        }
        Err(_) => COPY_LEN_MAX,
    }
}

/// A rolling index from a k-gram to the latest position it started at, over
/// the committed token sequence. Used by the context-copy drafter: when the
/// trailing k-gram recurs, the tokens that followed that occurrence are a
/// strong (but verified) draft.
pub struct CopyIndex {
    pub k: usize,
    map: std::collections::HashMap<u64, Vec<u32>>,
    indexed: usize,
}

impl CopyIndex {
    pub fn new(k: usize) -> Self {
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
    pub fn extend(&mut self, s: &[u32], upto: usize) {
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
    pub fn lookup(&self, s: &[u32]) -> Vec<usize> {
        if s.len() < self.k {
            return Vec::new();
        }
        self.map
            .get(&self.hash(&s[s.len() - self.k..]))
            .map(|v| v.iter().rev().map(|&p| p as usize).collect())
            .unwrap_or_default()
    }
}

/// Echo score of a token sequence: the fraction of 4-gram start positions
/// whose 4-gram also occurs EARLIER in the sequence. ~1.0 on repetitive
/// (echo/copy) text, ~0 on novel prose. This is the router signal: prompt-lookup
/// drafts only fire where n-grams recur, so the score predicts the PLD hit rate.
pub fn echo_score(s: &[u32]) -> f32 {
    let k = COPY_K;
    if s.len() < k + 1 {
        return 0.0;
    }
    let mut seen: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut idx = CopyIndex::new(k);
    let mut hits = 0usize;
    let mut total = 0usize;
    for start in 0..=s.len() - k {
        let h = idx.hash(&s[start..start + k]);
        if !seen.insert(h) {
            hits += 1;
        }
        total += 1;
    }
    idx.map.clear();
    if total == 0 {
        0.0
    } else {
        hits as f32 / total as f32
    }
}

/// A `depth`-token copy proposal for the token after the committed tail, or
/// `None` when the trailing k-gram has no earlier occurrence with enough
/// committed continuation.
pub fn copy_lookup(s: &[u32], idx: &CopyIndex, k: usize, depth: usize) -> Option<Vec<u32>> {
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

/// Minimum committed continuation a copy source must offer to be trusted. A
/// 1-token continuation is a coincidence, not a copy: the guard rejects it.
pub const MIN_COPY_CONT: usize = 2;

/// Guarded variant of [`copy_lookup`] (O4). Rejects the two degenerate sources
/// that make prompt-lookup emit a false draft:
/// 1. **self-overlap** — the candidate's source window overlaps the trailing
///    k-gram being predicted (`p + k > s.len() - k`), so the "continuation" is
///    just the text we are trying to predict echoed back;
/// 2. **short continuation** — fewer than [`MIN_COPY_CONT`] committed tokens.
pub fn copy_lookup_guarded(
    s: &[u32],
    idx: &CopyIndex,
    k: usize,
    depth: usize,
) -> Option<Vec<u32>> {
    let want = depth.max(MIN_COPY_CONT);
    if k != idx.k || s.len() < k + want {
        return None;
    }
    let query_start = s.len() - k;
    for p in idx.lookup(s) {
        // (1) never copy from a window that overlaps the query itself. The
        // source must END strictly before the query begins.
        if p + k >= query_start {
            continue;
        }
        let start = p + k;
        if start + want <= s.len() {
            return Some(s[start..start + depth.min(want)].to_vec());
        }
    }
    None
}

/// O4: geometric width ramp for the PLD draft. A round that accepted every
/// drafted token widens the next width (`w -> 2w + 1`, capped at `max`); a
/// partial accept or a copy-miss snaps back to the base. Rounded DOWNWARDS to
/// the fast verify lane — never up (see `cli::FAST_S_MAX`).
pub struct WidthRamp {
    base: usize,
    max: usize,
    w: usize,
}

impl WidthRamp {
    pub fn new(base: usize, max: usize) -> Self {
        let base = base.clamp(1, max);
        Self {
            base,
            max,
            w: base,
        }
    }
    pub fn width(&self) -> usize {
        self.w
    }
    /// Every drafted token was accepted: widen.
    pub fn on_full_accept(&mut self) {
        self.w = (2 * self.w + 1).min(self.max);
    }
    /// A partial accept or a copy-miss: snap back to the base.
    pub fn on_partial(&mut self) {
        self.w = self.base;
    }
}

/// O4: decides whether the copy-draft is still worth its verify cost, from the
/// observed throughput of the two paths. Shape mirrors the MTP `DepthController`:
/// an EWMA (window 32 => alpha = 2/33) on both, with a headroom margin and a
/// one-way hysteresis latch (once dropped, it stays dropped until the copy path
/// clears the margin again — no flapping).
pub struct CopyDraftGate {
    alpha: f32,
    copied: Option<f32>,
    plain: Option<f32>,
    headroom: f32,
    armed: bool,
}

impl Default for CopyDraftGate {
    fn default() -> Self {
        Self::new()
    }
}

impl CopyDraftGate {
    /// EWMA window 32 (`alpha = 2/(32+1)`); keep copying only while the copy
    /// path is at least `1.5x` the plain path.
    pub fn new() -> Self {
        Self {
            alpha: 2.0 / 33.0,
            copied: None,
            plain: None,
            headroom: 1.5,
            armed: true,
        }
    }

    fn ewma(prev: &mut Option<f32>, x: f32, alpha: f32) -> f32 {
        let v = match *prev {
            Some(p) => p + alpha * (x - p),
            None => x,
        };
        *prev = Some(v);
        v
    }

    /// Feed one round's observations (tok/ms on each path) and return whether
    /// the copy-draft should stay on.
    pub fn note(&mut self, copied_tok_per_ms: f32, plain_tok_per_ms: f32) -> bool {
        let c = Self::ewma(&mut self.copied, copied_tok_per_ms, self.alpha);
        let p = Self::ewma(&mut self.plain, plain_tok_per_ms, self.alpha);
        if self.armed {
            // Arm stays on until the copy path fails to clear the margin.
            if c < self.headroom * p {
                self.armed = false;
            }
        } else if c >= self.headroom * p {
            self.armed = true;
        }
        self.armed
    }

    pub fn armed(&self) -> bool {
        self.armed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn echo_score_separates_repetition_from_novelty() {
        let echo: Vec<u32> = (0..200).map(|i| (i % 7) as u32).collect();
        assert!(echo_score(&echo) > 0.9, "echo {}", echo_score(&echo));
        let novel: Vec<u32> = (0u32..200).map(|i| (i.wrapping_mul(2654435761) % 99991) as u32).collect();
        assert!(echo_score(&novel) < 0.05, "novel {}", echo_score(&novel));
        assert_eq!(echo_score(&[1, 2, 3]), 0.0);
    }

    /// The mixed regime (novel paragraphs + repeated summary blocks) must land
    /// strictly between the two pure regimes — the router's discrimination
    /// margin, measured on the bench prompts (specs/router).
    #[test]
    fn echo_score_mixed_is_intermediate() {
        let novel: Vec<u32> = (0u32..400).map(|i| (i.wrapping_mul(2654435761) % 99991) as u32).collect();
        let block: Vec<u32> = (0..40).map(|i| (i % 5) as u32).collect();
        let mut mixed = novel.clone();
        for _ in 0..4 {
            mixed.extend_from_slice(&block);
        }
        let e_novel = echo_score(&novel);
        let e_mixed = echo_score(&mixed);
        assert!(
            e_mixed > e_novel && e_mixed < 0.95,
            "novel {e_novel} mixed {e_mixed}"
        );
    }

    /// O4: the guards reject the degenerate sources that make PLD emit a false
    /// draft, and still accept a genuine copy. Fixtures are built FROM `COPY_K`
    /// so the test survives a key-length change.
    #[test]
    fn guarded_rejects_self_overlap_and_accepts_a_real_copy() {
        let k = COPY_K;
        let unit: Vec<u32> = (1..=k as u32).collect();
        // The only occurrence of the trailing k-gram IS the text itself: the
        // "continuation" would be the query echoed back, so no draft.
        let mut s = unit.clone();
        s.extend_from_slice(&unit);
        let mut idx = CopyIndex::new(k);
        idx.extend(&s, s.len() - k);
        assert!(copy_lookup_guarded(&s, &idx, k, 2).is_none());

        // A genuine earlier occurrence with a committed continuation IS copied.
        let mut s2 = unit.clone();
        s2.extend_from_slice(&[9, 9]);
        s2.extend_from_slice(&unit);
        let mut idx2 = CopyIndex::new(k);
        idx2.extend(&s2, s2.len() - k);
        assert_eq!(copy_lookup_guarded(&s2, &idx2, k, 2), Some(vec![9, 9]));
    }

    /// The decisive one: a model that ECHOES a long repeated prompt must get a
    /// copy draft. This is the "predictable content" path where copy drafting
    /// has the largest payoff.
    #[test]
    fn copy_fires_on_an_echo_of_a_long_prompt() {
        let k = COPY_K;
        let unit: Vec<u32> = (1..=20u32).collect();
        let mut prompt: Vec<u32> = Vec::new();
        for _ in 0..40 {
            prompt.extend_from_slice(&unit); // 800 tokens, strictly repeating
        }
        // The model echoes the prompt: committed = prompt + the first tokens again.
        let mut committed = prompt.clone();
        committed.extend_from_slice(&unit[..6]);
        let mut idx = CopyIndex::new(k);
        idx.extend(&committed, committed.len() - k);
        let c = copy_lookup_guarded(&committed, &idx, k, 6);
        assert!(
            c.is_some(),
            "an echo of a long repeated prompt must yield a copy draft"
        );
        // The trailing k-gram is the tail of `unit[..6]` (…[4,5,6]); the copy
        // must return what FOLLOWED that k-gram the last time it appeared, i.e.
        // the continuation of the repeating unit: unit[6..12].
        assert_eq!(c.unwrap(), unit[6..12].to_vec());
    }

    #[test]
    fn width_ramp_widens_on_accept_snaps_back_on_partial() {
        let mut r = WidthRamp::new(4, 7);
        assert_eq!(r.width(), 4);
        r.on_full_accept();
        assert_eq!(r.width(), 7, "4 -> 9 capped at the fast lane 7");
        r.on_full_accept();
        assert_eq!(r.width(), 7, "cap holds");
        r.on_partial();
        assert_eq!(r.width(), 4);
    }

    #[test]
    fn gate_latches_off_when_the_copy_path_loses() {
        let mut g = CopyDraftGate::new();
        assert!(g.armed(), "starts armed");
        // Copy path well below the 1.5x margin => dropped.
        for _ in 0..40 {
            g.note(10.0, 20.0);
        }
        assert!(!g.armed(), "dropped when copy < 1.5x plain");
        // Back above the margin => re-armed.
        for _ in 0..40 {
            g.note(40.0, 20.0);
        }
        assert!(g.armed(), "re-armed when copy >= 1.5x plain");
    }
}
