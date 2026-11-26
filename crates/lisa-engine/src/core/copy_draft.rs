//! Context-copy drafting for MTP rounds: a rolling k-gram index over the
//! committed tokens. When the trailing k-gram recurs, the tokens that followed
//! its earlier occurrence form a draft, which the target always verifies.

/// k-gram size of the copy index.
pub const COPY_K: usize = 3;

/// Longest copy draft a round may propose. The verify width stays at
/// `S = draft + 1 <= 8`, the split-K verify lane's coverage. Independent MTP
/// chains stay capped at depth 6; only copy hits reach seven.
pub const COPY_LEN_MAX: usize = 7;

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

/// Minimum committed continuation a copy source must offer to be trusted. A
/// 1-token continuation is a coincidence, not a copy: the guard rejects it.
pub const MIN_COPY_CONT: usize = 2;

/// A copy proposal of up to `depth` tokens for the token after the committed
/// tail. Rejects the two degenerate sources that would yield a false draft:
/// 1. **self-overlap** — the candidate's source window overlaps the trailing
///    k-gram being predicted (`p + k > s.len() - k`), so the "continuation" is
///    just the text we are trying to predict echoed back;
/// 2. **short continuation** — fewer than [`MIN_COPY_CONT`] committed tokens.
pub fn copy_lookup_guarded(s: &[u32], idx: &CopyIndex, k: usize, depth: usize) -> Option<Vec<u32>> {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The guards reject degenerate sources and still accept a genuine copy. Fixtures are built FROM `COPY_K`
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
}
