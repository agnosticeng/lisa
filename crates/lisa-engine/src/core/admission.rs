//! O3: byte-level admission control.
//!
//! The only pre-batch gate today is a token-ratio pad-waste check; nothing sums
//! the bytes an in-flight request will hold. This is that missing byte budget
//! for live generation state (invariant
//! `admission.budget <= prompt_memory.budget` and `admission.used == held`).
//!
//! The rule is refuse-don't-attempt: state beyond the budget is refused (a 503
//! upstream), never allocated and hoped for.

/// A byte budget for live generation state.
pub struct MemoryBudget {
    budget: usize,
    used: usize,
}

impl MemoryBudget {
    pub fn new(budget: usize) -> Self {
        Self { budget, used: 0 }
    }

    /// Budget from the host: `LISA_RAM_CAP_GB`, else physical RAM − 8 GiB.
    pub fn from_host() -> Self {
        Self::new(lisa_mlx::memory::ram_budget())
    }

    pub fn budget(&self) -> usize {
        self.budget
    }
    pub fn used(&self) -> usize {
        self.used
    }
    pub fn headroom(&self) -> usize {
        self.budget.saturating_sub(self.used)
    }

    /// Admit `bytes` of new state, or refuse (the caller answers 503).
    pub fn try_admit(&mut self, bytes: usize) -> bool {
        if bytes > self.headroom() {
            return false;
        }
        self.used += bytes;
        true
    }

    /// Release on request completion — must pair with every successful
    /// `try_admit` (RAII at the call site).
    pub fn release(&mut self, bytes: usize) {
        self.used = self.used.saturating_sub(bytes);
    }
}

/// Bytes one stream of `tokens` will hold: KV rows, the model's per-stream state
/// (GDN conv/SSM, indexer tape), and a cache-growth overshoot factor.
///
/// The overshoot covers the allocator's doubling growth (measured ×2 on this
/// runtime); a value below 1 is clamped to 1.
pub fn stream_bytes(
    tokens: usize,
    kv_bytes_per_token: usize,
    state_bytes: usize,
    overshoot: usize,
) -> usize {
    tokens
        .saturating_mul(kv_bytes_per_token)
        .saturating_mul(overshoot.max(1))
        .saturating_add(state_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_beyond_budget_and_releases() {
        let mut b = MemoryBudget::new(100);
        assert!(b.try_admit(60));
        assert_eq!(b.used(), 60);
        // 60 + 60 > 100 => refused, and `used` must not move.
        assert!(!b.try_admit(60));
        assert_eq!(b.used(), 60);
        b.release(60);
        assert_eq!(b.used(), 0);
        assert!(b.try_admit(100));
    }

    /// The admission invariant: the admission layer never holds more than it
    /// budgeted, and a release returns the accounting to zero.
    #[test]
    fn used_tracks_held_and_returns_to_zero() {
        let mut b = MemoryBudget::new(32 << 20);
        let req = stream_bytes(64, 65536, 4096, 2); // 64 tok @ 64 KiB/tok ×2
        assert_eq!(req, 64 * 65536 * 2 + 4096);
        assert_eq!(b.budget() >= b.used(), true); // invariant floor
        assert!(b.try_admit(req));
        assert_eq!(b.used(), req);
        b.release(req);
        assert_eq!(b.used(), 0);
        assert_eq!(b.headroom(), 32 << 20);
    }

    #[test]
    fn overshoot_below_one_is_clamped() {
        assert_eq!(stream_bytes(10, 100, 5, 0), 10 * 100 + 5);
    }
}
