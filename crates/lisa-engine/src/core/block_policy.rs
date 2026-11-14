//! Request-local cost control for a block drafter. Decisions use observed wall
//! time per emitted token, not prompt repetition or model-specific priors.

use std::time::Duration;

pub struct BlockPolicy {
    enabled: bool,
    serial_samples: usize,
    serial_cost: [f64; 2],
    draft_seconds: f64,
    draft_tokens: usize,
    draft_rounds: usize,
    draft_warmed: bool,
    serial_left: usize,
    backoff: usize,
}

impl BlockPolicy {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            serial_samples: 0,
            serial_cost: [0.; 2],
            draft_seconds: 0.,
            draft_tokens: 0,
            draft_rounds: 0,
            draft_warmed: false,
            serial_left: 0,
            backoff: 8,
        }
    }

    pub fn depth(&self, configured: usize) -> usize {
        if self.enabled && (self.serial_samples < 3 || self.serial_left > 0) {
            0
        } else {
            configured
        }
    }

    /// Call only for an unshortened round. The first serial observation is
    /// discarded to avoid calibrating against a first-use compilation stall.
    /// The same production completion boundary is used in both modes.
    pub fn observe(&mut self, depth: usize, emitted: usize, elapsed: Duration) {
        let seconds = elapsed.as_secs_f64();
        if !self.enabled || emitted == 0 || seconds <= 0. || !seconds.is_finite() {
            return;
        }
        if depth == 0 {
            self.serial_cost[self.serial_samples % 2] = seconds / emitted as f64;
            self.serial_samples += 1;
            self.serial_left = self.serial_left.saturating_sub(1);
            return;
        }
        // The first draft can compile a completely different kernel family
        // from serial decode. Its one-time cost must not disable a fast path.
        if !self.draft_warmed {
            self.draft_warmed = true;
            return;
        }
        self.draft_seconds += seconds;
        self.draft_tokens += emitted;
        self.draft_rounds += 1;
        if self.draft_rounds < 6 {
            return;
        }
        let serial = (self.serial_cost[0] + self.serial_cost[1]) * 0.5;
        let draft = self.draft_seconds / self.draft_tokens as f64;
        // Require a material loss over a window; small timing noise should not
        // disable a useful drafter. Re-entry always probes the existing cache.
        if draft > serial * 1.10 {
            self.serial_left = self.backoff;
            self.backoff = (self.backoff * 2).min(64);
        } else {
            self.backoff = 8;
        }
        self.draft_seconds = 0.;
        self.draft_tokens = 0;
        self.draft_rounds = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn observe(p: &mut BlockPolicy, configured: usize, tokens: usize, ms: u64) {
        p.observe(p.depth(configured), tokens, Duration::from_millis(ms));
    }
    #[test]
    fn fixed_policy_never_calibrates_or_changes_depth() {
        let mut p = BlockPolicy::new(false);
        for _ in 0..100 {
            assert_eq!(p.depth(5), 5);
            observe(&mut p, 5, 1, 500);
        }
    }
    #[test]
    fn cost_control_recovers_without_permanent_prompt_gate() {
        let mut p = BlockPolicy::new(true);
        for ms in [900, 40, 40] {
            assert_eq!(p.depth(5), 0);
            observe(&mut p, 5, 1, ms);
        }
        observe(&mut p, 5, 4, 9000); // First-use draft compilation is discarded.
        for _ in 0..12 {
            assert_eq!(p.depth(5), 5);
            observe(&mut p, 5, 4, 80);
        }
        // Six losing rounds trigger fallback, despite previously good history.
        for _ in 0..6 {
            observe(&mut p, 5, 1, 80);
        }
        for _ in 0..8 {
            assert_eq!(p.depth(5), 0);
            observe(&mut p, 5, 1, 40);
        }
        assert_eq!(p.depth(5), 5);
        for _ in 0..6 {
            observe(&mut p, 5, 1, 80);
        }
        for _ in 0..16 {
            assert_eq!(p.depth(5), 0);
            observe(&mut p, 5, 1, 40);
        }
        for _ in 0..6 {
            assert_eq!(p.depth(5), 5);
            observe(&mut p, 5, 4, 80);
        }
        assert_eq!(p.backoff, 8);
        assert_eq!(p.depth(5), 5);
    }
    #[test]
    fn empty_or_invalid_timing_never_advances_calibration() {
        let mut p = BlockPolicy::new(true);
        p.observe(0, 0, Duration::from_secs(1));
        p.observe(0, 1, Duration::ZERO);
        assert_eq!(p.serial_samples, 0);
    }
}
