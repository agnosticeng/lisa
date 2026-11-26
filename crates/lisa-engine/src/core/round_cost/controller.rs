//! The expected-value depth controller: picks the draft depth per round from
//! trusted cost cells and observed acceptance.
use super::*;

/// The EV depth controller: per-index acceptance EMAs + cost model (table
/// prior, refined by in-run round timing), picking the depth every round.
pub struct DepthController {
    /// Per-index draft acceptance EMA (index i accepted = the round accepted
    /// more than i tokens). Index 0..5, seeded optimistically at 0.75.
    pub acc: [f64; 6],
    /// Per-depth acceptance EMAs. The shared `acc` is polluted by whichever
    /// depth the controller currently runs: a d2 incumbent never observes
    /// index-2 hits, so the shared EMA at index >= 2 decays toward its last
    /// stale trial value and permanently under-prices d3+ EV (spec 03: at
    /// 12.5k the shared EMA read `acc[2]`=0.49 while forced-d3 measured 0.66,
    /// freezing the pick on d2 — 30 vs 37.7 tok/s measured). Each depth's EV
    /// is priced from ITS OWN acceptance once that depth has MIN_SAMPLES
    /// rounds; before that the shared EMA is the warm-start prior.
    pub depth_acc: BTreeMap<usize, [f64; 6]>,
    /// In-run measured round wall ms per depth (EMA).
    pub round_ms: BTreeMap<usize, f64>,
    /// In-run measured serial step ms (probe + serial rounds).
    pub serial_ms: f64,
    /// Table priors, if the persisted table has a row for this model.
    pub table: Option<ModelCost>,
    pub cap: usize,
    /// Depth picks per round (including 0 = serial), for the end log.
    pub picks: BTreeMap<usize, usize>,
    /// In-run sample count per depth (when a cost cell is trusted).
    round_n: BTreeMap<usize, u32>,
    /// In-run measured tokens per round (EMA), paired with `round_ms` so the
    /// gate can price ms PER EMITTED TOKEN, never ms alone.
    round_tok: BTreeMap<usize, f64>,
    /// Depths settled as worse than serial on their trusted measurement: the
    /// plan only needs "not better", and every further trial block of a
    /// clearly worse depth is a several-% hit on the request that carries it
    /// (reference `CLEARLY_WORSE` / first-sample settle rule).
    settled_worse: BTreeSet<usize>,
    /// Trust counter for the serial cost. The constructor's serial ms comes
    /// from the persisted table or a median of 6 REAL committed serial steps
    /// — a measurement, not a constant — so it starts trusted when present.
    serial_samples: u32,
    /// Round counter for the exploration schedule.
    rounds: usize,
    /// Tokens expected per depth from the current acceptance EMAs.
    last_ev: BTreeMap<usize, f64>,
    last_pick: usize,
    // --- two-chunk extension state (P2, §2.3) ---
    /// Per-index CONDITIONAL acceptance EMAs (`mtp_ev_accept`): indices
    /// `< accepted` fold toward 1.0, the index at the first rejection toward
    /// 0.0, deeper indices are never conditionally reached. This is what
    /// prices the extension horizon; `depth_acc` (per-depth, shared-EMA
    /// warm-start) keeps pricing `pick()` — the two designs coexist, they
    /// serve different latches (§2.0: do not merge).
    pub ev_acc: [f64; 6],
    /// Last round's base depth (the `m_lo_max = m_lo + 1` cap).
    pub m_lo_prev: usize,
    // --- P3: trial schedule + in-run table folds ---
    trial_target: Option<usize>,
    trial_end: usize,
    next_trial: usize,
    trial_kv: Option<usize>,
    prev_round_w: Option<u32>,
    table_dirty: bool,
    // --- P7: sticky-disable floor (§2.8) ---
    /// First-draft outcome window: drafted always records 1 at base depth
    /// one; accepted records `accepted > 0`. Extension misses never enter.
    floor_window_drafted: [u8; DEPTH_WINDOW],
    floor_window_accepted: [u8; DEPTH_WINDOW],
    floor_window_idx: u32,
    /// EV rounds observed (`mtp_ev_rounds`): counts every chain round, warmup
    /// included, independent of `pick`'s `rounds` (which a trial can skip).
    ev_rounds: u32,
    /// Sticky runtime disable: set once the floor fires, consulted before a
    /// speculation round is armed. It is never re-enabled within a run.
    pub spec_disabled: bool,
}

/// Trial block length (EXPLORE_BLOCK = 3: transition, still-elevated,
/// measurement — the first round of a width change is a transition the
/// table will not count).
pub(super) const EXPLORE_BLOCK: usize = 3;

/// Cold trial period (EXPLORE_PERIOD_COLD = 8): while an unmeasured width
/// exists the schedule re-arms every 8 rounds until the table fills.
pub(super) const EXPLORE_PERIOD_COLD: usize = 8;

/// A depth whose FIRST trusted sample reads this much worse per emitted token
/// than serial is settled as worse and never trialled again (CLEARLY_WORSE):
/// the plan only needs "not better".
pub(super) const CLEARLY_WORSE: f64 = 0.20;

/// Serial ticks one in-run probe runs, and how many leading ticks are
/// discarded: MTP_ADAPTIVE_PROBE_TOKENS = 8 / MTP_ADAPTIVE_PROBE_WARM = 2. The
/// first decode ticks in a process pay one-time Metal JIT / page faults (the
/// MTP path never runs the serial path's `warmup: shape` kernels), and a
/// plain median over an EOS-truncated two-tick probe reads exactly the cold
/// sample — the 86.31 ms ghost.
pub const PROBE_TICKS: usize = 8;

/// Ticks of a probe discarded as warm.
pub const PROBE_WARM: usize = 2;

/// The probe's result: `(value, warm-fold count)`. The value is the mean of
/// the warm tail (the running-mean fold) — or of ALL ticks when the probe was
/// truncated to (or below) the warm window itself, in which case the count
/// reads 0. **Trust is the count**, never the value: below `MIN_SAMPLES` folds
/// the caller must treat the serial price as unmeasured.
pub fn probe_warm_stats(samples: &[f64]) -> (f64, u32) {
    if samples.is_empty() {
        return (0.0, 0);
    }
    let warm = &samples[PROBE_WARM.min(samples.len())..];
    let count = warm.len() as u32;
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    (
        if warm.is_empty() {
            mean(samples)
        } else {
            mean(warm)
        },
        count,
    )
}

/// Rounds the controller runs the DEFAULT depth before any EV/serial
/// decision (MTP_EV_WARMUP_ROUNDS = 10) — rounds exist from round 0, so the
/// acceptance EMAs and round costs always get fed, whatever the price table
/// says. An empty-picks state therefore cannot reach a controller
/// that decides serial-vs-round before its first round.
pub(super) const EV_WARMUP_ROUNDS: usize = 10;

/// The warmup / fallback depth (the `MtpDepth::Auto` start depth, session.rs).
pub(super) const DEFAULT_DEPTH: usize = 2;

/// Rounds in the first-draft outcome window (`MTP_DEPTH_WINDOW` = 16).
pub(super) const DEPTH_WINDOW: usize = 16;

/// The accepted-sum rate a full depth-one window must clear
/// (`MTP_DISABLE_BELOW` = 0.20) — below it speculation is disabled for the
/// rest of the request (sticky; no in-run re-enable).
pub(super) const DISABLE_BELOW: f64 = 0.20;

/// EV EMA seed for the conditional acceptance EMAs (`MTP_EV_PRIOR = 0.85`,
/// generate.zig:1032) — deliberately ABOVE the ~0.77 measured per-draft rate
/// so a deep index can ever get its first trial.
pub(super) const EV_PRIOR: f64 = 0.85;

/// Conditional-EMA beta (`MTP_EV_EMA_BETA = 0.15`).
pub(super) const EV_BETA: f64 = 0.15;

impl DepthController {
    pub fn new(table: Option<ModelCost>, serial_ms: f64, cap: usize) -> Self {
        // A constructor value is a MEASUREMENT unless the caller says
        // otherwise: table priors and warm probe folds arrive trusted, and
        // `session.rs` passes an explicit count through
        // `new_with_serial` when it is not.
        Self::new_with_serial(
            table,
            serial_ms,
            if serial_ms > 0.0 { MIN_SAMPLES } else { 0 },
            cap,
        )
    }

    /// The controller with an explicit serial-trust count: the serial price is
    /// trusted by `n >= MIN_SAMPLES`, not by being a positive number — a cold,
    /// EOS-truncated probe must not arm the gate).
    pub fn new_with_serial(
        table: Option<ModelCost>,
        serial_ms: f64,
        serial_samples: u32,
        cap: usize,
    ) -> Self {
        DepthController {
            acc: [0.75; 6],
            depth_acc: BTreeMap::new(),
            round_ms: BTreeMap::new(),
            serial_ms,
            table,
            cap: cap.clamp(2, 6),
            picks: BTreeMap::new(),
            round_n: BTreeMap::new(),
            round_tok: BTreeMap::new(),
            settled_worse: BTreeSet::new(),
            serial_samples: if serial_ms > 0.0 { serial_samples } else { 0 },
            rounds: 0,
            last_ev: BTreeMap::new(),
            last_pick: 0,
            ev_acc: [EV_PRIOR; 6],
            m_lo_prev: DEFAULT_DEPTH,
            trial_target: None,
            trial_end: 0,
            next_trial: EV_WARMUP_ROUNDS + EXPLORE_PERIOD_COLD,
            trial_kv: None,
            prev_round_w: None,
            table_dirty: false,
            floor_window_drafted: [0; DEPTH_WINDOW],
            floor_window_accepted: [0; DEPTH_WINDOW],
            floor_window_idx: 0,
            ev_rounds: 0,
            spec_disabled: false,
        }
    }

    /// Is the serial price a trusted measurement (count, not value)?
    fn serial_trusted(&self) -> bool {
        self.serial_ms > 0.0 && self.serial_samples >= MIN_SAMPLES
    }

    /// The acceptance EMA a candidate depth's EV is priced from: its OWN
    /// per-depth EMA once trusted, the shared EMA before that (spec 03).
    fn acc_for(&self, depth: usize) -> [f64; 6] {
        match self.depth_acc.get(&depth) {
            Some(a) if self.round_n.get(&depth).copied().unwrap_or(0) >= MIN_SAMPLES => *a,
            _ => self.acc,
        }
    }

    /// Serial tok/s under the current cost model.
    pub fn serial_tps(&self) -> f64 {
        if self.serial_ms > 0.0 {
            1000.0 / self.serial_ms
        } else {
            0.0
        }
    }

    /// The round-cost-table-first gate: is a
    /// round at `depth` MEASURED to emit tokens cheaper than the plain serial
    /// step it would yield to? Both sides must be trusted in-run measurements
    /// — ms per EMITTED token, never ms alone (a wide round costs more and
    /// buys more). `None` until both sides are trusted: the calibrated EV bar
    /// then stays the only signal, exactly as before this gate existed.
    pub fn round_beats_serial(&self, depth: usize) -> Option<bool> {
        let n = self.round_n.get(&depth).copied().unwrap_or(0);
        if n < MIN_SAMPLES || self.serial_samples < MIN_SAMPLES {
            return None;
        }
        let ms = *self.round_ms.get(&depth)?;
        let tok = *self.round_tok.get(&depth)?;
        if ms <= 0.0 || tok <= 0.0 || self.serial_ms <= 0.0 {
            return None;
        }
        Some(ms / tok < self.serial_ms)
    }

    /// The widest depth the gate keeps on (measured cheaper per token than
    /// serial), if any. Reference: `roundCost.Table.roundBeatsSerial`. The
    /// forced-continuation cap applies here too: at the tail of a
    /// forced-length decode the round cannot fill its width, so its per-token
    /// price is `ms / min(observed, remaining)`.
    fn gate_kept_depth(&self, kv_len: usize, remaining: usize) -> Option<usize> {
        let capped = |d: usize| -> Option<bool> {
            let n = self.round_n.get(&d).copied().unwrap_or(0);
            if n < MIN_SAMPLES || self.serial_samples < MIN_SAMPLES {
                return None;
            }
            let ms = *self.round_ms.get(&d)?;
            let tok = (*self.round_tok.get(&d)?).min(remaining as f64);
            if ms <= 0.0 || tok <= 0.0 || self.serial_ms <= 0.0 {
                return None;
            }
            Some(ms / tok < self.serial_ms)
        };
        // Table-trusted fallback: when the in-run side has no MIN_SAMPLES
        // rounds yet, price the round from the MEASURED table — verify+chain
        // at `bucket_to_read` — against the trusted serial price, with tokens
        // from the acceptance EMA (the `.accept` policy: the table prices round
        // time, the EMA prices tokens). Without this the gate could never fire
        // in the exact state it exists for: no rounds yet → no in-run samples
        // → None → serial forever.
        let table_price = |d: usize| -> Option<bool> {
            let ms = self.priced_cost(d, kv_len)?;
            let etok = self.expected_tokens(d, kv_len);
            let tok = etok.min(remaining as f64);
            if ms <= 0.0 || tok <= 0.0 || self.serial_ms <= 0.0 {
                return None;
            }
            Some(ms / tok < self.serial_ms)
        };
        (2..=self.cap)
            .filter(|d| !self.settled_worse.contains(d))
            .filter(|d| capped(*d) == Some(true) || table_price(*d) == Some(true))
            .max()
    }

    /// The MEASURED round cost at `depth`, in ms: the in-run EMA once the
    /// depth has two samples, else the persisted table at the
    /// `bucket_to_read` bucket (verify + draft/chain, trusted cells).
    /// `None` = nothing measured at this depth — the analytic prior below,
    /// which is proportional to the serial step, can only fill in when that
    /// step itself is a measurement: an untrusted serial price must not
    /// fabricate round costs.
    fn priced_cost(&self, depth: usize, kv_len: usize) -> Option<f64> {
        let n = self.round_n.get(&depth).copied().unwrap_or(0);
        if n >= 2
            && let Some(ms) = self.round_ms.get(&depth)
        {
            return Some(*ms);
        }
        let t = self.table.as_ref()?;
        let bucket = t.bucket_to_read(kv_len)?;
        // CostSource first (schema v2): interpolated measured ms with the
        // raw floor and the slope-capped extrapolation past the widest
        // measured width. The legacy per-cell direct read stays as the
        // fallback for partially measured tables.
        t.cost_source_ms(depth, bucket, self.serial_ms).or_else(|| {
            let v = t.cell(Which::Verify, (depth + 1) as u32, bucket)?;
            let dcost = self.draft_cost_ms(t, depth, bucket)?;
            Some(v as f64 + dcost)
        })
    }

    /// The tokens a round at `depth` is priced with: trusted MEASURED tokens
    /// first (in-run EMA, then the whole-round serving fold in the table —
    /// the reference `measuredTokens` replacing the EMA model where
    /// measured), else the acceptance-EMA model E(m) = 1 + a0 + a0·a1 + ….
    /// `None` never happens at call sites that already fall back to the EMA.
    fn tokens_model(&self, depth: usize, kv_len: usize) -> Option<f64> {
        let n = self.round_n.get(&depth).copied().unwrap_or(0);
        if n >= MIN_SAMPLES
            && let Some(tok) = self.round_tok.get(&depth)
            && *tok > 0.0
        {
            return Some(*tok);
        }
        let t = self.table.as_ref()?;
        let bucket = t.bucket_to_read(kv_len)?;
        t.measured_round_tok(depth, bucket)
    }

    /// The acceptance-EMA expected-token model for a depth.
    fn ema_tokens(&self, depth: usize) -> f64 {
        let mut p = 1.0f64;
        let mut etok = 1.0f64;
        let acc = self.acc_for(depth);
        for i in 0..depth {
            etok += p * acc[i.min(5)];
            p *= acc[i.min(5)];
        }
        etok
    }

    /// Expected tokens = measured where trusted, else the EMA model.
    fn expected_tokens(&self, depth: usize, kv_len: usize) -> f64 {
        self.tokens_model(depth, kv_len)
            .unwrap_or_else(|| self.ema_tokens(depth))
    }

    // ------------------------------------------------------------------
    // Two-chunk extension (P2, port spec §2.3): conditional EMAs + the
    // `mtpEvPlanSrc` plan (generate.zig:6204-6278).
    // ------------------------------------------------------------------

    /// `mtpEvObserve` (generate.zig:6280-6288): conditional/prefix-structured.
    /// Indices `< accepted` fold toward 1.0; the index AT the first rejection
    /// toward 0.0; deeper indices are never conditionally reached → no
    /// observation. `drafted` = the chain positions proposed this round.
    pub fn ev_observe(&mut self, drafted: usize, accepted: usize) {
        let n = drafted.min(6);
        for i in 0..n {
            if i < accepted {
                self.ev_acc[i] += EV_BETA * (1.0 - self.ev_acc[i]);
            } else if i == accepted {
                self.ev_acc[i] += EV_BETA * (0.0 - self.ev_acc[i]);
            }
        }
    }

    /// Sticky-disable window observe (port spec §2.8,
    /// `mtpFloorDisableObserve`): track the only evidence that can justify
    /// sticky-disable — whether the first draft landed while the EV base
    /// depth was exactly one. Wider base rounds reset the probation window,
    /// and later extension misses (drafted > 1 with a partial accept) do not
    /// count against the depth-one floor. Returns the accepted-sum rate only
    /// after a full fresh window has been observed.
    fn floor_disable_observe(
        drafted_window: &mut [u8; DEPTH_WINDOW],
        accepted_window: &mut [u8; DEPTH_WINDOW],
        window_idx: &mut u32,
        m_lo: usize,
        drafted: usize,
        accepted: usize,
    ) -> Option<f64> {
        debug_assert!(drafted >= 1);
        if m_lo != 1 {
            *window_idx = 0;
            return None;
        }
        let idx = (*window_idx as usize) % DEPTH_WINDOW;
        drafted_window[idx] = 1;
        accepted_window[idx] = u8::from(accepted > 0);
        *window_idx += 1;
        let n = (*window_idx as usize).min(DEPTH_WINDOW);
        if n < DEPTH_WINDOW {
            return None;
        }
        let accepted_sum: u32 = accepted_window[..n].iter().map(|&a| a as u32).sum();
        Some(f64::from(accepted_sum) / f64::from(n as u32))
    }

    /// EV-mode per-round update (port spec §2.8, `updateMtpEvRound`): the
    /// conditional EMAs always; during warmup the legacy controller keeps
    /// running on its own (nothing to do — `pick()` still owns the depth and
    /// the mixed warmup evidence belongs in no floor window, so the window
    /// resets at the last warmup round); post-warmup only the sticky disable
    /// floor is checked — EV owns depth.
    pub fn ev_round_update(&mut self, drafted: usize, accepted: usize, m_lo: usize) {
        self.ev_observe(drafted, accepted);
        self.ev_rounds += 1;
        if self.ev_rounds <= EV_WARMUP_ROUNDS as u32 {
            // Warmup may evaluate several depths; none of that mixed
            // evidence belongs in the post-warmup depth-one window.
            if self.ev_rounds == EV_WARMUP_ROUNDS as u32 {
                self.floor_window_idx = 0;
            }
            return;
        }
        let rate = Self::floor_disable_observe(
            &mut self.floor_window_drafted,
            &mut self.floor_window_accepted,
            &mut self.floor_window_idx,
            m_lo,
            drafted,
            accepted,
        );
        let Some(rate) = rate else { return };
        if rate < DISABLE_BELOW && !self.spec_disabled {
            eprintln!(
                "[mtp.disabled] (EV: depth-1 first-draft rate {rate:.2} < {DISABLE_BELOW:.2})"
            );
            self.spec_disabled = true;
        }
    }

    // ------------------------------------------------------------------
    // P3 (port spec §2.5 + §2.6): width trials + in-run table folds.
    // ------------------------------------------------------------------

    /// `mtpWidthTrialTarget` (generate.zig:7034-7063), reduced to lisa's
    /// single-chunk world: the table needs the next UNMEASURED width — the
    /// smallest depth in 2..=cap that is neither table-measured, nor
    /// in-run-trusted, nor settled worse. `m_lo-1` is deliberately never
    /// trialled; a fully-measured ladder yields None (the table persists, so
    /// the cold period is paid once per machine/model). Trials are ordinary
    /// rounds — lossless greedy — so they cost nothing in correctness.
    pub fn width_trial_target(&self, kv_len: usize) -> Option<usize> {
        let bucket = self.table.as_ref().and_then(|t| t.bucket_to_read(kv_len));
        let table_measured = |d: usize| -> bool {
            match (bucket, self.table.as_ref()) {
                (Some(b), Some(t)) => {
                    t.cell(Which::Verify, (d + 1) as u32, b).is_some()
                        && (t.cell(Which::Chain, d as u32, b).is_some()
                            || t.cell(Which::Draft, 0, b).is_some())
                }
                _ => false,
            }
        };
        (2..=self.cap).find(|&d| {
            !self.settled_worse.contains(&d)
                && !table_measured(d)
                && self.round_n.get(&d).copied().unwrap_or(0) < MIN_SAMPLES
        })
    }

    /// One trial-schedule tick, called from `pick` AFTER the round counter
    /// advanced: when a trial block is running, force its target; when the
    /// schedule comes due and an unmeasured width exists, open a
    /// 3-round block (EXPLORE_BLOCK = 3: transition, still-elevated,
    /// measurement) with the cold period 8 (EXPLORE_PERIOD_COLD — the table
    /// persists, so the cold period is paid once per machine/model).
    /// Solo-only by construction (lisa serves one stream; gate on an idle
    /// check when real concurrency lands).
    pub fn trial_tick(&mut self) -> Option<usize> {
        if let Some(d) = self.trial_target {
            if self.rounds <= self.trial_end {
                return Some(d);
            }
            self.trial_target = None;
        }
        if self.rounds >= self.next_trial {
            // The pick's own kv is not at hand here; the target search uses
            // the last-seen bucket via the caller's kv (stored at pick).
            match self
                .trial_kv
                .take()
                .and_then(|kv| self.width_trial_target(kv))
            {
                Some(d) => {
                    self.trial_target = Some(d);
                    self.trial_end = self.rounds + EXPLORE_BLOCK - 1;
                    self.next_trial = self.rounds + EXPLORE_PERIOD_COLD;
                    Some(d)
                }
                None => {
                    self.next_trial = self.rounds + 64; // nothing to learn now
                    None
                }
            }
        } else {
            None
        }
    }

    /// Feed the whole-round sample into the table (the reference
    /// `specObserveRound` rejection rules): bad samples are dropped by
    /// `observe_round_cell`; a TRANSITION (this round's width differs from
    /// the previous round's — width changes are pipeline disturbances the
    /// table must not count) is dropped here. The caller passes only
    /// single-chunk shapes (the same exclusion as `observe_round`).
    /// Returns true when a sample FOLDED (the caller persists at request end).
    pub fn observe_table_round(&mut self, kv_len: usize, width_s: u32, ms: f64, tok: f64) -> bool {
        let prev = self.prev_round_w.replace(width_s);
        if prev.is_some() && prev != Some(width_s) {
            return false; // transition round
        }
        let bucket = bucket_for(kv_len);
        let folded = match self.table.as_mut() {
            Some(t) => t.observe_round_cell(width_s, bucket, ms, tok),
            None => false,
        };
        if folded {
            self.table_dirty = true;
        }
        folded
    }

    /// Stale-stamp a restored table row against this host (call once at
    /// controller construction, after load): a row written by another
    /// machine keeps its trust counts but re-folds its first live sample at
    /// RESEED_WEIGHT 0.5. A sweep-written row (machine empty) is portable.
    pub fn stamp_machine(&mut self, chip: &str, model_key: &str, os: &str) {
        let mk = ModelCost::machine_key(chip, model_key, "w", os);
        if let Some(t) = self.table.as_mut() {
            if !t.machine.is_empty() && t.machine != mk {
                t.stale = true;
            }
            t.machine = mk;
        }
    }

    /// Take the folded table for persistence at request end (`None` = nothing
    /// new folded this request).
    pub fn take_folded_table(&mut self) -> Option<ModelCost> {
        if self.table_dirty {
            self.table_dirty = false;
            self.table.clone()
        } else {
            None
        }
    }

    // ------------------------------------------------------------------
    // P4 (port spec §2.4): the extension dry-spell gate, the regime gate,
    // and the live-cost EMAs that price them.
    // ------------------------------------------------------------------

    // ------------------------------------------------------------------
    // P6 (port spec §2.7): cross-request EV seed.
    // ------------------------------------------------------------------

    /// The seed state a finished request hands to the next one: the
    /// conditional acceptance EMAs + the last base depth (the reference
    /// head's `ev_seed_accept` / `m_lo`, generate.zig:313-322).
    pub fn ev_seed_state(&self) -> ([f64; 6], usize) {
        (self.ev_acc, self.m_lo_prev)
    }

    /// Seed a FRESH controller with a previous request's EV state and mark
    /// warmup consumed (reference `generate.zig:4868-4875`): the request
    /// re-warms the controller instead of paying warmup again. Call only
    /// before the first `pick`.
    pub fn seed_ev(&mut self, acc: [f64; 6], m_lo: usize) {
        self.ev_acc = acc;
        self.m_lo_prev = m_lo.clamp(1, 6);
        self.rounds = EV_WARMUP_ROUNDS; // warmup consumed
    }

    /// Cost model for a round at `depth`, in ms. Priority: in-run measurement
    /// (the live machine at the live width), then the persisted table at the
    /// bucket `bucket_to_read` resolves (nearest trusted, lower preferred),
    /// then a conservative analytic prior
    /// `serial * (1 + 0.6*depth)`. Draft cost prefers the measured CHAIN cell
    /// (the chain as served) and only falls back to `draft_step * depth` when
    /// no chain cell is trusted — that legacy cell is enqueue-priced and
    /// under-bills the chain several-fold at long KV.
    fn cost_ms(&self, depth: usize, kv_len: usize) -> f64 {
        self.priced_cost(depth, kv_len)
            .unwrap_or(self.serial_ms * (1.0 + 0.6 * depth as f64))
    }

    /// The draft cost the EV math bills for a `depth` chain at `bucket`:
    /// the measured chain cell when trusted, else the legacy enqueue-priced
    /// `draft_step` cell × depth.
    fn draft_cost_ms(&self, t: &ModelCost, depth: usize, bucket: usize) -> Option<f64> {
        if let Some(c) = t.cell(Which::Chain, depth as u32, bucket) {
            return Some(c as f64);
        }
        t.cell(Which::Draft, 0, bucket)
            .map(|d| d as f64 * depth as f64)
    }

    /// EV pick for the next round: expected tokens per round
    /// `E = 1 + a0 + a0*a1 + ...` over the cost model, vs the serial rate.
    /// Returns 0 (serial) when no depth clears the serial rate by 3%.
    /// Hysteresis: an incumbent depth is only demoted when the challenger is
    /// 5% better, so timing noise cannot churn the pick.
    ///
    /// `remaining` is the FORCED-CONTINUATION term: tokens still owed by the
    /// request (max_tokens - emitted; `usize::MAX` = unbounded). A verify
    /// round always runs at full width and costs its full price, but a round
    /// at the tail of a forced-length decode can only emit `remaining` tokens
    /// — so the EV of a depth is priced with `min(E, remaining)` tokens. This
    /// stops the controller from paying a wide round it cannot fill (forced
    /// benchmarks like `--max-tokens 192` used to be penalized: the EV math
    /// counted tokens the decode was never allowed to produce).
    pub fn pick(&mut self, kv_len: usize, remaining: usize) -> usize {
        // P7 sticky disable: consulted before any speculation round is armed
        // (the reference checks `spec_disabled_runtime` at the round entry).
        // No in-run re-enable — the flag is sticky for the request.
        if self.spec_disabled {
            self.last_pick = 0;
            *self.picks.entry(0).or_insert(0) += 1;
            return 0;
        }
        let cap = |etok: f64| etok.min(remaining as f64);
        self.rounds += 1;
        // Bootstrap (MTP_EV_WARMUP_ROUNDS): the first EV_WARMUP_ROUNDS rounds
        // run the DEFAULT depth with no EV-vs-serial decision at all. Rounds
        // exist from round
        // 0, so the acceptance EMAs and round-cost samples always get fed —
        // an empty-picks state cannot reach a
        // controller that decides serial-vs-round before its first round.
        if self.rounds <= EV_WARMUP_ROUNDS {
            self.last_pick = DEFAULT_DEPTH;
            *self.picks.entry(DEFAULT_DEPTH).or_insert(0) += 1;
            return DEFAULT_DEPTH;
        }
        // P3 width trials: force the trial target for its 3-round block
        // (solo only — lisa serves one stream; never inside a regime block,
        // which does not exist yet). A trial is an ordinary round.
        self.trial_kv = Some(kv_len);
        if let Some(d) = self.trial_tick() {
            self.last_pick = d;
            *self.picks.entry(d).or_insert(0) += 1;
            return d;
        }
        // Exploration: a candidate whose in-run cost is not yet trusted gets a
        // trial round. Two gates keep the trials cheaper than what they learn:
        // - the table prior must already put the depth within 5% of serial EV
        //   (a depth the persisted costs say is hopeless is never tried — on
        //   Flash-Next prose the table is right, and trials would burn ~5
        //   ms/token against a pure serial run);
        // - the first in-run sample of a depth is discarded (pass-1 rule: the
        //   first verify width in a process pays Metal JIT / page faults).
        // A trial is an ordinary round — lossless greedy, whatever the depth.
        if self.rounds > 2 && self.rounds.is_multiple_of(6) {
            // The gate reads the SAME bucket `bucket_to_read` resolves
            // (nearest trusted, lower preferred). No trusted bucket at this kv: the
            // trial is the only pricing (the `_ => true` arm below).
            let read_bucket = self.table.as_ref().and_then(|t| t.bucket_to_read(kv_len));
            let stps = self.serial_tps();
            let trial = (2..=self.cap).find(|d| {
                if self.round_n.get(d).copied().unwrap_or(0) >= 3 || self.settled_worse.contains(d)
                {
                    return false;
                }
                match &self.table {
                    Some(t) => {
                        let etok = self.expected_tokens(*d, kv_len);
                        match read_bucket.and_then(|bucket| {
                            Some((
                                t.cell(Which::Verify, (*d + 1) as u32, bucket)?,
                                self.draft_cost_ms(t, *d, bucket)?,
                            ))
                        }) {
                            Some((v, dr)) => cap(etok) / (v as f64 + dr) * 1000.0 >= stps * 0.95,
                            None => true, // no trusted prior: the trial is the only pricing
                        }
                    }
                    None => true,
                }
            });
            if let Some(d) = trial {
                self.last_pick = d;
                *self.picks.entry(d).or_insert(0) += 1;
                return d;
            }
        }
        // Untrusted serial price: lisa never lets an UNMEASURED
        // serial step win a decision — serial is a candidate only where a
        // serial cell was measured. Argmax over the depths something has
        // actually priced (in-run EMA, else the table); with nothing priced at all,
        // run the default depth — rounds are how both sides get measured, so
        // the request must keep making them.
        if !self.serial_trusted() {
            let mut best_d = 0usize;
            let mut best_tps = 0.0f64;
            let mut ev: BTreeMap<usize, f64> = BTreeMap::new();
            for d in 2..=self.cap {
                let Some(cost) = self.priced_cost(d, kv_len) else {
                    continue;
                };
                if cost <= 0.0 {
                    continue;
                }
                let etok = self.expected_tokens(d, kv_len);
                let tps = cap(etok) / cost * 1000.0;
                ev.insert(d, tps);
                if tps > best_tps {
                    best_d = d;
                    best_tps = tps;
                }
            }
            let d = if best_d == 0 { DEFAULT_DEPTH } else { best_d };
            self.last_ev = ev;
            self.last_pick = d;
            *self.picks.entry(d).or_insert(0) += 1;
            return d;
        }
        // The EV loop prices at the SAME resolved bucket (inside cost_ms via
        // bucketToRead); with no trusted bucket the analytic prior carries
        // every depth.
        let mut best = (0usize, self.serial_tps()); // serial baseline
        let mut ev: BTreeMap<usize, f64> = BTreeMap::new();
        for d in 2..=self.cap {
            let etok = self.expected_tokens(d, kv_len);
            let tps = cap(etok) / self.cost_ms(d, kv_len) * 1000.0;
            ev.insert(d, tps);
            // Tie-break toward the deeper draft: equal EV buys more accepted
            // tokens per round (fewer rounds). A shallower challenger must be
            // clearly (3%) better; against the serial baseline 3% also.
            let beats = if best.0 == 0 {
                tps > best.1 * 1.03
            } else if d > best.0 {
                tps >= best.1 * 0.97
            } else {
                tps > best.1 * 1.03
            };
            if beats {
                best = (d, tps);
            }
        }
        // The table's own measure outranks the calibrated bar: a depth
        // MEASURED to emit tokens cheaper than the plain step it would yield to
        // stays on — no serial fallback, no hysteresis demotion. This early
        // return in front of the yield gate took a sampled prose request from
        // 12 cold disables per boot to 1 with no sub-serial requests after the
        // disable.
        if best.0 == 0
            && let Some(d) = self.gate_kept_depth(kv_len, remaining)
        {
            self.last_pick = d;
            *self.picks.entry(d).or_insert(0) += 1;
            return d;
        }
        // Hysteresis on the incumbent speculative depth.
        if best.0 != self.last_pick
            && self.last_pick >= 2
            && best.0 >= 2
            && ev.get(&best.0).copied().unwrap_or(0.0)
                < ev.get(&self.last_pick).copied().unwrap_or(0.0) * 1.05
        {
            best.0 = self.last_pick;
        }
        self.last_ev = ev;
        self.last_pick = best.0;
        *self.picks.entry(best.0).or_insert(0) += 1;
        best.0
    }

    /// Feed a round's outcome back: acceptance EMAs + measured round ms.
    pub fn observe_round(&mut self, depth: usize, accepted: usize, wall_ms: f64) {
        for i in 0..depth.min(6) {
            let hit = if accepted > i { 1.0 } else { 0.0 };
            self.acc[i] = self.acc[i] * 0.9 + hit * 0.1;
        }
        {
            let da = self.depth_acc.entry(depth).or_insert([0.75; 6]);
            for i in 0..depth.min(6) {
                let hit = if accepted > i { 1.0 } else { 0.0 };
                da[i] = da[i] * 0.9 + hit * 0.1;
            }
        }
        let n = self.round_n.entry(depth).or_insert(0);
        *n += 1;
        // Pass-1 rule: the first rounds at a depth pay one-time JIT /
        // page faults (the MTP path never runs the serial path's shape
        // warmup, and every new verify width compiles on its first trial) —
        // PROBE_WARM rounds are counted but NEVER folded. Folds then run as
        // a running MEAN until MIN_SAMPLES before the EMA (BETA 0.10) takes
        // over (the MTP_ADAPTIVE_PROBE_WARM = 2 discard).
        if *n <= PROBE_WARM as u32 {
            return;
        }
        let tok = (accepted + 1) as f64; // committed main_tokens[..=a]
        let folds = *n - PROBE_WARM as u32; // 1-based fold index for this depth
        if folds == 1 {
            self.round_ms.insert(depth, wall_ms);
            self.round_tok.insert(depth, tok);
        } else if folds <= MIN_SAMPLES {
            let e = self.round_ms.entry(depth).or_insert(wall_ms);
            *e += (wall_ms - *e) / folds as f64;
            let t = self.round_tok.entry(depth).or_insert(tok);
            *t += (tok - *t) / folds as f64;
        } else {
            let e = self.round_ms.entry(depth).or_insert(wall_ms);
            *e = *e * 0.9 + wall_ms * 0.1;
            let t = self.round_tok.entry(depth).or_insert(tok);
            *t = *t * 0.9 + tok * 0.1;
        }
        // First-trusted-sample settle (CLEARLY_WORSE, settled on the first
        // post-warmup sample): a depth that reads CLEARLY_WORSE per emitted
        // token than serial never gets another trial block.
        if folds == 1 && self.serial_trusted() {
            let ms = self.round_ms.get(&depth).copied().unwrap_or(wall_ms);
            let tok = self.round_tok.get(&depth).copied().unwrap_or(tok);
            if tok > 0.0 && ms / tok > self.serial_ms * (1.0 + CLEARLY_WORSE) {
                self.settled_worse.insert(depth);
            }
        }
    }

    /// Feed a serial fallback step's wall ms back into the serial cost.
    /// `transition` marks a tick that starts or warms a serial block: the
    /// first `PROBE_WARM` ticks of every block, and every tick that follows
    /// a speculative round, pay pipeline drain / width transitions (a
    /// post-round tick measured ~139 ms against a 40 ms steady step). These
    /// `.transition` ticks are dropped exactly.
    pub fn observe_serial(&mut self, wall_ms: f64, transition: bool) {
        if transition {
            return;
        }
        self.serial_samples += 1;
        if self.serial_ms <= 0.0 {
            self.serial_ms = wall_ms;
        } else {
            self.serial_ms = self.serial_ms * 0.8 + wall_ms * 0.2;
        }
    }

    /// End-of-run log line: depth picks, acceptance EMAs, EV per depth.
    pub fn log_summary(&self) -> String {
        let picks: Vec<String> = self.picks.iter().map(|(d, n)| format!("{d}x{n}")).collect();
        let acc: Vec<String> = self.acc.iter().map(|a| format!("{a:.3}")).collect();
        let ev: Vec<String> = self
            .last_ev
            .iter()
            .map(|(d, t)| format!("d{d}:{t:.1}"))
            .collect();
        format!(
            "[mtp.auto] picks [{}] serial {:.1} tok/s | acc EMA [{}] | EV tok/s [{}]",
            picks.join(" "),
            self.serial_tps(),
            acc.join(" "),
            ev.join(" ")
        )
    }
}

#[cfg(test)]
mod tests;
