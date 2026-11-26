//! Measured MTP round-cost table and the expected-value depth controller,
//! sized to what the MTP models need:
//!
//! - a persisted table `(model, S, KV-bucket) -> ms/step`, written by
//!   `lisa round-cost` (measured wall time of a serial step, a draft step, and
//!   a verify forward at each width), tracked in the repo and regenerable;
//! - a controller that keeps per-index acceptance EMAs across rounds, prices
//!   every candidate depth as `E[tokens]/round_ms` (EV), re-picks the depth
//!   every round with hysteresis, and falls back to serial (depth 0) when no
//!   speculative depth beats the serial step rate.
//!
//! Pure data + arithmetic; no model code imports here except the measurement
//! helper, which drives a [`LanguageModel`] through the same paths
//! `Session::generate_mtp` uses.

mod controller;
mod measure;

pub use controller::*;
pub use measure::*;

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::language::LanguageModel;
use crate::speculation::runtime::{SpeculativeRuntime, SpeculativeTensor};

/// Sentinel depth used by the CLI/serve plumbing to mean "EV auto" (the
/// `--depth auto` default). Any real depth stays 0 or 2..=6.
pub const AUTO_DEPTH: usize = usize::MAX;

/// KV-bucket edges: <2k, 2-4k, 4-8k, 8-16k, 16-32k, 32k+.
///
/// Grid SCOPE: the 9-bucket `.long` grid with separate 32-64k / 64-128k /
/// 128-256k / 256k+ cells is `qwen4_exp`-only. For our model class
/// (`qwen3_5`) everything past 32k folds into the last bucket, exactly what
/// this grid does. A >32k prompt reads bucket "32k+" (or, through
/// `bucket_to_read`, the nearest bucket with trusted data). Do not widen the
/// grid without a qwen4 model.
/// Samples an in-run cell needs before its measurement is trusted: a
/// one-sample cell is the seed, and the pass-1 rule already discards each
/// depth's first round (JIT / page faults).
const MIN_SAMPLES: u32 = 3;
const BUCKET_EDGES: [usize; 5] = [2048, 4096, 8192, 16384, 32768];
pub const BUCKET_NAMES: [&str; 6] = ["<2k", "2-4k", "4-8k", "8-16k", "16-32k", "32k+"];

pub fn bucket_for(kv_len: usize) -> usize {
    for (i, edge) in BUCKET_EDGES.iter().enumerate() {
        if kv_len < *edge {
            return i;
        }
    }
    BUCKET_EDGES.len()
}

/// One measured cell: an EMA of ms and the sample count (an unsampled cell is
/// `n == 0` and never read as data).
#[derive(Serialize, Deserialize, Clone, Copy, Debug)]
pub struct Cell {
    pub ms: f32,
    /// The tokens the measured round bought (schema v2). "Cost is never
    /// stored without the tokens it bought" (reference `Cell`, round_cost.zig
    /// :61-67): 0.0 = never folded (the cold-start sweep writes no tok —
    /// acceptance is a property of the served stream, not of the sweep), and
    /// the plan falls back to the acceptance-EMA model until serving folds
    /// fill it (n >= MIN_SAMPLES && tok > 0).
    #[serde(default)]
    pub tok: f32,
    pub n: u32,
}

impl Default for Cell {
    fn default() -> Self {
        Cell {
            ms: 0.0,
            tok: 0.0,
            n: 0,
        }
    }
}

impl Cell {
    /// The EMA fold weights for fold index `k` (1-based): running MEAN until
    /// `MIN_SAMPLES`, then EMA `BETA`. Shared by ms and tok so both columns
    /// stay in lockstep.
    fn fold_weight(n_so_far: u32) -> f32 {
        const BETA: f32 = 0.10;
        if n_so_far == 0 || n_so_far < MIN_SAMPLES {
            1.0 / (n_so_far + 1) as f32
        } else {
            BETA
        }
    }

    /// Fold one sample: the first `MIN_SAMPLES` folds are a running MEAN — an
    /// EMA seeded from sample 1 is still sample 1 at n=3 — and the EMA takes
    /// over after.
    pub fn observe(&mut self, ms: f32) {
        self.observe_pair(ms, None);
    }

    /// [`Cell::observe`] with the tokens the round bought (Some = fold the
    /// tok column with the same weights; None = leave it).
    pub fn observe_pair(&mut self, ms: f32, tok: Option<f32>) {
        if self.n == 0 || self.ms <= 0.0 {
            self.ms = ms;
            if let Some(t) = tok {
                self.tok = t;
            }
        } else {
            let w = Self::fold_weight(self.n);
            self.ms += (ms - self.ms) * w;
            if let Some(t) = tok {
                self.tok += (t - self.tok) * w;
            }
        }
        self.n += 1;
    }

    /// Trusted measured tokens: count-trust like ms, and never folded.
    pub fn trusted_tok(&self) -> Option<f64> {
        (self.n >= Self::TABLE_MIN_SAMPLES_CELL && self.tok > 0.0).then_some(self.tok as f64)
    }
}

/// Local alias for the cell-level trust threshold (ModelCost::TABLE_MIN_SAMPLES
/// lives on the model; the cell itself needs the same constant).
impl Cell {
    const TABLE_MIN_SAMPLES_CELL: u32 = 3;
}

/// Median helper for the measurement passes (thermal noise favors the median
/// over the mean).
fn median(v: &[f64]) -> f64 {
    let mut v = v.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Per-model cost surface: serial S=1 decode step, one draft step, the full
/// draft chain per depth (as served), and one verify forward per width
/// `S = depth + 1`, each per KV bucket (6 cells).
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct ModelCost {
    pub key: String,
    /// Serial decode step (forward `[1,1]` + head + argmax), ms per bucket.
    pub serial_ms: [Cell; 6],
    /// One synchronous `draft_step` (`[1,1]` token + multi row), ms per bucket.
    /// KEPT for schema compatibility but NOT used for pricing when a trusted
    /// chain cell exists: this cell is effectively enqueue-priced (one warm,
    /// serialized step), while the served round enqueues the whole chain lazy
    /// and drains it at the verify readback — the real GPU cost is several ×
    /// higher at 16k.
    pub draft_ms: [Cell; 6],
    /// Full depth-`d` draft chain AS SERVED (device-fed proposals, lazy
    /// enqueue, one eval of the chain tail at the end), ms per bucket,
    /// keyed by depth. This is the cell the EV controller prices with.
    #[serde(default)]
    pub chain_ms: BTreeMap<u32, [Cell; 6]>,
    /// Width `S` (= depth+1) verify forward (forward_capture + head + argmax
    /// over `[1,S]`, incl. its readback eval), ms per bucket.
    pub verify_ms: BTreeMap<u32, [Cell; 6]>,
    /// WHOLE-ROUND cells keyed by width `S` (schema v2, fed by serving
    /// traffic — the reconciliation with the reference's whole-round table,
    /// §1.6/§2.6): ms AND the emitted tokens per round, one Cell carrying
    /// both. The cold-start sweep never writes here; `lisa round-cost` stays
    /// the writer of the component cells above.
    #[serde(default)]
    pub round: BTreeMap<u32, [Cell; 6]>,
    /// Machine identity stamp (fnv1a64 over chip + model_dir + quant + OS,
    /// the reference's `rc1-fnv1a64` rule, round_cost.zig:519-527): "one
    /// machine's cliff is never served to another". Empty = the sweep wrote
    /// the row (component cells only, machine-portable ms tables).
    #[serde(default)]
    pub machine: String,
    /// Restored-from-disk cells whose machine stamp differs (or a stale
    /// restore): trust counts are KEPT but the next live sample re-folds at
    /// RESEED_WEIGHT = 0.5 instead of the EMA weight.
    #[serde(default)]
    pub stale: bool,
    /// When / on what host the row was measured (free text).
    pub measured: String,
}

impl ModelCost {
    pub fn cell(&self, which: Which, width: u32, bucket: usize) -> Option<f32> {
        let arr: &[Cell; 6] = match which {
            Which::Serial => &self.serial_ms,
            Which::Draft => &self.draft_ms,
            Which::Chain => self.chain_ms.get(&width)?,
            Which::Verify => self.verify_ms.get(&width)?,
        };
        let c = &arr[bucket.min(5)];
        // A 1-2 sample cell is a seed, never pricing — the same trust rule the
        // in-run controller applies.
        (c.n >= Self::TABLE_MIN_SAMPLES && c.ms > 0.0).then_some(c.ms)
    }

    /// Samples a table cell needs before it counts as measured data
    /// (MIN_SAMPLES = 3): a 1-2-sample cell is a seed, never pricing — the
    /// same trust rule the in-run controller applies.
    pub const TABLE_MIN_SAMPLES: u32 = 3;
    /// Trusted measured cells a bucket needs to be active (MIN_WIDTHS = 1:
    /// one trusted width anchors the bucket).
    pub const TABLE_MIN_WIDTHS: u32 = 1;

    fn trusted_cell(&self, which: Which, width: u32, bucket: usize) -> bool {
        let arr: &[Cell; 6] = match which {
            Which::Serial => &self.serial_ms,
            Which::Draft => &self.draft_ms,
            Which::Chain => match self.chain_ms.get(&width) {
                Some(a) => a,
                None => return false,
            },
            Which::Verify => match self.verify_ms.get(&width) {
                Some(a) => a,
                None => return false,
            },
        };
        arr[bucket.min(5)].n >= Self::TABLE_MIN_SAMPLES && arr[bucket.min(5)].ms > 0.0
    }

    /// The bucket has at least one trusted cell.
    pub fn active(&self, bucket: usize) -> bool {
        let mut widths = 0u32;
        if self.trusted_cell(Which::Serial, 0, bucket) {
            widths += 1;
        }
        if self.trusted_cell(Which::Draft, 0, bucket) {
            widths += 1;
        }
        for w in self.verify_ms.keys() {
            if self.trusted_cell(Which::Verify, *w, bucket) {
                widths += 1;
                break;
            }
        }
        widths >= Self::TABLE_MIN_WIDTHS
    }

    /// The bucket a plan at `kv_len` reads: its own when active, else the
    /// nearest active one — LOWER side preferred, because cost grows with KV
    /// so a lower bucket under-bills rather than over-bills. `None` = no active
    /// bucket, the prior applies. A bucket boundary crossed mid-generation must
    /// not snap the plan back to the prior.
    pub fn bucket_to_read(&self, kv_len: usize) -> Option<usize> {
        let own = bucket_for(kv_len);
        if self.active(own) {
            return Some(own);
        }
        for d in 1..BUCKET_NAMES.len() {
            if own >= d && self.active(own - d) {
                return Some(own - d);
            }
            if own + d < BUCKET_NAMES.len() && self.active(own + d) {
                return Some(own + d);
            }
        }
        None
    }

    /// The serial-step prior for a plan at `kv_len`: the resolved
    /// `bucket_to_read` bucket's serial cell, when that cell is MEASURED.
    /// Every table read goes through `bucket_to_read`, and a serial price
    /// reads only at `n >= MIN_SAMPLES`. Reading the request's OWN bucket only
    /// is how the empty 2-4k cell sent the runtime to a cold in-run probe while
    /// a trusted `<2k` cell sat next door.
    pub fn serial_prior_ms(&self, kv_len: usize) -> Option<f64> {
        let bucket = self.bucket_to_read(kv_len)?;
        self.cell(Which::Serial, 0, bucket).map(|ms| ms as f64)
    }

    // ------------------------------------------------------------------
    // CostSource (port spec §2.1 — the reference `MtpCostSource`,
    // generate.zig:6136-6202): the scaled-table resolution used by the
    // controller when `bucket_to_read` resolves an active bucket.
    // ------------------------------------------------------------------

    /// Interpolated chain+verify ms at `depth` in `bucket`: measured at the
    /// width, else LINEAR between the two nearest measured widths on each
    /// component (reference `roundMs`, round_cost.zig:236-266); `None` when
    /// neither component has any trusted width (the caller's prior fills in —
    /// never extrapolated from nothing).
    pub fn measured_round_ms(&self, depth: usize, bucket: usize) -> Option<f64> {
        let chain = self.interp_component(&self.chain_ms, depth as u32, bucket);
        let verify = self.interp_component(&self.verify_ms, (depth + 1) as u32, bucket);
        match (chain, verify) {
            (Some(c), Some(v)) => Some(c as f64 + v as f64),
            (Some(c), None) => {
                // One component measured, the other not: fall back to the
                // per-cell direct reads (the pre-v2 behavior) so a partial
                // table still prices instead of collapsing to the prior.
                let v = self.cell(Which::Verify, (depth + 1) as u32, bucket)?;
                Some(c as f64 + v as f64)
            }
            (None, Some(v)) => {
                let c = self.cell(Which::Chain, depth as u32, bucket)?;
                Some(c as f64 + v as f64)
            }
            (None, None) => None,
        }
    }

    /// One component map, interpolated at `w`: exact trusted cell, else
    /// linear between the nearest trusted widths below/above; `None` outside
    /// the measured span with only one anchor (never extrapolated).
    fn interp_component(&self, m: &BTreeMap<u32, [Cell; 6]>, w: u32, bucket: usize) -> Option<f32> {
        let b = bucket.min(5);
        let trusted = |k: u32| -> Option<f32> {
            m.get(&k)
                .map(|a| a[b])
                .filter(|c| c.n >= Self::TABLE_MIN_SAMPLES && c.ms > 0.0)
                .map(|c| c.ms)
        };
        if let Some(ms) = trusted(w) {
            return Some(ms);
        }
        // NEVER interpolate outside the measured span: past the widest (or
        // before the narrowest) trusted width the caller's prior /
        // extrapolation rule fills in (reference `roundMs`: "else null —
        // outside the measured span the caller's prior fills in, never
        // extrapolated").
        let has_lower = m.keys().rev().any(|k| *k < w && trusted(*k).is_some());
        let has_upper = m.keys().any(|k| *k > w && trusted(*k).is_some());
        if !has_lower || !has_upper {
            return None;
        }
        let (lk, l) = m
            .keys()
            .rev()
            .find(|&&k| k < w && trusted(k).is_some())
            .map(|&k| (k, trusted(k).unwrap()))
            .unwrap();
        let (uk, u) = m
            .keys()
            .find(|&&k| k > w && trusted(k).is_some())
            .map(|&k| (k, trusted(k).unwrap()))
            .unwrap();
        let t = ((w - lk) as f32) / ((uk - lk).max(1) as f32);
        Some(l + (u - l) * t)
    }

    /// The RAW (untrusted, n in 1..MIN_SAMPLES) cell ms at `depth`/width, if
    /// any: "one sample is evidence for WORSE, never for cheaper" (reference
    /// `measuredMarginal` floor rule, generate.zig:6173-6190) — a raw sample
    /// above the interpolated price floors it from below.
    fn raw_floor_ms(&self, depth: usize, bucket: usize) -> Option<f64> {
        let b = bucket.min(5);
        let raw = |arr: Option<&[Cell; 6]>| -> Option<f64> {
            arr.filter(|c| c[b].n >= 1 && c[b].n < Self::TABLE_MIN_SAMPLES && c[b].ms > 0.0)
                .map(|c| c[b].ms as f64)
        };
        let chain = raw(self.chain_ms.get(&(depth as u32)));
        let verify = raw(self.verify_ms.get(&((depth + 1) as u32)));
        match (chain, verify) {
            (Some(c), Some(v)) => Some((c + v).max(0.0)),
            (c, v) => c.or(v),
        }
    }

    /// Marginal ms of ONE extra draft position past the widest measured
    /// depth: `max(last measured slope, prior marginal)` where the prior
    /// marginal is the analytic `0.6 * serial` slope. `None` when the widest
    /// measured depth itself has no price.
    pub fn marginal_past_widest(&self, depth: usize, bucket: usize, serial_ms: f64) -> Option<f64> {
        let widest = self
            .chain_ms
            .keys()
            .copied()
            .filter(|&w| self.trusted_cell(Which::Chain, w, bucket))
            .max()? as usize;
        if depth <= widest {
            return self.measured_round_ms(depth, bucket);
        }
        let top = self.measured_round_ms(widest, bucket)?;
        let prev = self.measured_round_ms(widest.saturating_sub(1).max(2), bucket);
        let slope = prev.map_or(0.0, |p| {
            (top - p).max(0.0) / (widest - widest.saturating_sub(1)) as f64
        });
        let prior = 0.6 * serial_ms.max(0.0);
        Some(top + (depth - widest) as f64 * slope.max(prior))
    }

    /// The full CostSource resolution for a round at `depth`:
    /// interpolated measured ms, floored by any raw untrusted sample, then
    /// (past the widest measured width) the slope-capped extrapolation.
    pub fn cost_source_ms(&self, depth: usize, bucket: usize, serial_ms: f64) -> Option<f64> {
        let in_span = self.measured_round_ms(depth, bucket);
        let past = self.marginal_past_widest(depth, bucket, serial_ms);
        let price = match (in_span, past) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }?;
        // Raw floor: an untrusted sample can only make the price WORSE.
        let floor = self.raw_floor_ms(depth, bucket);
        Some(price.max(floor.unwrap_or(0.0)))
    }

    /// Measured tokens a round at `depth` bought (whole-round serving fold;
    /// The reference `measuredTokens`, generate.zig:6165-6171). `None` until
    /// serving folds fill the tok column (the sweep writes no tokens).
    pub fn measured_round_tok(&self, depth: usize, bucket: usize) -> Option<f64> {
        self.round
            .get(&((depth + 1) as u32))
            .map(|a| a[bucket.min(5)])
            .and_then(|c| c.trusted_tok())
    }

    /// Machine identity stamp: fnv1a64 over chip + model_dir + quant + OS
    /// (the reference `rc1-fnv1a64` rule). The sweep never sets it; serving
    /// persistence does.
    pub fn machine_key(chip: &str, model_dir: &str, quant: &str, os: &str) -> String {
        let mut h: u64 = 0xcbf29ce484222325;
        for part in [chip, model_dir, quant, os] {
            for &b in part.as_bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            h ^= 0x1f;
            h = h.wrapping_mul(0x100000001b3);
        }
        format!("rc1-{h:016x}")
    }

    /// Fold one serving round into the whole-round cells (the reference
    /// `Table::observe` rejection rules): bad samples (non-finite, non-positive)
    /// are dropped; trust/EMA follows `Cell::observe_pair`; a STALE restore
    /// re-folds its first live sample at RESEED_WEIGHT = 0.5, keeping the
    /// trust count.
    pub fn observe_round_cell(&mut self, width: u32, bucket: usize, ms: f64, tok: f64) -> bool {
        if !ms.is_finite() || !tok.is_finite() || ms <= 0.0 || tok <= 0.0 {
            return false;
        }
        let b = bucket.min(5);
        let arr = self.round.entry(width).or_default();
        let c = &mut arr[b];
        if self.stale && c.n > 0 {
            // Stale restore: blend at RESEED_WEIGHT, keep the trust count.
            c.ms = c.ms * 0.5 + ms as f32 * 0.5;
            c.tok = c.tok * 0.5 + tok as f32 * 0.5;
            self.stale = false;
            return true;
        }
        c.observe_pair(ms as f32, Some(tok as f32));
        true
    }
}

#[derive(Clone, Copy)]
pub enum Which {
    Serial,
    Draft,
    Chain,
    Verify,
}

/// The persisted table (`version` 1). One file, many models.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct RoundCostTable {
    pub version: u32,
    pub models: BTreeMap<String, ModelCost>,
}

impl RoundCostTable {
    /// Load a table file. Schema v2 (the `tok` column + whole-round serving
    /// cells); a v1 file loads through the `#[serde(default)]` fields (tok
    /// reads 0.0 = untrusted, `round` reads empty) and is re-saved as v2.
    pub fn load(path: &std::path::Path) -> Option<RoundCostTable> {
        let data = std::fs::read_to_string(path).ok()?;
        let mut t: RoundCostTable = serde_json::from_str(&data).ok()?;
        t.version = 2;
        Some(t)
    }

    pub fn save(&self, path: &std::path::Path) -> anyhow::Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }

    pub fn get_or_insert(&mut self, key: &str) -> &mut ModelCost {
        self.models
            .entry(key.to_string())
            .or_insert_with(|| ModelCost {
                key: key.to_string(),
                ..Default::default()
            })
    }
}

/// The host's chip string (`sysctl machdep.cpu.brand_string`), for the
/// machine identity stamp. Empty string when sysctl is unavailable.
pub fn host_chip() -> String {
    std::process::Command::new("sysctl")
        .arg("-n")
        .arg("machdep.cpu.brand_string")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default()
}

/// P5 (port spec §2.2): the per-silicon depth-cap row — the SHAPE of the
/// reference `adaptiveDepthCapForMachine` (mtp.zig:74-93), which exists
/// because the EV controller scores accepted-tokens-per-round and on an M1
/// Pro that IMPROVED (3.65 -> 4.00) while realized tok/s fell 21%. Measured
/// rows: M1 Pro -> 4, base M4 -> 4, base M5 -> 4; M4 Pro/Max and every
/// unmeasured chip (including M5 Max — no reference row exists
/// upstream, do not guess one) -> the default 6. Returns (cap, resolution
/// string for the one-per-process log).
pub fn adaptive_cap_for_machine() -> (usize, String) {
    const DEFAULT_CAP: usize = 6;
    let chip = host_chip();
    let big = chip.contains(" Pro") || chip.contains(" Max") || chip.contains(" Ultra");
    let row = if (chip.contains("M1") && chip.contains(" Pro"))
        || (!big && (chip.contains("M4") || chip.contains("M5")))
    {
        Some(4usize)
    } else {
        None
    };
    match row {
        Some(c) => (c, format!("measured per-silicon row '{chip}' -> cap {c}")),
        None => (
            DEFAULT_CAP,
            format!(
                "no measured per-silicon row for '{chip}' (rows: M1 Pro, base M4, base M5) \
                 -> default cap {DEFAULT_CAP}"
            ),
        ),
    }
}

/// Where the runtime table lives (`$HOME/.lisa/round_cost.json`). The tracked
/// repo copy under `docs/round-cost/` is what `lisa round-cost` refreshes into
/// both places (pass `--out` twice, or copy it).
pub fn default_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".lisa").join("round_cost.json")
}
