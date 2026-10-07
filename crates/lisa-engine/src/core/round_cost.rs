//! Measured MTP round-cost table + the EV depth controller (phase 2 of
//! specs/02). A measured cost table + EV controller, sized to what our two
//! models need:
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

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::core::cache::LayerCache;
use crate::models::LanguageModel;
use crate::models::speculate::argmax_id;

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
    /// Serial decode step (forward [1,1] + head + argmax), ms per bucket.
    pub serial_ms: [Cell; 6],
    /// One synchronous `draft_step` ([1,1] token + multi row), ms per bucket.
    /// KEPT for schema compatibility but NOT used for pricing when a trusted
    /// chain cell exists: this cell is effectively enqueue-priced (one warm,
    /// serialized step), while the served round enqueues the whole chain lazy
    /// and drains it at the verify readback — the real GPU cost is several ×
    /// higher at 16k (specs/01 §7).
    pub draft_ms: [Cell; 6],
    /// Full depth-`d` draft chain AS SERVED (device-fed proposals, lazy
    /// enqueue, one eval of the chain tail at the end), ms per bucket,
    /// keyed by depth. This is the cell the EV controller prices with.
    #[serde(default)]
    pub chain_ms: BTreeMap<u32, [Cell; 6]>,
    /// Width `S` (= depth+1) verify forward (forward_capture + head + argmax
    /// over [1,S], incl. its readback eval), ms per bucket.
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
    /// a trusted `<2k` cell sat next door (specs/09, the 86.31 ms ghost).
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
        let has_lower = m
            .keys()
            .rev()
            .any(|k| *k < w && trusted(*k).is_some());
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
        let slope = prev.map_or(0.0, |p| (top - p).max(0.0) / (widest - widest.saturating_sub(1)) as f64);
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
    /// the reference `measuredTokens`, generate.zig:6165-6171). `None` until
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
/// unmeasured chip (including this box's M5 Max — no reference row exists
/// upstream, do not guess one) -> the default 6. Returns (cap, resolution
/// string for the one-per-process log).
pub fn adaptive_cap_for_machine() -> (usize, String) {
    const DEFAULT_CAP: usize = 6;
    let chip = host_chip();
    let big = chip.contains(" Pro") || chip.contains(" Max") || chip.contains(" Ultra");
    let row = if chip.contains("M1") && chip.contains(" Pro") {
        Some(4usize)
    } else if !big && (chip.contains("M4") || chip.contains("M5")) {
        Some(4)
    } else {
        None
    };
    match row {
        Some(c) => (
            c,
            format!("measured per-silicon row '{chip}' -> cap {c}"),
        ),
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

/// Measure one model at one KV length: returns the median ms of a serial step,
/// a draft step, and a verify forward at each requested width `S`.
///
/// The KV state is built by prefilling `tokens` (the caller builds a synthetic
/// prompt of the target length), then timing over the live caches. Caches are
/// rolled back between verify passes, so the widths are comparable.
pub fn measure_at_kv(
    tower: &mut dyn LanguageModel,
    tokens: &[u32],
    widths: &[u32],
    steps: usize,
) -> anyhow::Result<(
    f64,
    f64,
    BTreeMap<u32, f64>,
    BTreeMap<u32, f64>,
)> {
    anyhow::ensure!(
        tower.has_drafter(),
        "round-cost measurement needs a model with an MTP drafter"
    );
    let warm: usize = 4;
    let mut sess = crate::core::session::Session::new(tower);
    let ctx_len = tower.context_window();

    // --- serial S=1 step ---
    let (logits, _multi) = sess.feed_multi(tower, tokens)?;
    let mut tok = argmax_id(&logits)?;
    let mut serial: Vec<f64> = Vec::new();
    for i in 0..(steps + warm) {
        let t0 = std::time::Instant::now();
        let logits = sess.feed(tower, &[tok])?;
        let id = argmax_id(&logits)?;
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        if i >= warm {
            serial.push(dt);
        }
        tok = id;
    }

    // --- one draft step ---
    // A fresh multi row from a capture forward (the [1,1] last row).
    let (_, m1) = {
        tower.set_context_tails(vec![sess_tail(&sess, ctx_len)]);
        let arr = lisa_mlx::Array::from_slice(&[tok as i32], &[1i32, 1]);
        // NOTE: this forward advances the full cache by one row; the draft
        // timings below do not touch the trunk caches, so the offset drift is
        // irrelevant for the verify passes (they snapshot/restore around each
        // pass and never rely on absolute offsets).
        let (mixed, multi) = tower.forward_capture(&arr, Some(&mut sess.caches), true)?;
        let logits = tower.head(&mixed)?;
        let _ = argmax_id(&logits)?;
        ((), multi)
    };
    let mut draft: Vec<f64> = Vec::new();
    {
        let mut tok_arr = lisa_mlx::Array::from_slice(&[tok as i32], &[1i32, 1]);
        let mut mul = m1.reshape(&[1, 1, m1.dim(-1)])?;
        for i in 0..(steps + warm) {
            let t0 = std::time::Instant::now();
            let (d, m) = tower.draft_step(&tok_arr, &mul)?;
            d.eval().map_err(|e| anyhow::anyhow!("{e}"))?;
            let dt = t0.elapsed().as_secs_f64() * 1e3;
            if i >= warm {
                draft.push(dt);
            }
            tok_arr = d.reshape(&[1, 1])?;
            mul = m.reshape(&[1, 1, m.dim(-1)])?;
        }
    }

    // --- draft chain per depth, AS SERVED ---
    // The served round (session.rs) enqueues every chain step LAZY — the draft
    // id stays on device, the next step consumes it — and drains the whole
    // chain at the verify readback. Timing one synchronous `draft_step` (the
    // old draft cell) therefore prices enqueue-only work and under-bills the
    // chain several-fold at long KV (specs/01 §7: cell 1.78 ms vs ~10-15 ms
    // real). Here: depth steps device-fed, ONE eval of the chain tail, wall
    // over the live caches at the bucket's KV.
    let mut chain: BTreeMap<u32, f64> = BTreeMap::new();
    for &s in widths {
        let depth = (s.max(2) as usize).saturating_sub(1).min(6);
        let mut times: Vec<f64> = Vec::new();
        for i in 0..(steps + warm) {
            let mut d_arr = lisa_mlx::Array::from_slice(&[tok as i32], &[1i32, 1]);
            let mut m_arr = m1.reshape(&[1, 1, m1.dim(-1)])?;
            let t0 = std::time::Instant::now();
            for _ in 0..depth {
                let (d2, m2) = tower.draft_step(&d_arr, &m_arr)?;
                d_arr = d2.reshape(&[1, 1])?;
                m_arr = m2.reshape(&[1, 1, m2.dim(-1)])?;
            }
            // Drain like the round: one readback over the chain tail forces
            // the whole lazy graph.
            d_arr.eval().map_err(|e| anyhow::anyhow!("{e}"))?;
            m_arr.eval().map_err(|e| anyhow::anyhow!("{e}"))?;
            let dt = t0.elapsed().as_secs_f64() * 1e3;
            if i >= warm {
                times.push(dt);
            }
        }
        chain.insert(depth as u32, median(&times));
    }

    // --- verify widths ---
    let mut verify: BTreeMap<u32, f64> = BTreeMap::new();
    for &s in widths {
        let s = s.max(2);
        let ids: Vec<i32> = std::iter::repeat(tok as i32).take(s as usize).collect();
        let arr = lisa_mlx::Array::from_slice(&ids, &[1i32, s as i32]);
        let mut times: Vec<f64> = Vec::new();
        for i in 0..(steps + warm) {
            // Snapshot the caches, verify wide, roll back to the snapshot.
            let snaps: Vec<_> = sess
                .caches
                .iter()
                .map(|c| match c {
                    LayerCache::Full(f) => (Some(f.offset), None),
                    LayerCache::Linear(g) => (None, Some(g.snapshot_state())),
                })
                .collect();
            tower.set_context_tails(vec![sess_tail(&sess, ctx_len)]);
            // Tail-ULP contract scope (AGENTS.md §9): round-cost must measure
            // the verify rows through the SAME dispatch the MTP round uses.
            let _verify_splitk = lisa_mlx::ops::enter_verify_splitk_scope();
            let t0 = std::time::Instant::now();
            let (mixed, _multi) = tower.forward_capture(&arr, Some(&mut sess.caches), true)?;
            let logits = tower.head(&mixed)?;
            let top = lisa_mlx::ops::indexing::argmax_axis(&logits, -1, None)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            top.eval().map_err(|e| anyhow::anyhow!("{e}"))?;
            drop(_verify_splitk);
            let dt = t0.elapsed().as_secs_f64() * 1e3;
            for (c, s) in sess.caches.iter_mut().zip(snaps.iter()) {
                match (c, s) {
                    (LayerCache::Full(f), (Some(off), _)) => f.trim(f.offset - off),
                    (LayerCache::Linear(g), (_, Some(st))) => g.restore_state(st),
                    _ => {}
                }
            }
            if i >= warm {
                times.push(dt);
            }
        }
        verify.insert(s, median(&times));
    }
    Ok((median(&serial), median(&draft), chain, verify))
}

fn sess_tail(sess: &crate::core::session::Session, ctx_len: usize) -> Vec<i64> {
    sess.fed[sess.fed.len().saturating_sub(ctx_len)..]
        .iter()
        .map(|&t| t as i64)
        .collect()
}

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
    /// 12.5k the shared EMA read acc[2]=0.49 while forced-d3 measured 0.66,
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
    /// Standing base width + its rate (the 5% SWITCH_MARGIN hysteresis).
    standing_lo: Option<usize>,
    standing_r: f64,
    /// Live chunk-A sync cost EMA (folded at the sync stopwatch; the dry
    /// gate and the plan both read it).
    pub sync_ms: f64,
    sync_n: u32,
    // --- P3: trial schedule + in-run table folds ---
    trial_target: Option<usize>,
    trial_end: usize,
    next_trial: usize,
    trial_kv: Option<usize>,
    prev_round_w: Option<u32>,
    table_dirty: bool,
    // --- P4: dry-spell + regime gates + live-cost EMAs ---
    ext_dry_streak: u32,
    ext_dry_cooldown: u32,
    live_round_ms: f64,
    live_round_n: u32,
    regime_two_ms_tok: f64,
    regime_two_n: u32,
    regime_two_m_lo: Option<usize>,
    regime_single_ms_tok: f64,
    regime_single_n: u32,
    regime_single_m_lo: Option<usize>,
    regime_prev_shape: Option<bool>,
    regime_two_inter_ms: f64,
    regime_single_inter_ms: f64,
    regime_worse: Option<bool>,
    regime_rounds: usize,
    regime_last_explore: usize,
    regime_explore_left: usize,
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
    /// speculation round is armed. No in-run re-enable (the reference's MTP
    /// path clears this flag only on its PLD side, which lisa does not have).
    pub spec_disabled: bool,
}

/// Dry-spell gate constants (§1.9): 16 consecutive considered rounds whose
/// tau gate never clears collapse consideration for 32 rounds; the
/// cost-aware threshold budgets the sync at 30% of the round.
const EXT_DRY_ROUNDS: u32 = 16;
const EXT_DRY_COOLDOWN: u32 = 32;
const EXT_SYNC_BUDGET: f64 = 0.30;
/// Regime-gate constants (§1.10): call two-chunk worse at ratio > 1.05,
/// 2-round trial blocks, explore drag 0.01, period clamped [8, 128].
const REGIME_MARGIN: f64 = 0.05;
const REGIME_EXPLORE_BLOCK: usize = 2;
const REGIME_EXPLORE_DRAG: f64 = 0.01;
const REGIME_EXPLORE_PERIOD: usize = 8;
const REGIME_EXPLORE_PERIOD_MAX: usize = 128;

/// Trial block length (EXPLORE_BLOCK = 3: transition, still-elevated,
/// measurement — the first round of a width change is a transition the
/// table will not count).
const EXPLORE_BLOCK: usize = 3;
/// Cold trial period (EXPLORE_PERIOD_COLD = 8): while an unmeasured width
/// exists the schedule re-arms every 8 rounds until the table fills.
const EXPLORE_PERIOD_COLD: usize = 8;

/// Samples an in-run cell needs before its measurement is trusted: a
/// one-sample cell is the seed, and the pass-1 rule already discards each
/// depth's first round (JIT / page faults).
const MIN_SAMPLES: u32 = 3;
/// A depth whose FIRST trusted sample reads this much worse per emitted token
/// than serial is settled as worse and never trialled again (CLEARLY_WORSE):
/// the plan only needs "not better".
const CLEARLY_WORSE: f64 = 0.20;

/// Serial ticks one in-run probe runs, and how many leading ticks are
/// discarded: MTP_ADAPTIVE_PROBE_TOKENS = 8 / MTP_ADAPTIVE_PROBE_WARM = 2. The
/// first decode ticks in a process pay one-time Metal JIT / page faults (the
/// MTP path never runs the serial path's `warmup: shape` kernels), and a
/// plain median over an EOS-truncated two-tick probe reads exactly the cold
/// sample — the 86.31 ms ghost of specs/09.
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
/// says. The empty-picks state of specs/08 has no path into a controller
/// that decides serial-vs-round before its first round.
const EV_WARMUP_ROUNDS: usize = 10;
/// The warmup / fallback depth (the `MtpDepth::Auto` start depth, session.rs).
const DEFAULT_DEPTH: usize = 2;

// --- P7: sticky-disable floor (port spec §2.8) ---
/// Rounds in the first-draft outcome window (`MTP_DEPTH_WINDOW` = 16).
const DEPTH_WINDOW: usize = 16;
/// The accepted-sum rate a full depth-one window must clear
/// (`MTP_DISABLE_BELOW` = 0.20) — below it speculation is disabled for the
/// rest of the request (sticky; no in-run re-enable).
const DISABLE_BELOW: f64 = 0.20;

/// Is the two-chunk extension machinery (P2, port spec §2.3) enabled? Env
/// kill-switch, default OFF (styled after the reference's
/// `MLX_SERVE_MTP_ADAPTIVE=0` revert): `LISA_MTP_TWO_CHUNK=1` arms it. The
/// repo's env list is closed — this flag is for A/B measurement only; if the
/// mechanism certifies, the kill switch is what gets removed, not the gate.
pub fn two_chunk_enabled() -> bool {
    matches!(
        std::env::var("LISA_MTP_TWO_CHUNK").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("on")
    )
}

/// One round's plan (the reference `MtpRoundPlan`, generate.zig:5982-5990):
/// draft `m_lo` positions, sync the chunk-A confidences, extend to `m_hi`
/// only when the chain log-confidence `c > tau_ln` (tau as ln — `c` and
/// `tau` both live on the log scale). `m_hi == m_lo` collapses to the
/// fixed-depth round: no confidence graph, no sync, byte-identical shape.
#[derive(Clone, Copy, Debug)]
pub struct MtpPlan {
    pub m_lo: usize,
    pub m_hi: usize,
    pub tau_ln: f64,
}

/// EV EMA seed for the conditional acceptance EMAs (`MTP_EV_PRIOR = 0.85`,
/// generate.zig:1032) — deliberately ABOVE the ~0.77 measured per-draft rate
/// so a deep index can ever get its first trial.
const EV_PRIOR: f64 = 0.85;
/// Conditional-EMA beta (`MTP_EV_EMA_BETA = 0.15`).
const EV_BETA: f64 = 0.15;
/// Exploration valve: the horizon closed at `m_lo` itself but the base rate
/// reads this much better than the floor units analog (tokens per ms over
/// the serial step) (`MTP_EV_EXPLORE_MIN_R = 1.10` maps to 1.10 * the serial
/// rate here — lisa prices in ms, not floor units).
const EV_EXPLORE_MIN_R: f64 = 1.10;
/// Marginal floor for off-table marginals (`MTP_EV_TABLE_MIN_MARGINAL`
/// = 0.02 floor units; here 0.02 * the serial step).
const TABLE_MIN_MARGINAL: f64 = 0.02;

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
    /// EOS-truncated probe must not arm the gate, specs/09 M2).
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
            standing_lo: None,
            standing_r: 0.0,
            sync_ms: 0.0,
            sync_n: 0,
            trial_target: None,
            trial_end: 0,
            next_trial: EV_WARMUP_ROUNDS + EXPLORE_PERIOD_COLD,
            trial_kv: None,
            prev_round_w: None,
            table_dirty: false,
            ext_dry_streak: 0,
            ext_dry_cooldown: 0,
            live_round_ms: 0.0,
            live_round_n: 0,
            regime_two_ms_tok: 0.0,
            regime_two_n: 0,
            regime_two_m_lo: None,
            regime_single_ms_tok: 0.0,
            regime_single_n: 0,
            regime_single_m_lo: None,
            regime_prev_shape: None,
            regime_two_inter_ms: 0.0,
            regime_single_inter_ms: 0.0,
            regime_worse: None,
            regime_rounds: 0,
            regime_last_explore: 0,
            regime_explore_left: 0,
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
        // → None → serial forever (specs/09 M4).
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
            .filter(|d| !self.settled_worse.contains(&d))
            .filter(|d| capped(*d) == Some(true) || table_price(*d) == Some(true))
            .max()
    }

    /// The MEASURED round cost at `depth`, in ms: the in-run EMA once the
    /// depth has two samples, else the persisted table at the
    /// `bucket_to_read` bucket (verify + draft/chain, trusted cells).
    /// `None` = nothing measured at this depth — the analytic prior below,
    /// which is proportional to the serial step, can only fill in when that
    /// step itself is a measurement (specs/09 M1/M3: an untrusted serial
    /// must not fabricate round costs).
    fn priced_cost(&self, depth: usize, kv_len: usize) -> Option<f64> {
        let n = self.round_n.get(&depth).copied().unwrap_or(0);
        if n >= 2 {
            if let Some(ms) = self.round_ms.get(&depth) {
                return Some(*ms);
            }
        }
        let t = self.table.as_ref()?;
        let bucket = t.bucket_to_read(kv_len)?;
        // CostSource first (schema v2): interpolated measured ms with the
        // raw floor and the slope-capped extrapolation past the widest
        // measured width. The legacy per-cell direct read stays as the
        // fallback for partially measured tables.
        t.cost_source_ms(depth, bucket, self.serial_ms)
            .or_else(|| {
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
        if n >= MIN_SAMPLES {
            if let Some(tok) = self.round_tok.get(&depth) {
                if *tok > 0.0 {
                    return Some(*tok);
                }
            }
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

    /// `mtpEvExpectedTokens` (generate.zig:6118-6128): the always-committed
    /// token plus the chain's conditional expectation
    /// `E(m) = 1 + a0 + a0·a1 + … + Π_{i<m} a_i`.
    pub fn ev_expected_tokens(&self, m: usize) -> f64 {
        let mut p = 1.0f64;
        let mut e = 1.0f64;
        for i in 0..m.min(6) {
            e += p * self.ev_acc[i];
            p *= self.ev_acc[i];
        }
        e
    }

    /// `mtpChainLogConf` (generate.zig:6289-6296): Σ clamp(log p_head, ≤ 0)
    /// over the chunk-A confidences (the INPUTS are log-probabilities); a NaN
    /// or missing entry reads −inf.
    pub fn chain_log_conf(confs: &[f64]) -> f64 {
        confs
            .iter()
            .map(|&lp| {
                if lp.is_nan() {
                    f64::NEG_INFINITY
                } else {
                    lp.min(0.0)
                }
            })
            .fold(0.0, |x, y| x + y)
    }

    /// `mtpEvPlanSrc` (generate.zig:6204-6278): pick the base width `m_lo`
    /// (argmax of E(m)/T(m) over 1..=lo_cap with the standing-base 5%
    /// hysteresis), then the extension horizon `m_hi` and `tau`.
    ///
    /// `from_table` = the CostSource resolved an active bucket (a measured
    /// slope agrees with the prior by construction). Marginals price in ms;
    /// the floor-units analog of the reference's constants is the serial step.
    pub fn plan_src(&mut self, cap_in: usize, kv_len: usize, from_table: bool) -> MtpPlan {
        let cap = cap_in.min(6).max(1);
        // The EV pick is the FLOOR for the base (§6): the plan may extend the
        // horizon above what `pick()` chose (step 2), but the base must never
        // re-base BELOW it. The old `lo_cap = cap.min(m_lo_prev + 1)` ratchet
        // let the conditional EMAs (poisoned by novel-arm rounds in the same
        // stream) collapse the base to m_lo = 1 while `pick()` held depth 6
        // at 5.4 accepted/round on the same arm — `m_lo_prev + 1` never
        // recovers once it ratchets down. The base search therefore runs over
        // the pick itself; the floor below (`base.max(cap)`) is the ratchet's
        // replacement.
        let lo_cap = cap;
        // 1. base: argmax over m in 1..=lo_cap of E(m)/T(m), tokens from the
        //    conditional EMAs (the plan prices the CHAIN's acceptance; the
        //    per-depth measured tokens price `pick`, not this horizon).
        let mut rates: BTreeMap<usize, f64> = BTreeMap::new();
        for m in 1..=lo_cap {
            let t = self.cost_ms(m, kv_len);
            if t <= 0.0 {
                continue;
            }
            rates.insert(m, self.ev_expected_tokens(m) / t);
        }
        let (mut base, best_r) = match rates.iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()) {
            Some((&m, &r)) => (m, r),
            None => {
                // Nothing priced: collapse to a fixed-depth round at the
                // pick itself — the safest shape (no sync, no extension),
                // still never below the EV pick.
                return MtpPlan {
                    m_lo: cap,
                    m_hi: cap,
                    tau_ln: 0.0,
                };
            }
        };
        // Hysteresis: the standing base keeps its place unless a challenger
        // beats it by 5% (a width change is a transition round the table
        // will not count).
        if let (Some(sl), sr) = (self.standing_lo, self.standing_r) {
            if let Some(&r) = rates.get(&sl) {
                if base != sl && r < sr * 1.05 {
                    base = sl;
                }
            }
        }
        // Floor (§6): the argmax and the hysteresis both lose to the pick.
        let base = base.max(cap);
        let base_r = rates.get(&base).copied().unwrap_or(best_r);
        self.standing_lo = Some(base);
        self.standing_r = base_r;
        self.m_lo_prev = base;
        // 2. extension horizon: walk while the conditional probability of
        //    the round reaching the next index still pays its marginal.
        let floor_units = self.serial_ms.max(1.0); // 1.0 floor unit ≈ the serial step
        let min_marginal = TABLE_MIN_MARGINAL * floor_units;
        let marginal = |m: usize| -> Option<f64> {
            let hi = self.cost_ms(m + 1, kv_len);
            let lo = self.cost_ms(m, kv_len);
            if hi <= 0.0 || lo <= 0.0 {
                return None;
            }
            Some((hi - lo).max(min_marginal))
        };
        let mut cond = self.ev_acc[(base - 1).min(5)];
        let mut s_sum = 0.0f64;
        let mut t_sum = 0.0f64;
        let mut m_hi = base;
        // The horizon walks to the controller's ABSOLUTE cap, not the pick:
        // with the floor in place the base can already sit at the pick, and
        // the plan's remaining freedom is exactly the conditional extension
        // above it (§6 — the base never sinks below the pick; the horizon
        // still reaches the cap).
        let hi_cap = self.cap.max(cap);
        while m_hi < hi_cap {
            let Some(mc) = marginal(m_hi) else { break };
            if cond <= base_r * mc {
                break;
            }
            s_sum += cond;
            t_sum += mc;
            m_hi += 1;
            cond *= self.ev_acc[(m_hi - 1).min(5)];
        }
        // 3. tau = clamp(best_r · t_sum / s_sum, 0.05, 0.95).
        let mut tau = if s_sum > 0.0 {
            (base_r * t_sum / s_sum).clamp(0.05, 0.95)
        } else {
            0.0
        };
        // Exploration valve: the horizon closed at the base itself, but the
        // base rate reads clearly above the floor-units analog and the next
        // marginal is cheap relative to the floor — take ONE extension with
        // its own tau (`MTP_EV_EXPLORE_MIN_R = 1.10`). From a measured table
        // the valve only fires while the extension does not price at a whole
        // floor unit (the reference's `fromTable` arm, mapped to ms).
        if m_hi == base && base_r * floor_units > EV_EXPLORE_MIN_R {
            if let Some(mc) = marginal(base) {
                let costly = from_table && base_r * mc >= 1.0;
                if !costly && base + 1 <= hi_cap {
                    m_hi = base + 1;
                    tau = (base_r * mc / self.ev_acc[(base - 1).min(5)].max(1e-6))
                        .clamp(0.05, 0.95);
                }
            }
        }
        if m_hi == base {
            tau = 0.0; // collapsed: byte-identical to the fixed-depth round
        }
        MtpPlan {
            m_lo: base,
            m_hi,
            tau_ln: if tau > 0.0 { tau.ln() } else { 0.0 },
        }
    }

    /// Fold one chunk-A sync stopwatch sample into the live sync EMA
    /// (seed-on-first, beta 0.10 — the `mtpEmaMs` shape).
    pub fn observe_sync_ms(&mut self, ms: f64) {
        if ms <= 0.0 || !ms.is_finite() {
            return;
        }
        self.sync_n += 1;
        self.sync_ms = if self.sync_n == 1 {
            ms
        } else {
            self.sync_ms * 0.9 + ms * 0.1
        };
    }

    /// Whether the plan's costs resolve from a MEASURED table bucket (the
    /// reference's `fromTable` arm).
    pub fn from_table(&self, kv_len: usize) -> bool {
        self.table
            .as_ref()
            .and_then(|t| t.bucket_to_read(kv_len))
            .is_some()
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
        (2..=self.cap)
            .find(|&d| {
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
            match self.trial_kv.take().and_then(|kv| self.width_trial_target(kv)) {
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
    pub fn observe_table_round(
        &mut self,
        kv_len: usize,
        width_s: u32,
        ms: f64,
        tok: f64,
    ) -> bool {
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

    /// Remember the request's kv so the trial target can resolve it.
    pub fn note_trial_kv(&mut self, kv_len: usize) {
        self.trial_kv = Some(kv_len);
    }

    // ------------------------------------------------------------------
    // P4 (port spec §2.4): the extension dry-spell gate, the regime gate,
    // and the live-cost EMAs that price them.
    // ------------------------------------------------------------------

    /// Live round-cost EMA across shapes (the dry gate's denominator). Folded
    /// on EVERY round; seed-on-first, beta 0.10.
    pub fn observe_live_round_ms(&mut self, ms: f64) {
        if ms <= 0.0 || !ms.is_finite() {
            return;
        }
        self.live_round_n += 1;
        self.live_round_ms = if self.live_round_n == 1 {
            ms
        } else {
            self.live_round_ms * 0.9 + ms * 0.1
        };
    }

    /// `mtpExtDryThresholdFor` (generate.zig:6665-6675):
    /// clamp(round(SYNC_BUDGET / (sync_ms/round_ms)), DRY_MIN, DRY_ROUNDS) —
    /// a measured-expensive sync backs off sooner.
    fn ext_dry_threshold(&self) -> u32 {
        const DRY_MIN: u32 = 3;
        if self.sync_ms <= 0.0 || self.live_round_ms <= 0.0 {
            return EXT_DRY_ROUNDS;
        }
        let per_round = self.sync_ms / self.live_round_ms;
        if per_round <= 0.0 {
            return EXT_DRY_ROUNDS;
        }
        ((EXT_SYNC_BUDGET / per_round).round() as u32).clamp(DRY_MIN, EXT_DRY_ROUNDS)
    }

    /// One extension-considered round: `cleared` = the tau gate fired (a
    /// single firing resets the streak); a streak past the (cost-aware)
    /// threshold collapses consideration for EXT_DRY_COOLDOWN rounds.
    pub fn ext_dry_observe(&mut self, cleared: bool) {
        if cleared {
            self.ext_dry_streak = 0;
            return;
        }
        self.ext_dry_streak += 1;
        if self.ext_dry_streak >= self.ext_dry_threshold() {
            self.ext_dry_cooldown = EXT_DRY_COOLDOWN;
            self.ext_dry_streak = 0;
        }
    }

    /// `mtpExtDryAllows`: pure policy step — false while the cooldown runs
    /// (each considered round consumes one cooldown tick).
    pub fn ext_dry_allows(&mut self) -> bool {
        if self.ext_dry_cooldown > 0 {
            self.ext_dry_cooldown -= 1;
            return false;
        }
        true
    }

    /// Diagnostic/test view of the cost-aware dry threshold.
    pub fn ext_dry_threshold_public(&self) -> u32 {
        self.ext_dry_threshold()
    }

    /// Feed one round to the regime gate: `two_chunk` = the round's shape,
    /// `m_lo` the base depth, `ms_per_tok` the round's wall per emitted
    /// token, `inter_round_ms` the wall since the previous round's end
    /// (charged to the PREVIOUS shape — `mtpRegimeWallMs`). A shape change
    /// is a TRANSITION and is dropped; a shape observed at a new m_lo
    /// reseeds that shape's EMA.
    pub fn regime_observe(
        &mut self,
        two_chunk: bool,
        m_lo: usize,
        ms_per_tok: f64,
        inter_round_ms: f64,
    ) {
        // Inter-round wall: charge to the previous round's shape.
        if inter_round_ms > 0.0 {
            if let Some(prev) = self.regime_prev_shape {
                let e = if prev {
                    &mut self.regime_two_inter_ms
                } else {
                    &mut self.regime_single_inter_ms
                };
                *e = if *e <= 0.0 {
                    inter_round_ms
                } else {
                    *e * 0.9 + inter_round_ms * 0.1
                };
            }
        }
        if !ms_per_tok.is_finite() || ms_per_tok <= 0.0 {
            self.regime_prev_shape = Some(two_chunk);
            return;
        }
        // Transition: a round whose shape differs from its predecessor is
        // dropped (the minority shape read 5-7% slow on transitions).
        if self.regime_prev_shape.is_some() && self.regime_prev_shape != Some(two_chunk) {
            self.regime_prev_shape = Some(two_chunk);
            return;
        }
        self.regime_prev_shape = Some(two_chunk);
        let (ema, n, key) = if two_chunk {
            (self.regime_two_ms_tok, self.regime_two_n, self.regime_two_m_lo)
        } else {
            (self.regime_single_ms_tok, self.regime_single_n, self.regime_single_m_lo)
        };
        // New base depth: reseed (a shape's ms/tok at a different m_lo is a
        // different price).
        let value = if n == 0 || key != Some(m_lo) {
            ms_per_tok
        } else {
            ema * 0.9 + ms_per_tok * 0.1
        };
        if two_chunk {
            self.regime_two_ms_tok = value;
            self.regime_two_n = n + 1;
            self.regime_two_m_lo = Some(m_lo);
        } else {
            self.regime_single_ms_tok = value;
            self.regime_single_n = n + 1;
            self.regime_single_m_lo = Some(m_lo);
        }
    }

    /// `mtpRegimeVerdict` with hysteresis: `Some(true)` = two-chunk is the
    /// worse shape (ratio > 1.0 + MTP_REGIME_MARGIN 0.05 to call it worse,
    /// but a standing "worse" flips only at ratio > 1.0). `None` until both
    /// shapes are measured. Updates the standing latch.
    pub fn regime_two_chunk_worse(&mut self) -> Option<bool> {
        if self.regime_two_n == 0 || self.regime_single_n == 0 {
            return None;
        }
        let ratio = self.regime_two_ms_tok / self.regime_single_ms_tok;
        let v = match self.regime_worse {
            Some(w) => {
                if w && ratio <= 1.0 {
                    false // a standing worse flips only at ratio <= 1.0
                } else if !w && ratio > 1.0 + REGIME_MARGIN {
                    true
                } else {
                    w
                }
            }
            None => ratio > 1.0 + REGIME_MARGIN,
        };
        self.regime_worse = Some(v);
        Some(v)
    }

    /// `mtpRegimeForce` (idempotent per round): when two-chunk is measured
    /// worse, force single-chunk EXCEPT during a 2-round trial block
    /// (MTP_REGIME_EXPLORE_BLOCK = 2) every `period` rounds, period =
    /// ceil(2*gap/0.01) clamped [8, 128].
    pub fn regime_force_single(&mut self) -> bool {
        match self.regime_two_chunk_worse() {
            Some(true) => {
                let gap = self.regime_rounds - self.regime_last_explore;
                let period = (((2.0 * gap as f64) / REGIME_EXPLORE_DRAG).ceil() as usize)
                    .clamp(REGIME_EXPLORE_PERIOD, REGIME_EXPLORE_PERIOD_MAX);
                if gap >= period {
                    // Open a trial block: two-chunk runs again for 2 rounds.
                    if self.regime_explore_left == 0 {
                        self.regime_explore_left = REGIME_EXPLORE_BLOCK;
                        self.regime_last_explore = self.regime_rounds;
                    }
                }
                if self.regime_explore_left > 0 {
                    self.regime_explore_left -= 1;
                    false // trial block: let the plan run two-chunk
                } else {
                    true
                }
            }
            _ => false,
        }
    }

    /// The regime gate's round counter (advanced by the caller per round).
    pub fn regime_tick(&mut self) {
        self.regime_rounds += 1;
    }

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
    /// under-bills the chain several-fold at long KV (specs/01 §7).
    fn cost_ms(&self, depth: usize, kv_len: usize) -> f64 {
        self.priced_cost(depth, kv_len)
            .unwrap_or_else(|| self.serial_ms * (1.0 + 0.6 * depth as f64))
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
        // the empty-picks state of specs/08 incidental 1 has no path into a
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
        if self.rounds > 2 && self.rounds % 6 == 0 {
            // The gate reads the SAME bucket `bucket_to_read` resolves
            // (nearest trusted, lower preferred). No trusted bucket at this kv: the
            // trial is the only pricing (the `_ => true` arm below).
            let read_bucket = self
                .table
                .as_ref()
                .and_then(|t| t.bucket_to_read(kv_len));
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
                            Some((v, dr)) => {
                                cap(etok) / (v as f64 + dr) * 1000.0 >= stps * 0.95
                            }
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
        // Untrusted serial price (specs/09 M3): lisa never lets an UNMEASURED
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
        if best.0 == 0 {
            if let Some(d) = self.gate_kept_depth(kv_len, remaining) {
                self.last_pick = d;
                *self.picks.entry(d).or_insert(0) += 1;
                return d;
            }
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
mod tests {
    use super::*;

    /// The planner floor (§6): a ratcheted-down `m_lo_prev` (the novel-arm
    /// poisoning path — the base re-based to 1 and hysteresis never let it
    /// recover) must not sink the plan's base below the EV pick. `pick()`
    /// held depth 6 at 5.4 accepted/round on the 16k arm while the plan
    /// collapsed to m_lo = 1 and lost −51 %.
    #[test]
    fn plan_base_never_rebases_below_the_ev_pick() {
        let mut c = DepthController::new(None, 30.5, 6);
        // Poison the conditional EMAs so every cheap depth out-prices deep
        // ones, and ratchet the hysteresis state down to m_lo_prev = 1.
        c.ev_acc = [0.01; 6];
        c.m_lo_prev = 1;
        c.standing_lo = Some(1);
        c.standing_r = 1e9; // hysteresis keeps the collapsed base if allowed
        let p = c.plan_src(6, 1000, false);
        assert!(
            p.m_lo >= 6,
            "plan base {} sank below the EV pick 6",
            p.m_lo
        );
        // The extension horizon may still reach above the pick, never below.
        assert!(p.m_hi >= p.m_lo);
    }

    /// Reference gate shape (b92bde4): a depth MEASURED cheaper per emitted
    /// token than the plain step stays on even when the EV bar (3% margin
    /// over serial) would have parked the request on serial.
    #[test]
    fn gate_keeps_a_measured_winning_depth_on_when_the_ev_margin_fails() {
        // Depth 2 measured: 150 ms for 5 tokens = 30.0 ms/tok. Serial 30.5 ms
        // = 32.8 tok/s: d2's EV (33.3 tok/s) does NOT clear the 3% margin
        // (33.8), so the calibrated bar alone would park the request on
        // serial; the measurement (30.0 < 30.5) keeps it on.
        let mut c = DepthController::new(None, 30.5, 6);
        for _ in 0..4 {
            c.observe_round(2, 4, 150.0);
        }
        assert_eq!(c.round_beats_serial(2), Some(true));
        // Drive the acceptance EMAs down with empty rounds so EVERY depth's
        // EV falls under the serial bar (the calibrated bar alone would park
        // the request on serial); the gate then re-arms on the measurement.
        for _ in 0..3 {
            c.observe_round(6, 0, 200.0);
        }
        // Serial trusted (constructor measurement), so the gate is armed.
        // Past the bootstrap warmup so this pick IS an EV-vs-gate decision.
        c.rounds = EV_WARMUP_ROUNDS;
        assert_eq!(c.pick(1000, usize::MAX), 2, "gate outranks the 3% EV margin");
    }

    /// Forced-continuation term: at the tail of a forced-length decode the
    /// remaining token budget caps a depth's EV (`min(E, remaining)`), so a
    /// round that cannot fill its width yields to serial.
    #[test]
    fn forced_length_decode_parks_on_serial_for_the_tail() {
        // d2 is a clear EV winner at full budget: 5 tokens / 150 ms beats
        // serial 30.5 ms/step (32.8 tok/s).
        let mut c = DepthController::new(None, 30.5, 6);
        for _ in 0..4 {
            c.observe_round(2, 4, 150.0);
        }
        // Drive the shared EMA down so every depth's EV falls under serial
        // (the d3 analytic prior otherwise out-prices the trusted d2); the
        // gate then re-arms on the d2 measurement.
        for _ in 0..3 {
            c.observe_round(6, 0, 200.0);
        }
        // Past the bootstrap warmup; land on non-trial rounds (the schedule
        // fires when rounds % 6 == 0).
        c.rounds = EV_WARMUP_ROUNDS;
        assert_eq!(c.pick(1000, usize::MAX), 2, "full budget keeps the depth");
        // One token left: a d2 round still costs its full ~150 ms but can
        // only emit 1 token — 1/150 vs serial 1/30.5 → serial wins.
        c.rounds = 13;
        assert_eq!(
            c.pick(1000, 1),
            0,
            "a round that cannot fill its width is priced down to serial"
        );
        // Two tokens left: d2 can just fill its width (E=5 → capped 2):
        // 2/150 = 13.3 tok/s vs serial 32.8 — still serial.
        c.rounds = 15;
        assert_eq!(c.pick(1000, 2), 0);
    }

    /// spec 03: a depth's EV pricing must not inherit the shared acceptance
    /// EMA's decayed deep indexes once the depth is trusted in-run.
    #[test]
    fn per_depth_acceptance_is_not_polluted_by_shallower_picks() {
        // spec 03: a long d2 incumbency (rounds that never see index-2 hits)
        // must not sink d3's EV pricing. The shared EMA decays acc[2] toward
        // 0.49-ish; d3's own trusted EMA keeps its measured 0.66.
        let mut c = DepthController::new(None, 35.0, 6);
        for _ in 0..40 {
            c.observe_round(2, 2, 60.0); // d2 full accepts: acc[2] never updated
        }
        for _ in 0..20 {
            c.observe_round(3, 3, 70.0); // d3 rounds accept indexes 0..2
        }
        // d3 is trusted now: its EV uses ITS OWN acc, where index 2 reads the
        // measured ~1.0, not the shared EMA's value.
        let own = c.depth_acc.get(&3).copied().unwrap();
        assert!(own[2] > 0.9, "d3's index-2 EMA tracks its own rounds: {}", own[2]);
        assert!(c.acc_for(3)[2] > 0.9, "acc_for(d3) uses the per-depth EMA");
        // An untrusted depth still prices from the shared EMA.
        assert!((c.acc_for(5)[2] - c.acc[2]).abs() < 1e-9);
    }

    /// `None` until BOTH sides are trusted: the calibrated bar stays the only
    /// signal, and an untrusted serial cost never arms the gate.
    #[test]
    fn gate_is_none_until_both_sides_are_trusted() {
        let mut c = DepthController::new(None, 0.0, 6); // serial unmeasured
        c.observe_serial(40.0, false);
        c.observe_serial(40.0, false);
        for _ in 0..4 {
            c.observe_round(2, 4, 150.0);
        }
        assert_eq!(c.round_beats_serial(2), None, "serial not trusted yet");
        // One more serial sample does not flip it: serial_ms is an EMA seeded
        // at the FIRST sample, trust is a count.
        c.observe_serial(40.0, false);
        assert_eq!(c.serial_samples, 3);
        assert_eq!(
            c.round_beats_serial(2),
            Some(150.0 / c.round_tok[&2] < 40.0)
        );
    }

    /// A measured LOSING depth never flips the gate: round beats serial is a
    /// measurement question, and the measurement says no.
    #[test]
    fn gate_does_not_rescue_a_measured_losing_depth() {
        let mut c = DepthController::new(None, 20.0, 6);
        for _ in 0..4 {
            c.observe_round(2, 4, 150.0); // 37.5 ms/tok vs serial 20
        }
        assert_eq!(c.round_beats_serial(2), Some(false));
    }

    /// CLEARLY_WORSE settle: a depth whose first trusted sample reads >20%
    /// worse per emitted token than serial stops consuming trial rounds.
    #[test]
    fn a_clearly_worse_depth_is_settled_and_never_retried() {
        let mut c = DepthController::new(None, 30.0, 6);
        for _ in 0..3 {
            c.observe_round(6, 1, 200.0); // 200 ms/tok vs 30 * 1.2 = 36
        }
        assert!(c.settled_worse.contains(&6));
        // Trial rounds go to the un-settled depths only.
        loop {
            let d = c.pick(1000, usize::MAX);
            assert_ne!(d, 6, "a settled-worse depth is never picked");
            if c.rounds % 6 == 0 && c.rounds > 42 {
                break;
            }
        }
    }

    /// A depth inside the noise band (<=20% worse) is NOT settled: the trial
    /// schedule keeps pricing it, as before the gate existed.
    #[test]
    fn a_marginal_depth_stays_a_trial_candidate() {
        let mut c = DepthController::new(None, 30.0, 6);
        for _ in 0..3 {
            c.observe_round(6, 0, 35.0); // 35 ms/tok vs 36: noise, not clearly worse
        }
        assert!(!c.settled_worse.contains(&6));
        assert_eq!(c.round_beats_serial(6), Some(false));
    }

    /// P1 (§2.1): the CostSource — interpolation between measured widths,
    /// the raw-sample floor, and the slope-capped extrapolation past the
    /// widest measured width.
    #[test]
    fn cost_source_interpolates_floors_and_extrapolates() {
        let mut m = ModelCost::default();
        let mut ch = [Cell::default(); 6];
        ch[0] = Cell {
            ms: 10.0,
            tok: 0.0,
            n: 3,
        }; // depth 2
        m.chain_ms.insert(2, ch);
        let mut ch4 = [Cell::default(); 6];
        ch4[0] = Cell {
            ms: 20.0,
            tok: 0.0,
            n: 3,
        }; // depth 4
        m.chain_ms.insert(4, ch4);
        let mut v3 = [Cell::default(); 6];
        v3[0] = Cell {
            ms: 90.0,
            tok: 0.0,
            n: 3,
        };
        m.verify_ms.insert(3, v3);
        let mut v5 = [Cell::default(); 6];
        v5[0] = Cell {
            ms: 100.0,
            tok: 0.0,
            n: 3,
        };
        m.verify_ms.insert(5, v5);
        // depth 2: exact cells -> 10 + 90 = 100.
        assert!((m.measured_round_ms(2, 0).unwrap() - 100.0).abs() < 1e-4);
        // depth 3: chain 15 (midpoint), verify 95 -> 110.
        assert!((m.measured_round_ms(3, 0).unwrap() - 110.0).abs() < 1e-4);
        // depth 5: chain past the widest measured depth (4) -> extrapolation:
        // slope (20-10)/2 = 5 vs prior 0.6*40 = 24 -> max(24) per step:
        // 20 + 24 = 44 chain; verify 100 -> 144.
        let past = m.marginal_past_widest(5, 0, 40.0).unwrap();
        assert!((past - 144.0).abs() < 1e-4, "slope capped by the prior: {past}");
        // Raw floor: an untrusted verify sample ABOVE the interpolated price
        // floors depth 3's price.
        let mut raw = [Cell::default(); 6];
        raw[0] = Cell {
            ms: 130.0,
            tok: 0.0,
            n: 1,
        };
        m.verify_ms.insert(4, raw);
        let with_floor = m.cost_source_ms(3, 0, 40.0).unwrap();
        assert!(
            (with_floor - 130.0).abs() < 1e-4,
            "raw floor wins over interp when worse: {with_floor}"
        );
        // Depth 4: chain trusted 20 + verify interp 95 = 115; the raw sample
        // sits at verify width 4, which is depth 3's row — it must not leak
        // into depth 4's price.
        let d4 = m.cost_source_ms(4, 0, 40.0).unwrap();
        assert!((d4 - 120.0).abs() < 1e-4, "chain 20 + verify exact 100: {d4}");
    }

    /// P1: measured tokens replace the EMA model only when count-trusted.
    #[test]
    fn measured_tokens_are_trusted_by_count() {
        let mut m = ModelCost::default();
        let mut r3 = [Cell::default(); 6];
        r3[0] = Cell {
            ms: 100.0,
            tok: 3.2,
            n: 3,
        };
        m.round.insert(3, r3);
        let mut r4 = [Cell::default(); 6];
        r4[0] = Cell {
            ms: 120.0,
            tok: 1.0,
            n: 1,
        };
        m.round.insert(4, r4);
        assert!(
            (m.measured_round_tok(2, 0).unwrap() - 3.2).abs() < 1e-5,
            "trusted tok reads through"
        );
        assert_eq!(m.measured_round_tok(3, 0), None, "1-sample tok is a seed");
        // Folding keeps both columns in lockstep.
        let mut c = Cell::default();
        c.observe_pair(100.0, Some(3.0));
        c.observe_pair(120.0, Some(4.0));
        assert!((c.ms - 110.0).abs() < 1e-4 && (c.tok - 3.5).abs() < 1e-4);
        c.observe_pair(140.0, Some(5.0));
        assert!((c.tok - (3.5 + (5.0 - 3.5) / 3.0)).abs() < 1e-4, "mean at n=2");
        c.observe_pair(140.0, Some(5.0));
        assert!((c.tok - (4.0 * 0.9 + 5.0 * 0.1)).abs() < 1e-4, "EMA at n=3");
    }

    /// P1: the machine identity stamp is deterministic and order-bound.
    #[test]
    fn machine_key_binds_chip_model_quant_os() {
        let a = ModelCost::machine_key("Apple M5 Max", "/m", "4bit", "macOS 27.0");
        let b = ModelCost::machine_key("Apple M5 Max", "/m", "4bit", "macOS 27.0");
        let c = ModelCost::machine_key("Apple M4 Max", "/m", "4bit", "macOS 27.0");
        assert_eq!(a, b);
        assert_ne!(a, c);
        assert!(a.starts_with("rc1-"));
    }

    /// P3 (tested here for the fold rule): the whole-round serving fold with
    /// the reference rejection rules + the stale reseed.
    #[test]
    fn serving_round_fold_rejects_bad_and_reseeds_stale() {
        let mut m = ModelCost::default();
        assert!(!m.observe_round_cell(4, 0, f64::NAN, 3.0), "non-finite rejected");
        assert!(!m.observe_round_cell(4, 0, 0.0, 3.0), "non-positive rejected");
        assert!(m.observe_round_cell(4, 0, 150.0, 3.0));
        assert!(m.observe_round_cell(4, 0, 160.0, 3.5));
        assert!(m.observe_round_cell(4, 0, 155.0, 3.2));
        let c = &m.round[&4][0];
        assert_eq!(c.n, 3);
        assert!(c.trusted_tok().is_some());
        // A different machine's row restores STALE: the first live sample
        // re-folds at weight 0.5 and keeps the trust count.
        let mut m2 = ModelCost::default();
        m2.machine = "rc1-deadbeef".into();
        m2.stale = true;
        let mut r = [Cell::default(); 6];
        r[0] = Cell {
            ms: 100.0,
            tok: 4.0,
            n: 9,
        };
        m2.round.insert(4, r);
        assert!(m2.observe_round_cell(4, 0, 200.0, 3.0));
        let c = &m2.round[&4][0];
        assert_eq!(c.n, 9, "trust count kept");
        assert!((c.ms - 150.0).abs() < 1e-4 && (c.tok - 3.5).abs() < 1e-4);
        assert!(!m2.stale, "reseed consumed");
    }

    /// P2 (§2.3): the conditional EMAs fold prefix-structured — indices below
    /// the first rejection toward 1.0, the rejecting index toward 0.0, deeper
    /// indices untouched.
    #[test]
    fn ev_observe_is_conditional_on_the_first_rejection() {
        let mut c = DepthController::new(None, 40.0, 6);
        c.ev_observe(4, 2);
        assert!(c.ev_acc[0] > EV_PRIOR && c.ev_acc[1] > EV_PRIOR, "accepted folds up");
        assert!(c.ev_acc[2] < EV_PRIOR, "the rejecting index folds down");
        assert!((c.ev_acc[3] - EV_PRIOR).abs() < 1e-12, "unreached index untouched");
        assert!((c.ev_acc[5] - EV_PRIOR).abs() < 1e-12);
    }

    /// P2: the plan collapses (m_hi == m_lo, tau 0) when the horizon closes —
    /// steep marginals with a cold deep index never sync.
    #[test]
    fn plan_collapses_when_the_horizon_closes() {
        let mut m = ModelCost::default();
        let mut serial = [Cell::default(); 6];
        serial[0] = Cell {
            ms: 40.0,
            tok: 0.0,
            n: 3,
        };
        m.serial_ms = serial;
        // Steep verify curve (~+80 ms/row) with a flat chain: the horizon
        // closes once cond (seeded 0.85) <= best_r * marginal.
        for (s, vms) in [(2u32, 90.0f32), (3, 170.0), (4, 250.0), (5, 330.0), (6, 410.0), (7, 490.0)] {
            let mut v = [Cell::default(); 6];
            v[0] = Cell {
                ms: vms,
                tok: 0.0,
                n: 3,
            };
            m.verify_ms.insert(s, v);
        }
        for d in 1u32..=6 {
            let mut ch = [Cell::default(); 6];
            ch[0] = Cell {
                ms: 60.0,
                tok: 0.0,
                n: 3,
            };
            m.chain_ms.insert(d, ch);
        }
        let mut c = DepthController::new(Some(m), 40.0, 6);
        c.rounds = EV_WARMUP_ROUNDS;
        let p = c.plan_src(6, 1000, false);
        assert_eq!(p.m_lo, p.m_hi, "steep marginals close the horizon: {p:?}");
        assert!(
            (p.tau_ln).abs() < 1e-9,
            "collapsed: tau_ln 0 (ln 1.0): {p:?}"
        );
    }

    /// P2: flat costs + optimistic EMAs open a horizon; tau clamps to
    /// [0.05, 0.95]; the standing base survives a <5% challenger.
    #[test]
    fn plan_extends_clamps_and_holds_the_standing_base() {
        let mut m = ModelCost::default();
        let mut serial = [Cell::default(); 6];
        serial[0] = Cell {
            ms: 40.0,
            tok: 0.0,
            n: 3,
        };
        m.serial_ms = serial;
        // Flat verify: every width ~100 ms; chain flat 10 ms.
        for d in 2u32..=6 {
            let mut v = [Cell::default(); 6];
            v[0] = Cell {
                ms: 100.0,
                tok: 0.0,
                n: 3,
            };
            m.verify_ms.insert(d + 1, v);
            let mut ch = [Cell::default(); 6];
            ch[0] = Cell {
                ms: 10.0,
                tok: 0.0,
                n: 3,
            };
            m.chain_ms.insert(d, ch);
        }
        let mut c = DepthController::new(Some(m), 40.0, 6);
        c.rounds = EV_WARMUP_ROUNDS;
        // Optimistic EMAs: everything accepts.
        for i in 0..6 {
            c.ev_acc[i] = 0.95;
        }
        // Floor (§6): the base IS the pick, so a pick of 6 leaves no
        // conditional extension room — pass 3 and the horizon extends to the
        // absolute cap.
        let p = c.plan_src(3, 1000, false);
        assert!(p.m_hi > p.m_lo, "flat costs + high EMAs extend: {p:?}");
        assert!(p.m_hi <= 6, "cap respected");
        let tau = p.tau_ln.exp();
        assert!(tau <= 0.95 + 1e-9 && (tau > 0.05 || p.m_hi == p.m_lo));
        // Hysteresis: standing base 2; a challenger needs >5% to move it.
        let c2p = c.plan_src(3, 1000, false);
        assert_eq!(c2p.m_lo, p.m_lo, "standing base kept on a tie");
    }

    /// P2: chain_log_conf — NaN poisons to -inf (never extends), log clamps.
    #[test]
    fn chain_log_conf_sums_clamped_logs() {
        assert!(
            (DepthController::chain_log_conf(&[0.0, -0.5]) + 0.5).abs() < 1e-9,
            "inputs are logs: sum of clamped logs"
        );
        assert_eq!(DepthController::chain_log_conf(&[f64::NAN]), f64::NEG_INFINITY);
        assert!(DepthController::chain_log_conf(&[-2.0]) > DepthController::chain_log_conf(&[-5.0]));
    }

    /// P4: the dry-spell gate — a cost-aware threshold, streak collapse into
    /// the cooldown, a single firing resetting the streak.
    #[test]
    fn ext_dry_gate_collapses_after_a_dry_spell() {
        let mut c = DepthController::new(None, 40.0, 6);
        // No sync measured yet: threshold = DRY_ROUNDS (16).
        assert_eq!(c.ext_dry_threshold_public(), 16);
        for _ in 0..15 {
            c.ext_dry_observe(false);
            assert!(c.ext_dry_allows(), "streak below threshold: still allowed");
        }
        c.ext_dry_observe(false); // 16th dry round
        assert!(!c.ext_dry_allows(), "cooldown open");
        for _ in 0..31 {
            assert!(!c.ext_dry_allows());
        }
        assert!(c.ext_dry_allows(), "cooldown consumed after 32 considered rounds");
        // A firing resets the streak mid-spell.
        let mut c2 = DepthController::new(None, 40.0, 6);
        for _ in 0..10 {
            c2.ext_dry_observe(false);
        }
        c2.ext_dry_observe(true);
        for _ in 0..15 {
            c2.ext_dry_observe(false);
        }
        assert!(c2.ext_dry_allows(), "streak reset by the firing");
        // Cost-aware threshold: an expensive sync (30% of the round) backs
        // off sooner.
        c2.sync_ms = 12.0;
        c2.live_round_ms = 40.0; // sync = 30% of the round -> threshold 1? round(0.30/0.30)=1 -> clamp 3
        assert_eq!(c2.ext_dry_threshold_public(), 3, "expensive sync -> DRY_MIN");
        c2.sync_ms = 4.0;
        assert_eq!(c2.ext_dry_threshold_public(), 3, "0.30/0.10 = 3");
        c2.sync_ms = 1.0;
        assert_eq!(c2.ext_dry_threshold_public(), 12, "0.30/0.025 = 12");
    }

    /// P4: the regime gate — hysteresis verdict + explore blocks.
    #[test]
    fn regime_gate_holds_a_verdict_and_explores() {
        let mut c = DepthController::new(None, 40.0, 6);
        assert_eq!(c.regime_two_chunk_worse(), None, "nothing measured yet");
        // Single-chunk 20 ms/tok, two-chunk 24 ms/tok (ratio 1.2 > 1.05).
        // Shape changes are transitions (dropped), so fold in BLOCKS.
        for _ in 0..3 {
            for _ in 0..6 {
                c.regime_observe(false, 2, 20.0, 0.0);
            }
            for _ in 0..6 {
                c.regime_observe(true, 2, 24.0, 0.0);
            }
        }
        assert_eq!(c.regime_two_chunk_worse(), Some(true));
        // Force logic: single-chunk forced, but a 2-round explore block opens.
        let mut forced = 0;
        let mut saw_trial = false;
        for r in 0..200 {
            c.regime_tick();
            if !c.regime_force_single() {
                saw_trial = true;
            } else {
                forced += 1;
            }
        }
        assert!(saw_trial, "the worse shape still gets its trial block");
        assert!(forced > 150, "forced mostly single: {forced}");
        // A standing worse flips only at ratio <= 1.0 (ratio 1.02: holds).
        let mut c2 = DepthController::new(None, 40.0, 6);
        for _ in 0..6 {
            c2.regime_observe(false, 2, 20.0, 0.0);
        }
        for _ in 0..6 {
            c2.regime_observe(true, 2, 21.5, 0.0); // 1.075 -> worse
        }
        assert_eq!(c2.regime_two_chunk_worse(), Some(true));
        for _ in 0..5 {
            c2.regime_observe(true, 2, 20.3, 0.0); // ratio 1.015: inside the band
        }
        assert_eq!(c2.regime_two_chunk_worse(), Some(true), "hysteresis holds");
        for _ in 0..12 {
            c2.regime_observe(true, 2, 18.0, 0.0); // EMA crosses below single: ratio <= 1.0
        }
        assert_eq!(c2.regime_two_chunk_worse(), Some(false), "standing worse flips at ratio <= 1.0");
        // Transition drop: alternating shapes fold nothing new.
        let mut c3 = DepthController::new(None, 40.0, 6);
        c3.regime_observe(false, 2, 20.0, 0.0);
        c3.regime_observe(true, 2, 100.0, 0.0); // transition: dropped
        assert_eq!(c3.regime_two_n, 0, "transition not folded");
    }

    fn cell_at(n: u32) -> Cell {
        Cell {
            ms: 40.0,
            tok: 0.0,
            n,
        }
    }

    fn table_with(active_bucket: usize) -> ModelCost {
        let mut m = ModelCost::default();
        m.serial_ms[active_bucket] = cell_at(ModelCost::TABLE_MIN_SAMPLES);
        m
    }

    /// Reference `bucketToRead` semantics on the legacy grid: own bucket when
    /// active; else the nearest ACTIVE one, LOWER preferred; `None` when the
    /// whole table is cold (the prior applies). 1-2-sample cells are seeds,
    /// never data (reference `trusted` / `MIN_SAMPLES`).
    #[test]
    fn bucket_to_read_mirrors_the_reference() {
        // Cold table: no active bucket anywhere -> the prior.
        assert_eq!(ModelCost::default().bucket_to_read(32768), None);
        // Own bucket active reads itself — including the 32k+ fold.
        let t = table_with(5);
        assert_eq!(t.bucket_to_read(32768), Some(5));
        assert_eq!(t.bucket_to_read(100_000), Some(5));
        // Active LOWER bucket wins the tie at equal distance.
        let t = table_with(3);
        assert_eq!(t.bucket_to_read(4096), Some(3)); // own
        assert_eq!(t.bucket_to_read(6000), Some(3)); // 4-8k: lower beats upper
        assert_eq!(t.bucket_to_read(1000), Some(3)); // <2k walks UP to it
        // A nearer UPPER active bucket wins when no lower exists.
        let t = table_with(4);
        assert_eq!(t.bucket_to_read(1000), Some(4));
        // 1-2 samples are not trusted: falls through to the trusted bucket.
        let mut t = table_with(0);
        t.serial_ms[3] = cell_at(1);
        assert_eq!(t.bucket_to_read(6000), Some(0));
    }

    /// specs/09 M1: the serial prior walks to the nearest TRUSTED bucket
    /// (every table value is read through `bucket_to_read`), and a
    /// 1-2 sample seed is never a prior.
    #[test]
    fn serial_prior_walks_past_an_empty_or_seeded_bucket() {
        let mut m = ModelCost::default();
        assert_eq!(m.serial_prior_ms(2343), None, "cold table: no prior");
        // Own bucket (2-4k) seeded at n = 1 — the old `cell()` read priced
        // this; the `trusted` rule does not.
        m.serial_ms[1] = cell_at(1);
        assert_eq!(m.cell(Which::Serial, 0, 1), None, "a seed is not pricing");
        assert_eq!(m.serial_prior_ms(2343), None, "own seed does not anchor");
        // Neighbour trusted: kv 2343 reads the <2k cell — the exact repro24k
        // shape (empty 2-4k cell, n = 36 under <2k).
        m.serial_ms[0] = Cell { ms: 38.5, tok: 0.0, n: 36 };
        assert_eq!(m.serial_prior_ms(2343), Some(38.5));
        // Own bucket trusted: it reads itself.
        m.serial_ms[1] = Cell { ms: 41.0, tok: 0.0, n: 3 };
        assert_eq!(m.serial_prior_ms(2343), Some(41.0));
    }

    /// The first MIN_SAMPLES folds are a running mean — an EMA seeded from
    /// sample 1 is still sample 1 at n = 3 — and the EMA takes over after.
    #[test]
    fn a_cell_folds_a_running_mean_until_it_is_trusted() {
        let mut c = Cell::default();
        c.observe(40.0);
        assert_eq!((c.ms, c.n), (40.0, 1));
        c.observe(60.0);
        assert!((c.ms - 50.0).abs() < 1e-6, "first fold is a mean: {}", c.ms);
        c.observe(60.0);
        assert!((c.ms - 53.3333).abs() < 1e-3, "still a mean at n=2: {}", c.ms);
        c.observe(60.0); // n = 3 before this call: EMA (BETA 0.10) takes over
        assert_eq!(c.n, 4);
        assert!((c.ms - 54.0).abs() < 1e-3, "EMA after trust: {}", c.ms);
    }

    /// specs/09 M2: the probe discards PROBE_WARM cold ticks and TRUST IS A
    /// COUNT. The repro24k b.arm fingerprint — an EOS-truncated two-tick
    /// probe whose median read the cold sample (86.31 ms) — must never arm
    /// any serial-priced path.
    #[test]
    fn a_truncated_probe_never_trusts_its_cold_samples() {
        let (val, warm) = probe_warm_stats(&[120.0, 86.31]);
        assert_eq!(warm, 0, "no warm fold survives a 2-tick probe");
        assert!((val - 103.155).abs() < 1e-9, "display value only: {val}");
        let mut c = DepthController::new_with_serial(None, val, warm, 6);
        for _ in 0..4 {
            c.observe_round(2, 4, 150.0);
        }
        assert_eq!(c.serial_samples, 0);
        assert_eq!(
            c.round_beats_serial(2),
            None,
            "an untrusted serial never arms the gate"
        );
        // A full probe: 8 ticks, 2 discarded, 6 warm folds -> trusted.
        let samples = [90.0, 88.0, 41.0, 40.5, 40.2, 40.9, 40.1, 40.6];
        let (val, warm) = probe_warm_stats(&samples);
        assert_eq!(warm, 6, "PROBE_TICKS - PROBE_WARM folds");
        let expect = (41.0 + 40.5 + 40.2 + 40.9 + 40.1 + 40.6) / 6.0;
        assert!((val - expect).abs() < 1e-9, "value = mean of warm folds: {val}");
        let c = DepthController::new_with_serial(None, val, warm, 6);
        assert!(c.serial_trusted(), "warm folds >= MIN_SAMPLES");
    }

    /// specs/09 M3: warmup runs the default depth, and an UNMEASURED serial
    /// step is never a candidate — rounds keep running (an undecided vote keeps
    /// the arm; serial is a candidate only where measured).
    #[test]
    fn warmup_and_an_untrusted_serial_keep_rounds_running() {
        // The ghost number with no fold count behind it (old constructor
        // trusted any positive serial_ms).
        let mut c = DepthController::new_with_serial(None, 86.31, 0, 6);
        for r in 1..=EV_WARMUP_ROUNDS {
            assert_eq!(
                c.pick(2343, 128 - r),
                2,
                "warmup round {r} runs the default depth"
            );
        }
        assert_eq!(c.picks.get(&2), Some(&(EV_WARMUP_ROUNDS as usize)));
        // Post-warmup, serial is still unmeasured and nothing prices a round
        // (no table, no in-run samples): default depth, NOT a serial park.
        for _ in 0..5 {
            assert_eq!(c.pick(2343, 64), 2, "an untrusted serial is never picked");
        }
        assert!(!c.picks.contains_key(&0), "no serial pick anywhere");
        // With a measured TABLE the untrusted-serial argmax prices depths:
        // d4 (70 ms, E[tok] 3.05 -> 43.6 tok/s) beats d2 (140 ms -> 16.5),
        // so the pick must come from the table, not from the default.
        let mut m = ModelCost::default();
        let mut serial = [Cell::default(); 6];
        serial[0] = Cell { ms: 40.0, tok: 0.0, n: 3 };
        m.serial_ms = serial;
        let mut v3 = [Cell::default(); 6];
        v3[0] = Cell { ms: 100.0, tok: 0.0, n: 3 };
        m.verify_ms.insert(3, v3);
        let mut v5 = [Cell::default(); 6];
        v5[0] = Cell { ms: 60.0, tok: 0.0, n: 3 };
        m.verify_ms.insert(5, v5);
        let mut ch2 = [Cell::default(); 6];
        ch2[0] = Cell { ms: 40.0, tok: 0.0, n: 3 };
        m.chain_ms.insert(2, ch2);
        let mut ch4 = [Cell::default(); 6];
        ch4[0] = Cell { ms: 10.0, tok: 0.0, n: 3 };
        m.chain_ms.insert(4, ch4);
        let mut c2 = DepthController::new_with_serial(Some(m), 86.31, 0, 6);
        c2.rounds = EV_WARMUP_ROUNDS;
        assert_eq!(
            c2.pick(1000, 128),
            4,
            "table-priced argmax decides while serial is unmeasured"
        );
    }

    /// specs/09 M4: the gate falls back to the MEASURED table when no in-run
    /// round exists (it reads the persisted cells) — MIN_SAMPLES must not block
    /// the very state the gate exists to rescue.
    #[test]
    fn the_gate_prices_from_the_table_before_any_round_ran() {
        let mut m = ModelCost::default();
        // serial 40 ms -> 25.0 tok/s. d2's table price: verify S3 87.6 +
        // chain 3.53 = 91.13 ms for E[tok] = 2.3125 (acc seed 0.75) ->
        // 25.375 tok/s: beats serial outright, but NOT the EV loop's 3%
        // margin (25.75) — EV parks, the gate must keep d2.
        let mut serial = [Cell::default(); 6];
        serial[0] = Cell { ms: 40.0, tok: 0.0, n: 3 };
        m.serial_ms = serial;
        let mut v3 = [Cell::default(); 6];
        v3[0] = Cell { ms: 87.6, tok: 0.0, n: 3 };
        m.verify_ms.insert(3, v3);
        let mut v4 = [Cell::default(); 6];
        v4[0] = Cell { ms: 500.0, tok: 0.0, n: 3 };
        m.verify_ms.insert(4, v4);
        let mut ch2 = [Cell::default(); 6];
        ch2[0] = Cell { ms: 3.53, tok: 0.0, n: 3 };
        m.chain_ms.insert(2, ch2);
        let mut c = DepthController::new(Some(m), 40.0, 6);
        c.rounds = EV_WARMUP_ROUNDS;
        assert_eq!(
            c.pick(1000, usize::MAX),
            2,
            "table-priced gate keeps d2 with zero in-run rounds"
        );
        assert!(
            c.round_n.iter().all(|(_, n)| *n == 0),
            "the decision needed no in-run samples"
        );
    }

    // --- P7: sticky-disable floor (port spec §2.8, the reference's three
    // `mtpFloorDisableObserve` / `updateMtpEvRound` tests) ---

    /// Extension misses do not poison depth one: a wide round at base depth
    /// one that accepted its first draft counts as a HIT (accepted > 0), and
    /// a window of them reads rate 1.0; alternating hits read 0.5 — both
    /// above the breakeven floor.
    #[test]
    fn floor_extension_misses_do_not_poison_depth_one() {
        let mut drafted = [0u8; DEPTH_WINDOW];
        let mut accepted = [0u8; DEPTH_WINDOW];
        let mut idx = 0u32;
        let mut rate = None;
        for _ in 0..DEPTH_WINDOW {
            rate = DepthController::floor_disable_observe(
                &mut drafted, &mut accepted, &mut idx, 1, 8, 1,
            );
        }
        let rate = rate.expect("a full window of base-depth-one rounds returns a rate");
        assert!((rate - 1.0).abs() < 1e-5);
        assert!(drafted.iter().all(|&s| s == 1));

        let mut accepted = [0u8; DEPTH_WINDOW];
        let mut idx = 0u32;
        let mut rate = None;
        for i in 0..DEPTH_WINDOW {
            rate = DepthController::floor_disable_observe(
                &mut drafted,
                &mut accepted,
                &mut idx,
                1,
                8,
                usize::from(i % 2 == 0),
            );
        }
        let rate = rate.expect("full window");
        assert!((rate - 0.5).abs() < 1e-5);
        // A 50% depth-one window is comfortably above the breakeven floor —
        // this rate must never disable.
        assert!(rate >= DISABLE_BELOW);
    }

    /// Disable needs 16 fresh failures at base depth one; a wider base round
    /// resets the probation window.
    #[test]
    fn floor_disable_needs_sixteen_fresh_failures_at_base_depth_one() {
        let mut drafted = [0u8; DEPTH_WINDOW];
        let mut accepted = [0u8; DEPTH_WINDOW];
        let mut idx = 0u32;
        for _ in 0..DEPTH_WINDOW - 1 {
            assert_eq!(
                DepthController::floor_disable_observe(
                    &mut drafted, &mut accepted, &mut idx, 1, 1, 0,
                ),
                None
            );
        }
        // A wider base round invalidates the probation window.
        assert_eq!(
            DepthController::floor_disable_observe(&mut drafted, &mut accepted, &mut idx, 2, 4, 0),
            None
        );
        assert_eq!(idx, 0);
        // Another complete run of depth-one failures produces rate 0.
        for _ in 0..DEPTH_WINDOW - 1 {
            assert_eq!(
                DepthController::floor_disable_observe(
                    &mut drafted, &mut accepted, &mut idx, 1, 1, 0,
                ),
                None
            );
        }
        let rate = DepthController::floor_disable_observe(
            &mut drafted, &mut accepted, &mut idx, 1, 1, 0,
        )
        .expect("sixteenth failure");
        assert!(rate.abs() < 1e-5);
        assert!(rate < DISABLE_BELOW);
    }

    /// Sticky disable fires from the round update (a full window of
    /// first-draft rejections, whatever the drafted width), and only
    /// post-warmup.
    #[test]
    fn ev_round_update_sticky_disable_uses_the_first_draft_at_base_depth_one() {
        let mut good = DepthController::new(None, 30.5, 6);
        good.ev_rounds = EV_WARMUP_ROUNDS as u32; // warmup consumed
        for _ in 0..DEPTH_WINDOW * 2 {
            good.ev_round_update(8, 1, 1);
        }
        assert!(!good.spec_disabled);
        assert!(good.floor_window_drafted.iter().all(|&s| s == 1));
        assert!(good.floor_window_accepted.iter().all(|&s| s == 1));

        let mut bad = DepthController::new(None, 30.5, 6);
        bad.ev_rounds = EV_WARMUP_ROUNDS as u32; // warmup consumed
        for _ in 0..DEPTH_WINDOW - 1 {
            bad.ev_round_update(8, 0, 1);
            assert!(!bad.spec_disabled);
        }
        bad.ev_round_update(8, 0, 1);
        assert!(bad.spec_disabled);
        // Sticky: the flag survives and pick parks on serial.
        assert_eq!(bad.pick(1000, usize::MAX), 0);
        assert_eq!(bad.pick(1000, usize::MAX), 0, "still serial");
    }

    /// Warmup consumes the window (mixed warmup evidence belongs in no
    /// floor window), and a wider base round resets fresh floor evidence.
    #[test]
    fn ev_round_update_warmup_and_wider_base_rounds_reset_floor_evidence() {
        let mut g = DepthController::new(None, 30.5, 6);
        g.ev_rounds = EV_WARMUP_ROUNDS as u32 - 1;
        g.floor_window_idx = 7;
        g.ev_round_update(4, 0, 1);
        assert_eq!(g.ev_rounds, EV_WARMUP_ROUNDS as u32);
        assert_eq!(g.floor_window_idx, 0);

        for _ in 0..DEPTH_WINDOW - 1 {
            g.ev_round_update(1, 0, 1);
        }
        assert!(!g.spec_disabled);
        assert_eq!(g.floor_window_idx, DEPTH_WINDOW as u32 - 1);

        g.ev_round_update(4, 0, 2);
        assert_eq!(g.floor_window_idx, 0);
        assert!(!g.spec_disabled);
    }
}
