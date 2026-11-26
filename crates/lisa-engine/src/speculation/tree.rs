//! Host control of greedy tree rounds, after the pinned reference's decode
//! cycle: the prompt-lookup chain and its learned acceptance rates, the
//! draft's learned acceptance model, the measured cost of a round per
//! verified row count, and the choice of how many rows a round verifies.
//! The tree itself grows on the device ([`DeviceTree`]).

/// Proposal positions of a draft block.
pub const POSITIONS: usize = 7;
/// Most rows a round verifies, the anchor's included.
pub const MOST_ROWS: usize = 16;
/// Most host chain tokens.
pub const CHAIN_TOKENS: usize = MOST_ROWS - 1;
/// A chain token's agreement classes with the draft (top candidate, another
/// candidate, absent, past the draft's positions).
pub const AGREEMENTS: usize = 4;
/// Children a draft row takes.
pub const CHILDREN: usize = 4;
/// Rows of one verify pass of the attention kernels' narrow cap.
pub const NARROW_CAP: usize = 8;
/// Candidates per proposal position.
pub const CANDIDATES: usize = 16;

/// One round's tree request.
#[derive(Clone, Debug, PartialEq)]
pub struct TreePlan {
    /// Deepest proposal a walk may retain.
    pub depth: usize,
    /// Host chain tokens (at most [`CHAIN_TOKENS`]) and their rates per
    /// agreement, `[CHAIN_TOKENS * AGREEMENTS]`.
    pub chain: Vec<u32>,
    pub rates: Vec<f32>,
    pub children: usize,
    pub acceptance: Acceptance,
    /// Verified rows, the anchor's included, and the lane cap.
    pub rows: usize,
    pub lane_cap: usize,
    /// Whether the round commits the path it accepts (a cost measurement
    /// does not), which lets a model accept and commit on the device right
    /// behind its verification.
    pub committing: bool,
}

/// The device outputs of a tree round before readback.
pub struct DeviceTree<T> {
    /// Verify IDs `I32 [rows]` and tree `I32 [2, rows]` (parents, depths).
    pub ids: T,
    pub tree: T,
    /// The grown tree in its growth order: tokens `U32 [16]`, rows
    /// `U32 [16, 2]` (ancestors; parent | depth << 16), scores
    /// `F32 [16, 2]` (draft probability, estimate), caps `F32 [16]`, size
    /// `U32 [1]` and the sorted candidates `U32 [7, 16]`.
    pub tokens: T,
    pub rows: T,
    pub scores: T,
    pub caps: T,
    pub size: T,
    pub candidates: T,
}

/// The grown tree read back after a round.
#[derive(Clone, Debug, Default)]
pub struct GrownTree {
    pub tokens: Vec<u32>,
    pub parents: Vec<u32>,
    /// Draft probability (-1 for a chain row no draft step joined) and
    /// estimate.
    pub q: Vec<f32>,
    pub estimate: Vec<f32>,
    pub caps: Vec<f32>,
    /// Sorted candidates `[7 * 16]`.
    pub candidates: Vec<u32>,
}

// ---------------------------------------------------------------------------
// Lookup

/// Shortest and longest suffix a lookup match counts.
pub const MINIMUM_MATCH: usize = 3;
pub const LONGEST_MATCH: usize = 64;

/// The longest suffix of `history` that also ended earlier, at least
/// [`MINIMUM_MATCH`] tokens: its length (capped at [`LONGEST_MATCH`]) and the
/// most recent earlier end of it (of its last [`LONGEST_MATCH`] tokens when
/// longer).
pub fn lookup_match(history: &[u32]) -> Option<(usize, usize)> {
    let n = history.len();
    if n <= MINIMUM_MATCH {
        return None;
    }
    let mut best: Option<(usize, usize)> = None;
    for end in (MINIMUM_MATCH - 1..n - 1).rev() {
        if history[end + 1 - MINIMUM_MATCH..=end] != history[n - MINIMUM_MATCH..] {
            continue;
        }
        let mut length = MINIMUM_MATCH;
        while length < LONGEST_MATCH
            && length <= end
            && history[end - length] == history[n - 1 - length]
        {
            length += 1;
        }
        if best.is_none_or(|(b, _)| length > b) {
            best = Some((length, end));
            if length == LONGEST_MATCH {
                break;
            }
        }
    }
    best
}

/// The lookup chain after `history`: up to `count` tokens that followed the
/// match, never past the latest token, and the match length.
pub fn lookup_chain(history: &[u32], count: usize) -> (Vec<u32>, usize) {
    match lookup_match(history) {
        Some((length, end)) if count > 0 => (
            history[end + 1..history.len().min(end + 1 + count)].to_vec(),
            length,
        ),
        _ => (vec![], 0),
    }
}

const LENGTH_BUCKETS: usize = 6;
const DEPTH_BUCKETS: usize = 6;
const PRIOR_LABELS: f64 = 4.0;
const HALF_LIFE: f64 = 4096.0;

/// The pinned reference's prior acceptance of a lookup chain token by match
/// length bucket, depth bucket and agreement (top, candidate, absent,
/// unknown).
#[rustfmt::skip]
const LOOKUP_PRIOR: [[[f64; AGREEMENTS]; DEPTH_BUCKETS]; LENGTH_BUCKETS] = [
    [[0.966, 0.214, 0.125, 0.648], [0.973, 0.307, 0.232, 0.709], [0.980, 0.516, 0.559, 0.787],
     [0.979, 0.652, 0.732, 0.830], [0.973, 0.569, 0.551, 0.856], [0.979, 0.634, 0.617, 0.911]],
    [[0.975, 0.249, 0.153, 0.749], [0.980, 0.349, 0.276, 0.798], [0.985, 0.564, 0.614, 0.857],
     [0.984, 0.694, 0.775, 0.888], [0.980, 0.616, 0.607, 0.906], [0.984, 0.678, 0.670, 0.943]],
    [[0.981, 0.307, 0.163, 0.812], [0.985, 0.419, 0.292, 0.851], [0.989, 0.634, 0.633, 0.897],
     [0.988, 0.753, 0.789, 0.920], [0.985, 0.682, 0.626, 0.933], [0.989, 0.738, 0.688, 0.960]],
    [[0.987, 0.327, 0.220, 0.867], [0.990, 0.441, 0.374, 0.896], [0.992, 0.655, 0.714, 0.929],
     [0.992, 0.770, 0.844, 0.946], [0.990, 0.702, 0.708, 0.955], [0.992, 0.756, 0.761, 0.973]],
    [[0.995, 0.648, 0.439, 0.935], [0.996, 0.750, 0.624, 0.950], [0.997, 0.878, 0.874, 0.967],
     [0.997, 0.927, 0.937, 0.975], [0.996, 0.899, 0.871, 0.979], [0.997, 0.921, 0.898, 0.988]],
    [[0.999, 0.955, 0.859, 0.993], [1.000, 0.972, 0.928, 0.995], [1.000, 0.988, 0.982, 0.997],
     [1.000, 0.993, 0.992, 0.997], [1.000, 0.990, 0.981, 0.998], [1.000, 0.993, 0.986, 0.999]],
];

const UNKNOWN: usize = 3;

fn bit_width(x: usize) -> usize {
    (usize::BITS - x.leading_zeros()) as usize
}
fn length_bucket(length: usize) -> usize {
    (bit_width(length) - 2).min(LENGTH_BUCKETS - 1)
}
fn depth_bucket(depth: usize) -> usize {
    bit_width(depth).min(DEPTH_BUCKETS - 1)
}

/// Learned acceptance of greedy lookup chain tokens: decayed counts over the
/// prior.
#[derive(Clone, Debug)]
pub struct LookupAcceptance {
    /// (accepted, labels) per length, depth and agreement.
    counts: [[[(f64, f64); AGREEMENTS]; DEPTH_BUCKETS]; LENGTH_BUCKETS],
}

impl Default for LookupAcceptance {
    fn default() -> Self {
        Self {
            counts: [[[(0.0, 0.0); AGREEMENTS]; DEPTH_BUCKETS]; LENGTH_BUCKETS],
        }
    }
}

impl LookupAcceptance {
    pub fn rate(&self, length: usize, depth: usize, agreement: usize) -> f32 {
        let (l, d) = (length_bucket(length), depth_bucket(depth));
        let (accepted, labels) = self.counts[l][d][agreement];
        ((accepted + PRIOR_LABELS * LOOKUP_PRIOR[l][d][agreement]) / (labels + PRIOR_LABELS)) as f32
    }
    /// The chain's rates `[CHAIN_TOKENS * AGREEMENTS]` and its chances, the
    /// running products of its tokens' unknown-agreement rates.
    pub fn rates(&self, length: usize, chain: &[u32]) -> (Vec<f32>, Vec<f32>) {
        let mut rates = vec![0f32; CHAIN_TOKENS * AGREEMENTS];
        let mut chances = vec![];
        let mut chance = 1f32;
        for at in 0..chain.len() {
            for a in 0..AGREEMENTS {
                rates[at * AGREEMENTS + a] = self.rate(length, at, a);
            }
            chance *= self.rate(length, at, UNKNOWN);
            chances.push(chance);
        }
        (rates, chances)
    }
    /// One round's outcome: each verified chain token's agreement and how
    /// many of them the target accepted.
    pub fn observe(&mut self, length: usize, verified: &[usize], accepted: usize) {
        let l = length_bucket(length);
        let labels = (accepted + 1).min(verified.len());
        let decay = (-(labels as f64) / HALF_LIFE).exp2();
        for depths in &mut self.counts {
            for agreements in depths {
                for c in agreements {
                    c.0 *= decay;
                    c.1 *= decay;
                }
            }
        }
        for (depth, &agreement) in verified.iter().enumerate().take(labels) {
            let label = if depth < accepted { 1.0 } else { 0.0 };
            let cells = &mut self.counts[l][depth_bucket(depth)];
            cells[agreement].0 += label;
            cells[agreement].1 += 1.0;
            if agreement != UNKNOWN {
                cells[UNKNOWN].0 += label;
                cells[UNKNOWN].1 += 1.0;
            }
        }
    }
}

/// A chain token's agreement with the draft's sorted candidates at its depth.
pub fn agreement(token: u32, depth: usize, candidates: &[u32]) -> usize {
    if (depth + 1) * CANDIDATES > candidates.len() {
        return UNKNOWN;
    }
    let at = &candidates[depth * CANDIDATES..][..CANDIDATES];
    if at[0] == token {
        0
    } else if at.contains(&token) {
        1
    } else {
        2
    }
}

/// The verified chain tokens' agreements and how many the target accepted:
/// walk the chain through the first `verify_rows` rows of the grown tree.
pub fn lookup_outcome(
    chain: &[u32],
    tree: &GrownTree,
    verify_rows: usize,
    retained: &[u32],
) -> (Vec<usize>, usize) {
    let (mut verified, mut accepted, mut row) = (vec![], 0, 0usize);
    for (depth, &token) in chain.iter().enumerate().take(retained.len()) {
        let Some(child) = (row + 1..verify_rows.min(tree.tokens.len()))
            .find(|&at| tree.parents[at] as usize == row && tree.tokens[at] == token)
        else {
            break;
        };
        verified.push(agreement(token, depth, &tree.candidates));
        if retained[depth] != token {
            break;
        }
        accepted += 1;
        row = child;
    }
    (verified, accepted)
}

// ---------------------------------------------------------------------------
// Draft acceptance model

/// A draft step's chance `sigmoid(slope * logit(q) + intercepts[position])`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Acceptance {
    pub slope: f32,
    pub intercepts: [f32; POSITIONS],
}

/// The pinned reference's prior.
pub const ACCEPTANCE_PRIOR: Acceptance = Acceptance {
    slope: 0.7,
    intercepts: [0.0, -0.35, -0.8, -0.75, -0.75, -0.85, -0.45],
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Label {
    pub position: usize,
    pub feature: f32,
    pub accepted: bool,
    pub cap: f32,
}

const PARAMETERS: usize = 1 + POSITIONS;
const LABELS: usize = 4096;
const REFIT_ROUNDS: u64 = 64;
const SLOPE_PRECISION: f64 = 50.0;
const INTERCEPT_PRECISION: f64 = 4.0;

fn logit(p: f64) -> f64 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}
fn precision(i: usize) -> f64 {
    if i == 0 {
        SLOPE_PRECISION
    } else {
        INTERCEPT_PRECISION
    }
}
fn parameters(a: &Acceptance) -> [f64; PARAMETERS] {
    let mut p = [0.0; PARAMETERS];
    p[0] = a.slope as f64;
    for (i, &b) in a.intercepts.iter().enumerate() {
        p[1 + i] = b as f64;
    }
    p
}
fn capped(label: &Label, z: f64) -> bool {
    label.cap < 1.0 && z > logit(label.cap as f64)
}
fn objective(labels: &[Label], theta: &[f64; PARAMETERS], prior: &[f64; PARAMETERS]) -> f64 {
    let mut value = 0.0;
    for label in labels {
        let z = theta[0] * label.feature as f64 + theta[1 + label.position];
        if capped(label, z) {
            let cap = (label.cap as f64).clamp(1e-6, 1.0 - 1e-6);
            value -= if label.accepted {
                cap.ln()
            } else {
                (-cap).ln_1p()
            };
            continue;
        }
        let margin = if label.accepted { z } else { -z };
        value += if margin > 0.0 {
            (-margin).exp().ln_1p()
        } else {
            -margin + margin.exp().ln_1p()
        };
    }
    for i in 0..PARAMETERS {
        value += 0.5 * precision(i) * (theta[i] - prior[i]).powi(2);
    }
    value
}
fn solve(mut h: [[f64; PARAMETERS]; PARAMETERS], mut g: [f64; PARAMETERS]) -> [f64; PARAMETERS] {
    for column in 0..PARAMETERS {
        let mut pivot = column;
        for row in column + 1..PARAMETERS {
            if h[row][column].abs() > h[pivot][column].abs() {
                pivot = row;
            }
        }
        h.swap(column, pivot);
        g.swap(column, pivot);
        for row in column + 1..PARAMETERS {
            let factor = h[row][column] / h[column][column];
            for k in column..PARAMETERS {
                h[row][k] -= factor * h[column][k];
            }
            g[row] -= factor * g[column];
        }
    }
    let mut x = [0.0; PARAMETERS];
    for row in (0..PARAMETERS).rev() {
        let mut sum = g[row];
        for k in row + 1..PARAMETERS {
            sum -= h[row][k] * x[k];
        }
        x[row] = sum / h[row][row];
    }
    x
}

/// The maximum a posteriori acceptance model of `labels` under the prior,
/// by damped Newton steps from `start`.
pub fn fit(labels: &[Label], start: &Acceptance) -> Acceptance {
    let prior = parameters(&ACCEPTANCE_PRIOR);
    let mut theta = parameters(start);
    let mut current = objective(labels, &theta, &prior);
    for _ in 0..16 {
        let mut gradient = [0.0; PARAMETERS];
        let mut hessian = [[0.0; PARAMETERS]; PARAMETERS];
        for label in labels {
            let b = 1 + label.position;
            let f = label.feature as f64;
            let z = theta[0] * f + theta[b];
            if capped(label, z) {
                continue;
            }
            let p = 1.0 / (1.0 + (-z).exp());
            let residual = p - if label.accepted { 1.0 } else { 0.0 };
            let weight = p * (1.0 - p);
            gradient[0] += residual * f;
            gradient[b] += residual;
            hessian[0][0] += weight * f * f;
            hessian[0][b] += weight * f;
            hessian[b][0] += weight * f;
            hessian[b][b] += weight;
        }
        for i in 0..PARAMETERS {
            gradient[i] += precision(i) * (theta[i] - prior[i]);
            hessian[i][i] += precision(i);
        }
        let step = solve(hessian, gradient);
        let (mut next, mut value, mut scale, mut moved) = ([0.0; PARAMETERS], current, 1.0, false);
        for _ in 0..30 {
            for i in 0..PARAMETERS {
                next[i] = theta[i] - scale * step[i];
            }
            value = objective(labels, &next, &prior);
            if value <= current {
                moved = true;
                break;
            }
            scale *= 0.5;
        }
        if !moved {
            break;
        }
        let largest = (0..PARAMETERS)
            .map(|i| (next[i] - theta[i]).abs())
            .fold(0.0, f64::max);
        theta = next;
        current = value;
        if largest < 1e-9 {
            break;
        }
    }
    let mut result = Acceptance {
        slope: theta[0] as f32,
        intercepts: [0.0; POSITIONS],
    };
    for (i, b) in result.intercepts.iter_mut().enumerate() {
        *b = theta[1 + i] as f32;
    }
    result
}

/// The labels of a grown tree after a walk whose target selections along
/// its path are `selections`: every row under a path row, whether the
/// target selected its token.
pub fn tree_labels(tree: &GrownTree, selections: &[u32]) -> Vec<Label> {
    let mut place = vec![usize::MAX; tree.tokens.len()];
    let mut labels = vec![];
    if place.is_empty() {
        return labels;
    }
    place[0] = 0;
    for row in 1..tree.tokens.len() {
        let parent = tree.parents[row] as usize;
        if parent >= row {
            continue;
        }
        let at = place[parent];
        if at == usize::MAX || at >= selections.len() {
            continue;
        }
        let accepted = tree.tokens[row] == selections[at];
        if accepted {
            place[row] = at + 1;
        }
        let (q, cap) = (tree.q[row], tree.caps[row]);
        if (0.0..=1.0).contains(&q) && (0.0..=1.0).contains(&cap) && at < POSITIONS {
            labels.push(Label {
                position: at,
                feature: logit(q as f64) as f32,
                accepted,
                cap,
            });
        }
    }
    labels
}

/// The learned greedy acceptance model.
#[derive(Clone, Debug)]
pub struct VerifyBudget {
    pub acceptance: Acceptance,
    labels: Vec<Label>,
    next: usize,
    rounds: u64,
}

impl Default for VerifyBudget {
    fn default() -> Self {
        Self {
            acceptance: ACCEPTANCE_PRIOR,
            labels: vec![],
            next: 0,
            rounds: 0,
        }
    }
}

impl VerifyBudget {
    pub fn observe(&mut self, labels: &[Label]) {
        for &label in labels {
            if self.labels.len() < LABELS {
                self.labels.push(label);
            } else {
                self.labels[self.next] = label;
                self.next = (self.next + 1) % LABELS;
            }
        }
        self.rounds += 1;
        if self.rounds.is_multiple_of(REFIT_ROUNDS) && !self.labels.is_empty() {
            let mut ordered = self.labels[self.next..].to_vec();
            ordered.extend_from_slice(&self.labels[..self.next]);
            self.acceptance = fit(&ordered, &self.acceptance);
        }
    }
}

// ---------------------------------------------------------------------------
// Chances and step costs

/// The estimates of the last tree whose draft made a child of its anchor:
/// its draft rows and the rows on its lookup chain's path, best first.
#[derive(Clone, Debug, Default)]
pub struct Chances {
    pub draft: Vec<f32>,
    pub joined: Vec<f32>,
}

impl Chances {
    /// The chances of a grown tree, or none when its draft made no child of
    /// its anchor.
    pub fn of(tree: &GrownTree, chain: &[u32]) -> Option<Self> {
        let n = tree.tokens.len();
        let drafted = |r: usize| (0.0..=1.0).contains(&tree.q[r]);
        if !(1..n).any(|r| tree.parents[r] == 0 && drafted(r)) {
            return None;
        }
        let mut chained = vec![false; n];
        let mut row = 0usize;
        for &token in chain {
            match (row + 1..n).find(|&r| tree.parents[r] as usize == row && tree.tokens[r] == token)
            {
                Some(r) => {
                    chained[r] = true;
                    row = r;
                }
                None => break,
            }
        }
        let mut chances = Self::default();
        for r in 1..n {
            if drafted(r) {
                if chained[r] {
                    chances.joined.push(tree.estimate[r]);
                } else {
                    chances.draft.push(tree.estimate[r]);
                }
            }
        }
        let descending = |v: &mut Vec<f32>| v.sort_by(|a, b| b.total_cmp(a));
        descending(&mut chances.draft);
        descending(&mut chances.joined);
        Some(chances)
    }
}

/// A round's chances: the last tree's draft chances (a first round's seven
/// ones) merged with this round's chain chances (or the last chain's joined
/// rows without a chain), best first, at most `depth` and 15.
pub fn lane_chances(last: Option<&Chances>, chain: &[f32], depth: usize) -> Vec<f32> {
    let draft = last.map_or_else(|| vec![1.0; POSITIONS], |c| c.draft.clone());
    let chain = if chain.is_empty() {
        last.map_or_else(Vec::new, |c| c.joined.clone())
    } else {
        chain.to_vec()
    };
    // Stable merge, the draft's first among equal ones.
    let (mut i, mut j, mut out) = (0, 0, vec![]);
    while i < draft.len() || j < chain.len() {
        if j >= chain.len() || (i < draft.len() && draft[i] >= chain[j]) {
            out.push(draft[i]);
            i += 1;
        } else {
            out.push(chain[j]);
            j += 1;
        }
    }
    out.truncate(depth.min(MOST_ROWS - 1));
    out
}

fn passes(rows: usize) -> f64 {
    rows.div_ceil(NARROW_CAP) as f64
}

const CONTEXT_DECAY: f64 = 1.0 - 1.0 / 64.0;

/// Measured round costs per verified row count and a learned cost per pass
/// over each thousand tokens of history.
#[derive(Clone, Debug, Default)]
pub struct StepCosts {
    /// Milliseconds of a round at 1..=16 rows (index rows - 1); empty until
    /// measured.
    pub table: Vec<f64>,
    per_pass: f64,
    passes_times_excess: f64,
    passes_squared: f64,
}

/// A round's choice: verified rows, the anchor's included, and lane cap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Step {
    pub rows: usize,
    pub lane_cap: usize,
}

impl StepCosts {
    pub fn measured(&self) -> bool {
        self.table.len() == MOST_ROWS && self.table.iter().all(|&ms| ms > 0.0)
    }
    fn price(&self, rows: usize) -> f64 {
        self.table[rows.clamp(1, MOST_ROWS) - 1]
    }
    fn cap(rows: usize) -> usize {
        if rows > NARROW_CAP {
            MOST_ROWS
        } else {
            NARROW_CAP
        }
    }
    /// The rows of the best expected tokens per millisecond, ties to fewer
    /// rows; before the table is measured, a draft's block as far as the
    /// chances fill it.
    pub fn choose(&self, chances: &[f32], context: usize) -> Step {
        if !self.measured() {
            let rows = (1 + chances.len().min(NARROW_CAP - 1)).min(MOST_ROWS);
            return Step {
                rows,
                lane_cap: NARROW_CAP,
            };
        }
        let history = context as f64 / 1000.0;
        let mut expected = vec![1.0f64];
        for &c in chances.iter().take(MOST_ROWS - 1) {
            expected.push(expected.last().unwrap() + c.clamp(0.0, 1.0) as f64);
        }
        let ms = |rows: usize| self.price(rows) + self.per_pass * passes(rows) * history;
        let (mut best, mut best_expected, mut best_ms) = (1, 1.0, ms(1));
        for rows in 2..=expected.len().min(MOST_ROWS) {
            let m = ms(rows);
            if expected[rows - 1] / m > best_expected / best_ms {
                (best, best_expected, best_ms) = (rows, expected[rows - 1], m);
            }
        }
        Step {
            rows: best,
            lane_cap: Self::cap(best),
        }
    }
    /// A measured round of `rows` rows at `context` tokens of history.
    pub fn observe(&mut self, rows: usize, context: usize, milliseconds: f64) {
        if !self.measured() || !milliseconds.is_finite() || milliseconds <= 0.0 {
            return;
        }
        let passes = passes(rows.max(1)) * context as f64 / 1000.0;
        let table = self.price(rows);
        let wall = milliseconds.min(2.0 * (table + self.per_pass * passes));
        self.passes_times_excess =
            CONTEXT_DECAY * self.passes_times_excess + passes * (wall - table);
        self.passes_squared = CONTEXT_DECAY * self.passes_squared + passes * passes;
        self.per_pass = if self.passes_squared > 0.0 {
            (self.passes_times_excess / self.passes_squared).max(0.0)
        } else {
            0.0
        };
    }
}

/// All learned state of greedy tree rounds for one model, and the current
/// request's chances.
#[derive(Clone, Debug, Default)]
pub struct TreeControl {
    pub lookup: LookupAcceptance,
    pub budget: VerifyBudget,
    pub costs: StepCosts,
    /// The current request's last tree's chances; none before its first
    /// round.
    pub chances: Option<Chances>,
}

#[cfg(test)]
#[path = "tree_tests.rs"]
mod tests;
