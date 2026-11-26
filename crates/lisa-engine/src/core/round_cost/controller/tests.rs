use super::*;

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
    assert_eq!(
        c.pick(1000, usize::MAX),
        2,
        "gate outranks the 3% EV margin"
    );
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
    assert!(
        own[2] > 0.9,
        "d3's index-2 EMA tracks its own rounds: {}",
        own[2]
    );
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
        if c.rounds.is_multiple_of(6) && c.rounds > 42 {
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
    assert!(
        (past - 144.0).abs() < 1e-4,
        "slope capped by the prior: {past}"
    );
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
    assert!(
        (d4 - 120.0).abs() < 1e-4,
        "chain 20 + verify exact 100: {d4}"
    );
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
    assert!(
        (c.tok - (3.5 + (5.0 - 3.5) / 3.0)).abs() < 1e-4,
        "mean at n=2"
    );
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
    assert!(
        !m.observe_round_cell(4, 0, f64::NAN, 3.0),
        "non-finite rejected"
    );
    assert!(
        !m.observe_round_cell(4, 0, 0.0, 3.0),
        "non-positive rejected"
    );
    assert!(m.observe_round_cell(4, 0, 150.0, 3.0));
    assert!(m.observe_round_cell(4, 0, 160.0, 3.5));
    assert!(m.observe_round_cell(4, 0, 155.0, 3.2));
    let c = &m.round[&4][0];
    assert_eq!(c.n, 3);
    assert!(c.trusted_tok().is_some());
    // A different machine's row restores STALE: the first live sample
    // re-folds at weight 0.5 and keeps the trust count.
    let mut m2 = ModelCost {
        machine: "rc1-deadbeef".into(),
        stale: true,
        ..ModelCost::default()
    };
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
    assert!(
        c.ev_acc[0] > EV_PRIOR && c.ev_acc[1] > EV_PRIOR,
        "accepted folds up"
    );
    assert!(c.ev_acc[2] < EV_PRIOR, "the rejecting index folds down");
    assert!(
        (c.ev_acc[3] - EV_PRIOR).abs() < 1e-12,
        "unreached index untouched"
    );
    assert!((c.ev_acc[5] - EV_PRIOR).abs() < 1e-12);
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

/// the serial prior walks to the nearest TRUSTED bucket
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
    m.serial_ms[0] = Cell {
        ms: 38.5,
        tok: 0.0,
        n: 36,
    };
    assert_eq!(m.serial_prior_ms(2343), Some(38.5));
    // Own bucket trusted: it reads itself.
    m.serial_ms[1] = Cell {
        ms: 41.0,
        tok: 0.0,
        n: 3,
    };
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
    assert!(
        (c.ms - 53.3333).abs() < 1e-3,
        "still a mean at n=2: {}",
        c.ms
    );
    c.observe(60.0); // n = 3 before this call: EMA (BETA 0.10) takes over
    assert_eq!(c.n, 4);
    assert!((c.ms - 54.0).abs() < 1e-3, "EMA after trust: {}", c.ms);
}

/// the probe discards PROBE_WARM cold ticks and TRUST IS A
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
    assert!(
        (val - expect).abs() < 1e-9,
        "value = mean of warm folds: {val}"
    );
    let c = DepthController::new_with_serial(None, val, warm, 6);
    assert!(c.serial_trusted(), "warm folds >= MIN_SAMPLES");
}

/// warmup runs the default depth, and an UNMEASURED serial
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
    assert_eq!(c.picks.get(&2), Some(&{ EV_WARMUP_ROUNDS }));
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
    serial[0] = Cell {
        ms: 40.0,
        tok: 0.0,
        n: 3,
    };
    m.serial_ms = serial;
    let mut v3 = [Cell::default(); 6];
    v3[0] = Cell {
        ms: 100.0,
        tok: 0.0,
        n: 3,
    };
    m.verify_ms.insert(3, v3);
    let mut v5 = [Cell::default(); 6];
    v5[0] = Cell {
        ms: 60.0,
        tok: 0.0,
        n: 3,
    };
    m.verify_ms.insert(5, v5);
    let mut ch2 = [Cell::default(); 6];
    ch2[0] = Cell {
        ms: 40.0,
        tok: 0.0,
        n: 3,
    };
    m.chain_ms.insert(2, ch2);
    let mut ch4 = [Cell::default(); 6];
    ch4[0] = Cell {
        ms: 10.0,
        tok: 0.0,
        n: 3,
    };
    m.chain_ms.insert(4, ch4);
    let mut c2 = DepthController::new_with_serial(Some(m), 86.31, 0, 6);
    c2.rounds = EV_WARMUP_ROUNDS;
    assert_eq!(
        c2.pick(1000, 128),
        4,
        "table-priced argmax decides while serial is unmeasured"
    );
}

/// the gate falls back to the MEASURED table when no in-run
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
    serial[0] = Cell {
        ms: 40.0,
        tok: 0.0,
        n: 3,
    };
    m.serial_ms = serial;
    let mut v3 = [Cell::default(); 6];
    v3[0] = Cell {
        ms: 87.6,
        tok: 0.0,
        n: 3,
    };
    m.verify_ms.insert(3, v3);
    let mut v4 = [Cell::default(); 6];
    v4[0] = Cell {
        ms: 500.0,
        tok: 0.0,
        n: 3,
    };
    m.verify_ms.insert(4, v4);
    let mut ch2 = [Cell::default(); 6];
    ch2[0] = Cell {
        ms: 3.53,
        tok: 0.0,
        n: 3,
    };
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
        rate =
            DepthController::floor_disable_observe(&mut drafted, &mut accepted, &mut idx, 1, 8, 1);
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
            DepthController::floor_disable_observe(&mut drafted, &mut accepted, &mut idx, 1, 1, 0,),
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
            DepthController::floor_disable_observe(&mut drafted, &mut accepted, &mut idx, 1, 1, 0,),
            None
        );
    }
    let rate =
        DepthController::floor_disable_observe(&mut drafted, &mut accepted, &mut idx, 1, 1, 0)
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
