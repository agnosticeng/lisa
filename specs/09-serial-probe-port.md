# Spec 09 — The adaptive depth controller: serial prior, probe, bootstrap, trusted gate

## Verdict

**Designed, landed, certified.** The controller's failure mode at 2–4k KV
("EV controller picks serial-only … every depth loses EV → `picks []`") was
misdiagnosed once: the picks are not lost to the EV math — they are never
computed. Three anomalies were one chain: **empty own-bucket cell →
wrong-bucket prior read → cold, EOS-truncated probe median auto-trusted → the
probe consumes the request to EOS → the controller never runs.** This spec
states the design that closes the chain; its validation section holds the
measured before/after.

## 1. The failure chain it closes (lisa's own code, before this design)

kv 2343, prompt `bench/data/router_prose.txt`, EOS active (arm `b.auto.eos`,
`bench/data/repro24k/`):

1. `~/.lisa/round_cost.json` bucket `2-4k` was **all-zero** (`serial_ms[1] =
 (0.0, 0)`, every chain/verify cell `n = 0`).
2. `session.rs:508-513` read the serial prior from **the request's own bucket
 only** — `bucket_for(self.fed.len)` then `t.cell(Serial, 0, bucket)` —
 an empty own cell silently discarded a trusted neighbour (`<2k` held
 `(38.89, n = 36)`).
3. The probe (`session.rs:518-541`) took a plain median of up to 6 serial
 steps — no warm-drop, no minimum sample count — and broke on EOS
 (`session.rs:535`). The prose stream emits EOS at token 3, so it collected
 **2 samples, both cold** → median over 2 = the larger cold one →
 `serial step 86.31 ms` (`b.auto.eos.err:81`).
4. Trust was auto-granted: `DepthController::new` set
 `serial_samples = MIN_SAMPLES` for any `serial_ms > 0`
 (`round_cost.rs:462`) — the gate armed on a 2-cold-sample seed.
5. The probe's last tick **was the EOS token**, so the round-loop entry
 condition (`session.rs:555-556`) was false: no round ever ran, `pick`
 was never called, `observe_round` never fired, the acceptance EMA stayed
 at its seeded `[0.750 ×6]`. Zero rounds, zero picks, zero EV. Controls:
 the same kv with `--ignore-eos` ran rounds fine (`c.auto.base.err:81-83`,
 `a.auto.oracle.err:81-83`) — a full 6-sample probe reads warm ~40-41 ms
 and the stream survives it.

## 2. The design — six decisions (M1-M6)

**M1 — the serial prior reads the nearest trusted bucket, never the own
bucket blindly.** `ModelCost::serial_prior_ms(kv_len)` resolves the read
bucket as: the request's own bucket if it is the active layout, else the
nearest active bucket with the lower side preferred (`round_cost.rs:472-481`
walk logic, ported as `bucket_to_read`); a cell is priced only at
`n >= TABLE_MIN_SAMPLES` (`round_cost.rs:398-400`) — seeds (`n = 1..2`) are
never data. The adaptive switch resolves `bucket_to_read`, falling back to
the layout bucket, and never the free grid. Lower-preferred: cost grows with
KV, so under-billing beats over-billing.

**M2 — the serial probe is an 8-tick warm-dropped running mean with count
trust.** One probe = `MTP_ADAPTIVE_PROBE_TOKENS = 8` serial ticks, of which `PROBE_WARM = 2` are
discarded as transitions (drop at `round_cost.rs:303-306`). The value is the
**running mean of the warm folds**, and trust is a **count**: a cell reads
only at `n >= MIN_SAMPLES = 3` (`round_cost.rs:377-381, 398-400`); mature
cells drop `> 3×` self-spikes (`round_cost.rs:354-357`). Folds follow the
cell rule: running mean until `MIN_SAMPLES = 3`, EMA `BETA = 0.10` after
(`round_cost.rs:359-375`). No cold pair can ever arm the gate.

**M3 — the controller bootstraps instead of deciding on untrusted prices.**
First `MTP_EV_WARMUP_ROUNDS = 10` rounds run the default depth with no
EV/serial comparison; the mtp↔serial vote
stays `.undecided` while any of its three prices is missing, and
`.undecided` **keeps the arm**. Below `MTP_ADAPTIVE_MIN_KV = 32768` the serial
switch does not exist at all. Serial enters the width argmax only where a
serial cell was actually measured (`round_cost.rs:675-686`) — serial is never trialled on a
guessed price. On our side `DepthController::pick` prices the EV argmax over
depths 2..cap by table → in-run → analytic shape, defaulting to depth 2 when
nothing prices (the argmax starts at depth 1, so an unmeasured depth can
never win on a guessed price). An unknown serial
price can no longer park — and can no longer kill — a request.

**M4 — the round-vs-serial gate is trusted on both sides, from the persisted
table.** The table's `round_beats_serial` reads the **persisted** table's own cells
on both sides (`width` row or `serial` row), each `n >= MIN_SAMPLES`; it
returns null until both are trusted (`round_cost.rs:383-391`). The old gate
required **in-run** counts — precisely unavailable in the empty-picks state
it existed to rescue.

**M5 — the serial cell is watered mid-request, and the folds persist.** The
probe is armed when the vote lacks the serial price ("teach the bucket a
serial token, once"), only when solo/idle/useful
and the cell is untrusted, at most `MAX_SERIAL_PROBES = 3` per bucket per
process (`round_cost.rs:88-89`). Each tick folds
into the **persisted** table, so
an empty cell heals across requests. (Our serving table is written only
by `lisa round-cost` — the instrument-of-truth rule — so the in-process fold
stays in-run; the empty cell is healed by the round-cost fill, §3.)

**M6 — the grid and walk are unchanged.** The six-bucket grid for `qwen3_5`
(`round_cost.rs:55-57`), `MIN_WIDTHS = 1`, `MIN_SAMPLES = 3`,
lower-preferred walk (`round_cost.rs:77-95, 463-481`) were already faithful
as `active` / `bucket_to_read` (`round_cost.rs:128-187`); only M1's call
site bypassed them.

## 3. What landed

1. **M1** — `ModelCost::serial_prior_ms(kv_len)`: serial prior from
 `bucket_to_read` + a trusted cell; `cell` prices only
 `n >= TABLE_MIN_SAMPLES`, seeds excluded (`round_cost.rs:398-400`).
2. **M2** — 8-tick probe (steps stay in-stream, committed greedy tokens),
 first 2 discarded, value = mean of the warm folds, trust = a count of
 warm folds, trusted only at `>= 3`. `Cell::observe` folds running-mean →
 EMA exactly per `round_cost.rs:370`.
3. **M3** — bootstrap in `DepthController::pick`: 10 warmup rounds at the
 default depth; untrusted serial is not a candidate.
4. **M4** — `gate_kept_depth(kv_len, remaining)` falls back to table pricing
 when the in-run side is under `MIN_SAMPLES`: ms = verify+chain cells at
 `bucket_to_read`, tok = `min(E[tok] from the acceptance EMA, remaining)`,
 serial = the trusted prior. Tokens from the acceptance EMA with table
 pricing time only is the table's `.accept` policy.
5. **Table fill** — `lisa-bench round-cost --kv
 1024,1536,2048,2343,3072,4096,5120,6144 --docs`: three 2-4k points
 (2048/2343/3072) are the minimum for `TABLE_MIN_SAMPLES = 3` (two points
 would land a seed, which rule 1 then refuses); 5120/6144 lift the 4-8k
 cell from its `n = 1` seed to trusted.

## 4. Deviations (and why, by code)

- **Probe placement.** Arming the probe mid-request, after its price window
 fills, would work only for a planner that never needs the serial price
 below 32k. Our EV bar IS
 a serial comparison from round 0 (`round_cost.rs:643`), so the probe stays
 before the loop. That is harmless here: warm-drop + count-trust mean a
 truncated probe yields `serial_samples < 3` → serial stays untrusted →
 M3 keeps rounds running instead of dying on the number. (An interrupted
 probe cannot be re-armed later in our process: the CLI process is one
 request.)
- **No chip-row fitted prior.** A hand-fitted floor-unit fallback for the
 no-active-bucket case was considered. Our analytic
 prior is proportional to `serial_ms` (`round_cost.rs:556`), so with an
 unknown serial it cannot price absolutely; M3's default-depth arm covers
 exactly that case. A per-chip fitted row table is out of scope.
- **No cross-request persistence of in-run serial folds.** Our table is
 written only by `lisa round-cost`, so a serving run's probe folds stay
 in-run; the empty cell is healed by the round-cost fill above, not by
 serving traffic.
- **No tok column in our table.** Our cells store verify+chain cost cells
 (specs/02 schema), so M4's table-side tok comes from the acceptance EMA —
 same code path as our existing EV math.
- **Warmup depth is fixed at 2.** Our controller has a single default
 (`session.rs:413`), so warmup = 2; there is no incumbent depth to nudge.

## 5. Validation (measured, this box,

- **Repro BEFORE** (fresh pre-fix binary, `bench/data/repro24k.before/
 b.auto.eos.FRESH.*`): `serial step 99.79 ms` (86.31 at HEAD) from a
 2-tick cold probe, `picks []`, `EV tok/s []`, `acc EMA [0.750 ×6]`,
 3 tokens, **zero rounds** — the run dies inside the probe on the probe's
 own forked EOS.
- **Repro AFTER** (`bench/data/repro24k/`, final tree, all three arms
 identical — greedy deterministic): `serial step 35.12 ms (table prior,
 bucket 2-4k)`, `picks [0x1 2x22 3x22 4x3]`, `[mtp.round] … rounds 47
 accepted 79`, EV populated; the arm now emits all 128 tokens (the
 fork-only EOS is gone with the feed-carry fix).
- **Table**: 2-4k serial n=3 (35.12 ms), chain d2-d6 n≥3, verify S3-S7 n≥3;
 <2k n=38; 4-8k n≥3 — 1k-8k trusted end to end, `docs/round-cost/
 qwen3_5-64L-5120h.json` byte-identical to `~/.lisa/round_cost.json`.
- **In-run folds under debug** (temporary instrumentation, removed):
 `round_ms` settles at d2 53 / d3 57 / d4 65 ms — table-like from the
 first folded round — and serial stays pinned to the 35.12 prior; a
 post-round serial tick measured ~139 ms and is dropped as a transition.
- **e2e `--depth auto --ignore-eos`, kv 2343, 128 tok**: avant 20.6 (b arm
 1.1 — dead) → après 20.9-24.3 over five runs (a 20.7-21.4, b 21.8-22.6,
 c 20.9-24.3; prefill-chunk jitter of ±0.4 s swings the metric ±1.5 tok/s).
 Decode-phase only (total − prefill − head-prime): avant 35.4 → après
 ~41-45 tok/s; serial control decode 25.4 tok/s.
- **Goldens ×2**: `TOTAL: 310/310 (100.0%) OK` twice. Workspace lib tests
 66 pass / 11 ignored (5 new controller tests for M1-M4). `tools/check.sh`
 green (build + tests + smoke).

## 6. Design alternatives rejected here (do not reopen)

- **EV auto-K** (an auto-tuned K controller) — falsified; our
 EV argmax over measured table prices already caps the depth where the cost
 model says it pays.
- Depth-pick features already covered by the EV: a `w_max` bound / difficulty
 stop (the EV + `settled_worse` already halts), top-2 margin as a pick
 feature (per-depth acceptance EMAs already carry it, specs/03), halting by
 chain convergence (dominated by the EV), short/long EMA fusion (per-depth
 EMAs must never be merged — the shared-EMA d2 freeze cost ~40 % headroom,
 specs/03), width bounded by first-draft entropy (width is already priced;
 an uncalibrated signal), strict break-even EV → serial (**already in
 place** — `round_beats_serial`, M4).
