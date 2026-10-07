# Spec 08 — Tree drafting: the top-k oracle says the signal is real, the M4 verify row price says it does not pay

## Verdict

**NO-GO — closed.** Two independent measurements, one on each side of the
question:

1. **Signal (oracle): real.** On the rounds the linear chain rejects, the
 target's true next token sits in the draft head's own top-2 **49%** of the
 time (27B: 53% of chain first-failures, Flash-Next: 29%) and in its top-4
 **72%** of the time — far above the 20% "continue" gate on the literal
 reachability measure. Converted to tokens (the only currency that matters)
 the branch is worth a **floor** of +4%…+19% per round depending on k and
 model: the repair itself buys exactly +1 token, never more, and the
 repaired branch's children are unmeasured.
2. **Price (verify width): killing.** A useful tree needs 2–5× the verify
 rows (2× for a two-path beam, 3× unpadded / 5× padded for chain+siblings,
 ~6× for a full top-2 tree of depth 4). Measured on this box: verify is
 **40–65 ms at S≤7, 85 ms at S=8, 117–152 ms at S≥9** — the dominant
 split-K qmm lane covers only `M in 2..=7`, and at S=9 every verify fast arm
 (`S≤8`: fused GDN, `track_qk_norm_rope_rows`, MoE wide, mixer) is left
 behind. 2.6× round cost for ≤+16% tokens projects to **−40…−70% tok/s**.

The brief's cost assumption — "verify S>1 inchangé (même batch)" — is the one
thing this spec falsifies: verify is *not* flat in rows, and the rows are
exactly what a tree adds. Do not build the beam; the A/B outcome is already
bounded by the width curve below, and the gap (2.6× cost vs ≤16% tokens) is
far outside the measurement error.

**Reopen when** one of these is true, re-measure with `lisa run --oracle`
first, cost second:

- the verify forward is flat (or near-flat) across 9–25 rows — i.e. specs/07
 batched decode kernels land, or a tree-mask attention verifies the flattened
 tree (~`2d+1` nodes, not `Σ path lengths`) in ONE sequence; or
- the repair can be bought with ≤1 extra verify row per round at a row price
 under ~0.20 tokens (current sibling row earns 0.10–0.15 and costs ~7 ms,
 against a 0.0282 tok/ms serial opportunity cost → break-even 0.20).

## Context

The DFlash2 prototype (`b92bde4`) verified candidate **branch trees** on
tensor units — hardware this box does not have, so that tree verify is not
portable here. What *is* portable is the generic idea: one
forward at `S>1` that checks several candidate paths in parallel and retains
the sub-path that matches, rejecting the rest. Our verify validates one linear
chain `d2→d6` — one path.

The question this spec answers: **is there enough signal in our draft head's
own ranking for a tree to harvest, and does exploiting it pay on M4?**

Honest restatement of what a tree can buy here (derived from the accept scan,
not assumed):

- Verify inputs are `[carry, d_0 … d_{S-1}]`; row `i` is the target's next
 token after inputs `0..i`, i.e. after `[carry, d_0 … d_{i-1}]` — **the draft
 at index `i` never conditions row `i`**. Acceptance `a` = first index where
 `main[a] != d_a`; the round emits `main[0..=a]` = `a+1` tokens.
- Therefore the emitted token at the failure index is *already* the target's
 correct token (row `a` only ever saw the correct prefix). A sibling branch
 that repairs index `a` is worth exactly **+1 token** — the repaired row's own
 `row a+1` output, which the chain can only produce in the *next* round — and
 nothing for the repair alone. Children of the repair extend it further and
 are **unmeasured** (the chain never drafts from the corrected token).
- So the tree's worth = `P(first failure repairable) × (1 + children)`.

## Constraints (user directive)

- HARD CAP 120 GB RAM (detected 128 − 8); sequential runs, one model at a
 time, `memory_pressure` before every run (91–94% free throughout).
- max-tokens ≤ 128; flaky tests → isolated; English in `specs/`.
- Zero invention: every number below is a tool output in
 `bench/data/tree/`.

## Phase 1 — The oracle (instrumentation, `--oracle`)

`crates/lisa-engine/src/core/oracle.rs` + hooks in both `draft_step`
implementations, enabled by **`lisa run --oracle`** (a CLI flag — the env-flag
set stays closed per AGENTS.md §5).

| model | what is recorded per draft step |
|---|---|
| `qwen3_5` | the coarse top-32 shortlist ids **and** the exact re-scored logits (the 32 rows the argmax already reads) |
| `qwen4` | the shortlist-head readout + its row→id map (98 584 rows) |

The driver correlates record *k* with proposal *k* (record 0 = the restart /
priming tail, record *i+1* = chain iteration *i*), then after verify ranks the
target's true token inside it. Classification per chain index:

- `Chain(rank)` — head's own proposal; rank = `1 + #(score > target)`, with a
 score tie lost to the argmax, so a first failure can never rank 1 (the
 `1:` histogram bucket reads **0 in every run** — instrumentation pin).
- `Chain(None)` — the target's token is not in the head's candidate set at all
 (a coarse-shortlist miss): no top-k of *this* head can carry it.
- `Copy` — the prompt-lookup proposal supplied that index: no distribution
 exists, excluded from the denominator (echo text is copy-dominated: 11 of
 17 first failures).
- `NoRecord` — an instrumentation gap, counted separately so it can never hide
 in the numbers. **0 in every run.**

Nothing recorded feeds a scored path (proposal, verify forward, commit,
rollback all untouched).

**Controls (oracle changes no output):**

- `prose d4` with and without `--oracle`: *identical* — 41 rounds, 86
 accepted, per-index `[0.805 0.634 0.439 0.220]`; the rank histogram is
 byte-identical across the two oracle runs as well (deterministic).
- `prose d6` with and without `--oracle`: identical (5 rounds, 1 accepted) —
 that run is void for a different reason, see *Incidental*.
- Goldens ×2 below.

## Phase 1 measurements — where was the true token?

All runs: 128 tok, `--depth 4` forced, kv 2 343 (`router_prose` / `router_mixed`
/ `router_echo`, `--raw`), 27B-4bit unless noted. Logs: `bench/data/tree/oracle_*.txt`.

| run | rounds | accepted (mean) | first failures → chain-fail / copy | rank ≤2 of chain-fail | ≤4 | ≤8 | floor tok/round |
|---|---|---|---|---|---|---|---|
| prose d4 (×2, identical) | 41 | 86 (2.10) | 32 → 30 / 2 | 20/30 = **66.7%** | 25/30 = 83.3% | 83.3% | +0.488 (**+23.3%** of accepted) |
| mixed d4 | 47 | 83 (1.77) | 41 → 39 / 2 | 17/39 = **43.6%** | 24/39 = 61.5% | 79.5% | +0.362 (+20.5%) |
| echo d4 | 27 | 55 (2.04) | 17 → 6 / 11 | 3/6 = 50% | 4/6 = 66.7% | 66.7% | +0.111 (+5.5%) |
| **27B aggregate (unique texts)** | **115** | **224 (1.95)** | 90 → **75** / 15 | **40/75 = 53%** | 53/75 = 71% | 80% | **+0.348 (+11.8% of emitted)** |
| Flash-Next d4 (prose, qwen4) | 37 | 90 (2.43) | 21 → 17 / 4 | 5/17 = **29.4%** | 13/17 = 76.5% | 94.1% | +0.135 (+3.9% of emitted) |

`chain-fail` = head-ranked + outside-the-head (the denominator every
reachability rate uses: a copy proposal has no distribution, an outside miss
is unreachable for any top-k of that head). `no-record 0` everywhere.

Rank histogram over the 27B chain first-failures (n=75, unique texts):
`1:0 2:40 3-4:13 5-8:7 9-32:6 outside:9`.

Per-index acceptance, prose d4, linear chain → oracle top-2 / top-4:

```
linear [0.805 0.634 0.439 0.220]
top-2 [0.927 0.780 0.561 0.317]
top-4 [0.927 0.805 0.610 0.366]
```

(These are *repair-at-that-index* values: only the first failure is repaired
per round, children not drafted. A full tree would also lift the indices after
the repair — that lift is exactly the unmeasured "children" term.)

Reading: the head's miss is **usually a near-miss**, not a confident wrong
answer — 64% of first failures (27B, duplicated run) are a rank-2 miss. The
`qwen3_5` coarse top-32 is almost never the binding constraint (11/94 outside);
the `qwen4` shortlist head spreads the miss wider (only 29% rank-2, 76% rank-4)
because it ranks a 98k-row set with no coarse truncation.

## Phase 1 — the price side (the measurement the design assumed away)

`lisa-bench round-cost --kv 1024 --widths 3,5,6,7,8,9,10,12,15,20,25 --steps 48`
→ `bench/data/tree/round_cost_tree{,2}.json` (written to the tree dir on
purpose — the tracked `~/.lisa/round_cost.json` EV table is untouched, no
verify kernel changed):

| S | 3 | 5 | 6 | 7 | 8 | 9 | 10 | 12 | 15 | 20 | 25 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| verify ms | 40.5 | 51.1 | 57.8 | 65.4 | **84.7** | **116.9** | 119.0 | 146.6 | 142.1 | 143.4 | 152.0 |

serial 35.5 ms · draft 1.6 ms · chain d2 3.3 / d4 6.2 / d6 9.7 ms.

Two regimes, and the cliff is structural, not noise (reproduced in two
independent sessions):

- **S ≤ 7**: ~6–7 ms/row over a ~35 ms weight-stream floor — the designed
 verify region (split-K qmm `M in 2..=7`; GDN fused arm, rope-rows, MoE wide
 and mixer arms all `S≤8`).
- **S ≥ 9**: +70% step, then a flat 117–152 ms plateau — the stock dispatch,
 off every verify fast path. **This is precisely where a tree lives**: chain +
 one sibling per level is 15 rows at d4 (25 padded as B=5 slots), a full
 top-2 tree is 31 nodes.

Cross-check that a *real* batched tree is not cheaper than this single-sequence
proxy: specs/13's B=4 batched decode costs 57.5 ms/step at M=4 while a
single-sequence S=4 verify measures 44.4 ms — batched is **dearer** at equal
M, so the proxy is optimistic for the tree.

### Break-even arithmetic

Opportunity cost of a GPU millisecond = the serial step: `1/35.5 = 0.0282
tok/ms` (and the bar to *not regress vs current chain* is the chain's own
rate, 3.10 tok / 57.3 ms = 0.0541 tok/ms).

A verify row near S=5 costs ~7 ms ⇒ **a row must earn ≥0.20 tok** (≥0.38 tok
to beat the chain). Measured sibling-row yield: `P(a = i) × reachability`
≈ `0.19 × 0.53` ≈ **0.10–0.15 tok/row** — below both bars, and below the
0.22 tok of the chain's *own* last row (7 ms for 0.22 tok = 0.031 tok/ms:
above serial, below the chain average — which is exactly why the EV
controller already stops at d3–d4).

Cost of every candidate structure vs the d4 chain (measured widths, prose,
27B — tok = accepted + rounds + floor):

| structure | rows | ms/round | tok/round | tok/ms | vs chain |
|---|---|---|---|---|---|
| chain d4 (baseline) | 5 | 57.3 | 3.10 | 0.0541 | — |
| chain d3 | 4 | 49.1 | 2.88 | 0.0586 | best chain |
| chain + siblings (leaf), d4, ragged rows | 15 | 148 | 3.59 | 0.0243 | **−55%** |
| chain + siblings, d4, padded to batch slots | 25 | 158 | 3.59 | 0.0227 | **−58%** |
| d2 + siblings, levels 0–1 | 6 | 61.1 | 2.68 | 0.0439 | −19% |
| d3 + one sibling (index 0) | 6 | 62.5 | 3.01 | 0.0482 | −18% |
| d4 + one sibling (index 0) | 7 | 71.6 | 3.23 | 0.0451 | −17% |

Caveats stated, not hidden: (a) rows are counted ragged/summed, but our engine
has no tree-mask attention and no ragged batch — a real tree pays the padded
figure (path lengths rounded up to the longest sibling) or several forwards;
(b) the `[1,S]` curve above is the measurement; a `[B,S]` tree forward at equal
row count is *dearer*, directionally, from specs/13 (B=4, M=4: 57.5 ms/step vs
44.4 ms for a single-sequence S=4 verify). Every structure loses either way.

The `K=2` beam of the brief is worse than the leaf-sibling tree: two full
chains diverge at whatever index the draft score chose, and a path only helps
if it diverges *exactly* at the failure index (otherwise it is already wrong
where the chain is right) — so it pays 2× rows for
`P(a = divergence) ≈ 0.13`.

## Validation

- Goldens ×2, both models, with the oracle code in the tree:
 `27B 310/310 OK`, `Flash-Next 256/256 OK`.
- `cargo test --workspace --lib --release`: 61 pass, 0 fail (2 new: the rank
 semantics, including the tie-loses-to-argmax pin).
- `no-record 0` and rank-bucket `1: 0` in every run (instrumentation pins).
- Oracle on/off output-identical (controls above).

## Incidental findings (not investigated, recorded so they are not rediscovered)

1. **The EV controller picks serial-only at 2–4k KV.** `~/.lisa/round_cost.json`
 has no serial cell for bucket `2-4k` (all-zero), so the controller probes
 in-run and measured **84.77 ms** on a warm box → every depth loses EV →
 `picks []`, no MTP rounds at all (`bench/data/tree/oracle_prose_auto.txt` is
 therefore empty). Auto runs at ~2.3k are void until the cell exists:
 `lisa-bench round-cost --kv 2343 --docs`.
2. **`--depth 6` on the prose prompt stops after 6 tokens (EOS)**, identically
 with and without `--oracle` — deterministic, so it is the S=7 verify path
 disagreeing with S=5 on something (§9.6 near-tie class), not drift. It
 voids the d6 sample (n=5); d4 is the measured depth.

## Artifacts

- Instrument: `crates/lisa-engine/src/core/oracle.rs`, hooks in
 `models/qwen3_5/tower.rs`, `models/qwen4/tower.rs`, `core/session.rs`,
 flag in `crates/lisa-cli/src/main.rs` (`lisa run --oracle`).
- Logs: `bench/data/tree/oracle_{prose_d4,prose_d4b,prose_d6,prose_auto,mixed_d4,echo_d4,flashnext_prose_d4}.txt`
- Width curve: `bench/data/tree/round_cost_tree.json`, `round_cost_tree2.json`.
## Dispositions recorded herecampaign)

- **Tree verify via a beam over tensor units (the DFlash2 prototype)** —
 FALSIFIED for our hardware: **0.94× vs MTP**; the prototype's premise ran
 on silicon this box does not have (§ Context). Do not reopen.
- **Continue-from-correction — FALSIFIED, already implemented**: the
 correction token *is* emitted (`main_tokens[..=a]`); the trimmed rows were
 computed from the failed draft and are provably invalid (the § Context
 derivation says it verbatim). Do not reopen.
- **Changing the verify's logits (draft↔verify contrast re-scoring,
 orthogonal re-ranking) — set aside**: the verify is lossless; changing its
 logits breaks the contract (00-contracts §4).

## The NAX verify lane — the planned work item for widths S >= 8

Measured verify ladder at kv 1024, pool 512 MiB, current tree (serial-audit /
round-cost, serial 37.64 ms):

  S5 41.78 | S7 59.79 | S8 89.04 | S9 110.52 | S11 141.25 | S13 129.73 ms
  per token: S7 8.54 | S8 11.13 | S9 12.28 | S11 12.84 | S13 9.98 ms/token

Findings, measured:

- The S=8 cliff is real (+29 ms over S7 for one token of width): the split-K
  verify lane covers M 2..=7 and the tiled lane takes over at a per-token
  penalty of 30-50%.
- The WIDE end is not a cliff: **S13 prices 9.98 ms/token — BETTER than our
  current 5.86-tokens-per-64-ms state (10.92 ms/token)**, and beats S9/S11.
  The middle (S8-S11) is where the lane transition bites; the wide end does not.

Work item, two steps:

1. Measure S=13 acceptance in-situ. If per-position acceptance holds at width 13
   the way it holds at width 6, a 13-wide round at 129.7 ms yields ~10-11
   accepted tokens at ~12 ms/token — comparable to today. Interleaved pairs,
   accepted/round recorded, COPY_LEN_MAX raised behind a flag for the test.
2. If acceptance holds: either land a fast NAX m16 verify lane for M in 8..=16
   (the middle S8-S11 is where the transition penalty is), or raise COPY_LEN_MAX
   straight to the measured sweet spot (S=13). Goldens x2 + the footprint ledger
   before any default flip.

The reopen condition from the earlier verdict ("verify flattens over 9-25 rows")
is PARTIALLY met by the S13 point — the ladder is not monotonic, and the wide
end prices at parity with S7.

## Addendum — long-copy frequency measurement (the default-flip's second condition)

The wide lane pays only through the copy source (above: acceptance 1.000 at
indices 6–12, +22…+50 % in-pair), so the default flip from `COPY_LEN_MAX = 6`
is justified iff long copy hits (`copy_len >= 8`) are FREQUENT in serving-like
traffic. Measured here; verdict below.

**Instrument (trace-only, no scored-path change):** the `mtp.copy` span now
carries the round's copy-hit length as its trace detail
(`core/session.rs`, one `span` → `span_detail` line). A sweep
(`bench/data/tree/copyfreq/sweep.sh`) runs 21 sequential requests
(`lisa run --raw --temperature 0 --max-tokens 256 --ignore-eos`, depth auto,
27B-4bit, one model at a time, memory ≥ 92 % free throughout), parses the
`LISA_TRACE_JSON` chrome events per request, and tallies the per-round
`copy_len` distribution.

**Corpus — stated limitation:** the repo holds no novel-prose serving corpus.
What exists (recovered from git history, `8e882ff`, the router corpora) is the
engine's own serving-test text: `router_prose.txt` (entry-structured
workplace prose), `router_mixed.txt`, `router_echo.txt`, and
`long_prompt.txt` (infrastructure/journal entries). All four are real English
text, but they carry deliberate internal repetition (entry structure, echo
regime), so the copy-hit frequency measured here is an UPPER-side bias for
this class of text, not a census of generic novel prose. On a purely novel
corpus the frequency would be lower; nothing in the repo can measure that.

**Design note:** `copy_lookup_guarded` proposes the FULL window — with the cap
raised (`LISA_COPY_LEN_MAX=13`) a hit fires only when a source offers the
entire 13-token continuation. Therefore every measured hit reads exactly 13
and the `>= 8` count is a LOWER bound of long hits (sources offering 8–12
tokens are skipped, not shortened). Both configs were run: the cap-13 pass
(the flipped-default scenario) and the shipped cap-6 pass (21 + 21 requests).

**Measurements** (986 + 1212 rounds; `tally_cap13.json`, `tally_def6.json`):

| config | rounds | copy-hit rounds | hits >= 8 | copy-proposed tok | committed tok | committed/round | median tok/s |
|---|---|---|---|---|---|---|---|
| cap 13 (wide) | 986 | 448 (**45.4 %**) | 448 (45.4 %) | 5824 | 5376 | 5.45 | 43.4 |
| cap 6 (shipped) | 1212 | 695 (57.3 %) | 0 | 4170 | 5376 | 4.44 | 28.2 |

- Every copy hit under the wide cap is a full 13-window; the copy-round
  fraction 45.4 % is a lower bound of `>= 8` frequency (§ design note).
- Copy-sourced tokens: under cap 6 the copy already proposes 77.6 % of what
  gets committed (4170 of 5376) and copy rounds accept to their ceiling
  (measured above: 1.000 at the proposed indices), so committed-copy share is
  far above the 15 % bar under the shipped default too; under cap 13 the
  committed/round rises 4.44 → 5.45 and median decode 28.2 → 43.4 tok/s
  (cross-config medians, ±40 % box drift, NOT interleaved — the in-pair
  numbers above govern the speed claim, this median only shows direction).

**Verdict (threshold stated before the run):** the flip is justified iff
rounds with `copy_len >= 8` are >= ~10 % of rounds AND copy tokens >= ~15 %
of committed tokens. Measured: **45.4 % >= 8** (lower bound) and copy-source
tokens ≈ 60–78 % of committed — **BOTH BARS CLEARED on this corpus.** The
corpus limitation stands: the text is entry-structured with internal
repetition, so the conclusion reads "long copy hits are frequent in the
serving-like traffic lisa can measure", not "in all prose". The flip itself
stays gated on ITEM 2 (a fast verify lane for `M in 8..=16`) — without it the
wider rounds land off every fast arm at long KV.

## Results addendum — S=13 acceptance in-situ (`LISA_COPY_LEN_MAX`)

Knob added: `LISA_COPY_LEN_MAX` overrides the copy cap (`COPY_LEN_MAX`, const
untouched) — 0/absent/parse-error all fall back to 6, so the default path is
byte-identical. It flows through both consumers (the copy-lookup length cap and
the round-width cap, `w = max(depth, copy_len)`) and, when raised, also lifts
the FIXED-depth clamp so `--depth 12` runs chain-only wide rounds. The
controller's own picks stay capped at 6.

Round-cost re-check in the current tree (`lisa-bench round-cost
--kv 1024,16384 --steps 24 --widths 7,13`, `bench/data/tree/round_cost_wide.json`):
kv 1024 S7 48.84 ms / S13 119.08 ms; kv 16384 S7 61.91 / S13 258.54. The wide
end still prices ~2.4× S7 per row-count but ~9 ms/token vs 8.4 — parity, no
cliff. At 16k the wide lane dears badly; wide rounds are a short-KV tool.

Acceptance, measured (27B, `lisa run --oracle`, greedy, `--ignore-eos`,
`LISA_MTP_TWO_CHUNK=0`):

- **Chain at width 13 does NOT hold.** Fixed `--depth 12` (chain-only,
  S = 13): per-index acceptance `[0.706 0.456 0.309 0.235 0.088 0.029 0.015
  0.000 …]` — the geometric decay reaches zero at index ~7. 1.84 tok/round on
  a ~130-150 ms round → **13.7 tok/s vs 25.7-26.5 for the same prompt at
  depth 6**. Wide chain rounds are falsified: the draft head's signal is
  exhausted by index ~5, exactly as the d4 curve extrapolated.
- **Copy at width 13 DOES hold — fully.** Paired runs (same prompt, greedy,
  `--depth 6`, knob OFF vs 13, two interleaved pairs): OFF 66 rounds, 1.91
  tok/round, copy hits capped at 6; ON 45 rounds, **3.27 tok/round**, 8 copy
  hits at mean 13.00 with per-index acceptance **1.000 at indices 6-12**
  (0.875 at 12) — the widened round accepts to its ceiling. In-pair tok/s
  17.9/18.3 (ON) vs 11.9/15.0 (OFF): **+22…+50%**, on prose with ~8 long copy
  hits per 192-token request. Serve arm (temp 1.0, interleaved OFF/ON pairs,
  charge ~2.2k tok): ON copy hits all 13.00-long and accepted; echo-regime
  runs (repetitive prompt, greedy) show 12.00 tok/round at 43.1 tok/s.

Verdict: **the wide lane pays ONLY through the copy source.** The bet "a
13-wide round yields ~10-11 accepted at ~12 ms/token" is confirmed for copy
rounds (13-14 tokens land on a ~130-150 ms round, ~11 ms/token, better than
the current 10.92) and falsified for the chain. `COPY_LEN_MAX` stays 6 (the
const and the controller cap are untouched); the knob is the documented
wide-mode switch for copy-heavy workloads. Raising the DEFAULT is deferred
until long-copy frequency is measured across
real workloads (the m16 lane is measured-out — see its step 3 section) — on general prose the widened hits are what pay, and their
yield depends entirely on how much text repeats.

## The NAX m16 lane — step 2 result: M=8 joined the split-K verify tile

The ladder's middle penalty was attributed before any kernel change: S9-S12
(M 9-12) ride `qmv_wide` in situ (353 dispatches/verify) while the tiled NAX
lane is flat ~220-245 us at every M in 8..=16 — the middle bites on the
dispatch GATE, not on kernel physics. New gpu-slot shapes (`bench_verify_qmm_
m_sweep`, eval+sync per dispatch, warmup excluded) price each M.

Landed, bit-exact (goldens 310/310 x2, Flash-Next 256/256, tail-ULP pin
extended to s=8, lib tests 30/30 + 54/54):

  the split-K verify tile's MROWS range extends 2..=7 -> 2..=8. M=8 prices
  180 us isolated vs qmv_wide's 594 us, and does not spill. In-situ
  interleaved pairs at S=8: verify-audit 72.4 -> 57.5 ms (-20.6%), tok/s
  -8/-23/-24% better per pair. S=7 and serial: unchanged (wash).

Falsified, measured: split-K at MROWS >= 9 (register spill, 1500-4500
us/dispatch); a fused two-tile pass (weight-stream conflict, >= 1550 us);
qmm_nax for the middle (argmax flip at M=11 — contract-illegal). The
reference's MPP m16 tile transcribes and compiles but does not execute under
lisa's `newLibraryWithSource` path (the minimal MPP probe hangs); executing it
requires the additive-binary linkage its MLX fast-kernel path uses. The
opt-in kernel stays at `jit::affine_verify_qmm_nax_m16`, kernel-level only,
unwired — the same shape as the msg-tile precedent.

## The NAX m16 lane — step 3 result: the kernel executes; the lane LOSES

The "probe hangs / needs additive-binary linkage" diagnosis was WRONG. The
`mpp_probe` test module (`jit/mpp_probe.rs`, `#[ignore]`, staged timestamps +
watchdog) reproduced the probe in isolation and the MPP `matmul2d` op compiles
AND executes bit-correct through lisa's ordinary `newLibraryWithSource` path —
no binary archive, no metallib build step. Three concrete defects blocked it,
each fixed:

1. Compile class: the wrapper used `compile_builtin` (Math::SafeNoLang ->
   MSL 3.2). MSL 3.2 cannot compile the MPP+bfloat16 source at all (the
   `bfloat16` name is ambiguous against the stdlib's `metal::bfloat16`, and
   tensors reject the resolved reserved type). Every MPP kernel must ride
   `compile_nax_jit` (Math::Safe -> Metal 4 language) — done.
2. Template type: the specialization instantiates with the bare name
   `bfloat16`; under Metal 4 that is ambiguous. The MLX preambles' typedef
   `bfloat16_t` is the correct spelling (same convention as `qmm_nax`/
   `gather_qmm_rhs_nax`) — done.
3. Dispatch axis: the grid rode Y (`(1, N/32)`, `tg_n = tgp` flat). MPP
   tensor-op kernels SILENTLY SKIP every threadgroup launched with
   grid.y > 0 — no GPU error, no completion failure, just missing output
   (the probe's sentinel check: with grid (1,4) only tg0 wrote 512/512;
   with (4,1) all four wrote). The reference dispatches (256, N/32, 1),
   keeping the varying axis on X. Fixed to `(N/32, 1, 1)`.

With all three, the m16 kernel runs and is numerically correct to bf16 output
quantization (probe: worst rel diff 0.38% = half a bf16 ulp; minimal-probe
512/512 exact).

Measured (`bench_verify_qmm_m_sweep`, eval+sync wall/32 dispatches, warmup
excluded, [*,5120]->17408 gs64 4-bit): the m16 tile is M-independent (it
always streams the fixed [16, K] tile) at 394.6-438.0 us across two runs —
vs the stock path's 282.4-398.3 us at M 9..=12 and qmm_nax's 222.8-241.6 us.
It never wins: worse than stock at every M (incl. M=12, the best case for
it), ~1.7x qmm_nax everywhere. NO-LAND: the lane stays
kernel-level opt-in, unwired; M 9..=12 keep `qmv_wide`. Nothing to re-measure
in situ — a slower per-dispatch kernel cannot win round-cost.

The diagnosis above ("additive-binary linkage") is retired; the m16 follow-up
is closed as measured-out, not blocked.

Footprint: unchanged — t4-draft-head / t5-map-cleared at 15038 MB exactly.

Verdict: the wide lane now pays at S=8 in situ. Widths 9..=16 are closed —
the m16 lane executes (step 3, below) and measured LOSES to stock, so no
fast lane exists for M 9..=12 at these trunk shapes; S=8 alone makes an
8-wide default viable pending the long-copy frequency bar (above) holding on
a novel-prose corpus.

## Re-run — the S=9 decision (2026-10-07, HEAD 55fe498, this box)

The reopen question, re-priced on the current tree: is S=9 within ~15 % of
S=8 per-token? `lisa-bench round-cost --widths 8,9,10,13 --steps 24`, two
passes per kv in reversed width order (order-drift control; in-pair = 8 and
9 measured adjacently within a pass; cooldowns between passes). Table
written to `bench/data/s9/` (scratch, NOT merged into the EV table).

| kv | pass (order) | S8 | S9 | S10 | S13 | S9/S8 | per-row S9 vs S8 |
|----|--------------|----|----|----|-----|-------|------------------|
| 1024 | A (8→13) | 54.84 | 90.29 | 92.44 | 134.17 | 1.65× | +46 % |
| 1024 | B (13→8) | 72.10 | 104.71 | 101.41 | 113.41 | 1.45× | +18 % |
| 16384 | A (8→13) | 84.99 | 233.56 | 232.32 | 257.48 | 2.75× | +145 % |
| 16384 | B (13→8) | 110.99 | 263.88 | 263.06 | 280.68 | 2.38× | +112 % |

S=9 lands off every fast arm in both orders at both KV — 2.4-2.7× S=8 at
16k, 1.5-1.7× at 1k; the marginal row 9 costs more than the whole S=8
verify. **COPY_LEN_MAX stays 7; the depth ladder stays S ≤ 8 on the split-K
lane.** The S=9/S=10 step is nearly free (~1 ms) — the cliff is the S=8→9
boundary (the fast lanes end at 8 rows), consistent with the earlier
S≥9 curve; this does not change the verdict.
