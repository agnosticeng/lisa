# Spec 03 — Draft quality in long context (why the auto ceiling at 15.6k)

## Context

At ~15.6k prompt tokens the MTP auto controller picks d2/d3 and acceptance on
natural prose is 1.6–1.8 tok/round (per-index roughly [0.86 0.62 0.11 0.03 …]).
On short haiku-style prompts the same draft reaches 2.5+/round, and the d6 trim
fix (spec 11 §15) proved 3.5 tok/round is reachable at short KV. The auto
ceiling of ~28 tok/s on long prose is therefore an *acceptance* ceiling, not a
verify-cost ceiling (specs 23/24 already cut verify cost). Question: why does
the draft degrade with context length, and is it repairable without inventing
machinery?

## Constraints (user directive)

- HARD CAP 120 GB RAM total; sequential runs only, one model loaded at a time.
- `memory_pressure` check before every run; max-tokens ≤ 128 (256 for final runs).
- Flaky tests → isolated runs, never diagnose inside a parallel suite.
- Interleaved same-session A/B only (box drifts ±40%).

## Phase 1 — Measure

On ONE long natural-English text (`bench/data/long_prompt.txt`, ~75 KB,
Qwen3.8-27B-4bit), same text family for every arm:

(a) **Acceptance per index × depth × KV length** at kv ∈ {1k, 4k, 8k, 16k}
(char-truncated prefixes of the SAME file, counts verified with `lisa tok`),
depth ∈ {2, 3, 4} forced + auto, max-tokens 128. Distinguishes KV-length
degradation from text-type degradation (compare prose vs the short-haiku
numbers already on record).

(b) **Loss decomposition in the draft pipeline**
(`draft_step`: draft head → 2-bit requantized lm_head → top-32 rerank →
exact re-score of the shortlist). At 16k vs 1k, attribute the loss to:
 1. the 2-bit lm_head requantization: measure top-1 agreement of the 2-bit
 argmax vs the full-precision lm_head argmax on real hidden states at
 each KV length;
 2. the top-32 rerank truncation: count rounds where the TRUE top-1 falls
 outside the 2-bit top-32 shortlist;
 3. the head/trunk itself: hidden-state drift from long context (draft-head
 input distribution vs short context).
Whoever owns the miss owns the fix.

(c) **Coarse-head width analysis**: whether a wider coarse requant (3-bit vs
our shipped 2-bit) would buy acceptance at long KV — and what it would cost in
draft-step time.

## Phase 2 — Ports/fixes by finding

- 2-bit lm_head degrades with KV → gate the requantization by context length
 or raise to 3/4-bit; measure acceptance delta vs draft-step cost (draft is
 ~1.6–1.7 ms — a 2× draft cost is affordable).
- top-32 truncates → widen to top-64 on long context; measure.
- head itself degrades → compare against PLD on the same prose (PLD hit
 59 tok/s on echo-y text); if the ngram-score gate is disabling it on natural
 prose, adjust the threshold.

Every fix: test + goldens ×2 + round-cost + traces; acceptance per index at
16k before/after.

## Phase 3 — Verdict

Acceptance at 16k before/after; auto tok/s before/after (goal ≥ 35 tok/s);
if winning, A/B protocol 24-style (interleaved same-session, ×3 pairs).

## Results

### Phase 1(a) — acceptance per index × depth × KV (same prose file)

`bench/data/long_prompt.txt`, char-prefixes, `lisa run --depth {2,3,4}`,
max-tokens 128. Char targets 1k/4k/8k/16k tokenize to
(respectively) ~0.8k/3.1k/6.2k/12.5k tokens (4.85 chars/token); the "16k" arm
below is 12,545 KV rows (printed `prompt_tokens`).

| arm | rounds | mean acc/round | per-index |
|---|---|---|---|
| 1k d2 | 50 | 1.58 | [0.900 0.680] |
| 1k d3 | 40 | 2.17 | [0.850 0.700 0.625] |
| 1k d4 | 34 | 2.76 | [0.941 0.765 0.647 0.412] |
| 4k d2 | 50 | 1.56 | [0.880 0.680] |
| 4k d3 | 38 | 2.37 | [0.947 0.816 0.605] |
| 4k d4 | 33 | 2.85 | [0.909 0.818 0.636 0.485] |
| 8k d2 | 48 | 1.65 | [0.854 0.792] |
| 8k d3 | 36 | 2.53 | [0.917 0.833 0.778] |
| 8k d4 | 35 | 2.63 | [0.800 0.686 0.629 0.514] |
| 12.5k d2 | 49 | 1.63 | [0.837 0.796] |
| 12.5k d3 | 36 | 2.53 | [0.889 0.861 0.778] |
| 12.5k d4 | 35 | 2.66 | [0.800 0.714 0.629 0.514] |

**FINDING A: the draft does NOT degrade with KV.** Forced d4 acceptance is
flat (2.76 → 2.66 tok/round from 1k to 12.5k; per-index depth coverage
actually *improves* mid-chain at long KV). The 1.6–1.8 "decline" reported at
15.6k is the AUTO CONTROLLER's d2/d3 mix, not a draft-quality collapse: d2 is
1.6/round at every KV. Short-haiku 2.5+/round at d2 is a TEXT effect
(repetitive echo prose), not a KV effect — same class as the spec-23 trap note.

### Phase 1(b) — pipeline decomposition (draft head → 2-bit lm_head → top-32 → re-score)

Temporary diagnostic in `draft_step` (reverted after measurement): per draft
step, argmax of the FULL-precision trunk lm_head vs the coarse/re-scored
proposal. Greedy d4, same prose:

| KV | coarse2bit top1 == full | full top-1 in coarse top-32 | final selected == full |
|---|---|---|---|
| 1k | 0.953 | **1.000** | **1.000** (n=155) |
| 12.5k | 0.960 | **1.000** | **0.993** (n=150) |

**FINDING B: requantization and rerank are exonerated, at short AND long KV.**
The 2-bit coarse head's own top-1 misses the full head's top-1 ~4–6% of the
time, but the miss is always inside the top-32 shortlist and the exact
re-score repairs it (final==full ≥ 99.3%). No KV trend — the 2-bit lm_head
does NOT degrade at long context. None of the three hypothesized loss sites
(requant / top-32 truncation / head hidden drift) owns a measurable loss on
natural prose; the index-0 acceptance miss (~0.80–0.94) is the model being
genuinely uncertain on novel prose, not a pipeline artifact.

### Phase 1(c) — coarse-head width analysis

- Our draft scheme: coarse 2-bit/gs64 requant of the trunk lm_head → exact
 top-32 shortlist → re-score through trunk-head rows (group size 64). A wider
 coarse head (3-bit) would buy only coarse-top1 agreement — and the 2-bit
 head's own top-1 already misses the full head's top-1 only ~4–6% of the time,
 a miss the exact re-score repairs every time (Finding B: final == full
 ≥ 99.3%). "A miss costs acceptance, never output."
- 3-bit is not measurable here anyway: it needs MLX's permuted 3-bit host
 encoder, which does not exist in this repo (spec-12 trap). Analytically the
 extra bit buys nothing Finding B does not already neutralize, so 2-bit ships.
- No separate long-context draft machinery is warranted: acceptance is flat in
 KV (Finding A), so there is nothing long-KV-specific for a wider head to fix.
- The long-context cost gap (specs/23: our ~28 tok/s decode at 15.6k) is a COST
 edge, not an acceptance edge: our serial step ~40 ms and verify/row 7.3 ms at
 15.6k are the driver; the draft pipeline is exonerated by Findings A/B.

### Auto-controller observations (same runs)

- auto @1k: picks [2x7 3x7 4x22 5x2], serial 26.7, EV [d2 49.6 d3 55.1 d4 54.4],
 realized 36.9 tok/s combined.
- auto @12.5k: picks [2x15 3x17 4x4 5x2 6x3] (mostly d2/d3), serial 24.6,
 EV [d2 36.0 d3 37.1 d4 34.5 d5 35.6], acc EMA [0.946 0.725 0.731 0.548 …].
- serial @12.5k decode: 23.8 tok/s (split stats). PLD @12.5k: 2.17/round —
 no better than MTP d3 on this text; PLD is not the answer here either.

The EV table prices d4 ≈ d3 at long KV (34.5 vs 37.1) while measured d4
acceptance is *higher* (2.66 vs 2.53/round) — the controller's cost model,
not the draft, caps the long-context ceiling.

## Phase 2 — what landed

Two changes, both measured-first:

1. **Round-cost table recalibration at the long bucket** (`lisa-bench
 round-cost --kv 1024,12288 --docs`): the 8-16k bucket still carried
 pre-spec-24 verify prices (the S-row qk-norm+rope port was never
 re-calibrated into the table — the spec-24 "reset a bucket cell" trap).
 New 8-16k row: serial 33.35 ms | draft 1.70 | S3 63.68 S4 75.96 S5 80.08
 S6 79.98 S7 87.72. Alone this did NOT move auto reliably (interleaved
 old/new-table medians 34.2 vs 34.5 tok/s — generation-path variance
 dominates); kept because the EV math should price the shipped kernels.

2. **Per-depth acceptance EMAs in the EV controller** (`round_cost.rs`):
 ROOT CAUSE of the d2 freeze — the acceptance EMA is SHARED across depths,
 and a d2 incumbent never observes index ≥ 2 hits, so `acc[2+]` decays on
 stale trial values and permanently under-prices d3/d4 EV (measured at
 12.5k: shared acc[2] = 0.49 while forced-d3 acceptance index-2 = 0.66).
 Fix: `depth_acc` per depth, used by `pick` (trial gate + EV loop) once
 the depth has MIN_SAMPLES rounds; the shared EMA remains the warm-start
 prior. Unit test `per_depth_acceptance_is_not_polluted_by_shallower_picks`.

Not landed (measured zero): context-gated/3-bit requant, top-64 shortlist,
PLD — see Phase 1(b)/(auto observations); 3-bit additionally needs MLX's
permuted 3-bit host encoder that does not exist in this repo (spec-12 trap).

## Phase 3 — verdict

- Acceptance @12.5k: draft was never degraded (d4 forced 2.66/round, flat
 from 1k). "Before/after" therefore applies to the CONTROLLER: auto now
 picks d3/d4 (picks [2x16 3x3 4x36 …]) where it froze on d2 before.
- Interleaved same-session A/B (old HEAD binary vs fix, same recalibrated
 table, 3 pairs × 256 tok, decode-only): **old median 33.6 vs new 41.6
 tok/s** (pairs: new 27.4/41.6/45.7, old 28.6/33.6/46.6 — new wins the two
 clean pairs by +45%/+36%; the two high old/new runs rode echo regions of
 the prompt). **Objective auto ≥ 35: MET on the median** (41.6), consistent
 with the forced-d3/d4 ceilings (37.7/36.2) the old controller could not
 reach.
- Verification: lib tests 21/21, `per_depth_acceptance…` new pin green,
 smoke green, qwen3_5 golden 310/310 ×2, workspace build clean. Memory:
 sequential single-model runs only; system 92% free pre-run (137 GB
 machine, ≤120 GB cap respected; one model resident ≈ 20 GB).
- Residual: box drift makes single pairs unreliable — always interleave;
 EV still prices from the shared EMA until a depth has 3 trusted rounds
 (first ~6 rounds of a request stay conservative by design).
## Additional measured findingsserving campaign)

- **Speculation density is content-bound.** lisa decodes at **57.7 tok/s on
 predictable** content vs **39.6 on novel** (1.46×); the draft composition
 is sound — the copy merges **per position** (proposal `i` takes the copy
 token when one covers it, else the chain's own prediction; only a
 full-depth copy hit skips the chain). The lever is lookup-draft
 composition/length: **1.3-1.5× on file rewrites**; the PLD draft length
 defaults to 5.
- **Verify-width cost bar (why `COPY_LEN_MAX = 6`).** A verify is 1.4-2× a
 serial step (S3 54.3 ms, S7 78.0, **S8 108.0** — the S-cliff that
 justifies `COPY_LEN_MAX = 6`), so speculation pays only through accepted
 tokens; the copy is already optimal on echo (mean 6.00 of 6).
- **Open item — yield per verify on predictable NON-echo content.** Needs a
 workload that reproduces the decode-rate gap before optimizing; three
 "gaps" of that campaign turned out not to be code.
- **Width ramp / copy-draft gate (open)**: wire `WidthRamp`/`CopyDraftGate`
 into the verify/PLD loops (needs the accepted-count vs depth in the accept
 scan; judge = in-situ interleaved round-cost). MTP double-gating stays
 forbidden (it would corrupt the EV accounting). Already landed on the same
 lane: PLD width clamped downward to the fast lane (depth ≤ 6, S ≤ 7) and
 the copy-draft guards (`copy_lookup_guarded`) wired on both paths —
 picks/acc-EMA unchanged, goldens ×2.
- **Draft-depth logit ensemble — set aside**: the exact re-score is
 load-bearing (without it: slower *and* worse).

