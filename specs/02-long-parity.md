# specs/02 — Long-context parity analysis (16k): MTP-or-serial, round economics, gap decomposition

Campaign: certified long A/B (707ee82/f8912d8) = lisa 25.6–30.3 tok/s
(15.6k natural prompt, 256 tokens FORCED ignore_eos, greedy, same-session interleaved).
Mandate: explain and close the long-context gap.

## 1. MTP or serial at 16k? → **MTP (auto depth ≤ 6), + prompt-lookup drafts**

Answered from our certified anchor's own logs, not a new run:

Readings:
- Our auto controller picks MTP at every 16k rep; depth caps at 6.
- **Our acceptance**: 256/77..92 rounds = **2.8–3.3 tok/round** (mean accepted
 1.77–2.32, ~1.5–1.9 accepts/attempt) — acceptance is not the gap.
- Our PLD (prompt-lookup, k=7) runs concurrently with MTP.

## 2. Our round cost at 16k (the actual gap)

From the certified JSONs (decode window = (ctok−1)/(t_last−t_first)):

| side | decode tok/s | decode wall s | rounds | wall/round | accept/round |
|------|-----|------|-----|-------|-------|
| lisa rep1 (d4-heavy) | 22.2 | 11.5 | 86 | ~134 ms | 1.95 |
| lisa rep2 (d2-heavy) | 25.6 | 9.97 | 92 | ~108 ms | 1.77 |
| lisa rep3 (d3-heavy) | 30.3 | 8.42 | 77 | ~109 ms | 2.32 |

**The gap is round COST, not acceptance.** Our served rounds cost 108–134 ms
wall. Our in-situ round-cost harness says S7@16k = 79.1 ms — our real served
rounds carry ~30–50 ms the harness does not see, and our EV controller picks
shallow depths whose realized cost does not scale down.

## 3. Hypotheses to falsify (Phase 2)

H1 (round-boundary overhead): our served round carries ~30–50 ms outside the
measured verify window — commit/rollback copies, drafter trim, keep-mask, EV
bookkeeping, sampling. Our mtp.* spans + LISA_TRACE on a real 16k serve round
must localize it.
H2 (shallow-width cost inversion at 16k): our S3/S5 rows at kv16384 are stale or
genuinely expensive (slot-price class), so the controller's EV math picks depths that
don't pay. Check the fresh round-cost table (measured this spec) vs what e2e rounds realize.
H3 (PLD): prompt-lookup drafts could add acceptance on natural prose; our MTP arm is
the driver. Given acceptance parity (§1), PLD is NOT the driver of the gap.
H4 (serial step): falsified as the primary driver — we never go serial at 16k
(our auto always picks a depth).

## 4. Measurements log

- round-cost kv=1024,16384 (fresh, this spec, post-707ee82 tree):
 - kv1024: serial 40.05 ms | draft 1.73 | verify S3:47.6 S4:48.1 S5:50.0 S6:59.8 S7:67.3
 - kv16384: serial 35.86 | draft 1.78 | verify S3:58.9 S4:72.7 S5:79.1 S6:76.0 S7:84.2
- Trace anatomy of a real served MTP round at 15.6k (`lisa run`, LISA_TRACE_JSON,
 span nesting):
 - forced d2 (S3): round mean 77.1 ms = draft chain ~1.5 + verify_build 75.6
 (tower.forward 46.4 + readback drain 29.1).
 - forced d4 (S5): round mean 92.9 ms (verify_build 90.3, readback 37.7).
 - The draft chain's GPU cost does NOT show in its own span (async enqueue) —
 it lands inside verify_build's throttled window; the round-cost table's
 1.78 ms "draft" cell is enqueue-only and under-counts by ~10 ms at 16k.
 - mtp.commit is 0.06 ms; the round-boundary host cost is negligible. H1's
 "30-50 ms of hidden boundary overhead" is thereby REFINED: ~12-16 ms is
 draft-chain GPU + drain (real work), the rest of the served-vs-run gap is
 acceptance mix (deeper rounds) + thermal noise.
- verify-audit S3@16k, 24 reps: wall 66.1-67.0 ms, 944 dispatches, GPU-attributed
 65.4/66.1 (GPU-BOUND): splitk qmm 23.7, **copy_strided 9.6 + copy2 3.1**,
 fused_add_rms 8.4, swiglu2 4.5, GDN (prep+two_row+gated_rms) 8.0, sdpa 3.05,
 qk_norm_rope_rows 1.4 (ENGAGED), copy_rows_into 1.4.
- LISA_CONTIG_TRACE backtrace census of the copy class: 128 contiguous/verify =
 32 DEAD q/gate de-interleave copies (attention.rs built q/gate eagerly at
 s>1 even though the fused rows arm reads qg_raw; the copies were never
 consumed) + 48 GDN z reshape copies (slice→4D reshape of the fused in_proj
 slice) + 48 small tail/reshape copies. All are tiny (<40 KB) and pay the
 ~75 µs GPU slot each — slot price, not bandwidth.

## 5. PORT A landed: deferred q/gate de-interleave (this spec)

attention.rs: at s>1 the q/gate `.contiguous` materialization is DEFERRED
until after the fused `qk_norm_rope_rows` arm declines (the arm consumed qg_raw
anyway — the two copies per layer were pure dead slots). s==1 keeps eager
materialization (its own fused arm needs the dense row).
- verify-audit S3@16k: 944→912 dispatches, contiguous 128→96/verify; wall wash
 in isolation (66.1→67.0) — expected: slots overlap the splitk tail; judged on
 interleaved e2e only (specs/02 lesson, see SKILL).
- Golden 310/310 (bit-identity: when the arm engages the copies were dead code;
 when it declines the composed chain builds identical arrays). Lib 23/23.

## 6. CERTIFIED verdict (this spec, same-session interleaved, 15.6k / 256 forced)

Post-PORT-A HEAD `1073466`, 3 interleaved pairs, sequential arms, memory check
per arm (peak well under cap):

| rep | lisa decode tok/s |
|-----|------|
| 1 | 24.61 (256 tok) |
| 2 | 27.13 (199†) |
| 3 | 30.80 (204†) |

† our serve stopped at EOS before 256 despite `ignore_eos` in 2/3 reps —
harness caveat to fix before the next certification round.

**VERDICT: NOT at parity — median 27.1 tok/s; objective ratio ≥ 0.90 NOT met.**

## 7. Residual gap, quantified per class (the ledger)

Answer to the mandate question — **at 16k we run MTP; the gap is per-round
COST, not acceptance** (§1–§2: acceptance parity, our realized 108–134 ms
served). Decomposition of our S3@16k verify (GPU-bound, 65.4 of 66 ms
attributed):

- splitk verify qmm 23.7 ms — at parity family (specs/15 closed).
- copies (copy_strided 9.6 + copy2 3.1) — 128 small slot-priced copies;
 PORT A removed the 32 dead ones (e2e wash — slots overlap the splitk tail).
 Remaining 96: GDN z reshape ×48 + tail transposes ×48. Next candidate port,
 same falsification risk.
- fused_add_rms 8.4 ms — floor (specs/16 §5.2 closed: norm-in-qmv loses).
- GDN kernels 8.0 ms, sdpa 3.1, swiglu2 4.5, qk-norm-rope 1.4 (engaged).
- Round extras outside the verify cell: draft chain GPU ~10–15 ms (NOT visible
 in the round-cost draft cell — enqueue-only measurement, fix the table), plus
 the depth-mix problem: the auto controller's realized d4–d6 rounds cost
 93–156 ms for ~1.0–2.6 tok/round when acceptance dies at index 2 — the
 collapse we need to avoid by NOT picking those depths (our table drops w2 34×,
 yet our EV picks them).

Named next levers, in order: (1) fix the EV controller's cost table to price
the draft chain + real readback drain (the 1.78 ms draft cell is enqueue-only
— the controller is buying rounds the table says are cheap but cost +15 ms);
(2) force-ignore-EOS correctness in our serve; (3) GDN z + tail copy removal
(96 slots, wash-risk); (4) batched decode kernels (specs/07 item 2, the
standing structural lever).
## 8. Open anchor

The certified long-context ratio still owes its M5-era re-measurement
(latest certified same-session medians: base 32.1 → 36.1 tok/s, specs/04
§7; the long-context TTFT ratio is likewise an open anchor, specs/05).

