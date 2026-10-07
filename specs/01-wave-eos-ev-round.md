# specs/01 §7 wave — ignore_eos fix, EV re-pricing, round decomposition

HEAD base 045f28f → wave commits 7eed729 (ignore_eos), c503bf1 (EV chain pricing).

## 1. ignore_eos root-cause + fix (7eed729)
- `set_eos_ids` was `OnceLock::set`: first write wins. `lisa run` arms `im_end`
 ids then clears them for `--ignore-eos`; the clear was a SILENT NO-OP, so the
 decode loop kept `is_eos` armed and stopped at EOS (199/204 of 256 in 2/3
 certified reps). The flag never reached the decode loop — not a race, not a
 token budget: a lost write.
- Fix: `static EOS_IDS: RwLock<Vec<u32>>` — LAST call wins. Regression test
 `eos_set_is_overwritable`. Proof (forced-256 15.6k, 3 reps): 3/3 complete
 (rounds+accepted = 257/256/258), decode 30.9/37.5/33.6 tok/s.

## 2. EV table re-pricing (c503bf1)
- Old draft cell = ONE synchronous `draft_step` (1.73-1.79 ms) × depth. New
 `chain_ms` cell times the depth-d chain AS SERVED (device-fed proposals, lazy
 enqueue, one tail eval). Measured (27B, 24 iters, median):
 - kv1024: d2-d6 = 3.54/5.13/6.72/8.60/10.34 ms
 - kv16384: d2-d6 = 3.63/5.26/6.85/8.53/9.82 ms (nearly FLAT in KV)
- Picks before/after (3 interleaved forced-256 reps): UNCHANGED (d2-d4 mix).
 The x-depth heuristic was within ~1% of the measured chain at every depth;
 in-run `round_ms` dominates pricing after 2 samples anyway. e2e wash
 (median 33.6 → 30.8, within drift). The controller was NOT buying
 under-priced deep rounds — ledger hypothesis falsified by measurement.

## 3. Round decomposition (xctrace Metal System Trace, attach, forced-256
## @15.6k, the run)
- lisa-serve (23.0 tok/s, 1.42 tok/round → round ≈ 62 ms): steady 5 s window =
 10040 intervals; small (≤50 µs) intervals are ~0.5 µs — the "75 µs slot" is
 per-KERNEL GPU time (our LISA_TRACE attribution), visible to Instruments only
 inside the big encoders. Big-encoder exec ≈ 880 ms/s → ~55 ms GPU per round,
 GPU ~88-100% busy (matches specs/19).
- CONCLUSION: the 40-50 ms/round advance is GPU CONTENT per round, not round
 structure, not cadence, not acceptance (our accepted 1.42-2.0/round). Our S=3
 verify attributes 65.4 ms of GPU (splitk qmm 23.7, copies 12.7, fused_add_rms
 8.4, GDN 8.0, swiglu2 4.5, sdpa 3.1 — specs/01 §5) while we run GPU-busy
 88-100% — the residual is the per-launch GPU slot price inside our kernels,
 not idle structure. Micro-ports (copies/rms) are measured washes because slots
 overlap the splitk qmm tail. Remaining named lever: batched decode kernels
 (specs/07 item 2) + serial-step floor.
- Artifacts: bench/data/spec26/{xc.mlx,xc.lisa}.{trace,req.json,server.log},
 round-anatomy.{json,err} (LISA_TRACE_JSON d4 round: 84.5 ms = 1.6 host +
 chain ~6.9 GPU hidden in the verify window + readback drain 34 ms).

## 4. Certified verdict AFTER the fixes (4316f5f, verdict-24 protocol)
3 interleaved pairs, forced 256, ignore_eos HONORED (this run's entries are the
LAST per cert.*.json — the files also hold the morning's §6 reps):

| rep | lisa decode tok/s |
|-----|------|
| 1 | 24.32 (256 tok) |
| 2 | 26.66 (256 tok) |
| 3 | 22.55 (256 tok) |

**VERDICT: NOT at parity — median 24.3 tok/s (24.32/26.66/22.55); objective
≥0.75 NOT met, 0.90 out of reach this wave.** The fixes were correctness/
pricing, not speed — the §3 decomposition shows why: the residual is GPU CONTENT
per round (~55 ms at 16k), attributable to splitk qmm + rms + copies bulk. The
only named lever left is batched decode kernels (specs/07 item 2); micro-ports
in the copies/rms class are measured washes (specs/01 §5-§6, PORT A) because
their slots overlap the splitk qmm tail.
Correctness wins this wave: 3/3 lisa reps complete 256 forced (was 1/3), the
EV table now bills the chain as served, goldens 310/310, engine lib 22/22.
## 5. Measured additions from theserving campaign

- **Speculation at temperature > 0 — the hole the probe cannot see.**
 Everything speculative was greedy-only (four CLI guards + the scheduler's
 `ensure!(greedy)`), and the packed checkpoints declare `temperature: 1.0`,
 so the default request got **no speculation at all**: 1.31 s vs 0.60 s for
 the same echo request. The acceptance rule landed per 00-contracts §2. It
 is NOT bit-exact (near-tie class): at temp 1.0 the draw is an argmax over
 `v/T + gumbel`, so the verify batch's ≤ 1 ULP flip moves it — validate
 statistically (seeds 1/2/3 must differ), never by diffing; a first cut
 claimed equality with the serial stream and diverged at the FIRST token.
- **Multi-client batching.** B=2 aggregate **1.70×** single-stream; B=4
 aggregate **2.73×** (1.47×/stream). The earlier "63 %/stream" claim was
 measured on a corrupted path — a cross-row `track_swiglu2_packed` bug
 (gate of row 0 × up of row 1 at B>1, since the gate|up merge of
 00-contracts §4) is fixed, with a row-generic pin at B∈{1,2,4}.
 Structural floor: the qmm streams ~13.5 GiB of weights per step regardless
 of B → aggregate caps at ≈ 2.6-2.8× at 4-bit (bandwidth-bound; derivations
 in specs/07 §2). Batched decode is row-invariant — the residual is the
 stream-weights floor, not the row path.
- **Wave batching policy (the 5× TTFT gap).** Wave eligibility required
 depth 0, so with the default depth no request was ever batchable and N
 concurrent clients paid N sequential prefill+decode passes. A wave of
 ≥ 4 eligible requests now drops per-stream speculation and runs the
 continuous batch (`BATCH_WAVE_MIN`, serve-side; only the depth clause is
 relaxed — 00-contracts §2). Crossover measured on both sides (4 streams ×
 64 engine steps, medians of 3): B=1 MTP 1.11 s; B=2 MTP 2.22 vs batched
 2.81 s (MTP wins); B=4 MTP 4.71 vs batched 3.91 (**1.20×**). On the
 serving probe: worst 4-stream TTFT **7069 → 1302 ms**, aggregate
 42.8 → **47.7** tok/s against 41.7 solo.
- **Short-context speculative parity (certified)**: 58.7 vs 58.3 tok/s
 interleaved same-session, 64 tokens forced both sides (ratio 1.01);
 certified short path also measured 60.2 tok/s (single sample).
- **Closed as NOT gaps** (verified, not assumed): the multi-row head is
 already ONE `lm_head.forward` over `[1,S,D]`; the probe's global
 `speculative: null` verdict is client-side inference (the echo passes
 report per-rung speculation); decode density is now **3.37-3.76
 tok/step**, so the remaining decode gap is not per-step cost.
- **Prefix-homogeneous batching — set aside**: needs real concurrency,
 like the shortest-tail scheduling idea (specs/05).

