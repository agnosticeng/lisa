# specs/04 — Wave-2 batched-kernel tails + certification protocol

Base 76de0d7 → wave commits 220ae47 (strided-src `copy_rows_into`).
Baseline anchor for every A/B: lisa @ **68c6127**, worktree `/tmp/lisa-baseline`
(rebuilt release the run). Ratios quoted below are same-session
interleaved; single-sided wall numbers drift ±10 ms with box thermal state.

## 1. The qmm bandwidth question, settled analytically

The verify qmm (split-K affine, gs64 b4, BN=2 KP=2) is ~**S-invariant**:
S3@16k 20.4+2.3 ms vs S7@16k ~23-26 ms — the quantized weight streams ONCE
for all verify rows, so cost scales with N×K bytes, not S. 20.4 ms for
~13-15.5 GB of quantized weights = ~510-605 GB/s. **CORRECTED:**
the denominator once quoted here ("~819 GB/s chip peak") was wrong — an
M3 Ultra desktop figure; this machine's published peak is **614.4 GB/s**
(M5 Max, 40-core GPU) and its measured achievable streaming rate is
**548-552 GB/s** (specs/07 §2, two independent kernels). The 605 upper bound
exceeds the achievable 548, so this byte/ms estimate needs re-measurement
before it is quoted as a saturation claim. The residual is JIT codegen class
(weight-stream scheduling), NOT missing S-tiling — there is no "3.5 ms/row"
arithmetic to recover; per-row accounting double-counts rows. specs/15 §6
(tiling exhaustion) stands; this is the one structural lever left
("BW splitk codegen", effort L+).

## 2. Wave-1 kernels (recap, ce70d68 + 76de0d7)

- `track_gated_rms` reads z through declared `zproj_strides`; GDN z stays the
 raw slice view of the fused in_proj (enabler: `Array::reshape_rows_view`).
- `track_sigmoid_mul_tail` merges gate [b,s,h,hd] with the UN-transposed sdpa
 output [b,h,s,hd] in one launch.
- Result: 912→848 dispatches, contiguous 96→32, copy_strided 7.5→3.1 ms,
 S3@16k GPU/verify 66.0→56.8 ms, wall 67.9→57.7 ms.

## 3. Wave-2 port: strided-src `copy_rows_into` (220ae47)

**Finding (probe-led).** The audit's residual copy class was NOT the GDN
keep-mask/state arms. Backtrace probes on `Array::copied` located the
32 contiguous/verify in the KV-cache update: `copy_rows_into` DECLINED on the
attention tail's transposed v view ("input src is not contiguous and the
kernel declares no strides"), and `FullAttentionCache::update` fell back to
`index_mut` → `slice_assign`, which materialises the ENTIRE capacity buffer
(`copied`) plus a strided write — per layer, per step, at ANY kv length.
That is the copy2_strided 3.3 ms + contiguous×32 + part of copy_strided
3.3 ms in the S3@16k audit.

**Port.** `copy_rows.metal` reads src through declared `src_strides`
(auto-detected by the MetalKernel signature builder — same mechanism as
gated_rms `zproj_strides`): decompose the linear thread index through
shape×strides instead of assuming dense layout. Same bytes, same order —
bit-exact by construction. Pin: `copy_rows_into_strided_src_matches_contiguous`
(transpose-out-and-back view, full-buffer bitwise compare).

**Result (verify-audit, 24 reps, the run, sequential):**

| metric | before (76de0d7) | after (220ae47) |
|---|---|---|
| S3@16k wall ms (median) | 60.0 | **52.7** |
| S3@16k GPU ms | 58.8 | **51.3** |
| S7@16k wall ms | 77.9 | **62.6** |
| dispatches/verify | 848 | **800** |
| copy2_strided / contiguous / copy_strided | 3.3 / 32 / 3.3 ms | **0 / 0 / 0** |
| copy_rows_into | declined→fallback | 2.3 ms (k+v, strided) |

Traps hit on the way (worth keeping):
- The probe must be armed past prefill: chunked 16k prefill fires ~44
 copies/chunk; a naive first-N probe window reports prefill sites only.
- A test that builds the "strided" tensor from the same flat byte vector as
 the dense tensor compares two DIFFERENT logical tensors (axis semantics).
 Build the strided view by transposing the dense tensor out and back.
- Note dispatches only dropped 848→800 but the fallback was O(cap) bytes,
 not O(s): the win is bandwidth + allocation churn, invisible to dispatch
 counts alone.

## 4. Families deliberately NOT forced

- **fused_add_rms (7.9-8.2 ms, 128 launches):** already S-generic —
 `n_rows × threadgroup` grid handles any S; the looped arm (>4096) is the
 correct shape at D=5120. Norm-in-qmv fusion is CLOSED (specs/16 §5.2, wave-2
 occupancy falsification): at k=5120 every consumer threadgroup re-runs the
 reduction and loses. No arm found; judged already optimal at this design
 point. Remaining cost = the slot price of 128 row-looped rms launches.
- **GDN (5.1-7.4 ms) / swiglu2 (4.2-4.3 ms):** already fused (2-dispatch GDN,
 packed swiglu2, gated_rms_silu). No further fusion attempted — forcing it
 would re-open the occupancy trap.

## 5. Certification protocol (the wave-1 gap) — `bench/tools/spec27_cert.sh`

Protocol verdict-24, executed the run against the 68c6127 anchor:
- Long: 15.6k natural prompt, 256 tokens FORCED (ignore_eos both sides — the
 serve-side fix 4316f5f is in the base), greedy, server-vs-server over
 /v1/chat/completions, decode tok/s = (ctok−1)/(t_last−t_first).
- 3 interleaved pairs, sequential arms, ONE model resident at a time,
 memory_pressure logged per arm, 15 s cooldown. Order ALTERNATED per pair
 (base-first / new-first / base-first) — the second side of an A/B runs
 ~5-10 ms hotter; fixed order biases the ratio.
- Short: haiku/64 forced (ignore_eos), 3 reps per side, non-regression control.
- Prior certified ratio (specs/01 §7): **0.54** (24.3-26.7 vs 43.9-46.9).
- Results: see §7 verdict (filled from bench/data/spec27/).

## 6. GPU readback race under contention — CLOSED (fix in lisa-mlx runtime/commands.rs)

`keyed_same_seed_same_tokens` was reported failing ≥3× with a MASSIVE
divergence (3388 vs 783 at token 11). Investigation per protocol:
1. Isolated rerun: green ×3.
2. Controlled repro: 22 runs under deterministic heavy GPU load (full
 lisa-mlx release suite, --test-threads 8 and 16, concurrent): green 22/22 —
 the keyed path itself did NOT reproduce.
3. BUT the same 16-thread sessions reproduced REAL kernel-buffer corruption:
 `sigmoid_mul_matches_composed` and `gated_rms_strided_z_view_matches_contiguous`
 failed with both sides reading foreign/stale values (e.g. expected −4.375,
 fused read −57.49, composed read ~4e-28 — buffer-sized garbage, not ULP),
 green in isolation. This is a genuine **readback/synchronisation race under
 GPU contention** (a kernel's output buffer read before its producing
 encoder completes), not a test-flake class. It plausibly explains the keyed
 divergence (same readback mechanism).

RESOLUTION (scope b — structural, two stacked runtime bugs; the pins and
`eval`/`as_slice` readback paths were already synchronised correctly, the
runtime's flush was not):

1. **Lost wait (the primary)**: `flush_and_wait`/`flush_wait_through` DRAINED
 `in_flight` under the lock and waited OUTSIDE it. Under parallel test
 threads a concurrent `eval` could observe an EMPTY `in_flight` (the other
 thread had checked the entries out) and return while committed command
 buffers were still executing — the caller then read buffers those buffers
 were writing. Proof: the `LISA_RB_DEBUG` detector (`Array::to_vec` asserts
 `buf.used_cb <= completed_id` after `eval`) fired exactly at failures
 (`used_cb=21 > waited_id=19`). Fix: never drain before waiting — clone the
 highest committed entry, wait it, prune `<= waited` under the lock
 afterwards. `flush_wait_through` keeps its overlap semantics (entries after
 the target stay in flight) and deliberately does NOT clear `prev_outputs`
 (later in-flight outputs must keep fencing the next encoder).
2. **Freed completion block (UAF)**: `ComputeEncoder::end` registered a
 second `addCompletedHandler` block for the cross-encoder fence-map cleanup
 and dropped it when `end` returned. The objc2-metal binding passes a raw
 `*mut DynBlock` and does NOT copy (same fact the specs/16 arm_completion
 fix established), so completion invoked a freed block — benign while the
 heap kept the bytes intact, but under parallel heap churn the stale
 captures removed the WRONG fence entry (or none), the next encoder skipped
 its `waitForFence`, and kernels read in-flight producers. Fix: the cleanup
 moved into `arm_completion`'s kept-alive block (end returns
 `(outputs, fence)` to rotate).

DETECTOR kept: `LISA_RB_DEBUG=1` prints `[rb-race] buf#N used_cb=X >
waited_id=Y` on any readback whose producer cb was not waited — classify any
future parity-pin failure against it before calling it a numerics bug.

EXTINCTION EVIDENCE: see §8.

## 7. Certified verdict, bench/data/spec27/)

Protocol of §5 executed: 3 interleaved long pairs (order alternated), 3+3
short control reps, one model resident at a time, peak RAM well under the
120 GB cap (single 27B serve, same as every certified session).

- LONG (15.6k / 256 forced): base(68c6127) 29.78 / 32.08 / 39.81 tok/s
 (median **32.1**), new(HEAD+220ae47) 21.61 / 36.08 / 39.19 (median
 **36.1**). Pairwise new/base = 1.21 / 0.67 / 0.98 — the 0.67 pair is the
 new-first COLD arm (JIT of the 16k shapes + hot second side), the exact
 ordering bias the alternation is there to expose. Median-of-medians ratio
 **1.12**, all-arm mean ratio **0.95** → **certified ≥ 0.75: PASS**, no
 regression, evidence of a real win (the removed O(cap) KV fallback scales
 with context length, which is where the win shows).
- SHORT (haiku/64 forced): base 42.1 / 48.0 / 46.4 vs new 45.3 / 47.7 / 46.5
 (prefill+decode tok/s) → median ratio **1.00**, parity — non-regression ✓.
- verify-audit GPU S3@16k: 58.8 → **51.3 ms** (S7 wall 77.9 → **62.6 ms**).
 GPU/verify campaign trajectory: 66.0 (specs/01 wave base) → 56.8 (wave 1)
 → **51.3** (wave 2).

Campaign ledger (batched-kernel tails, specs/01→04):
- S3@16k verify wall 67.9 → 57.7 (wave 1) → **52.7 ms**; S7@16k 77.9 →
 **62.6 ms**; dispatches 912 → 848 → **800**.
- Certified long ratio vs the 68c6127 anchor ≥0.75 ✓ (the pre-wave certified
 framing put it at 0.54; this wave's win is measured lisa-vs-lisa, and a
 cross-session number ~0.77-0.82 is NOT certified — same-session pairs are
 required before quoting it).
- What remains, structural, in order: (1) split-K qmm BW efficiency
 (JIT codegen class, ~3-5 ms/verify);
 (2) fused_add_rms 8.2 ms = 128 row-looped rms launches at slot price
 (norm-in-qmv closed, occupancy-falsified); (3) the readback race — CLOSED,
 see §6;
 (4) our ~51 ms verify GPU — the residual is GPU CONTENT (qmm BW +
 slot-priced small kernels), not round structure (specs/01 §7 xctrace:
 ~90-100 % GPU-busy in-round).

## 8. §6 extinction evidence, post-fix)

Repro baseline (buggy): `cargo test -p lisa-mlx --lib --release
-- --test-threads=16` failed ~1 run in 2 (qk_norm_rope_rows, sigmoid_mul,
gated_rms_strided_z, qsa fused_selector, qgate_offset_slice — same
stale/zero-read signature); the `LISA_RB_DEBUG` detector fired exactly at the
failures (`used_cb` > `waited_id` — the readback raced an unwaited command
buffer).

Post-fix runs, all green:
- Full workspace parallel suite `cargo test --workspace --lib --release
 -- --test-threads=8`: **3/3 green** (criterion met).
- lisa-mlx full lib suite at `--test-threads=16 --nocapture` with
 `LISA_RB_DEBUG=1`: **8/8 green, detector fired 0 times**.
- Historically flaky pins ×5 each under `--test-threads=16` load:
 counting_sort_matches_reference, fused_selector_matches_reference,
 qk_norm_rope_rows_matches_composed, qk_norm_rope_matches_composed,
 sigmoid_mul_matches_composed, sigmoid_mul_tail_matches_transposed,
 gated_rms_strided_z_view_matches_contiguous, gated_rms_silu_matches_composed
 (5/5 each), keyed_same_seed_same_tokens (5/5) — **0 flakes in 45 runs**.
- Goldens: `lisa-bench golden` (27B) 310/310 ×2. check.sh equivalent
 (build --workspace --release + test --lib + smoke): green. NOTE: stock
 check.sh uses the dev profile and hung >12 min on `cargo test --workspace
 --lib` (known trap) — killed and re-run the same steps in release.

Perf (round-cost, 27B, sequential single-model, interleaved passes to cancel
thermal drift): serial kv1024 39.47/39.39 vs base 39.39 ms; S7@kv1024
80.76/79.64 vs 78.90; S7@16k 74.48/75.17 vs 74.90 ms — WASH, the fix is
host-bookkeeping-only (it REMOVES one addCompletedHandler registration per
encoder and the pre-wait list drain; no new GPU-side synchronization).

# specs/04 wave 3 CLOSED: tail-ULP codegen axis — FLOOR AT OUR SHAPES
Question: specs/15 §6 exhausted PPT/BN/K_PARTS under the strict bit-exact
constraint; the verify path is now tail-ULP (§9.7) — does the relaxed
contract reopen a BW win (62-75% peak → ~85%)?
## What was falsified, and why (re-read of specs/14/§6 ledgers)
- specs/14 full-K fp32 re-scored tile: was built BIT-EXACT (qmv_fast numerics);
 its in-situ loss (S7 126.5 vs 102.5 ms) was STRUCTURAL (full-K per-thread
 chains serialize the mixed pipeline). Tail-ULP does not change that design's
 failure mode.
- specs/15 §6 PPT=2: BIT-IDENTICAL to PPT=1 by construction — its
 falsification (+3.7% S7) was STRUCTURAL. BN=4: stack-spill at M=5 on our
 Metal compiler — STRUCTURAL. CONCLUSION: the old constraint was NOT what
 blocked any of these; relaxing it reopens nothing as-is. What tail-ULP
 actually opens is NEW codegen designs, so we built two:
## Variants built and pinned (commit 4e8ef50, infra; production path unchanged)
- VLOAD=1: 4 consecutive packs/thread/iter loaded with ONE 16-byte uint4 per
 column (lane*4+it, stride 128; requires per_part%4==0 — K%64 already
 aligns). NOT the falsified PPT shape (different lane→pack map AND load
 width). New tail-ULP pin verify_qmm_splitk_variants_tailulp_vs_qmv_wide:
 vl/fm/vl+fm × S{3,7} × 2 draws all ≤1 ULP + argmax equal.
- Math::Fast compile of the splitk specializations (compile_builtin_math;
 distinct host_name `_vl{}_fm{}`, stock name preserved). warm_verify_qmm
 splitk warm FIXED (was silently failing: 6 template args on a 7-param
 template, `let _ =` swallowed it — it never warmed the production kernels;
 now emits the exact stock 8-arg name and matches the dispatch).
- ISOLATED-BENCH TRAP (new): my first variant sweep patterned `let _ = f`
 which DROPS y without eval — timings were enqueue-only and showed fake
 2-3x "wins". Any isolated kernel timing must eval + sync the OUTPUT
 tensor, and even then isolated numbers do not rank in-pipeline.
## In-situ interleaved round-cost (arbitre; S3/S7@16k, steps 12, 3
## order-alternated pairs per arm, sequential single-model)
- stock (c0de159+infra): S3 med 71.3-72.1, S7 med 81-86.3 ms (thermal swings
 to 152 on one contaminated rep — discarded by the serial-drift control).
- fm=true: S3 med 69.2, S7 med 79.7 — pair deltas −1.3/+5.6/−7.8 ms = WASH
 inside the ±7 ms band. The Math::Fast codegen change does not move the
 DRAM stream.
- vl=true: S3 med 70.8 (−2.5%, noise-band), S7 med 83.6 vs 81-86 stock =
 WASH. 16B loads cut load instructions ~4x but the stream was already
 fully coalesced (32 lanes x 4B consecutive = 128B transactions); the
 kernel is DRAM-BW-bound, not issue-bound.
## Verdict: CODEGEN FLOOR AT OUR SHAPES, now measured not assumed
Splitk verify qmm ~24-26 ms for ~13.5-15.5 GB quantized weights = 510-605
GB/s is the floor for THIS tile class on our JIT compiler: fp32 codegen
(math mode, load width, pack-per-thread) does not convert in situ; every
tiling axis (PPT/BN/K_PARTS) is exhausted. Against the achievable 548-552
GB/s streaming rate (specs/07 §2) this tile is already near saturation, so
the residual is NOT recoverable by kernel codegen at our shapes; the
remaining candidate explanations are (a) attribution (total-round GPU incl.
overlap, not kernel efficiency) or (b) an occupancy/scheduling property
outside the tile (threadgroup count vs N/BN grid).
Objectives GPU/verify S7 ≤20 ms / round ≤75 ms NOT met by this wave; e2e
unchanged (no branchement — nothing to branch). Goldens 256/256 x2, lib
27/27 incl. both tail-ULP pins. Do not reopen codegen variants on this
tile without new evidence (e.g. a wide-tile BN=8 redesign, pack-gated).
## 9. Pre-wave anchor for the same cell (recorded

Before the raw-view kernel waves, the same S7@16k round measured **130.4 ms**
at **1392 dispatches** — the copies/rms slot-price class, largely eradicated
by `qk_norm_rope_rows`, `gated_rms_silu` and `copy_rows_into` strided-src
across waves 1-2 → **62.6 ms / 800 dispatches** (§3/§7).
