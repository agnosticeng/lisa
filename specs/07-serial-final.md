# specs/07 — Serial-step FINAL audit and verdict

Question: after all waves, is there still a real ≥1 ms lever in the serial decode step
(37-40 ms measured here), or is the residual fully attributed to
the lazy-graph MLX-C runtime class? Method: fresh `lisa-bench serial-audit` at HEAD with
the full current equipment (LISA_TRACE=1 + LISA_TRACE_JSON + CB_RECORD), then every
remaining fusion/launch/reorder candidate checked against the falsification ledger.

## 1. Final decomposition (27B-4bit, kv1024, 32 counted steps, sequential, single model)

```
wall ms/step median 41.18 (mean 41.16)
dispatches/step 640 (was 704 @ specs/16, 672 @ fde22cd)
GPU ms/step 40.04 un-attributed (idle/gaps) 1.14 ms
```

| class | ms/step | launches/step | note |
|---|---|---|---|
| qmv (all projections) | 18.63 | 289 | see §2 — 88 %-of-peak claim RETRACTED (wrong peak) |
| fused_add_rms (64L × 2 norms) | 8.52 | 128 | 66 µs/launch — see §3 |
| swiglu2_packed | 4.40 | 64 | 68 µs each, slot-bound |
| gdn_decode_complete | 3.25 | 96 | fused 2-dispatch, 34 µs each |
| copy_rows_into (KV write) | 1.54 | 64 | post specs/04-wave2 strided-src |
| sdpa 2-pass | 1.57 | 48 | dense measured cheaper at kv1024 (specs/23) |
| sigmoid_mul_tail | 0.82 | 16 | fused arm engaged |
| qk_norm_rope | 0.76 | 16 | specs/24 rows arm (serial s=1 kernel) |
| argmax (draft head) | 0.31 | 1 | keyed |
| everything else | ~0.25 | ~45 | scalar 49, zeros 1, array 3, casts ~8, index 3 |

Launch-count history: 798 (pre) → 704 (specs/16) → 672 (q\|gate merge) → **640 (HEAD)**.
The 45 cbs/step cadence ceiling stays 1.33 ms (specs/16 phase 3, specs/11 §13); current
idle measurement 1.14 ms is inside it.

## 2. NEW FACT — the serial qmv bandwidth gap is FALSIFIED

Exact per-step streamed weight bytes from the safetensors index (4-bit gs64: n/2 B words
+ bf16 scales + bf16 biases; 48 GDN layers, 16 attention layers, MLP everywhere):

13.47 GiB/step across the projections (gate|up|down 8.96 GiB, GDN in/out 4.20 GiB,
attention 0.62 GiB).

**CORRECTED— the peak in this section was WRONG.** This spec used
"~819 GB/s" as the M5 Max peak. **819 GB/s is the Mac Studio M3 Ultra's figure**;
it is not an M5 Max number. Apple's own tech-spec pages for the 2026 MacBook Pro
give **460 GB/s (32-core-GPU die) / 614 GB/s (40-core-GPU die)**, and the 40-core
M5 Max is **512-bit LPDDR5X-9600 ⇒ 614.4 GB/s** exactly (Tom's Hardware). This
box is the 614.4 one: a saturated streaming-read probe measures **548-552 GB/s =
~89 % of peak** (`bench_peak_stream`, two independent kernels agreeing), and 548
is therefore the *achievable* denominator.

Two consequences:
1. **The old arithmetic is impossible.** 13.47 GiB ÷ 18.63 ms = 723 GB/s *exceeds*
 the real 614.4 peak — a read cannot beat the memory system. The one
 reconciliation that fits is that 18.63 ms covers the **8.96 GiB MLP subset**
 (=> 516 GB/s ≈ 84-94 % of achievable), not the full 13.47 GiB it was credited
 with. Treat the 18.63 ms figure as un-attributed until re-measured.
2. **The floor verdict below does NOT hold.** At 614.4 peak, a full step's ~14.5 GB
 of weights take 26.4 ms to stream at the achievable 548 GB/s. Our step is
 38.69 ms => **374 GB/s effective = 61 % of peak / 68 % of achievable**, and the
 spare ~12.3 ms is work that does not overlap the stream. A step that overlapped
 its weight stream fully would sit at the 26.4 ms floor (548 GB/s effective).
 **The qmv path is RE-OPENED as a target** — the entry was closed against a peak
 33 % too high for this machine.

The 62-75 %-of-peak figure in the ledger is a **splitk verify-path @16k** number
(specs/04), not the serial path. specs/04 wave 3 already showed codegen variants do
not move GB/s on a DRAM-BW-bound tile — that finding stands, but it was a
*saturating-stream* result, and at 61-68 % of achievable we are demonstrably not
saturated. **The "BW codegen" entry on the assumed-deltas list is re-opened for the
serial step.**

## 3. Candidates examined and falsified/dissolved (none ≥ 1 ms)

(a) Untried bit-exact fusions:
- "rms_out 81 launches = GDN pre-norm" premise is STALE: at HEAD `rms_out` is 1/step —
 the GDN output gate is already the `gated_rms_silu` single-launch arm (specs/01 wave 2).
- fused_add_rms looked like TWO launches (`fused_add_rms_sum` + `fused_add_rms_norm`
 labels, 128+128/step). Read of jit/norm.rs: it is ONE dispatch writing two output
 buffers; the label counts are per-output attribution. A single-launch merge therefore
 already exists — no 4 ms lever there.
- Its 66 µs/launch equals the STOCK `rms_looped` price at the same shape (0.064 ms for
 one launch in the same trace) — our kernel is at stock-kernel parity; the price is the
 per-launch GPU frontend cost of a looped single-row reduction at D=5120, which any
 implementation at this width pays (add+rms fusion declines D>4096, so add and rms run
 as two ops).
- norm-in-qmv: falsified twice (occupancy specs/11-wave2, reduction-redistribution
 specs/16 §5.2). swiglu-in-down-qmm is the same O(N×tgs) prologue-recompute class
 (5120 consumer threadgroups re-reading a 70 KB intermediate) — falsified by the same
 law without a run.
- sdpa+sigmoid_mul_tail already fused; cross-layer batching impossible (sequential
 dependency chain); qk_norm_rope rows arm exists (specs/24).

(b) Avoidable launches: the entire non-top tail (scalar 49, zeros 1, array 3, casts ~8)
is ≈0.25 ms of GPU time; killing all of it cannot reach 1 ms. Dispatch merges are bounded
by the 1.14 ms idle measurement.

(c) Reordering/hoisting: idle 1.14 ms/step is the only un-attributed wall; cos/sin and
const hoists landed in specs/16; cb-cadence merges falsified (specs/11 §13, specs/19).

## 4. Verdict — **RE-OPENED**

**The "STRUCTURAL FLOOR, CLOSED" verdict below was reached against a wrong peak
(819 GB/s = an M3 Ultra desktop figure, not this M5 Max's 614.4).** At the
*measured* achievable 548 GB/s our step runs at **68 % of achievable** (374 GB/s
effective), while a fully-overlapped step would sit at the 26.4 ms streaming floor
(548 GB/s = 100 % of achievable). A step that is demonstrably not saturating is not
at a floor. See §2's correction for sources and the full arithmetic.

What survives: the individual *phase attributions* (qmv bytes, norm/act slot price
at stock-kernel parity, GDN/attention fusion, cadence) were measured and stand. What
does not: the conclusion drawn from them — that no candidate ≥1 ms exists — because
"we are at 88 % of peak" was the premise, and it was false.

The live target is now the **~12.3 ms of work that does not overlap the weight
stream** — the gap between our 38.69 ms step and the 26.4 ms unavoidable streaming
floor at 548 GB/s.

--- (original text, superseded) ---

The serial step is GPU-busy-bound at 40.0/41.2 ms with idle 1.14 ms. Every ms is now
attributed: qmv 18.6 at 88 % peak BW (floor ≈ 16.4 at 100 %, unreachable), norm/act slot
price 13.0 ms at stock-kernel parity (closable only by norm/act-inside-qmm — falsified
×2 by occupancy and by O(N×tgs) prologue recompute), GDN/attention/copies 7.2 ms already
in fused arms, cadence ≤1.33 ms (falsified). No candidate ≥1 ms survives triage.

Assumed deltas ledger (revised):
1. ~~Serial qmv BW 62-75 % vs 85-97 % on the saturating path~~ — FALSIFIED for serial
 (§2): the splitk-verify BW number does not transfer.
2. The lazy-graph MLX-C runtime advantage: unquantifiable from outside and the
 dispatch count is not the limiter (specs/22 xctrace); per-op GPU content at
 D=5120 is the same class (add+rms two ops, stock rms price). What remains of the
 ~34-vs-41 comparison is benchmark-inference vs in-pipeline measurement (the serial
 cell never serves at 16k; specs/22 showed our GPU busy/token FASTER on the haiku
 path). Closing the serial-step file with floor accepted; the only named campaign
 lever left remains batched decode kernels (specs/07 item 2) and verify-round GPU
 content (specs/01 §7), not the serial step.

Serial ledger anchors: 39.3-41.2 ms wall, 640 dispatches, GPU 40.0, idle 1.1.
Reproduction: `LISA_TRACE=1 LISA_CB_RECORD=1 cargo run --release -p lisa-bench -- \
 serial-audit --model agnosticeng/Qwen3.8-27B-4bit --kv 1024 --steps 32`
Artifacts: bench/data/spec30/serial_audit.txt, serial.trace.json.

Machine safety: sequential single-model run, memory_pressure 89 % free before both runs,
no second model loaded; prior certified serve peak 31.8 GB — cap respected (this run's
peak RSS not sampled; short 8-step re-check wall 47.8 = thermal drift, same 640
dispatches — in-session medians only).
## 5. Campaign-day anchors and the M4 → M5 round-cost table recalibration

- Anchors: goldens 310/310; decode 30.6-30.7 tok/s at HEAD; serial step
 **39.67 ms** median at **640 dispatches/step** (matches §1's count);
 certified short path **60.2 tok/s** (single sample).
- The persisted round-cost table was **re-calibrated M4 → M5** — the EV
 controller had been over-pricing every round by ~26-32 %, depressing the
 picked depths:

| cell | M4 (<2k) | **M5 (<2k)** | Δ |
|---|---|---|---|
| serial | 45.0 ms | **39.44** | −12 % |
| draft step | 2.3 | **1.65** | −28 % |
| verify S3/S4/S5/S6/S7 | 60.3 / 69.5 / 74.7 / 91.0 / 103.6 | **44.40 / 48.44 / 55.48 / 63.57 / 70.75** | −26…−32 % |

- `@8-16k (kv 12288)`: serial 34.24 · draft 1.70 · chain d2..d6
 3.51/5.44/7.22/9.10/10.77 · verify S3..S7 57.95/60.10/66.79/66.08/74.99.
- Structural floors, falsified — do not chase: **M=1 decode is out of the
 NAX range** (structural, confirmed by the GPU audit); norm-inside-consumer
 fusions (TG-memory/occupancy — 00-contracts §4); lazy-runtime cadence
 (0 % idle; inside the 1.33 ms ceiling of §1); `msg lane lm_head` /
 `lane_qmm` / `lane_matmul mpp` (already in place in `shaders/nax/`).
 (The splitk verify 62-75 %-of-peak figure is specs/04's number, not this
 path's — see §2's correction for the serial-path arithmetic.)

## 6. MTP-round addendum — the per-round head-read census (specs/07 item 2 scoped to the MTP path)

 The 87.7-vs-191.5 gap was traced to per-round GPU content (specs/01 §3). This
 section scopes item 2 onto the MTP path only (the DFlash sidecar's batched
 block-head mechanism is NOT engaged in MTP mode and is not ported).

 **Trunk-head reads per round, both engines (source census):**

 | read | reference MTP mode | lisa MTP mode (default) |
 |---|---|---|
 | draft chain, per tick | coarse 2-bit/gs64 full-vocab readout → top-32 → exact 32-row re-score (auto-built at load) | **full 4-bit trunk lm_head per tick** — the coarse path exists (`LISA_DRAFT_HEAD_BITS=2`, tower.rs `draft_step`) but is OFF by default |
 | verify block | one trunk forward + one lm_head over all block rows | identical (one batched `tower.head` over w+1 rows) |

 So the head-read mechanism the reference runs by default is ALREADY
 IMPLEMENTED in lisa — `draft_step`'s coarse shortlist is the same
 coarse→top-32→exact-re-score scheme — and the only difference is the default.

 **Counter-balanced in-situ A/B** (`lisa-bench round-cost`, 27B-4bit, kv16384,
 medians of 2 pairs per arm, alternating boot order OFF,ON,ON,OFF, ~2 min
 cooldown between boots — the box drifts ~2× on chain/verify cells under
 sustained load, cooldown restores them; serial is stable across all boots):

 | cell @16k | head OFF | head ON (coarse) | Δ |
 |---|---|---|---|
 | chain d2 | 5.38 ms | **4.26** | −21 % |
 | chain d3 | 7.73 | **5.97** | −23 % |
 | chain d4 | 10.12 | **7.90** | −22 % |
 | chain d5 | 12.45 | **9.82** | −21 % |
 | chain d6 | 14.29 | **11.20** | −22 % |
 | serial step | 39.4 | 39.4 | — |
 | verify S3 | 58.6-67.6 (boot noise ±15 %) | same | — |

 The coarse head saves ~1.1-3.1 ms per round (the per-tick 4-bit full-vocab
 read replaced by the 2-bit coarse read + 32-row gather). Against a 55-65 ms
 round that is ~2-3 % e2e — and the coarse head costs **+283 MB resident**
 (measured `t4-draft-head` delta, 14755 → 15038 MB), which violates the
 footprint contract (specs/00 Rule 3: weights are never duplicated at load;
 the recorded decision in tower.rs is "RAM wins"). **The default therefore
 stays OFF**; the mechanism ships behind `LISA_DRAFT_HEAD_BITS` for
 latency-first runs.

 ## 7. Item-2 verdict at HEAD (0f054a1) — the streaming/non-overlap slot class,
## measured, falsified where falsified (this session, one box, sequential)

### 7.1 The fresh census (verify-audit / serial-audit at HEAD)

Serial step, kv 1024, 32 counted steps (`lisa-bench serial-audit`, GPU probe on):
`wall 37.78 ms | GPU 38.15 (fully attributed, idle 0)` —
`affine_qmv_fast 34.67 ms` (ALL projections, one kernel name) + phases
`fused_add_rms 0.96 + gdn_decode_complete 0.77 + swiglu2_packed 0.40 +
copies ~0.5 + sdpa 0.50 + misc < 0.2` ≈ 1.9 ms. The GPU is BUSY; the phase
fusions of specs/16-24 landed. **The specs/07 §2 "12.3 ms that does not
overlap the stream" is, at HEAD, inside the qmv stream itself**: ~14.5 GB of
quantized weights in 34.67 ms = **417 GB/s effective = 76 % of the 548 GB/s
achievable** (§2's corrected denominator). The 18.63/8.52/4.40 phase table of
§1 is a pre-fusion census and no longer describes HEAD.

Verify forward, S7, kv 1024, 24 counted (`lisa-bench verify-audit`):
`wall 50.49 | GPU 49.01` — splitk qmm 21.03 ms (280 kp2 + 28 kp4 launches),
then the phase classes 8.08 (fused_add_rms) + 4.39 (swiglu2) + 5.74 (gdn
prep/lean) + 2.86 (gated_rms) + 3.1 (copies) by the CB counter. BUT the same
kernels measured back-to-back through the real op path (`bench_gpu_slot`,
extended this session to the M=3/7 shapes) cost
`fused_add_rms [7,5120] 16.2 µs | swiglu2 [7,17408] 4.2 µs | rms [7,5120]
7.0 µs` gpu-slot — and the GPU-probe p50 inside a verify-only window reads
`fused_add_rms 5.7 µs | swiglu2 3.6 µs | gdn_prep 6.8 µs`. **The phase
kernels are cheap at verify widths; their in-situ CB-attributed ms is
launch-context (dependency-boundary slot), not kernel execution.** A
row-parallel phase redesign therefore has nothing to win — the row-scaling
S3→S7 measured ~1.2×, not 2.3× (a fixed per-launch slot, matching specs/01
§3's "per-launch GPU slot price" and the micro-port washes of specs/01 §5).

Census caveat, now measured: the CB per-kernel gpu-ms attribution
OVER-COUNTS when exec windows overlap (the 16k S7 run: per-kernel sum
180.7 ms/verify against a 115 ms wall) and UNDER-COUNTS pipelined
back-to-back launches (the gpu-slot method at n=256: `qmv_fast 5120→17408`
62.5 µs for a 45 MB read = 770 GB/s, above the 614.4 physical peak —
impossible, therefore under-attributed). Per-class census stands; per-launch
absolutes from either counter do not. The only un-attributed-free instrument
remains xctrace (specs/22/26), artifacts gone with bench/data/.

### 7.2 The occupancy/tile axis of the splitk verify tile — FALSIFIED in situ

specs/04 wave 3 closed codegen (PPT/BN/K_PARTS/VLOAD/Math::Fast) isolated and
in situ, leaving "(b) occupancy/scheduling outside the tile" untested. This
session closed it in situ: `LISA_VERIFY_QMM_TILE=bn:kp` (host-side A/B seam,
landed) × `verify-audit S7 @16k`, interleaved stock-vs-candidate pairs,
steps 16, ~1 min cooldowns (the boot drifts ±40 % — in-pair deltas only):

| tile | stock pairs (ms) | cand pairs (ms) | verdict |
|---|---|---|---|
| 1:2 (BN=1, 4× threadgroups) | 76.1 / 81.9 | 100.8 / 106.4 | **loses +30 %** |
| 2:4 (kp=4) | 70.3 / 79.3 | 73.7 / 75.9 | wash |
| 1:4 | 66.2 / 84.0 | 110.6 / 103.1 | **loses +60 %** |
| 2:8 (kp=8) | 58.2 / 64.3 | 79.8 / 77.4 | **loses +30 %** |
| 4:2 (BN=4 @M7) | 60.6 / 131.2 | 1769 / 2114 | **stack-spill (specs/15 §2 reproduced in situ)** |

The shipped (BN=2, kp=2) tile wins or ties every interleaved pair. Threadgroup
count and warps-per-threadgroup are not the residual; the "wide-tile BN=8,
pack-gated" reopen idea is dead at M=7 (NACC=56) and unmeasured-worse at every
measured shape.

### 7.3 Verdict

- **Verify path (the MTP round's bulk): CLOSED.** The splitk tile is at its
  measured floor; the occupancy axis is falsified in situ; the phase kernels
  are cheap (real-exec µs class) and their in-situ slot price is not
  reachable by kernel redesign (falsified twice: specs/01 §5 micro-ports,
  this session's row-parallel hypothesis).
- **Serial path: the 417-GB/s qmv deficit (~8 ms/step) is REAL but
  STRICT-locked.** specs/00 pins serial M=1 bit-exact; any reassociation
  (splitk, stock qmm_splitk) is illegal there, and a load-width-only tweak is
  an issue-rate change on a DRAM-bound kernel (specs/04 wave 3's measured
  wash class). No legal kernel lands this session.
- **What remains named:** (a) the S≥8 NAX verify lane (specs/08) — the width
  lever the reference's 191.5 actually rides; its reopen condition ("the
  verify forward must be flat across 9-25 rows") is unchanged; (b) a
  clock-controlled re-measure of the serial/verify streaming rate — this
  box's ±40 % sustained-load drift straddles the 417-vs-548 arithmetic, and
  any future GB/s claim must come from in-pair interleaved boots at settled
  clocks; (c) an xctrace ground-truth timeline of the verify forward to price
  the per-launch slot independently of the CB counters.

Landed this session: the instrumentation (gpu-slot M=3/7 phase shapes in
`bench_gpu_slot`; `LISA_VERIFY_QMM_TILE` seam + pure parser pin), the in-situ
falsification tables above, artifacts under `bench/data/probe7/` (censuses +
sweep logs). Goldens 310/310; `verify_qmm*` lib tests green. No production
kernel changed — the default tile and every dispatch are byte-identical.

## 8. The serial split-K qmv port — unlocked by portage-1:1, falsified in situ

specs/00 §1 replaced the bit-exact pin with portage-1:1 (reorders legal, goldens
re-capturable), which unlocked §7.3's "STRICT-locked" serial deficit for one
measured attempt. The port: the reference's split-K accumulation structure on
the serial M=1 GEMV — `shaders/common/matmul/verify_qmm.metal`'s
`affine_verify_qmm_splitk` body instantiated at MROWS = 1 (K partitions of whole
quantization groups across K_PARTS simdgroups of one threadgroup, fp32 partition
accumulators, per-part lane-0 spill + barrier + part-ordered combine, single
bf16 write — the in-kernel ordered reduction; no separate reduce dispatch is
needed at M=1 since one threadgroup owns all K parts of its columns). Lane
`jit::qmv_splitk_lane`: 4-bit, N ≥ 512, N < 100000, K % 64 == 0, N % 2 == 0,
K_PARTS = 2 for N ≥ 4096 else 4, BN = 2.

- **Correctness**: with the lane WIRED into `quantized_matmul`'s M=1 branch,
  all three goldens pass unchanged (27B 310/310, 9B 310/310, Flash-Next
  256/256) — the reorder did not flip a single greedy token.
- **In situ, 27B kv1024 paired interleaved audits (base,base vs new,new,
  48 steps, `lisa-bench serial-audit`)**: 36.66 / 37.21 ms median before vs
  **36.75 / 36.20 after** — a WASH (−1 %, inside noise). The lane engaged
  (288 `qmv_splitk` dispatches/step, lm_head 1/step on qmv_fast), so the wash
  is real, not a silent fallback.
- **K_PARTS sweep**: the wider-partition hypothesis (K_PARTS = 4 on the
  wide-N projections, deeper chains) REGRESSES hard: 85.37 / 87.79 ms median
  (+135 %). Falsified.
- **Verdict: NO-LAND.** Per the falsified-lane precedent (specs/15 PPT,
  specs/19 msg), the serial dispatch stays `affine_qmv_fast`;
  `jit::affine_qmv_splitk` + `qmv_splitk_lane` remain as kernel-level opt-in
  infra. Goldens were NOT re-captured: the shipping path's numerics are
  unchanged (the wired-lane run proved the tracked goldens still pass).
- `bench_gpu_slot::bench_qmv_cold_stream` (cold-stream qmv bench, distinct
  ~50 MB weight sets cycled) landed with this port as the isolated-cold
  instrument.
