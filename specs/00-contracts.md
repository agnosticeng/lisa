# specs/00 — Consolidated contracts

The standing contracts consolidated from the retired capability-spec set,
rewritten in lisa's own voice. These are the rules that outlive any single
campaign: the numerics regime, the serving surface, the machine-safety
discipline, and the kernel / cache / loader clauses that only ever existed as
capability-spec requirements.

Measured results and campaign verdicts stay in the numbered specs (25-32); this
file carries invariants, not narratives — except where a contract *is* a measured
threshold, which is kept.

## 1. Numerics contracts — the portage-1:1 regime (supersedes the bit-exact pin)

### Scope: portage-1:1 everywhere

The old **bit-exact STRICT pin is REPLACED**: reduction reorders are LEGAL on
every path (serial, prefill, batched (B>1), PLD, verify). A kernel that
replaces or fuses into another may accumulate in a different order — the
portage-1:1 rule is 1:1 behaviour against the reference's accepted numerics
class, not word-level identity. Goldens are RE-CAPTURABLE: after any numerics
change, re-capture the golden baselines and the new captures become the
reference (run the capture twice and diff for determinism before trusting
one). Evidence duty is unchanged: a kernel change ships only with goldens run
to 100% on all tracked models, and with an in-situ measured win — a wash
ships nothing.

The MTP verify forward keeps its **tail-ULP class** (≤ 1 bf16 ULP + argmax
equality vs the `qmv_wide` chain, thread-local guard
`ops::enter_verify_splitk_scope`); portage-1:1 makes that class the floor, not
the exception.

### Bit-exactness traps (measured — do not re-learn)

- **Reduction reassociation is caught by the goldens.** A fused add+norm that
 changes the reduction association (2-simdgroup fp32 vs the 1024-lane stock
 order) flipped ~1% of bf16 roundings and the GEMV amplified to **56 ULP** on
 outputs — the golden read **210/310**. Fix by construction: the same hardware op
 on the same operands in the same order.
- **Transcendentals need exhaustive proof.** A fused kernel containing a
 transcendent (silu, sigmoid) proves bit-parity over the **exhaustive
 65536-pattern bf16 input sweep**, not random sampling.
- **Near-ties are governed semantics, not bugs.** A wide verify can round a
 genuine top-2 near-tie (a **0.125 f32 gap** at serial becomes exactly 0.0 at
 S=6); the flip is recorded as semantics, is deterministic across commits, and
 must be attributed to the wide forward's rounding (forced-acceptance
 experiment) before any change is attempted.
- **The goldens are the final net, not the contract.** qwen3_5 310/310 and qwen4
 256/256 pass after every numeric change, run at 100%, twice per landed phase.
 Two engine paths (serial vs speculative, serial vs batched-collapsed) diverging
 on identical weights outside a documented contract class is an engine bug,
 never acceptable model nondeterminism — and a kernel is never forced to parity
 just to make a golden pass.
- **Fused kernels need explicit per-op bf16 casts + `metal::precise::exp`** (the
 composed MLX rounds per-op in bf16 with precise math; custom kernels default to
 Math::Jit → ~0.03% divergence). Never simulate the hardware `simd_sum` tree
 with a butterfly — map emulated lanes 16-consecutive-per-real-lane and run REAL
 hardware `simd_sum` over the same 32 partials in lane order. Uniform-draw
 bit-exact proofs do NOT transfer to real activations: a fp32 association
 differing ~3e-5 rel flips ~1% of bf16 roundings and the GEMV amplifies to tens
 of ULP — only in-situ goldens are evidence. FMA codegen differs per kernel:
 source-equivalence proves nothing.

### The compiled-pipeline cache key

The JIT pipeline cache key encodes every input that changes the emitted
arithmetic — math mode, consts, and the NAX language version — besides the
caller-built name. Because the key is computed once **per dispatch**
(`compile_full` / the fast-path lookup), it holds only cheap fields and **no
shader-source hashing**; a per-dispatch cost in the key passes the goldens and is
only admissible evidence via an interleaved end-to-end A/B (measured:
**30.6 → 14.3 tok/s, −54%**). A same-name compile with a different body is
detected (twin-check) and refused under `LISA_STRICT_KERNELS`. The NAX axis
cannot collide: the same kernel name compiled with and without the NAX target
under one math mode occupies distinct cache entries.

The key's `nax` axis was once absent — a latent NAX collision — and the
numerical-law infrastructure (`fnv1a`, `record_pipeline` twin-check,
`kernel_fingerprint`) landed with it; measured non-regression
32.0/31.9/31.7 tok/s decode, no twin-mismatch.

## 2. Serving surface (OpenAI-compatible HTTP)

The server implements chat completions (SSE streaming with usage on request, plus
non-streamed JSON), model listing, a decisions endpoint exposing speculation
state, health, `/v1/embeddings`, tool-result turns, and structured output.
Streamed and non-streamed paths produce identical tokens for identical requests.
Core conformance coverage is **9/9**; every MUST-level gap is either fixed or
recorded as an explicit future requirement, never a silent omission.

### Refusal path (armed only by `LISA_RAM_CAP_GB`)

- When byte-level admission is armed (`LISA_RAM_CAP_GB` set — kill-switched,
 default OFF), a request whose round cannot fit the admission budget is refused
 **at wave intake, before any forward**, answered **HTTP 503** (not 500) with
 the computed numbers in the body and a `Retry-After` hint. A refusal is a
 distinct outcome from an engine error. Admitted bytes are released at the end of
 the step.
- Measured end-to-end: with `LISA_RAM_CAP_GB=8`, `max_tokens=100000` → **503**
 `out of memory budget: round needs 12500 MiB, headroom 8192 MiB`; the server
 stays alive after the refusal; `max_tokens=16` → **200**. With the variable
 unset the identical oversized request is **200** (byte-for-byte the pre-existing
 behaviour).
- The internal **`\u{1}refused\u{1}` sentinel** is what routes the status; it is
 stripped before serialization, so the body carries the human message only.

### Determinism and parity

- The same request with the same seed (including the explicit **`seed: 0`**)
 reproduces its text; streamed and non-streamed agree per surface (chat,
 messages, responses); stop sequences truncate both paths identically. Direct
 tests `test_parity_seed.py`, `test_depth_gating.py`.
- `/v1/models` returns the actually loaded checkpoint id (the HF repo id for
 hub-cached checkpoints, the directory name otherwise), matching the `model`
 echoed on every completion payload.
- Spec-driven generation serves through the same surface: `[mtp.auto]` /
 `[mtp.round]` engagement lines appear in the server log and throughput meets
 the committed bench figures.
- Forced-length decode is not EV-penalized at the tail: each candidate depth is
 priced with the forced-continuation term `min(E[round], remaining)`
 (`DepthController::pick(kv_len, remaining)`), so a full-width verify round that
 cannot fill its width yields to serial.
- Capability quality does not depend on the decode path: divergence between
 speculative and serial serving on identical weights is an engine defect.

### Endpoints and capabilities

- **Decisions endpoint**: picked depth per round, per-index acceptance, EV tok/s,
 and the round-cost source consulted.
- **`/v1/embeddings`**: string or array input → OpenAI shape; the trunk's last
 hidden states are mean-pooled over the sequence and L2-normalized — unit-norm,
 deterministic per input, distinct across inputs (dim 5120).
- **`response_format: json_schema`** (and `json_object`) yields a JSON value
 satisfying the declared subset (types, `required`, `enum`,
 `properties`/`items`, `additionalProperties: false`). The engine has no
 grammar-constrained decoding: prompt instruction + extraction + repair +
 validation with **ONE corrective retry**, and grammar-constrained sampling is
 documented as a future upgrade, not a silent omission.
- **`role: "tool"` turns** round-trip the OpenAI way: declared tools → assistant
 `tool_calls` (valid JSON object args, `finish_reason: "tool_calls"`) → tool
 result referencing the call id → a text answer grounded in the result.
- **Speculation at temperature > 0**: verify accepts a proposed token with
 `min(1, p_target[proposal])` and, at the first rejection, emits a draw from the
 residual `norm(max(p_target − one_hot(proposal), 0))` — exact for a
 deterministic draft (both prompt-lookup copy and the MTP head's argmax give
 `q = delta`); a fully accepted round earns one bonus draw; greedy keeps the
 certified argmax verify. This path is NOT bit-exact and is validated
 statistically, never by diffing against the serial non-greedy stream (at
 temperature > 0 the draw is an argmax over `v/T + gumbel`, so the verify
 batch's ≤ 1 ULP Tail-ULP difference flips it). Rationale: the packed
 checkpoints declare `temperature: 1.0`, so the default request is non-greedy —
 a greedy-only gate would give the default traffic NO speculation.

### Batching and `ignore_eos`

- A wave of **≥ 4** eligible speculative requests is NOT serialized: it runs
 through the continuous batch, trading per-stream speculation for shared work.
 The decision is **per wave**, so a lone request keeps its speculation. Only the
 depth clause is relaxed; requests needing the serial paths (tool calls, stop
 sequences, per-token logprobs, non-chat surfaces, persistent sessions) stay
 serial. The threshold is 4 **by measurement**, not interpolation, and the
 batched path emits different tokens than the speculative one (near-tie class).
- **`ignore_eos: true`** means EOS never ends the reply — the engine keeps
 decoding to the budget and the serving callback does not stop accumulating on
 an EOS token. Both halves were once inverted, truncating forced-length replies
 to ~2 of 64 tokens and invalidating every batched/concurrency measurement.
 `ignore_eos + min_tokens` is exactly what forced-length benchmarks send.

## 3. Machine safety

- **Admission budget = `LISA_RAM_CAP_GB` when set, else detected RAM − 8 GiB,
 always non-zero** (`memory::ram_budget`), stated in the startup log. Never a
 hardcoded constant. The budget is HARDWARE-DERIVED: on a 128 GB machine that
 is 120 GB, on a 64 GB machine 56 GB — it moves with the box, and this document
 must never quote a fixed number as the rule.
- A request whose live state would exceed the budget is refused **before any
 forward** with a 503 carrying the computed numbers (see §2). Enforcement is
 kill-switched: armed only when `LISA_RAM_CAP_GB` is set, so the default path
 never refuses a request.
- The budget bounds **request state, not resident weights**: a 27 GB model under
 `LISA_RAM_CAP_GB=8` still loads (`memory_limit` is informational — its sole
 consumer is the log line).
- **Never two models resident**; A/B measurements and benchmark arms run strictly
 sequentially with cooldown, never as concurrent processes.
- **Memory headroom check before every measured run** (the OS memory-pressure
 report); refuse to start below the required headroom.
- **No residual GPU/host state after a run or abort** — no orphaned engine
 process or pinned allocation; the next run's pre-check observes full headroom.
- Measured campaign-day safety anchors: the long-context runs held
 **86-89 %** of the memory-pressure free fraction; observed peak **73.7 GB
 RSS** in the long-context wave — both inside the discipline above. A
 9B→27B oracle A/B remains blocked by the one-model rule (it needs two
 models resident); owner decision.

### The footprint contract — how RAM is managed so nothing superfluous survives

Every rule below was learned by measurement on this box (128 GB M5 Max, 27B
4-bit). They are contracts, not advice. The ledger lines that verify them are in
Rule 4.

**Rule 1 — the buffer pool is always capped. It is never unlimited.**
`BufferPool::set_limit` defaults to `0`, which means *unlimited*: a buffer
released by a dropped `Array` is returned to the pool and STAYS an allocated
Metal buffer that MLX counts as `active`. An uncapped pool is invisible garbage.
Current cap: **512 MiB** at `MetalRuntime::new` — chosen so the engine's
non-weight footprint stays under 1 GB. The cap must ALWAYS be set: an uncapped
pool parks every released buffer and MLX counts it as `active`, which is how
~2.7 GB of released row-join originals appeared as footprint.
Careful with the two knobs: `set_cache_limit` caps MLX's OWN cache;
`pool.set_limit` caps OURS. Capping one does not cap the other — for a day we
capped MLX's cache while OUR pool held gigabytes.

**Rule 2 — every fused weight is materialized and its originals released.**
Any `cat`/`concatenate`/`gather` that produces a fused weight (a row-join) MUST
be followed by: (a) the joined buffers evaluated, (b) every original rebuilt as a
VIEW of the joined buffer, (c) proof that nothing else holds a reference. The
two row-joins currently: GDN `in_proj_all` (4 parts), MLP `gate_up_proj` (2 parts).
Measured cost of a surviving original: ~44 MB per layer — **2.85 GB on the 27B**.

**Rule 3 — weights are never duplicated at load.**
`serve-warm active` must equal the model's tensor bytes plus the known runtime
set. For the 27B that is **~14.7 GB**. Anything above it is superfluous: name it,
measure it, remove it — never tolerate it as "probably alignment". The trace
that proved this: `t0-mmap-only 14659 | serve-warm 17372` meant the constructors
created 2.7 GB that the weights could not explain, and it took five hypotheses
and eleven falsifications to find the pool.

**Rule 4 — the ledger is the instrument. Judge nothing without it.**
`mlx_mem_line` fires at fixed points: `t0-mmap-only`, `t1-heads`,
`t2-64-layers`, `t3-mtp-head`, `t4-draft-head`, `t5-map-cleared`, `serve-loaded`,
`serve-warm`. Read it as a delta table: the gap between `t0` and `serve-warm` is
the runtime's own footprint and every byte of it has a named owner. Cross-check
with `footprint -p <pid>` — `active`, `footprint` and RSS must agree; if they
disagree the measurement is lying and must not be used. `ps rss` ALONE is not a
comparator across engines (one engine's RSS was 6 GB against a 15 GB footprint).

**Rule 5 — goldens are model-specific, never model_type-specific.**
`model_type` is not a model: two checkpoints share `qwen3_5` (27B, 9B), and
scoring the 9B against the 27B's baseline reported `37/310 MISMATCH` — a verdict
that said nothing about the 9B. The harness picks the embedded golden whose
`model` names the checkpoint under test, and refuses the embedded default on
mismatch. A new model captures its own baseline before it is certified.

**Rule 6 — the teardown assertion reads zero.**
`[load] map teardown: N unclaimed tensors` must read `0`. A non-zero count means
a tensor the model never asks for: a name the accessors miss, or a tensor the
checkpoint ships that lisa does not load. It is the one line that would have
caught the load-time duplication immediately, had anyone read it.

**Rule 7 — the pool cap is the only place RAM may be traded for reuse.**
Anything else that trades RAM for speed (a retile, a draft head, a fused copy)
must state its byte cost and its measured speed effect at the decision point in
the code, as `qgate_retile_enabled` does (−0.77 GB / +3.56 ms on the serial
step). An unmeasured trade is a leak with a rationale.

## 4. Folded clauses from the retired capability specs

These existed only as capability-spec requirements. They are mechanisms and
invariants, not campaign results, so they are kept whole; restatements of
measured campaign outcomes already carried by specs/03-09 were dropped.

### Batching

- **Pad-waste cap at admission.** A stream that would push the group's padded-KV
 waste past **1.5× (MAX_PAD_WASTE)** waits for the next group instead of joining
 the batch. The cap is applied at admission, not per tick — the packed-cache
 layout makes per-tick re-formation a copy storm — and stays **< 2.0**, the only
 range in which a 2-slot group is ever vetoable.
- **Ragged lockstep decode.** Each slot's query attends its own live window
 `[origin_i, offset)` of the packed shared cache (origin from the next-position
 pack invariant); logits are sliced per slot without copy until evaluation.
- **Lone-stream collapse.** With one live stream (or a model reporting
 batch-unsafe), the scheduler runs the exact single-stream path, byte-identical
 to standalone, logging `batched: model batch-unsafe` when it clamps.
- **Prefill/decode interleave.** A cold admission prefill yields decode ticks at
 chunk boundaries (up to **8 ticks per 2048-token chunk**, budget a quarter of
 the chunk's wall time), with an environment kill-switch restoring the blocking
 admit.
- **Per-slot verdict.** Every slot that runs serial logs a reason at retire or
 per tick (`lone stream`, MTP depth, batch-safety clamp, pad-waste deferral);
 one-shot engagement markers fire on the first B>1 step, so a silent shape
 regression cannot ship unnoticed.

### Decode kernels

- The gate|up row-concat merge, the q|gate row-gather merge and the hoisted
 per-step constants are **decode-gated**: they are exact only on the GEMV (S=1)
 paths. Prefill keeps the unmerged stock path — the tiled prefill quantized
 matmul is not row-exact under N-concatenation (it diverges at token 0).
- The serial-step ledger prints wall/dispatch/per-kernel and per-op attribution
 **side by side** because the op-label and kernel-label universes do not join.
 Probe-mode timings (one dispatch per buffer, isolated cold execution) are
 excluded as evidence — cold single-buffer execution inflates small kernels
 ~10-30× (a qmv p50 ~1.04 ms/launch vs ~54 µs in-pipeline).

### Generation core

- Serial decode runs with **exactly one sync per token**: one GPU submission per
 step, one scalar readback per token, and no other fence may gate the step.
- Per-stream cache state keeps committed offsets; the attention keep mask is
 built from committed tokens only, so every emitted token is the target model's
 own output at the correct cache position. After a speculative round accepting
 `a` of `m` drafts, caches roll back to `committed + a`.
- Session restart prefills only the suffix; the next token matches a full prefill
 bit-for-bit (the session-check invariant).
- The sampler keys every draw on **(seed, absolute position, token id)** so
 identical seeds reproduce identical streams; greedy stays the exact
 masked-argmax path, untouched by sampling code.
- Sampling configurations outside the implemented semantics (e.g. a mode needing
 a vocabulary-wide multi-block sort the runtime does not implement on the
 248320-token vocab) are rejected/clamped **up front**, not mid-generation.

### Verify kernels

- **Split-K verify qmm scope.** For **M in 2..=7** and 4-bit gs64 shapes,
 quantized matmul routes through the split-K kernel — one threadgroup owns BN
 columns × all M rows, K reduction over K_PARTS simdgroups, deterministic
 part-order partial reduction — inside the thread-local verify scope. Serial,
 prefill, batched, PLD and lm-head (N ≥ 100000) shapes keep the stock dispatch.
- **Two host-dispatch facts are pinned by measurement — do not "re-fix" them.**
 The grid is `(1, N/BN)` threadgroups with threadgroup `(32*K_PARTS, 1, 1)` (the
 threadgroup-count misreading measured 64× oversubscription). **BN = 2 for every
 M** (BN=4 stack-spills on this compiler).
- The S-row verify attention fuses qk-norm, RoPE and the q|gate de-interleave
 into one dispatch (`track_qk_norm_rope_rows`, one threadgroup per (row, head)),
 bit-identical to the pinned s==1 kernel; serial, batched, masked and non-gated
 paths keep the stock dispatch. The GDN S-row recurrence (S=2..8) runs in one
 launch with per-position SSM/conv state captured for rollback and the
 conv-input rows folded in.
- **Fused add+norm stays unported at D=5120** — geometrically unreachable at our
 width: `threadsFor` gives ceil(5120/4) = 1280 threads > the 1024-thread
 threadgroup max, so add and rms run as two ops. The only near lever is the
 norm-inside-consumer family, already falsified in situ (prologue re-reduction
 per consumer threadgroup loses; TG-memory residency kills occupancy).
- Before counting an elementwise class (sigmoid+bmul, silu gates, casts) as
 recoverable slot price, **probe the LIVE dispatch** — a fused kernel may
 already own the site (the S7@16k bmul+sigmoid pair turned out to be the GDN
 `gated_rms_silu` chain, pinned 0 ULP over 74k values).

### Prefix cache & PLD

- **Law binding.** Each cache entry records the process-wide kernel fingerprint
 at capture (`runtime::kernel_fingerprint`); `insert` drops entries from a
 different law and `lookup`/`restore_session` skip them. A token-exact prefix is
 necessary but **not sufficient** — state from different kernel arithmetic is
 not a valid resume point. The fingerprint folds the registered (pipeline-name,
 source-hash) set once per compile and is stable across a run; it MUST NOT be
 recomputed from kernel source on a hot path (a per-dispatch source hash
 measured −54% decode).
- **Boundaries.** Snapshots are captured only at clean commit boundaries, keyed
 by token prefix, storing full-attention offsets plus GDN (conv, ssm, ple_conv)
 snapshots, bounded by an explicit RAM budget and entry count (GDN state is
 ~113 MB per entry at 27B); same-prefix re-inserts move to front. A mid-prompt
 boundary sits only past **2048 tokens** and on a prefill-chunk multiple —
 finer granularity was rejected (it flips near-ties vs full prefill).
- **Parity.** A restored session is argmax-identical to a full prefill, and only
 the unfed remainder is fed after resume (feeding full ids again doubles the
 offset). `lisa-bench prefix-check` pins this; cross-turn resume measured
 **16.7×** vs a full re-prefill.
- **Radix trie over the prefix — set aside**: its prerequisite (cross-turn
 resume) is already solved by the LRU + snapshot design above.
- **Supersequence / LCP prefix matches — rejected** (they break the
 next-position pack invariant; do not reopen).
- **PLD.** A rolling **3-gram** index (two most recent positions per n-gram;
 a 3-gram key (calibrated)), greedy-only, proposal depth clamped
 to **1..=6** (width `S = depth + 1 ≤ 7`, the widest the split-K verify lane
 covers), with a strong draft capped at 16 tokens after a long match. A
 non-greedy request declines PLD and falls back to the plain serial path.
- **Copy guard.** Before a copy draft is used, the lookup rejects (a) a source
 window overlapping the query itself (`p + k >= query_start`) and (b) a
 committed continuation shorter than `MIN_COPY_CONT = 2`. Rejecting a draft is
 output-neutral — the verify takes the longest accepting prefix, so a rejected
 proposal can only change a round's cost, never an emitted token. Both the MTP
 driver and `pld_decode` route through the shared `core/copy_draft.rs`
 (`CopyIndex` / `copy_lookup_guarded`).

### Speculative MTP

- The draft chain is produced step by step: the MTP head shortlists 32 candidates
 through a coarse 2-bit/gs64 full-vocab readout, then re-scores exactly those
 rows through the trunk head before taking the argmax.
- One `S = depth + 1` verify forward per round with per-position state capture;
 the accept scan emits `a + 1` tokens (the verify's own position-0 token plus
 `a` drafts), every one the target model's own output. Greedy speculation is
 lossless; divergence is only at the documented near-tie semantics.
- Rollback is a **length operation**, not a state copy: full-attention offsets,
 GDN/conv state and the draft head's history cache truncate to the commit point;
 the head history holds exactly the committed pairs, trimmed offset-exactly to
 `committed − 1`.
- The EV controller re-picks depth every round by maximizing expected tokens per
 unit cost, consulting in order: the in-run round EMA, the persisted
 per-(model, width, KV-bucket) round-cost table, then an analytic prior, with a
 table-first gate that never disables speculation into a sub-serial rate (3%
 margin, 5% hysteresis against timing noise, ties breaking toward the deeper
 draft).
- **Bucket reads** skip cells with fewer than **3 samples** (seeds are never
 data), walk to the **nearest trusted bucket with the lower side preferred**
 (cost grows with KV: under-billing beats over-billing), and fall back to the
 analytic prior only when no bucket is trusted; past 32k the qwen3_5 class folds
 into the last bucket.
- **Per-depth acceptance EMAs**: a shared EMA is only the warm-start prior until
 a depth reaches `MIN_SAMPLES`. A shallow-depth incumbent must not drag the
 index-≥2 estimates down — a d2 incumbent never observes index-≥2 hits, and the
 shared-EMA freeze cost ~40% headroom.

### Model loading

- Resolve a HF repo id or local directory to **exactly one** snapshot in the HF
 cache; an ambiguous or missing reference fails fast, before any GPU allocation.
- **mmap** each `*.safetensors` shard of the snapshot and read tensors directly
 from the mapping — no full-shard host copy, so load RSS stays near the weights
 actually touched.
- Read per-tensor quant metadata (group size, bits) from the safetensors headers,
 reject unsupported combinations (naming the tensor and the format) instead of
 silently dequantizing, and dispatch the matching quantized GEMV/GEMM lane
 (4-bit gs64 for the shipping trunk class).
- Build the draft head by **requantizing the trunk lm_head to 2-bit/gs64 at load
 time** (the sidecar ships no low-bit head tensors), one-time and disk-cached
 (`.lsdh`, geometry-guarded); the verify keeps the exact trunk head.
- Bind exactly one tower from `config.json` `model_type` (plus the MTP presence
 fields); unknown model types fail at load. `model_type` strings are external
 checkpoint contracts — never rename.
- Surface **one-shot engagement proof** at first engagement of a loaded
 acceleration, so a run proves which loaded paths actually engaged.

### CLI tooling

- Split surfaces: user commands (`run`, `chat`, `cbatch`, `serve`, `golden`) on
 `lisa`; measurement instruments (`round-cost`, `serial-audit`, `verify-audit`)
 on `lisa-bench` — benchmark tooling never loads into the user path by accident.
- **check.sh gates every landed phase** (workspace build + lib tests + smoke +
 goldens); known pre-existing flakes are named and pass standalone.
- **Instruments are the truth for in-situ decisions**: performance verdicts come
 from interleaved same-session instrument runs with dispatch-count engagement
 proof; isolated single-shot benches are hypotheses until they convert in situ
 (an isolated win that washes in-situ does not ship). An arm without its
 engagement line is void.

## 5. Measured serving anchors (`lisa serve`, sequential, one model at a time,

The probe run drifted thermally (lisa −22.4 %), so read the measured figures.
Single-stream (concurrency 1): decode **54 tok/s** (53.3-57.8), **TTFT
269 ms**, prefill of a 10.4k prompt **603.4 tok/s**; context curve
~512/4.2k/8.2k/16.3k → 45.5/45/37.4/34.5 tok/s; **2.34/2.31/2.34 tokens per
decode step**; prefix cache read "not detected" (1.1×); 4 concurrent streams
serialized (0.26 effective).

Three structural facts verified in our code locate the decode rate — not in
the kernels (prefill and TTFT are already at or above the certified class):

1. **Prefix-cache floor**: lisa only *inserts* for prompts **> 2048 tokens**
 (`lisa-serve/src/lib.rs:2716`), so a 1.5k-token prompt (1507/1538 tokens)
 is never cached — the cache is not broken; its floor is too high. (The
 probe's "not detected" reading and its global `speculative: null` verdict
 are worth one investigation, not two.)
2. **Speculation density**: 2.34 accepted tokens/step, content-bound —
 57.7 tok/s predictable vs 39.6 novel (measured, specs/03).
3. **MTP and batching are mutually exclusive in the serve path** unless the
 wave rule fires: with speculation on, no request is batchable; the
 documented B=4 2.73× exists with speculation off (specs/01 §5). Batching
 *with* MTP (fusing verify across streams) is not implemented — the worst
 4-stream TTFT measured **6367 ms** before the wave rule,
 **1302 ms** after it (specs/01 §5).
