# 00 — Standing contracts

These contracts apply to the current engine. [AGENTS.md](../AGENTS.md) and the
current task's explicit constraints take precedence over historical campaign
text. Numbered specs record scoped measurements; they do not grant permission
to weaken these requirements. GLM-specific execution and remaining acceptance
work are in [11 — GLM-5.3 Flash](11-glm53-flash.md).

## 1. Numerics

### Strict paths and the verify exception

Serial, prefill, B>1 batching and PLD preserve the applicable reference's
arithmetic, casts and accumulation order **bit for bit**. Matching greedy
output tokens alone does not establish kernel parity. Reduction reassociation,
changed FMA behavior or a fusion needs proof against the actual reference
operation; a performance gain does not relax this requirement.

The Qwen MTP verify exception is limited to `affine_verify_qmm_splitk` at
S=2..8, inside `ops::enter_verify_splitk_scope`: at most one bf16 ULP and equal
argmax against the `qmv_wide` chain. Serial, prefill, batch, PLD and large
output heads retain STRICT. A demonstrated wide-forward near-tie can change a
greedy choice; attribute it to that documented arithmetic class before treating
it as expected behavior. GLM uses DS4's selected shape-specific numerical law;
the Qwen verify exception does not authorize approximate GLM kernels.

Never recapture a golden solely to hide a regression or bless a new reduction.
A new checkpoint needs its own identified baseline. Existing goldens remain
regression evidence and must run twice for affected changes; per-operation
comparison remains the numerical contract.

### Kernel proof and cache keys

- Preserve per-operation bf16 casts, transcendental math mode, FMA behavior and
  reduction order. Equivalent source expressions can compile differently.
- Hardware `simd_sum` is not an XOR butterfly. Match operand layout and the
  reference reduction, not an algebraically equivalent tree.
- Prove fused bf16 transcendental operations against the composed chain over
  all 65,536 input patterns where that domain is sufficient, then exercise
  real activations. Synthetic equality alone does not cover every input path.
- Pipeline keys include math mode, constants and NAX language target. Source
  hashing and same-name/different-source checks belong on cache misses; a
  per-dispatch source hash once reduced measured decode from 30.6 to 14.3
  tokens/s while goldens remained green.
- `LISA_STRICT_KERNELS` makes a same-name/different-source collision fatal.
  Numerical state identity must be available when state is captured and checked
  before it is restored; it must not depend on expensive per-dispatch work.

## 2. Serving surface

The server exposes chat completions, model listing, health, decisions and
embeddings, together with its implemented messages/responses surfaces. Endpoint
availability is distinct from certification for every model. Historical Qwen
conformance does not certify newly changed GLM cache or scheduling paths.

### Determinism and response semantics

- Identical requests and seeds, including `seed: 0`, reproduce results under
  the same numerical law. Streamed and non-streamed responses agree in content,
  finish reason and usage for the same request.
- Stop strings never leak through a partial streamed suffix. `ignore_eos`
  applies both to generation and the serving callback; neither may terminate
  early on EOS when it is enabled.
- `/v1/models` and completion payloads identify the loaded checkpoint rather
  than a fixed placeholder.
- Tool calls carry valid arguments and call identifiers; tool-result turns
  preserve the model's prompt-template semantics. Reasoning, content and tool
  channels must not be mixed.
- Structured output supports the implemented JSON-schema subset through
  instruction, extraction, repair and validation, with one corrective retry.
  It is not grammar-constrained decoding; do not promise unsupported schema
  constraints or silently accept invalid output.
- Embeddings use mean-pooled, L2-normalized trunk states. Their dimension is
  model-dependent; the measured 27B case has 5,120 dimensions.
- A persistent session preserves committed state, sampling configuration and
  absolute token position. Reject an incompatible numerical law before reuse.

### Speculation and batching

Qwen's stochastic deterministic-draft path accepts proposal `t` with
`min(1, p_target(t))`; rejection samples from
`normalize(max(p_target - one_hot(t), 0))`. A fully accepted round earns a bonus
draw. Validate this distribution statistically; a different wide-forward law
and different random-draw consumption need not produce the serial sampled
stream. GLM has its own target-MTP transaction and RNG rules, described in
[11](11-glm53-flash.md).

The serving wave policy may route at least four eligible requests through
continuous batching instead of per-stream speculation. Eligibility restrictions
for tools, stops, logprobs, persistent sessions and other response surfaces
remain independent. Preserve each stream's seed, EOS/stop behavior and usage.
Native batched MTP is not implied by support for separate batch and MTP modes.

### Request admission

Server byte-level admission is enabled by `LISA_RAM_CAP_GB`. At wave intake,
a request whose estimated round cannot fit is refused before its forward with
HTTP 503 and the budget information; the server remains usable. A refusal is
not an internal engine error. This opt-in request policy is separate from the
always-armed resident GLM allocation reservation described below.

## 3. Machine safety and memory ownership

- Default budget: detected physical RAM minus **8 GiB**. Read
  `sysctl hw.memsize` and `memory_pressure` before a measured run. Use
  `LISA_RAM_CAP_GB` to choose a lower explicit budget. Never substitute a fixed
  machine capacity or assume detection succeeded.
- Load one model at a time. GPU tests, benchmark arms and inference runs are
  sequential. Stop residual processes after an abort before another load.
- Attribute memory by owner: mapped model storage, private tensor buffers,
  live cache state, retained captures and reusable scratch. RSS, physical
  footprint and allocator accounting measure different things; explain their
  differences rather than requiring them to be numerically identical.
- GLM's lifetime reservation counts unique mapped model bytes once plus private
  pool bytes against `memory::ram_budget()`. Admission happens before a new
  allocation. Completed unused buffers may be reclaimed; failure must be
  actionable and must not wait for GPU completion while holding an allocator
  lock. Dropping one reservation must not reset another model's reservation.
- `active_memory()` includes private buffers and model reservations;
  `private_allocated_memory()` excludes reservations. `peak_memory()` currently
  reports live allocation, not a high-water mark. Do not label it a peak.
- The pool defaults to a 512 MiB reuse limit; this is not the model RAM budget
  and does not bound live arrays to 512 MiB. `set_cache_limit` controls a
  separate growth-triggered trim in LISA's own runtime, not an external MLX
  allocator. An unlimited reuse setting is a diagnostic choice, not the
  production memory policy.
- Row-joined weights must release redundant originals or retain views of a
  shared owner. Any extra retile, draft head or fused copy needs explicit byte
  accounting and measured benefit. Model loading must not duplicate a full
  checkpoint silently.
- CPU imports and clears require completed ownership before reusing a buffer.
  GLM's scoped ordered GPU reuse may reuse an unleased output only after prior
  reads are encoded on the same serial queue. Partial writers need an ordered
  clear; only individually audited full writers may omit it. Captures retain
  leases on any values needed later.
- A model's buffers and residency declarations outlive every GPU user.
  Teardown releases mappings, state and reservations after ordered completion.
  Verify failure and cancellation paths as well as successful runs.

## 4. Execution and state

### Batching

- Preserve the padded-cache admission cap (`MAX_PAD_WASTE = 1.5`) and per-row
  live windows. A row attends its own committed positions; retirement and
  repacking must preserve retained rows exactly.
- A lone stream, or a model that declines native batching, uses its supported
  serial path. Record the actual native/fallback branch in validation.
- Chunked prefill can yield decode work between chunks. Interleave credit and
  chunk size are scheduler policies; changing them requires output/state and
  latency validation, including late arrivals and disconnects.
- A performance claim needs row identities, per-slot results and aggregate
  throughput. A batch speedup cannot excuse corrupted rows.

### Qwen kernels and generation

- Gate/up and q/gate merges retain their proven shape gates. Do not assume a
  row-concatenated GEMV has the same tiled-prefill arithmetic.
- Qwen's normal serial decode aims for one token readback per step. GLM instead
  follows DS4's submission and encoder boundaries. Neither path may gain
  unaccounted host synchronization inside its hot loop.
- Qwen verify's split-K kernel covers M=2..8 under its verify scope and supported
  4-bit/group-64 geometry; large output heads stay on their stricter route.
  Kernel availability does not change the public PLD depth clamp of six.
  The MTP copy proposal limit is a separate setting and currently defaults to
  seven; do not conflate copy length, trained draft depth and kernel tile width.
- Keep committed offsets and attention masks consistent. Rollback must restore
  all affected recurrent and attention state, not only a visible length.
- A session continuation feeds only the unseen suffix. Compare its next output
  and committed state with an equivalent full prefill under the same law.
- Key sampler draws by their specified seed/position/token identity. Validate
  unsupported sampling configurations before starting a request.

### Prefix cache and PLD

- Bind RAM entries, persisted spill and live sessions to the model's numerical
  law. GLM provides an explicit version before lazy compilation, including Q8,
  DSA, fusion, NAX and actual projection modes retained at load. Models without
  an explicit law retain the runtime fingerprint mechanism.
- Tokens alone are insufficient for a cache hit. Same-prefix entries from a
  foreign law must not win lookup, rewarm or restore.
- Snapshot only at clean commit boundaries, with all required attention,
  recurrent and draft state. Enforce both byte and entry budgets.
- Preserve the existing long-prefix/chunk-boundary insertion policy when
  changing the Qwen cache. Finer boundaries need independent numerical proof.
- PLD uses a rolling 3-gram index and verifies proposals. Reject self-overlap and
  insufficient committed source continuation (`MIN_COPY_CONT = 2`). Rejecting
  a proposal changes cost, not the emitted target token.
- The public PLD control clamps depth downward to six. A non-greedy request
  follows the explicitly supported sampling policy; it must not enter a
  greedy-only shortcut silently.

### Qwen MTP controller

Qwen's coarse 2-bit/group-64 draft head shortlists 32 candidates, which are
rescored through exact trunk-head rows. The verify forward commits the accepted
prefix and target correction/bonus. Draft history, GDN and attention caches
must truncate to the same commit point.

Cost decisions use trusted samples. `TABLE_MIN_SAMPLES = 3`; seeds are not
measurements. Read the nearest trusted KV bucket with the lower side preferred.
Maintain per-depth acceptance EMAs so shallow observations do not suppress
unobserved deeper proposals. Use expected emitted tokens capped by the remaining
budget when comparing depths. The detailed historical correction is in
[09 — Adaptive depth controller](09-serial-probe-port.md).

### Model loading

Resolve a local directory or HF repository ID to one checkpoint; fail clearly
on ambiguity or unsupported model type/quantization. Read the checkpoint's
actual dtype, shapes, quantization groups and per-tensor configuration. Do not
silently dequantize into a different model representation.

Ordinary safetensors loading uses mappings rather than full-shard host copies.
GLM resident loading requires a validated aligned index and shared backing
owners while preserving original MLX payload bytes and metadata. The explicit
CPU-only `lisa prepare --model /path/to/checkpoint` command creates that cache;
`--plan-only` reports its disk requirements. Resident decode may not fall back
to per-expert host copies.

The Qwen35 draft-head requantization/cache is an explicit model-specific
optimization; the exact trunk head remains the verifier. It does not authorize
requantizing GLM weights. Golden baselines are checkpoint-specific, not merely
`model_type`-specific.

### Commands and validation

Use CLI help for the current command inventory. User operations belong in
`lisa`; compiled instruments and embedded goldens belong in `lisa-bench`.
Before committing, follow the release workspace build, sequential library tests
and smoke sequence in [AGENTS.md](../AGENTS.md). Run affected numerical cases
twice, then before/after traces and interleaved end-to-end performance checks.
Model-dependent tests that skip without local data do not certify those cases.

Compare matching workloads, timing boundaries and token-count denominators.
Diagnostic captures, per-kernel probe mode and competing GPU work invalidate
scored throughput. Isolated microbenchmarks guide investigation; in-session
measurements decide a performance change.

## 5. Historical serving anchors

These are scoped Qwen campaign measurements, not current guarantees across all
models or machines: single-stream decode 53.3–57.8 tokens/s, 269 ms TTFT, and
603.4 prompt tokens/s on a 10.4K prompt. The run drifted thermally by 22.4%.
B=4 wave batching later reduced the measured worst-stream TTFT from 7,069 to
1,302 ms. See [01](01-wave-eos-ev-round.md), [03](03-draft-long.md) and
[05](05-ttft.md) for conditions and limitations.
