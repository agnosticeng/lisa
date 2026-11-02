# 11 — GLM-5.3 Flash: architecture, DS4 contract and status

Updated 2026-11-02. **Full port acceptance remains OPEN.** Resident execution
and native Q8/DSA/paired schedules are implemented and have the model-level
proofs below. Those measured cases do not certify every context, state or
serving transition.

## Scope and reference

LISA executes GLM in Rust on its own Metal runtime using the original MLX
checkpoint values. The execution reference is
[antirez/ds4](https://github.com/antirez/ds4) at commit
`b4e3faf5262b21db94ec309cf321640ff1eed460`. The admitted same-weight adapter
source fingerprint is
`b844de0fac75a8b02c4f30306099c059ace5572795af406c8852a30446c3a6dd`.
Record both identities when reproducing a comparison; an upstream commit alone
does not identify the affine adapter.

The adapter permits tensor lookup, original MLX affine storage access and
capture hooks. Preserve original payload bytes, dtype, shape, quantization
group, scales and biases. Requantization or widened metadata is not an allowed
shortcut. Independently validate shared adaptation code: if both engines use
the same affine helper, their agreement cannot prove that helper correct.

Same-weight adapter parity, comparison with an unchanged native DS4 kernel,
and performance of upstream DS4 on a different GGUF checkpoint are separate
claims. [Sushi](https://github.com/beamivalice/sushi) is an optimization reference;
DS4 remains the GLM numerical and execution reference. Refresh and pin source
before importing an upstream change. See [NOTICE](../NOTICE) for attribution.

The target is DS4's default resident GLM text engine. Its optional SSD streaming
policy, other model families and vision support are outside this port. Do not
substitute a DS4 subprocess for LISA execution.

## Implementation map

Checkpoint identifiers `glm5_next` and `glm5_next_text` are external contracts.
The exercised configuration has 45 layers: 34 KDA and 11 sparse-attention
layers, with mHC residual streams, 288 routed experts from layer three and one
MTP block. Runtime geometry comes from validated checkpoint configuration.

| Responsibility | Implementation |
| --- | --- |
| Configuration and tower orchestration | [GLM model](../crates/lisa-engine/src/models/glm53/) |
| KDA recurrence, convolution and rollback | `kda.rs` and GLM Metal kernels |
| DSA indexer, pooling, dense/sparse attention and compact caches | `sparse.rs` and GLM Metal kernels |
| mHC split/mix/expand | `mhc.rs` and GLM Metal kernels |
| Dense/shared/routed MoE | `moe.rs`, address-based dispatch and grouped workspace |
| Original tensor lookup and resident ownership | [loader](../crates/lisa-engine/src/core/loader.rs), [weight arena](../crates/lisa-engine/src/core/weight_arena.rs) |
| Native Q8 projection wrappers | [tiny Q8](../crates/lisa-mlx/src/jit/glm53_q8_tiny.rs), [paired Q8](../crates/lisa-mlx/src/jit/glm53_q8_pair.rs), NAX prefill wrappers |
| Metal operation bindings and source | [GLM bindings](../crates/lisa-mlx/src/models/glm53/), [GLM shaders](../crates/lisa-mlx/src/shaders/glm53/), common matmul/attention shaders |
| Generation, session, prefix state and batching | Shared [engine core](../crates/lisa-engine/src/core/) through `LanguageModel` |
| Reusable numerical fixtures | Owning [engine fixtures](../crates/lisa-engine/src/models/glm53/fixtures/) and [runtime fixtures](../crates/lisa-mlx/src/models/glm53/fixtures/) |
| Production-path measurement | `lisa-bench glm53-run`; use `--help` for current arguments |

## Resident weight and memory contract

Production loading requires `.lisa-resident/index.json` inside the model
directory. Use the native, CPU-only preparation command on a local checkpoint:

```sh
lisa prepare --model /path/to/checkpoint --plan-only
lisa prepare --model /path/to/checkpoint
```

[Preparation](../crates/lisa-engine/src/core/prepare.rs) reads bounded headers
for `--plan-only` and reports disk requirements without loading tensor payloads
or creating files. The full command streams unchanged payloads into aligned
storage, verifies them through SHA-256 readback and publishes atomically. It
refuses to overwrite an existing resident directory and leaves the original
checkpoint intact. No model load, Metal initialization or external conversion
helper is required. The prepared index is a derived cache; the original MLX
payloads, dtypes, scales and biases remain authoritative.

The measured checkpoint contains 113,127 tensor records and maps
**102,033,588,224 bytes (95.03 GiB) through three shared Metal views**. Tensor and
expert slices retain the shared owners. Runtime checks validate source inventory,
relative paths, opened-file identity, ranges and alignment; full payload hashing
belongs to preparation, not every model load or dispatch.

Expert addresses and residency remain stable throughout model lifetime. Resident
decode must not perform selected-expert CPU readbacks, staging joins, per-expert
weight mapping or address-table rebuilding. The older bounded selected-expert
cache is not the default resident policy and its old timings are not current
performance results.

Mapped bytes are reserved once and combined with private allocations against
`memory::ram_budget()`: detected RAM minus 8 GiB unless an explicit lower run
budget is selected. Allocation refuses before exceeding the budget. Grouped
MoE reuses gate/up/activation/down workspace; other GPU outputs use a scoped,
ordered reuse policy with live array leases. CPU writes still require completed
ownership. Partial writers receive an ordered clear; only audited full writers
omit it. Teardown and failed loads must release owners and reservations after
GPU users finish. These GLM mechanisms do not change Qwen's allocation policy.

## Numerical and scheduling contract

Follow DS4's actual source branches when prose descriptions disagree with code:

- Preserve mHC, KDA, indexer, attention, routing, expert sums and output-head
  arithmetic, including FMA, casts, reduction order and metadata precision.
- Native Q8 uses its row-count-specific M=1, M=2..16 and larger-prefill
  schedules, including the large-output-head SIMD-group branch. Paired
  gate/up/SwiGLU retains its supported geometry and exact affine residual terms.
- Preserve BF16 small-row/tiled boundaries, grouped MoE eligibility, 2048-token
  prefill chunks and one-token tails, DSA pool completion, dense/sparse
  selection, value-projection slice limits and selected-index bounds.
- Keep DS4's command submission cadence and encoder-only boundaries. Ending an
  encoder is distinct from committing or waiting on a command buffer.
- Diagnostic captures must retain production dispatch or explicitly compare a
  shadow replay. Captures that change dispatch do not certify the production
  path, and capture timing is not performance evidence.
- `LISA_GLM53_Q8_GATE_UP=separate` retains separate projections, selected at
  model load. Native Q8, DSA and supported paired fusion are defaults.
- Numerical-law revision **4** identifies Q8, DSA, paired fusion, NAX and the
  actual projection-storage modes retained at load. Bind prefix RAM/spill
  entries and live sessions to it before lazy compilation. Reject changes
  before state reuse; never infer loaded mode from a later environment read.
  Revision 4 also covers native MTP KV-A/output-head dispatch, the DS4
  indexed-attention fallback for private windows above 2,051 rows, and the
  GLM generated-token decode graph shared by serial generation and MTP seed,
  rejection replay and tail steps. Prompt and suffix ingestion remain prefill.

No relaxed tolerance is authorized for GLM serial, prefill or batch by Qwen's
separate MTP verify exception. Bit-exact output IDs alone remain insufficient:
compare requested full-vocabulary vectors and state at the relevant boundaries.

## State, batching and serving contract

Two MTP drafts may commit three output tokens. The resumable transaction owns
carry/proposal tokens, trunk and private MTP caches, sampling mode and RNG state
across scheduling calls. Preserve remaining-token budgets, rejection/correction,
EOS and tool-mode transitions. Token equality is not a cache-state proof.

Exact MTP uses DS4's probability-building and categorical-draw order. The
completed speculative cycle consumes its acceptance/residual RNG before the
parser can request a rewind. A parser-mode rewind replays the committed token,
uses the replayed target logits, clears the proposal, and takes a plain target
step before drafting again. Preserve the reference's private-tape writes from
the discarded cycle. Positive-temperature `top_k=1` still consumes the native
sampling RNG. Any failed transaction invalidates the session for reuse/export.

CLI and HTTP retain LISA's established seed handling, including deterministic
HTTP seed 0; native DS4 randomizes omitted/zero seeds. Cross-engine RNG
comparisons therefore use the same explicit positive seed at most 2^53, with
seed-0 checks scoped to LISA's API contract. Original MLX weights are unchanged.
LISA's `ignore_eos` also retains its configured sampler while suppressing EOS
termination; native DS4 HTTP instead selects a non-EOS argmax. Forced-length
runs must record that API difference and cannot certify stochastic equivalence.
The engine commits accepted/replacement EOS like DS4's native cycle API.
HTTP sessions select `StopBeforeVerified`: accepted EOS preserves the completed
cycle's private writes but rewinds target state to the carry, and a residual
replacement EOS leaves the replayed carry unextended. EOS remains in the
generation result for stop reporting, as distinct from committed HTTP state.

Keep independent row histories through native batch and serial fallback,
retirement, repacking and later admission. DS4 does not provide native batched
GLM MTP; two served MTP sessions exercise scheduling, not an invented batched
verify contract.

A configured context limit is an admission boundary. Test the final legal
append, MTP fallback when a full draft does not fit, and refusal of an
out-of-range append without state mutation. Internal raw-tape compaction and
compact-cache behavior are separate from that limit. Do not require a context
ring rollover that the reference does not support.

Match prompt IDs and supported response fields for reasoning modes, tools,
forced choice, tool-result turns, stops, streamed/non-streamed responses and
prefix reuse. Include late arrivals and disconnects; a completed single-user
request does not establish concurrent serving equivalence.

## Measured status — 2026-11-02

The source audit refreshed upstream on this date. The checked-out oracle is
`b4e3faf5262b21db94ec309cf321640ff1eed460`; fetched `origin/main` is now
`890898cfb437ed256be58ebcf7c58a6dfa8f2f53`. The intervening GLM-named change
is scoped off the GLM path, so it does not alter this oracle. At the initial audit checkpoint,
the revision-3 MTP corrections (projection dispatch, wide private attention,
sampling/RNG, parser transitions and failure invalidation) had not been
validated by a new model run. The MTP measurements below predate those
corrections and do not certify them. The follow-up record below covers the
subsequent decode-graph and kernel work on the current source.

Compilation for this correction pass: `cargo check --workspace --all-targets`
passes with zero warnings, and the new wide attention entry point compiles
through Metal's runtime compiler with default options. No test, inference,
benchmark or GPU dispatch was executed. The existing toy rejection-replay
fixture now injects its verifier-logit perturbation under separate one-row
head calls and asserts the verifier parent alongside the replayed next token;
its updated assertions have not been executed.

These results identify the admitted source above. Older campaign evidence is
valid only for unchanged dependencies and its recorded inputs; it does not
silently certify the current native path.

| Gate | Observed result | Scope |
| --- | --- | --- |
| 2,082 prompt tokens / 32 greedy outputs | 4,956,160 / 4,956,160 F32 logit values exact | Complete native path |
| 8,328 prompt tokens / 32 greedy outputs | 4,956,160 / 4,956,160 F32 logit values exact | Complete native path |
| Rebuilt default 2K/32 repeat | All 4,956,160 values exact | Candidate flags unnecessary |
| 2K/32 depth-two MTP | All 32 IDs agree across LISA MTP/serial and DS4 MTP/serial | Token agreement; full state not proved |
| Final workspace checks | 160 engine, 114 runtime and 18 serving tests plus smoke pass; 17 ignored | Ignored/data-gated cases are not certified |
| Final plain 2K/32 default run | 11.005 s prefill; 18.87 decode forwards/s; 12.648 s total generation | One production-path confirmation |
| Memory after 2K timing run | 95.03 GiB mapped plus 5,101,128,704 private bytes | Final allocation, not peak |

Focused tests cover resident reservation, encoder boundaries, shared workspace,
ordered reuse, target-MTP scheduling and numerical-law rejection. Native Q8
fixtures cover wide/tiny branches and metadata types. Paired fixtures include
nonzero affine residual bias; centered synthetic codes alone missed a real
weight mismatch. DSA fixtures cover scalar/matrix/fused boundaries, including
2,051 selected entries. These checks are scoped to their fixtures.

### Performance evidence and limits

One sequential same-weight DS4/LISA/LISA/DS4 block on the 2,082-token prompt
with 32 outputs measured LISA prefill at **8.972/10.292 s**, versus approximately
**9.670/11.842 s** reconstructed from DS4's reported rates. LISA's median
prefill latency was 10.4% lower. Reconstructed median total generation was
**11.230 s for LISA / 14.422 s for DS4**, 22.1% lower for LISA in that block.

LISA reported **19.78/19.03 decode forwards/s**; DS4 reported **9.13/8.36
output tokens/s**. LISA counts 31 forwards and excludes callbacks; DS4 counts
32 outputs and includes emission. Do not turn those raw rates into an exact
speedup ratio without aligning intervals and denominators. Page-cache state was
not controlled, so model-load and first-touch equivalence remain unproved.

A separate LISA old/new Hello/32 block measured old decode **3.4/2.9 tokens/s**
and new decode **20.8/23.1 tokens/s**, with identical emitted text: rounded-rate
medians **3.15 → 21.95 (6.97×)**. This is a LISA before/after result. It does not
compare upstream DS4 or certify token IDs by itself.

Final Hello/8 traces reduced selected-ID readbacks, stage joins and cache stages
from **336 each to zero**, weight mappings **4,746 → 0**, metadata mappings
**9,492 → 0** and address-table rebuilds **329 → 0**. These emitted-span counts
show the removed host work; traced duration is not a scored throughput result.
Depth-two MTP took **14.226 s** on the 2K/32 case and did not demonstrate a
speedup over serial.

### Follow-up correction and evidence — 2026-11-02

The main correctness issue found in this pass was that Session serial
generation fed each generated token through the prompt-prefill path, while
DS4 and LISA's MTP seed, rejection replay and tail used the normal decode
graph. GLM KDA therefore saw a different schedule after the prompt. Generated
tokens now use `decode_token` in Session, serving logprobs, round-cost and
SerialAudit paths; GLM implements decode with its normal one-token forward,
while prompt/suffix ingestion stays on prefill. Other model implementations
retain the former default behavior. The GLM cache-law revision is now 4.

The final release workspace build and `lisa-bench smoke` pass. On the final
source, focused checks pass: Q2 projection/reference checks **3/3**, BF16
projection checks **8/8** (twice), and Session/MTP transaction checks **18/18**.
Two real-model rejection replays took the traced reject-and-rollback branch
(draft 13, replacement 12089) and ended with committed target-cache contents
and layout bit-identical to serial after the replay tail. Captures on the same
39-token prompt produced 128 output IDs identical to frozen LISA serial/MTP
baselines. Those captures are LISA regression checks, not independent DS4
equivalence; they preceded removal of an experimental Q2 two-row kernel, and
the final retained Q2 dispatch was separately checked exactly against the
scalar reference.

The retained decode changes are shape-gated: BF16 row sharing applies only to
two rows with output width at least 1024; Q2 expert gate/up uses the exact
paired route and the down projection hoists affine metadata by group. An
experimental shared-expert Q2 MTP route passed scalar comparisons but showed
no repeatable end-to-end benefit and was removed. The remaining accepted-wide
MTP-vs-serial cache diagnostic reports 91 differing groups after the short
real-weight probe. This compares the one-row serial decode path with the
two-row verify path, which use different target kernels; it is not a DS4-vs-LISA
comparison. Both paths emitted the same nine tokens, next greedy token, and
multi-turn continuation. The first reported differences were KDA F32 state
rounding deltas below 1e-6. A separate accepted-cycle comparison against DS4
matched the trunk transaction across 112 arrays and the private MTP cache
exactly for its captured fixture (see the local
`bench/glm53/reports/ds4-mtp-private-cache-bitexact-20261031.txt`). Broader
accepted/rejected cycles, prompts, cache rollover, batching, and serving still
need direct DS4 state comparisons; the serial-vs-verify diagnostic does not
establish a remaining DS4 state delta.

The first DS4/LISA comparison recorded before these fixes was **32.16/30.29
tokens/s** for DS4 serial/MTP and **20.34/17.37 tokens/s** for LISA serial/MTP.
MTP acceptance was close (DS4 69.3%, LISA 68%), so acceptance scheduling does
not explain that gap. This is not a controlled same-weight comparison:
DS4's GGUF stores Q/K in Q4_K and V/O in Q8_0 (3.45 GiB measured payload),
while LISA's MLX checkpoint stores Q/K/V/O in BF16 (8.5 GiB estimated
payload). The LISA projection payload is about 2.5x larger; this is the leading
remaining bandwidth/compute explanation, not evidence that the two engine
implementations receive the same projection weights. Payload size is not a
measurement of DRAM traffic.

The latest curl A/B blocks are not performance acceptance evidence. GPU clocks
and thermal state drifted substantially during the sequence, and the apparent
decode differences were small and inconsistent. A diagnostic probe comparing
the pre-fix and candidate source snapshots measured serial host phase means of
66.72 → 54.09 ms and MTP verify means of 91.92 → 77.95 ms. Summed GPU command
buffer intervals per phase changed 69.21 → 56.69 ms for serial and 112.28 →
97.27 ms for MTP. The candidate includes the decode-graph correction and the
BF16/Q2 kernel changes together; the probe also changes submission/fence
behavior, and command-buffer intervals are not exclusive hardware counters.
These numbers localize improvement but do not establish end-to-end throughput.
Therefore no current speedup over the pre-fix LISA baseline, and no DS4
performance parity, is certified. A controlled comparison must first use
equivalent projection precision/weights or report the weight format difference
explicitly, then repeat warmed and first-touch A/B blocks under the same
machine conditions.

### Short-verify projection follow-up — 2026-11-02

The two-row MTP verify path now uses the existing fused mHC RMSNorm plus BF16
projection when its flattened row count is 1–8. The two-row fixture compares
the fused result bit-for-bit with the separate DS4-order RMSNorm and GEMV
chain. A 64-token production MTP request also returned identical output IDs
with and without this route.

Performance was noisy. Two fresh-process A/B/B/A blocks used the same 12-token
raw prompt, 64 greedy outputs, depth 2, 2K context, model snapshot and seed.
Across all four runs per side, median generation time was **2.9212 s without /
2.8972 s with** the fused mHC verify projection (**−0.82%**). The first block
alone appeared to improve 5.67% because its first baseline run took 3.215 s;
the second block improved only 0.58%. Treat this as numerically validated with
no measured regression, but **not a certified throughput gain**.

A second proposed fusion routed MTP's `[1,2,H]` KDA inputs through the
single-token fused six-projection kernel. Its projections matched the existing
DS4 two-row fixture bit-for-bit and all four 64-token output sequences matched.
However, its two-run A/B/B/A median was **2.902 s for the current path versus
3.328 s for the proposed fusion** (**+14.68% slower**). That code was removed.
The high-cost MTP families remain address-based Q2 routed-expert gate/up and
down projections plus BF16 two-row projections; no further exact fusion has
yet shown a repeatable end-to-end benefit.

### Same-weight short MTP comparison — 2026-11-02

One sequential DS4/LISA/LISA/DS4 bracket used the same MLX-derived checkpoint,
raw 12-token prompt, 64 greedy outputs, 2K context and depth 2. DS4 reported
**7.16 output tokens/s** in both runs. LISA completed in **2.8932/2.8900 s**
(**22.12/22.15 tokens/s** including its request generation interval); its
post-first-token interval works out to **25.43/25.43 tokens/s**. LISA's timer
includes TTFT/prefill while DS4 reports its own generation rate, so these are
not directly comparable rate intervals. DS4's per-cycle trace showed roughly
236–239 ms for verify-two and about 24.5 ms for accepted head/draft work.

The DS4 binary was built from the local affine adapter working tree at
`b4e3faf5262b21db94ec309cf321640ff1eed460` (binary SHA-256
`dca4ee7cecaec8ae539f51689af02977bec7b03eefa156007ea88125e9835b49`). The
working-tree source fingerprint is `de6f09b6fd8df4cf563ae97380a92379cdc97eb8c2db4eb250071e27d273ba36`;
it differs from the admitted bundle fingerprint because `ds4.c` has since been
locally modified. The DS4 adapter retained the MLX affine payload and metadata,
and both systems tokenized the prompt to the same 12 IDs. Therefore the bracket
shows no measured LISA performance deficit on this short MTP case, but the DS4
source identity mismatch and differing timer boundaries keep it diagnostic,
not a parity acceptance result. The raw record is in the local-only
`bench/glm53/reports/current-short-mtp-ds4-adapter-abba-20261102.txt`.

## Finite remaining acceptance matrix

Reuse an existing result if source, reference, inputs and affected dependencies
are unchanged. Add a cell only for a newly identified source branch or a concrete
missing invariant; explain why. Run affected numerical cases twice and all GPU
jobs sequentially under the machine-safety rules.

| Cell | Required coverage still tracked |
| --- | --- |
| Source/adaptation | Complete upstream-to-LISA branch map; independently validate shared affine Q2/Q8 adaptation and retain reproducible reference identity |
| Ownership/memory | Prepared tensor integrity, declared maximum context/batch footprint, allocation refusal, failed load and teardown |
| Kernel boundaries | Every mapped selector; rows 1/2, 8/9, 95/96, chunk/tail, pool completion and DSA position 2051 |
| Full model | Existing short/32, 2082/32, admitted 8328/32 and 16K final-logit cases; rerun only affected cells |
| MTP | Accept/reject, configured/exact sampling at one explicit positive seed (plus LISA seed-0 contract), tool-mode transitions, budgets 1/2/3, EOS cases and the existing 16K/512 compaction replay; compare decisions, RNG and committed trunk/private state |
| Cache/context | Prefix save/restore/append and spill/reload, raw-tape compaction, last legal append, MTP fallback and nonmutating over-limit refusal |
| Batch | Equal/ragged/reversed B=2, B=4 cohort, B=8/9 branch fixture, native/fallback, two retire/admit cycles and mixed EOS |
| Serving | Supported reasoning/sampling/tool/stop fixtures, short/8K prefix reuse, two serial sessions, two MTP sessions, late arrival and disconnect followed by a healthy request |
| Performance | Serial short/2K/8K/16K and MTP short/8K, all 64 outputs; short HTTP serial concurrency 1/2/4 and MTP concurrency 2 |

Performance acceptance requires two sequential A/B/B/A blocks per cell and
comparable phase boundaries, outputs, residency conditions and memory reporting.
The working band is at most 5% regression in each comparable metric; unresolved
noise leaves a cell open. Separate warmed-session from fresh-process/first-touch
claims. A faster warm case cannot offset a slower first request or concurrent
case. The current 32-output pilot does not close this matrix.

Completion requires one final source-bound PASS/FAIL table covering these cells,
with reproducible reference/fixture assets and no unreported deviation. Do not
replace it with a percentage estimate or drop a failing requirement.
