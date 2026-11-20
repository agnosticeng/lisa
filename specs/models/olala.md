# Olala v1.0 mini 7A1B

Experimental native Metal execution of the publisher's **Q5_K_M GGUF** through
the curated UI download, `lisa run` and `lisa serve`. The checkpoint contract is
`general.architecture=olala`. Other quantizations, ROCm,
vision, batching and speculative decoding are not qualified by this path.

```sh
./target/release/lisa run \
  --model OVHaiLLM/Olala-v1.0-mini-7A1B-GGUF \
  --prompt 'Explain what a compiler does in two sentences.' \
  --max-tokens 128
```

Repository resolution downloads only `Olala-v1.0-mini-7A1B-q5_k_m.gguf` and
publisher notices. A local GGUF file or directory containing that filename also
works. There is no conversion step. A 5-bit package includes mixed tensor types:
Q5_K, Q6_K, Q8_0, BF16 and F32. The reader and weight binder retain their original
payloads. Embedding rows are decoded on demand from disk; expert weights remain
packed, with selected experts addressed directly on the GPU.

## Checkpoint and source identity

- [GGUF repository](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B-GGUF),
  revision `b9f5d20adf2ac0aec05e9fa00b057cf26df263fc`.
- File: `Olala-v1.0-mini-7A1B-q5_k_m.gguf`, 5,990,551,040 bytes.
- SHA-256: `ed353d2632c406be9ca2022b35e4c13049d6f8f7390b1c1bb61d6a7d6b971860`.
- [Architecture paper](https://huggingface.co/OVHaiLLM/Olala-v1.0-mini-7A1B/blob/main/Olala-v1.0-mini.pdf),
  sections 4 and 8.2 describe the small latent experts and recommended Q5_K_M mix.
- Architecture and serial kernel reference:
  [OVHai-LLM/llama.cpp](https://github.com/OVHai-LLM/llama.cpp/tree/55c22dca0c987f247c21b427c3b3a09e1a28187f),
  revision `55c22dca0c987f247c21b427c3b3a09e1a28187f`.
- Kernel source hashes and adaptation details live beside the MIMO and Q6_K
  sources. MIT licenses are retained. Compact checkpoint metadata and chat
  template fixtures retain the publisher's Apache-2.0 license and notice;
  model weights are not tracked.

The released geometry has 29 Mamba3-MIMO layers and seven differential
tensor-product attention layers, hidden width 1536, and latent MoE with six of
256 experts. The loader rejects incompatible geometry, arithmetic metadata,
tensor shapes, encodings and EOS settings before inference. Metadata advertises
100,000 positions; the publisher describes a 32,768-token window. Short-context
checks do not qualify either full extent. Use `serve --ctx-size 32768` or lower
when setting an application budget.

## Execution and state

Model schemas and plans are in `lisa-models::olala`; the engine owns GGUF
access, weight bindings, memory admission and execution. Generic Metal
operations provide MIMO recurrence and geodesic residuals. A future device
provider can consume the same GGUF and model plan, but must implement and
validate the required operations. No ROCm provider is included.

Prefill and decode process rows serially with F32 activations. MIMO uses the
pinned serial recurrence with immutable input state and a separate output;
geodesic updates use the pinned kernel. Attention uses explicit F32
projections, token shift, softcapped attention and full KV history. MoE uses
GPU-indexed packed expert projections, squared ReLU and ordered weighted sums.
The six experts share one up-projection dispatch and one down-projection
dispatch. Routing, squared ReLU, ordered weighted accumulation, paired token
shifts and differential SiLU output gating have dedicated F32 fusions. Q8_0
uses the existing dense dot-product helper with selected base pointers; Q5_K/Q6_K use the existing indirect kernels. No selected matrix is
copied. Reductions, activation rounding and the ordered six-term sum are unchanged.
Every four layers are submitted asynchronously on the same serial queue;
prefill and multi-row forward wait at token boundaries to bound in-flight
recurrent storage. Single-token decode waits through normal readback.
No weights are requantized. Tiny projections pad only their device byte views
to the existing packed kernel tile size once at load, retaining the original
bytes. Attention plans distinguish empty/nonempty history; log-position is
runtime data evaluated once per token, so decoding reuses prepared plans.

Numerical cache law `0x4f4c414c41000001` identifies this execution path.
Snapshots share immutable tensors; continuation replaces state rather than
mutating saved prefixes. A model step commits its cache only after all layers
succeed. Native cache disk spill is unsupported. There is no draft head,
wide verification or B>1 path. Do not enable PLD for this implementation.

The embedded template supplies Olala's channel framing. UI and CLI chat generation
display the final channel; HTTP separates reasoning, final text and declared
tool calls, withholding partial control markers during streaming. Thinking
maps to `medium`, disabled thinking to `disabled`; independent low/high effort
selection is not implemented. The UI re-renders complete text history through
the embedded template each turn. Text-only Olala requests restore exact matching
prompt prefixes from two in-memory snapshots, gated by the model numerical
law; images bypass this cache. Loading/unloading a model, clearing a chat or
restoring a saved chat drops the snapshots. It does not reuse Qwen turn suffixes. The interactive `lisa chat` command retains its
existing Qwen4 restriction. Structured responses use the server's existing
repair/retry mechanism, not grammar-constrained decoding.

## First-token latency

The app previously created a fresh session for every Olala text or browser
request and replayed all history. It now retains immutable prompt-only states
for exact prefix reuse, including between browser-agent passes. Cache filling
uses 128-token chunks with cancellation checks. The final prompt token is left
for generation, so identical repeats restore at N−1 and consume that token
exactly once. Sampled output never enters the prompt cache. The activity log
reports total, reused and remaining prompt tokens; first-token timing includes
cache filling. A fresh or changed prompt still requires serial prefill.

Olala warmup now evaluates two tokens once per model instance, covering empty
and nonempty history with private caches. The app does this before declaring
the model loaded. The CLI previously warmed the entire input before performing
the real prefill, duplicating prompt work outside its reported prefill timer.
Warmup is now bounded independently of input length. Context-specific kernels
can still compile on first use; no universal cold-start latency is claimed.

An explicit checkpoint test in `lisa-app::prompt_cache` compares all logits
from initial, identical-repeat and next-turn cached requests with fresh
full-prefill sessions, checks immutable snapshots after decode, and verifies
that divergent prompts and image/non-admitted requests cannot reuse text state.
It requires `LISA_OLALA_MODEL` and fails if that checkpoint is missing. Both
explicit runs passed; the release workspace passed 589 library tests (34
model/environment tests ignored), plus compiled smoke. The
throughput table below predates this latency change; no new benchmark sweep
was run for it.

## Validation boundary

The native path reproduced the pinned reference's first 32 greedy tokens for
raw prompt `The capital of France is`. The reference used Metal, disabled flash
attention, and `-b 1 -ub 1`. This checks a continuation, not full numerical
parity. LISA's explicit attention, normalization, routing and projection graph
has not been certified bitwise equivalent to all fused reference operations.
Do not use this result to weaken existing STRICT contracts or label the model
reference-equivalent.

Incremental prefill at chunk size four for a 17-token prompt produced identical
final hidden state and logits. The opt-in checkpoint test also branches from a
saved prefix, performs a different continuation, restores it, and requires
bitwise-identical logits on replay:

```sh
LISA_OLALA_MODEL=/absolute/path/Olala-v1.0-mini-7A1B-q5_k_m.gguf \
  cargo test --release -p lisa-bench --lib olala_prefix_forks \
  -- --ignored --test-threads=1
```

Live HTTP checks covered matching streaming/non-streaming content, raw
completions, JSON output, a declared weather tool call, and a two-turn session
using suffix reuse. The workspace library suite and compiled smoke checks
cover integration with existing paths; they do not qualify other checkpoints.

An explicitly enabled checkpoint test fails when its model is missing.
Self-contained checks cover metadata and plan shapes, Q6_K decoding against a
pinned C oracle, packed GPU projections, MIMO state chunking and immutable
inputs, and channel/template behavior. Long-context behavior, full per-layer
reference captures, sampled generation, tool-use quality and performance
acceptance beyond the measured cases below remain open.


## Packed fusion and launch policy

The fused router selects six of 256 finite sigmoid-plus-bias scores in one SIMD
group, with ascending expert id as the tie-breaker. It preserves the existing
four-consecutive-read reduction for normalization, the clamp, and the original
Fast division. Other sigmoid/transcendental operations retain Safe + Precise
math. Products and subsequent additions remain separate F32 operations;
contraction is disabled. In particular, the expert sum is left-associated, not
a tree or fused multiply-add. ReLU uses the original comparison-based maximum,
including NaN-to-positive-zero behavior. The shader-side provenance records
hashes of the actual standalone oracle sources.

For Q8 expert inputs at K≤256 or K≤512, one or two SIMD groups replace four
(or eight for very large output matrices). Removed groups only contributed
positive zeros: active groups visit the same blocks once, and the final
32-lane reduction retains identical zero padding. Wider inputs retain the old
group count and block stride. Original Q5_K/Q6_K reductions are unchanged.

Attention matrix products use an explicit F32 schedule for two to eight query
rows: 8×32 output tiles with BK=16 retain the original eight-wide MMA chain
and K traversal. Transposed keys are addressed through strides, avoiding a
full key-cache copy. Ordinary matmul dispatch remains unchanged. Bitwise
checks cover NN and NT products, batches 1/12, query rows 2/4/8 and cache
lengths from 1 through 4,097, including partial tiles.

Self-contained tests compare fusions with their original operation chains:
all 65,536 BF16 bit patterns in F32 storage for ReLU; weighted sums with
cancellation, subnormals and a wide exponent range; random, tied, saturated and
clamped routing; first-token and continuation token shifts; differential gates;
and Q8 input-width boundaries 32/256/384/512/544/1280, plus Q5_K/Q6_K experts.
These checks pass bitwise twice, including the serialized workspace suite.

Optimization references were refreshed on 2026-11-20:

- [MLX-Serve GGUF kernels](https://github.com/ddalcu/mlx-serve-gguf/blob/c32bb7b06ec372551219604254193e18f9341bd4/src/kernels.zig):
  packed block access, row packing and occupancy-aware launch choices.
- [Olala MLX kernels](https://github.com/OVHai-LLM/mlx-engine/blob/58083ca0ac4ca0654b508f74e1bdf7b6e0e173f4/mlx_engine/models/olala/metal.py):
  fused prologues and short decode chains.
- [Olala GGUF kernels](https://github.com/OVHai-LLM/llama.cpp/blob/55c22dca0c987f247c21b427c3b3a09e1a28187f/ggml/src/ggml-metal/kernels/mamba3.metal):
  mixed-format persistent MoE and attention fusion.

Their alternative dot-product reductions and persistent full-MoE arithmetic
are not substituted for LISA's strict oracle. This change retains the original
Q5_K_M package and adds no affine conversion, requantization, speculative path,
or relaxed numerical mode.

## Measured performance

Release comparison against `a9d06df`, on Apple M5 Max (40 GPU cores, 128 GiB
physical RAM), with the pinned Q5_K_M above. The RAM budget was detected as
120 GiB and memory pressure checked before each run. Lisa was closed; GPU
workloads ran sequentially. Runs use greedy serial generation, 128 output
tokens, `--ignore-eos --depth 0`, CLI warmup, warm filesystem/JIT caches, and
one second between processes. Probe, capture and tracing are disabled for
scored runs. Each workload uses old/new/new/old. Toolchain: Rust 1.99.0
(`b940084d7`), Xcode 27.2 (`27B52`).

| Prompt tokens | Previous decode tok/s | Optimized decode tok/s | Mean speedup |
| ---: | ---: | ---: | ---: |
| 5 | 87.6–88.0 | 147.1–149.0 | 1.69× |
| 306 | 81.7–82.4 | 133.1–137.6 | 1.65× |
| 1,021 | 62.2–62.8 | 96.9–100.0 | 1.58× |
| 2,041 | 43.1–45.0 | 66.1–67.6 | 1.52× |

Each range contains two measurements. Optimized prefill was 148.5–150.8
tok/s for the 306-token chat prompt, 128.5–132.7 tok/s at 1,021 tokens,
and 96.5–100.0 tok/s at 2,041 tokens. Decode exceeds 100 tok/s in the
short/chat cases; the repeated 1K and 2K cases do not establish that target.
Longer-context performance still needs work.

The measurement campaign was stopped during the first 4K run; no 4K result
is reported. A baseline Metal System Trace was collected as `before.trace`;
the final after-trace was not collected.
These results therefore include numerical and interleaved throughput evidence,
but not a completed before/after system-trace comparison.

The context axis is the tokenized prompt length at the start of generation;
cache length grows during the 128-token continuation. Rates use the existing
CLI `decode` metric on both versions: generated-token count divided by timed
serial decoding, excluding prompt ingestion, text emission and periodic pool
trimming. The first generated token comes from prefill, so using only the 127
timed decode steps would reduce these reported rates by 0.78%. These are warm
single-request rates, not cold-start or full-response wall throughput.

The 5-token raw prompt is `The capital of France is`. The 306-token chat prompt
is `Explain what a compiler does in two sentences.` with the embedded template.
The longer raw prompts repeat the following paragraph 20/40 times,
including its trailing space, producing 1,021/2,041 input tokens:

> A compiler translates source code into machine instructions. It performs parsing, type checking, optimization, and code generation. Explain how each stage contributes to correctness and performance, and compare this with an interpreter. Give a concrete example involving a loop and a function call.

Validation of the final build:

- Full release workspace build; 589 library tests passed, zero failures,
  33 explicitly ignored model/environment tests; compiled smoke passed.
- The opt-in Olala chunked-prefill and prefix-fork/replay test passed twice.
- All 151,936 final-prefill logits match `a9d06df` bitwise for 17, 52, 307
  and 1,021 input tokens, with each candidate capture repeated twice.
- Fusion, Q8 launch-boundary and small-row NN/NT matrix checks passed twice,
  including the serialized workspace run. No golden was recaptured.

The run inventory uses `final-build.log`, `final-tests.log`, `final-smoke.log`,
`final-session{1,2}.log`, `final-logits-*`, `final-measurements.json` and
`final-<workload>-*.log`. Workspace cleanup retains the chart, measurements and
validation summaries under `target/olala/perf/extreme/`; full captures and
research checkouts are preserved in an external local archive. These artifacts
are not tracked. Checkpoint/source identities, measurement method, result
summary and numerical boundaries are retained here.
