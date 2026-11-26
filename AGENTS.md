# AGENTS.md — lisa-qwen

Repository guidance for changes to the engine, kernels and serving paths.
Write all documentation, comments and reports in English. Preserve unrelated
working-tree changes and make one commit per logical change.

## Project and layout

LISA is a Rust inference engine for Apple Silicon with its own objc2 Metal
runtime. Some kernels and plans are ported from other projects; preserve their
source attribution, adjacent licenses and [NOTICE](NOTICE) when changing them.
Name external projects only in NOTICE; specs, docs and comments refer to "the
pinned reference" and link to the provenance file that records it.

| Checkpoint identity | Module | Model |
| --- | --- | --- |
| `qwen4_exp`, `qwen4_exp_text` | `qwen4` | Qwen 3.8 Flash-Next: hybrid GDN + MoE + MTP |
| `qwen3_5`, `qwen3_5_text`, GGUF `qwen35` | `qwen3_5` | Qwen 3.8 27B and Qwen 3.5: dense hybrid GDN/attention, MTP |
| `glm5_next`, `glm5_next_text`, GGUF `glm5-next` | `glm53` | GLM-5.3 Flash: KDA + sparse attention + MoE + MTP |
| GGUF `deepseek41` | `deepseek41` | DeepSeek V4.1 Flash Coder: mixed IQ GGUF, experimental |
| GGUF `inkling2` | `inkling2` | Inkling 2 Mini: looped MoE on the pinned reference's decode step, DFlash2 tree drafting, vision, experimental |
| GGUF `olala` | `olala` | Olala v1.0 mini 7A1B: Q5_K_M, experimental |
| `lfm3` | `lfm3` | LFM3: compact hybrid text |
| `flux2-klein`, `flux3-klein` | `flux2`, `flux3` | Local image generation |
| `laya` | `laya` | Decision model |

These identifiers are checkpoint contracts; do not rename them to match product
names. Models are read from and downloaded to the Hugging Face Hub cache only:
model resolution accepts Hub repository IDs. Metal is the LLM backend; a CPU path exists for Laya.

Draft heads whose weights ship with the checkpoint are supported (Qwen and
Flash-Next MTP, GLM target MTP). The one separate drafter is Inkling's pinned
DFlash2 block-diffusion draft, which drafts its tree rounds; add other
separate drafters only with the same pinning,
provenance and serial-equality check (`lisa-bench spec-check`). DeepSeek's six-file GGUF package preserves packed
weights, reads its Engram table from disk and has no draft head; a working reply
does not certify reference equivalence ([DeepSeek](specs/models/deepseek41.md)).

| Crate | Responsibility |
| --- | --- |
| `lisa-backend` | Portable tensor-import, capability and prepared-kernel contracts |
| `lisa-models` | Backend-independent model schemas and plans; no I/O, device allocation or kernel execution |
| `lisa-metal` | Tensor runtime, allocation, JIT dispatch and Metal shaders |
| `lisa-cpu` | Host tensor storage and model-independent ordered f32 kernels |
| `lisa-mlx` | Portable MLX safetensors parsing, shard indices and mapped host storage |
| `lisa-gguf` | GGUF parsing, split packages, mapped storage and CPU block decoding |
| `lisa-engine` | Portable weight binding, generic plan execution, host memory admission and shared generation/session logic; no concrete device dependency |
| `lisa-serve` | HTTP APIs, request handling and application provider composition |
| `lisa-cli` | `lisa` CLI |
| `lisa-bench` | Compiled diagnostics, benchmarks and embedded goldens |
| `lisa-app` | Shared Apple app services and C bridge; SwiftUI interface in `apps/Lisa` |

Put Metal sources under `crates/lisa-metal/src/shaders/` and their Rust wrappers
under `jit/` or operation modules in `kernels/`. Device crates have no model
namespace; checkpoint configuration and model plans belong to `lisa-models`. Keep model-specific logic out of `lisa-engine/src/core/` when it
can be expressed through `LanguageModel`.

Organize engine code by owner: `bindings/<model>/` for execution bindings,
`weights/<model>/` for checkpoint imports, and `bindings/shared/` for common
operations. Keep session, cache, sampling and speculative-runtime helpers in
their owning module folders, with tests beside the implementation. Use short
child names such as `attention.rs` or `capture.rs`; do not grow flat families
of model- or subsystem-prefixed filenames. Use canonical nested paths inside
the workspace; public re-exports can preserve existing API entry points.

Keep `lisa-mlx` and `lisa-gguf` independent of the engine and all compute runtimes.
Model schemas and computation plans belong to `lisa-models`; the engine owns
file access, binds weights and executes plans. Model definitions must not read
files/environment variables or import device/format crates.
Device crates must not traverse models or own inference policy.
Device selection belongs to the engine/application, never to a format reader
or the Metal crate. Preserve original packed payloads across format imports. The engine's `weights::gguf` adapter owns generic device weight views and RAM admission.

Tests are self-contained: they run by default, need no model, no environment
variable and no external data, and keep their small fixtures with the owning
crate. Checks against real checkpoints are `lisa-bench` commands that take
`--model` (goldens, GLM token runs, session and prefix checks). Do not add
`#[ignore]` tests, feature gates or environment switches; a setting that must
vary is a CLI flag with a good default. Keep durable contracts in `specs/` and
required provenance beside source. Store detailed result summaries outside the
specs.

## Portable paths and artifact references

Do not record local filesystem locations anywhere in the repository, including
this file, documentation, comments, scripts and fixture metadata. Never commit
developer usernames, absolute workstation paths, home-directory expansions,
host cache locations or links to local evidence files. Keep resolved locations
in the invoking environment or external configuration outside the repository.
Use repository-relative references for project files and portable artifact
identifiers for provenance. Keep large external artifacts out of Git.

## Build and validation

Cargo builds generate a matching `kernels.bin` beside the executables. Preserve
it when copying standalone binaries; the app bundle places it in Resources.
`sh apps/Lisa/bundle.sh` builds the SwiftUI `dist` bundle and retains
the matching dSYM outside it. See the [Apple app spec](specs/apple-app.md).

Before committing, run the following sequence with the machine-safety checks
below.

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps --document-private-items
cargo build --release --workspace
cargo test --release --workspace --lib -- --test-threads=1
./target/release/lisa-bench smoke
```

Use CLI help as the command inventory. Useful model checks include:

```sh
MODEL=agnosticeng/Qwen3.8-27B-4bit
./target/release/lisa-bench golden --model "$MODEL"
./target/release/lisa-bench round-cost --model "$MODEL"
./target/release/lisa-bench session-check --model "$MODEL" --chunk 16
```

Qwen goldens are embedded under `crates/lisa-bench/golden/`. GLM uses
`lisa-bench glm53-run` and comparisons against the pinned GLM reference; its
`--capture-logits` runs are diagnostic and excluded from performance evidence.

Kernel or scheduling changes require affected numerical checks twice,
before/after traces, and interleaved end-to-end measurements. Use in-session
`round-cost` where supported. Cache-key changes also need an interleaved
end-to-end pair: goldens cannot detect throughput regressions. State which
models, shapes and modes were exercised; tests that skip without local model
data do not certify those cases.

## Machine safety

- On macOS and Simulator, compute the budget from detected physical RAM minus **8 GiB**. Read
  `sysctl hw.memsize` and `memory_pressure` before every benchmark. Never
  hardcode a machine's capacity. Use `--ram-cap-gb` for a lower run limit.
- Physical iOS devices use the smaller of detected physical RAM and the live
  `os_proc_available_memory()` allowance, minus **512 MiB**. Never cache the
  allowance or let `--ram-cap-gb` raise it. Release the model on memory warnings.
- Load one model at a time. Run GPU tests, inference and benchmarks
  sequentially; other GPU activity invalidates timing comparisons.
- After an abort, stop residual processes from that run before loading again.
  Prefer 64–256 output tokens unless measuring long generation.
- GLM's resident reservation counts mapped model bytes plus private allocations
  against `lisa_engine::memory::ram_budget()`. Server request admission is separately armed
  by `--ram-cap-gb`; these are distinct mechanisms.
- Preserve allocation lifetime and completion ordering. Never wait for GPU
  completion while holding the allocator lock.

## Numerical contracts

Follow the current task instructions and these constraints when older campaign
text conflicts. [Numerics](specs/numerics.md) provides additional
detail, but historical permission to recapture goldens does not authorize
weakening the strict requirements below.

- **Serial, prefill and B>1 batching are STRICT:** preserve the applicable
  reference's arithmetic and accumulation order. Do not change reductions,
  casts or fusions merely because greedy tokens still match.
- The Qwen MTP verify exception is limited to `affine_verify_qmm_splitk` at
  S=2..8: at most one bf16 ULP plus argmax equality versus `qmv_wide`, guarded
  by `ops::enter_verify_splitk_scope`. Serial, batch and large output heads
  retain STRICT semantics.
- A wide verify forward can differ from serial at a genuine near-tie. Prove
  and document that attribution before treating it as expected behavior.
  Never recapture a golden solely to conceal a regression or force an
  unproven kernel onto the scored path.
- Preserve FMA behavior, per-operation casts, transcendental math mode and
  reduction order. Equivalent source expressions can compile differently.
  `simd_sum` is not an XOR butterfly. Prove fusions against their actual oracle
  chain, including exhaustive bf16 checks where the domain is small.
- Pipeline keys must encode math mode, constants and the NAX language target.
  Keep source hashing and expensive checks on cache misses; lookup runs per
  dispatch.
- Prefix/session state belongs to an explicit numerical law. Do not restore
  state across arithmetic modes. GLM's law includes loaded projection modes
  and must be available before lazy kernel compilation.

## GLM-5.3 port rules

- The pinned GLM reference implementation is the execution oracle; its
  revision and adapter identity are recorded in the
  [GLM status](specs/models/glm53.md) and kernel provenance. Refresh the
  reference before applying upstream changes. Specs do not override observed
  reference code.
- The curated checkpoint is `antirez/glm-5.3-flash-gguf`, file
  `GLM-5.3-Flash-Q2.gguf`. Preserve its original Q8_0/Q4_K/IQ2_XXS/Q2_K
  payloads and embedded MTP block.
- MLX loading preserves tensor payloads, metadata precision, scales, biases
  and quantization groups, and requires a prepared `.lisa-resident/index.json`
  (`lisa prepare --model <dir>`; `--plan-only` reports disk needs; preparation
  preserves original bytes and refuses to overwrite). No requantization. Keep
  mappings alive for all GPU users, validate source identities and reject
  invalid ranges.
- Keep expert addresses and residency stable. Resident decode must not
  reintroduce selected-expert CPU readback, staging joins or address-table
  rebuilding. Bound temporary memory independently of layer count.
- Q8, DSA and fused gate/up schedules are defaults. The GGUF path groups
  prompt FFN projections across up to eight rows; attention and target
  verification keep serial arithmetic. Preserve the reference's shape gates,
  dense/sparse boundaries, value-projection slice sizes, submission cadence and
  encoder boundaries when changing dispatch.
- Paired Q8 gate/up projection is part of the GLM law and cache identity.
- Two MTP drafts can commit three tokens. Preserve resumable draft/cache/RNG
  state and remaining-token limits; token equality alone does not establish
  state or snapshot equivalence.
- Same-weight adapter agreement, independent kernel validation and upstream
  performance are separate claims; see the GLM status for gaps.

## Runtime and measurement pitfalls

- `ensure_row_contiguous` does not contiguate inputs. Bindings honor layout
  offsets and strides; materialize a contiguous array when a kernel needs one.
- Scoped ordered reuse is GLM-specific. Reuse GPU outputs only after earlier
  reads are encoded on the same serial queue and no live array leases remain.
  CPU imports retain completion requirements. Partial writers require a clear;
  only audited full writers may skip it.
- Rerun suspected GPU-contention flakes in isolation with `--test-threads=1`.
  Do not change arithmetic to accommodate an oracle returning stale zeros.
- Isolated microbenchmarks guide investigation; interleaved live workloads
  support performance decisions. Exclude captures and per-kernel probe mode
  from scored runs; align timing boundaries and token-count denominators.
- Use Metal System Trace for system tracing. Command-buffer intervals are not
  exclusive per-kernel hardware counters. A CLI trace can appear as `Lisa.app`;
  copy the binary to a distinct path when attribution is ambiguous.
- Qwen MTP depth is 2–6 (context-copy hits up to seven, verify width eight).
  Kernel tile width is not draft depth; GLM's two-draft MTP and Inkling's
  tree rounds of up to sixteen rows are separate contracts.
- Consult external evidence before reopening AOT, paged-KV decode,
  norm-inside-qmv, tree verification, ANE/GPU prefill or MTP-head grafting.
  Follow the [optimization admission rules](specs/numerics.md#optimization-admission).

## Diagnostics and documentation

Write specs as the current architecture and behavior: responsibilities,
interfaces, execution, invariants, supported modes and known limitations.
Do not add research diaries, investigation history, migration narratives,
abandoned approaches, chronological fixes, benchmark campaigns or test-run
logs. Keep those records in external evidence. Preserve required source
provenance and fixture identities alongside their implementation. Centralize
external-project citations in NOTICE; keep required copyright,
license and source-attribution notices with adapted code and in NOTICE.

LISA reads no environment variables of its own. Settings are flags:

| Flag | Commands | Purpose |
| --- | --- | --- |
| `--device metal\|cpu\|auto` | `lisa`, `lisa-bench` | Backend selection (Metal when present by default) |
| `--ram-cap-gb <GiB>` | `lisa`, `lisa-bench` | Lower the RAM budget; also arms HTTP request admission |
| `--kv-format auto\|native\|int8` | `lisa`, `lisa-bench` | Key/value cache format; `auto` (default) uses 8-bit caches where a model implements them |
| `--tree-nodes <n>` | `lisa`, `lisa-bench` | Cap on the rows a greedy tree round verifies (1–16, default 16) where the model supports tree rounds |
| `--trace [prefixes]`, `--trace-json <path>` | `lisa-bench` | Hierarchical spans and their report, optionally filtered; Chrome trace output |
| `--gpu-probe`, `--cb-record <path>` | `lisa-bench` | One dispatch per command buffer, or per-command-buffer recording |

Same-name/different-source JIT kernels always fail. Do not restore removed
controls (environment switches, capture hooks, the DSpine, `--pld` and
`--drafter` options) without a documented requirement.

- [README](README.md): user setup and entry points.
- [Runtime](specs/runtime.md): loading, memory, sessions, speculation and scheduling.
- [Specification index](specs/README.md): current architecture and capability status.
- [Numerics](specs/numerics.md): numerical laws, state identity and the validation protocol.
- [Roadmap](specs/roadmap.md): remaining work and acceptance conditions.
- [GLM status](specs/models/glm53.md): status and numerical details.
  Historical results do not certify newly changed paths.
