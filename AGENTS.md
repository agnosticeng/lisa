# AGENTS.md — lisa-qwen

Repository guidance for changes to the engine, kernels and serving paths.
Write all documentation, comments and reports in English. Preserve unrelated
working-tree changes and make one commit per logical change.

## Project and layout

LISA is a Rust inference engine for Apple Silicon with its own objc2 Metal
runtime. Kernels include ports from MLX and DS4; preserve source attribution,
adjacent licenses and [NOTICE](NOTICE) when changing them.

| Checkpoint `model_type` | Module | Model |
| --- | --- | --- |
| `qwen4_exp`, `qwen4_exp_text` | `qwen4` | Qwen 3.8 Flash-Next, hybrid GDN + MoE + MTP |
| `qwen3_5`, `qwen3_5_text` | `qwen3_5` | Qwen 3.8 27B, dense hybrid GDN/attention |
| `glm5_next`, `glm5_next_text` | `glm53` | GLM-5.3 Flash, KDA + sparse attention + MoE + MTP |
| `laya` | `laya` | Decision model |

These identifiers are checkpoint contracts; do not rename them to match product
names. Model resolution accepts local directories and Hugging Face repository
IDs. Metal is the LLM backend; a CPU path exists for Laya.

| Crate | Responsibility |
| --- | --- |
| `lisa-mlx` | Tensor runtime, allocation, JIT dispatch and Metal shaders |
| `lisa-engine` | Model implementations and shared generation/session logic |
| `lisa-serve` | HTTP APIs and request handling |
| `lisa-cli` | `lisa` CLI |
| `lisa-bench` | Compiled diagnostics, benchmarks and embedded goldens |
| `lisa-ui` | Native macOS app |

Put Metal sources under `crates/lisa-mlx/src/shaders/` and their Rust wrappers
under `jit/` or `models/`. Shared architecture code such as GDN belongs at the
models root. Keep model-specific logic out of `lisa-engine/src/core/` when it
can be expressed through `LanguageModel`.

Keep self-contained test fixtures with the owning crate. Large model-dependent
captures are external data selected through `LISA_GLM53_FIXTURE_DIR`; retain
their filenames, sizes and hashes in the crate's fixture manifest. An enabled
test must fail if required data is missing. Keep durable contracts, provenance
and result summaries in `specs/`, `docs/` or alongside source.

## Build and validation

Before committing, run the following sequence with the machine-safety checks
below.

```sh
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
`lisa-bench glm53-run` and comparisons against the pinned DS4 reference; its
`--capture-logits` runs are diagnostic and excluded from performance evidence.

Kernel or scheduling changes require affected numerical checks twice,
before/after traces, and interleaved end-to-end measurements. Use in-session
`round-cost` where supported. Cache-key changes also need an interleaved
end-to-end pair: goldens cannot detect throughput regressions. State which
models, shapes and modes were exercised; tests that skip without local model
data do not certify those cases.

## Machine safety

- Compute the budget from detected physical RAM minus **8 GiB**. Read
  `sysctl hw.memsize` and `memory_pressure` before every benchmark. Never
  hardcode a machine's capacity. Use `LISA_RAM_CAP_GB` for a lower run limit.
- Load one model at a time. Run GPU tests, inference and benchmarks
  sequentially; other GPU activity invalidates timing comparisons.
- After an abort, stop residual processes from that run before loading again.
  Prefer 64–256 output tokens unless measuring long generation.
- GLM's resident reservation counts mapped model bytes plus private allocations
  against `memory::ram_budget()`. Server request admission is separately armed
  by `LISA_RAM_CAP_GB`; these are distinct mechanisms.
- Preserve allocation lifetime and completion ordering. Never wait for GPU
  completion while holding the allocator lock.

## Numerical contracts

Follow the current task instructions and these constraints when older campaign
text conflicts. [Standing contracts](specs/00-contracts.md) provide additional
detail, but historical permission to recapture goldens does not authorize
weakening the strict requirements below.

- **Serial, prefill, B>1 batching and PLD are STRICT:** preserve the applicable
  reference's arithmetic and accumulation order. Do not change reductions,
  casts or fusions merely because greedy tokens still match.
- The Qwen MTP verify exception is limited to `affine_verify_qmm_splitk` at
  S=2..8: at most one bf16 ULP plus argmax equality versus `qmv_wide`, guarded
  by `ops::enter_verify_splitk_scope`. Serial, batch, PLD and large output heads
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

- Use the pinned [DS4](https://github.com/antirez/ds4) implementation as the
  execution reference. Record upstream revision and adapter identity in the
  [GLM campaign](specs/11-glm53-flash.md); refresh the reference before
  applying upstream changes. Specs do not override observed reference code.
- Preserve original MLX tensor payloads, metadata precision, scales, biases and
  quantization groups. No requantization to make the port easier.
- Production loading requires a prepared `.lisa-resident/index.json` inside
  the model directory. Use `lisa prepare --model <local-directory>`; its
  `--plan-only` mode reports disk needs without writing weights. Preparation
  preserves original bytes and refuses to overwrite existing output.
  Keep mappings alive for all GPU users, validate source
  identities and reject invalid ranges.
- Keep expert addresses and residency stable. Resident decode must not
  reintroduce selected-expert CPU readback, staging joins or address-table
  rebuilding. Bound temporary memory independently of layer count.
- Native Q8, DSA and supported fused gate/up schedules are defaults. Preserve
  DS4's shape gates, dense/sparse boundaries, value-projection slice sizes,
  submission cadence and encoder boundaries when changing dispatch.
- `LISA_GLM53_Q8_GATE_UP=separate` retains separate projections. Its loaded mode
  must remain part of cache identity.
- Two MTP drafts can commit three tokens. Preserve resumable draft/cache/RNG
  state and remaining-token limits. Token equality alone does not establish
  state or snapshot equivalence.
- Same-weight DS4 adapter agreement, independent kernel validation and native
  upstream performance are separate claims. The full acceptance matrix remains
  open; see the GLM campaign for measured cases and gaps.
- [Sushi](https://github.com/beamivalice/sushi) is an optimization reference.
  Refresh its source before using it; DS4 remains the GLM parity oracle.

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
- Preserve the Qwen PLD depth clamp at six (verify width seven). Kernel tile
  width is not draft depth; GLM's two-draft MTP is a separate contract.
- Consult prior findings before reopening AOT, paged-KV decode, norm-inside-qmv,
  tree verification, ANE/GPU prefill or MTP-head grafting. See the
  [serial](specs/07-serial-final.md), [tree](specs/08-tree-draft.md) and
  [catalogue](specs/10-catalogue.md) reports.

## Diagnostics and documentation

Selected controls; model tests have additional capture hooks in their source:

| Variable | Purpose |
| --- | --- |
| `LISA_DEVICE` | Backend selection (`metal` or `cpu`, where implemented) |
| `LISA_RAM_CAP_GB` | RAM budget in GiB and server admission opt-in |
| `LISA_TRACE`, `LISA_TRACE_JSON` | Hierarchical spans and Chrome trace output; `filter:<prefixes>` narrows spans |
| `LISA_GPU_PROBE`, `LISA_CB_RECORD` | Per-kernel probe or command-buffer recording |
| `LISA_STRICT_KERNELS` | Fail on same-name/different-source JIT kernels |
| `LISA_NO_INTERLEAVE`, `LISA_ADDNORM` | Existing scheduling/fusion controls |
| `LISA_GLM53_CAPTURE_PRESERVE_DISPATCH` | Preserve production dispatch during captures; some intermediates need shadow replay |
| `LISA_GLM53_Q8_GATE_UP` | `paired` (default) or `separate`, selected at load |
| `LISA_GLM53_MODEL`, `LISA_GLM53_FIXTURE_DIR` | Checkpoint and external captures for explicitly enabled model integration tests |

Do not restore removed controls (`LISA_KEEP`, `LISA_DUMP_TOKENS`, `LISA_TOPK`,
`LISA_GDN_OPS`, `LISA_METALLIB`) without a documented requirement.

- [README](README.md): user setup and entry points.
- [Standing contracts](specs/00-contracts.md): serving, state and memory rules.
- [Campaign specs](specs/): scoped measurements and previous decisions.
- [GLM campaign](specs/11-glm53-flash.md): status and numerical details.
  Historical results do not certify newly changed paths.
