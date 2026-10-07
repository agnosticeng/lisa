# AGENTS.md — lisa-qwen

Working knowledge for this repository. Read before touching numerics, kernels, or
serving paths. Dense by design: every claim was established by measurement; the
standing contracts under `specs/00-contracts.md` hold the requirement-by-requirement
story — go there for the full detail, not here.

**Language: everything in English.** This file, every spec, every verdict, every
code comment. No French, no mixed text, anywhere.

## 1. Mission & models

`lisa` is a from-scratch Rust inference engine for Qwen hybrid (GDN + MoE + MTP)
models on Apple Silicon, on its own objc2 Metal runtime (JIT-compiled kernels,
`lisa_mlx`-shaped op surface). The Metal kernels are a port of the upstream MLX
kernel library, vendored and byte-pinned under `shaders/` — provenance is kept
for license compliance, but all design, orchestration, and scheduling are ours
and are documented in this file and `specs/`.

| model_type (config.json) | module | commercial name | shape | golden |
|---|---|---|---|---|
| `qwen4_exp` | `qwen4` | Qwen 3.8 Flash-Next | 125B total / A6B active, 48L, 512-expert MoE | 256/256 (1024→256 greedy, serial + MTP d1..6) |
| `qwen3_5` | `qwen3_5` | Qwen 3.8 27B dense-hybrid | 27B, 64 hybrid GDN/attn layers | 310/310 |

The `qwen4`/`qwen3_5` vs "3.8" mismatch is historical — `model_type` is an
external checkpoint contract, do not rename.

Per-capability contracts live in `specs/00-contracts.md` (numerics-contracts,
serving-api, machine-safety, plus the batching / decode-kernel / generation-core
/ verify-kernel / prefix-cache-pld / speculative-mtp / model-loading /
cli-tooling clauses folded from the retired capability-spec tree), each an
invariant rather than a campaign narrative. Numbered campaign specs live under
`specs/` (01-10, plus `10-measures.md` for measured tables without a topical
spec); measured artifacts live under `bench/` (round-cost tables, llmprobe
runs, verdict harnesses, GPU traces).

## 2. Machine safety (any box — the rule is generic, not this machine's)

The working rule is **cap = detected total RAM − 8 GiB**. On this box that is
120 GiB (128 GiB physical − 8). Do not hardcode 120 anywhere.

- Read total RAM with `sysctl hw.memsize` (bytes) and headroom with
 `memory_pressure`; compute the cap from the detected value, never from memory.
- `LISA_RAM_CAP_GB` overrides the computed cap when a run must be bounded lower.
 It is now **implemented** (`memory::ram_budget`) — see §9.
- **Never hold two models at once.** The 125B model is ~77 GB RAM-resident, the
 27B ~28 GB; two can exhaust memory and GPU faults follow.
- **Runs are sequential.** No concurrent benches, no concurrent GPU users
 (a `cargo test` in the background swings numbers up to ~2×).
- **Memory check before every benchmark run** (see `memory_pressure` free
 fraction; the long-context runs held 86–89% free).
- **Kill residual processes after an abort** — an orphaned `lisa`/`lisa-bench`
 still holds the model.
- **Prefer short runs** (max-tokens 64–256) unless a run explicitly measures
 long generation; observed peaks stay far below the cap (max 73.7 GB RSS in
 the long-context wave).

## 3. Architecture

Six crates (Cargo virtual workspace, resolver 3):

| crate | role |
|---|---|
| `lisa-mlx` | tensor runtime + Metal kernels (lib `lisa_mlx`): runtime/, array/, ops/, jit/, shaders/, models/ |
| `lisa-engine` | inference library: `models/` (per-model towers, gdn, speculate) + `core/` (loader, quant, norm, cache, generate, session, batch, sched, sampler, tokenizer, admission, prefix_cache, copy_draft) |
| `lisa-serve` | OpenAI-compatible HTTP server (`/v1/chat/completions`, `/v1/models`, `/health`, `/v1/decisions` for Laya) |
| `lisa-cli` | the `lisa` user binary |
| `lisa-bench` | the `lisa-bench` test/bench binary |
| `lisa-ui` | macOS AppKit GUI app |

Key directory layout inside `crates/lisa-mlx/src/`:

- `shaders/common/{elementwise,matmul,reduce,attention,data}` — generic op
 kernels assembled by `jit/`; `shaders/nax/` — Metal 4 / NAX headers;
 `shaders/gdn/` — Gated DeltaNet (shared by qwen4 AND qwen3_5); `shaders/qwen4/`
 — model kernels. Convention: `shaders/` = every `.metal` source; `models/` =
 Rust wrappers only.
- `src/models/` — shared arch components + model host wrappers (gdn.rs, qwen4/).
- `src/core/` (in lisa-engine) — model-agnostic generation core; per-model code
 never leaks into it (GDN and `speculate` were extracted out of the qwen4
 namespace).
- `crates/lisa-bench/golden/` — embedded tracked goldens (`qwen3_5.json`,
 `qwen4.json`); `--capture` regenerates.

Models are addressed by local dir or HF repo id (`models::resolve_model_dir`,
standard hub cache layout). Shared arch (currently: GDN) lives at `models/` root in
both crates, not under one model. Devices: `LISA_DEVICE=metal|cpu`; Metal by
default, auto-fallback to CPU where a host path exists (Laya only).

## 4. Commands (verified against the clap enums)

Build: `cargo build --release`. Guardian: `tools/check.sh` (build workspace +
`cargo test --workspace --lib` + `lisa-bench smoke`). Run it green before any
commit.

User binary `lisa`: `run`, `serve`, `inspect`, `decide` (Laya), `tok`, `chat`,
`batch`.

Bench binary `lisa-bench`: `smoke`, `golden`, `layer-diff`, `logits`,
`cache-test`, `neg-slice`, `rounding`, `qmm-m`, `indirect-bench`,
`decode-moe-bench`, `prefill-bench`, `session-check`, `cbatch`, `round-cost`,
`serial-audit`, plus `prefix-check` and `verify-audit`.

```bash
M=agnosticeng/Qwen3.8-27B-4bit # or the Flash-Next repo id / snapshot dir
./target/release/lisa-bench smoke # no model needed
./target/release/lisa-bench golden --model "$M" # 310/310 (27B) / 256/256 (qwen4)
./target/release/lisa-bench golden --model "$M" --depth 5 # wide-verify semantics
./target/release/lisa-bench round-cost --model "$M" # in-situ round table
./target/release/lisa-bench session-check --model "$M" --chunk 16
```

`--model` accepts repo ids directly (resolved through the HF hub cache).

## 5. Numerics contracts

- **Bit-exact STRICT** on serial, prefill, batch (B>1), and PLD paths: every
 dispatched kernel must reproduce the oracle's accumulation order exactly
 (per-op details in `specs/00-contracts.md` and the kernel
 sources). Do NOT "simplify" a reduction tree, a norm layout, or a fusion that
 passes — 1 ULP can flip an argmax through 48 chaotic layers.
- **Tail-ULP exception, scoped to MTP verify only** (verify-kernels): the S=2..8
 verify quantized matmuls ride `affine_verify_qmm_splitk` under ≤ 1 bf16 ULP +
 argmax-equality vs the `qmv_wide` chain, enforced by a thread-local dispatch
 scope (`ops::enter_verify_splitk_scope`) and the test
 `verify_qmm_splitk_tailulp_vs_qmv_wide`. Serial M=1, prefill, B>1 batch, PLD,
 and lm_head (N ≥ 100k) keep STRICT.
- **Goldens are the final net**, not the contract: near-tie flips on the MTP
 wide path (§9.6 semantics) are expected — the MTP verify is one S=depth+1
 forward whose rounding class differs from serial. A depth that diverges from
 the serial golden at a genuine near-tie (top-2 gap rounding to 0) is
 semantics, not a kernel bug. NEVER force a not-yet-parity kernel into the
 path to "make goldens pass" — fix the kernel to parity or keep it off the
 scored path.
- Near-ties: measured qwen4 example — `golden --depth 5` flip at token 14 on a
 0.125 f32 gap that the S=6 verify rounds to exactly 0.0; deterministic across
 runs/depths; forced `a=0` flips at the same token → the flip lives in the
 wide forward's rounding (speculative-mtp).
- **The compiled-pipeline cache key is a numerics contract.** It must encode
 every axis that changes the emitted arithmetic — math mode, consts, and the
 NAX language version — while staying free of per-dispatch work (§7). Cached
 model state (prefix cache) is bound to that identity: state computed under a
 different numerical law is never restored.

## 6. Environment flags (complete list)

| flag | effect |
|---|---|
| `LISA_DEVICE` | `metal`/`cpu` backend selection (default: metal if present) |
| `LISA_TRACE` | hierarchical spans + `jit.compile` events, `JIT_COMPILES` counter |
| `LISA_TRACE_JSON` | Chrome trace JSON output |
| `LISA_RAM_CAP_GB` | memory **admission** budget (GiB). Arms byte-level admission + the 503 refusal path (§9); also the value `memory_limit` reports. Unset = historical behaviour (no refusals) |
| `LISA_STRICT_KERNELS` | turns the JIT twin-check (same name, different body) from a warning into a fatal error |
| `LISA_NO_INTERLEAVE` | disables prefill/decode interleave during admission (admission results identical) |
| `LISA_ADDNORM` | fused add+norm infra switch (off = stock path) |
| `LISA_GPU_PROBE`, `LISA_CB_RECORD` | runtime diagnostics: per-kernel probe mode; per-command-buffer JSON recorder |

Everything else (`LISA_KEEP`, `LISA_DUMP_TOKENS`, `LISA_TOPK`, `LISA_GDN_OPS`,
`LISA_METALLIB`) was removed — do not reinstate without a spec.

## 7. Gotchas (current, each one real)

- **Flaky GPU tests**: MoE `counting_sort`, `fused_selector_matches_reference`,
 and `keyed` tests can fail under parallel test execution; the oracle chain
 returns stale zeros while the fused kernel is correct. Isolated re-run
 (`--test-threads=1` / single test) = green. Don't "fix" the kernel on a flake.
- **Never do expensive work in a per-dispatch key.** `MetalRuntime::compile_full`
 and the JIT `apply_keys` fast path are the per-dispatch cache *lookup* (they
 early-return on a hit), so every byte costed there is a per-dispatch tax.
 Folding a multi-KB shader source hash into the key measured **−54 % decode**
 (30.6 → 14.3 tok/s) *with goldens still green* — only an interleaved
 end-to-end A/B caught it. Runtime-varying axes go in as cheap fields (the
 `nax` flag was previously **absent** from the key: a real latent M5
 collision); compile-side hashing and the twin-check stay on the slow path.
 Kernels compile once at launch; the hot path only reads the key.
- **`ensure_row_contiguous` is IGNORED** — the `metal_kernel` parameter is
 `_ensure_row_contiguous` (jit/kernel.rs:245); the bindings honor
 `layout.offset`/strides directly. Never assume the runtime contiguates inputs.
- **FMA codegen differs per kernel**: two kernels with the same inline expression
 tree are NOT bit-identical to each other. Fusion validity is proven per kernel
 against the oracle chain (exhaustive bf16 sweeps where the input domain is
 small), never by source equivalence.
- **`simd_sum` is not an xor-butterfly**: hardware simd reductions run lanes
 emulated strided; reproducing a reduction means reproducing the emulator's
 order, not writing a butterfly.
- **TG-memory kills qmv occupancy**: the norm-in-qmv fusion (norm inside the
 quantized GEMV kernel) was falsified — threadgroup memory pressure collapses
 occupancy and it loses to the split norm+qmv.
- **Isolated benches lie**: standalone micro-benchmarks favor whichever kernel
 they were warmed for. Only in-situ `round-cost` (interleaved A/B, medians, in
 a live session) is admissible evidence. **And goldens do not catch a
 throughput regression** — a change can be numerically perfect and still halve
 the decode rate, so a kernel/cache change needs *both* the golden gate and an
 interleaved e2e pair.
- **AOT is dead** (specs/21): a vendored-sources metallib built with the MLX
 flags measured no perf difference — the JIT content-addressed cache already
 amortizes compilation. Do not reinstate an AOT path.
- **No Metal GPU Counters on macOS 27**: xctrace has no such template — use
 **Metal System Trace**, and per-kernel names are NOT externally observable
 (only per-command-buffer GPU execution intervals).
- **xctrace resolves the CLI through the Lisa.app path**: `ps`/traces label the
 `lisa` CLI as `Lisa.app`/`lisa-ui` — a false positive, not a GUI process.
 Copy the CLI to a distinct path (e.g. `/tmp/gpudiff/lisa-cli`) when tracing.
- **The PLD depth clamp is downward, not upward.** The verify fast lane covers
 S ∈ 2..=7 (one precompiled kernel per M). Widening a draft past that lane puts
 *every* round on the stock qmm path (S7 = 78.0 → S8 = 108.0 → S9 = 136.8 ms).
 `pld_enable` clamps depth to 6 and logs it. The MTP head is trained and
 configured at `depth=6, profile=generic` (verify width 7) — 6 is the head's
 own contract, not a compromise; the 32-row tile is a kernel scheduling width,
 never a draft depth.

## 8. Performance state — one verdict per line (tables live in specs/)

- Speculative decode, short prompts: **PARITY certified** (58.7 vs 58.3 tok/s
 interleaved; `specs/10-measures.md` §1).
- Long context: MTP profitable from ~16k KV; S7 round 130.4 → 62.6 ms; certified
 15.6k A/B 36.1 vs 32.1 tok/s; per-depth acceptance EMAs (shared EMA froze EV);
 `bucketToRead` ported, 32k+ bucket trusted (`specs/02`, `specs/04`).
- TTFT: 12.3/17.2/24.2 s @1k/4.1k/8k → 4.5/9.6/16.8 s; prefix-cache resume
 10-20× faster (`specs/05`).
- Router: echo-score 4-gram routes prompt-lookup vs MTP — echo +76 %, prose
 +12 %, mixed parity with the best single engine (`specs/06`).
- Multi-client batching: B=2 aggregate 1.70×, B=4 2.73×; bandwidth-bound cap
 ≈2.6-2.8× at 4-bit (`specs/10-measures.md` §2).
- Decode/probe characterization + where the remaining decode gap sits
 (speculation density, prefix-cache floor >2048, MTP/batching exclusivity):
 `specs/10-measures.md` §3-4. Temperature>0 speculation, the serve-side
 `ignore_eos` bugs and the wave-batching crossover are recorded there too.
- M5 anchors + the M4→M5 round-cost re-calibration table:
 `specs/10-measures.md` §1.
- Structural floors (falsified — do not chase): splitk verify qmm at 62-75 % of
 peak BW (`specs/04`); serial step at 88 % of peak / 723 GB/s (`specs/07`);
 norm-inside-consumer fusions (§7); AOT (§7); M=1 decode out of NAX range.

## 9. Serve surface

All nine gaps of the original audit are **CLOSED** (see
`specs/00-contracts.md` §2 for the implemented requirements and
`tools/serve_tests/run_all.sh` for the live suite — 7/7 against a loaded
server, run manually per the one-model rule):

- streamed ≡ non-streamed parity; per-request seed threading (incl. `seed: 0`);
 stop sequences and per-token logprobs on every wave path; tool-result turns
 (`role=tool`) verified against the OpenAI round-trip; structured outputs
 (`response_format` json_schema, one corrective retry); `/v1/embeddings`
 (mean-pooled L2-normalized trunk states); depth gating decoupled; `/v1/models`
 serves the real loaded id; forced-length decode priced in the EV
 (`min(E, remaining)`); long-prefill TTFT halved; prefix-cache resume 16.7×.

**Memory admission.** `LISA_RAM_CAP_GB` exists in the code: `memory::ram_budget`
(cap or RAM − 8 GiB), `core::admission::MemoryBudget` + `stream_bytes`,
per-model KV geometry (27B = 64 KiB/token, Flash-Next = 27 KiB), and the
wave-intake hook in `Server::step` returning a **503** (`Reply::Refused`)
instead of an OOM. Kill-switched: armed only when the var is set. Measured E2E:
503 with the cap, 200 without.

Fixed in earlier work (do not reopen): thinking-scratchpad leak in the
batched plain path, tool_calls arguments string→object render failure, softmax
looped-path axis >4096, MTP head-history trim (d6 1.94 → 3.50 tok/round),
`ignore_eos` (OnceLock no-op + serve flag; specs/01), GPU readback race (lost
wait + fence-map UAF), cross-row swiglu2 corruption at B>1, empty-picks at 2-4k
(probe warm-drop + bucket_to_read prior + EV warmup, specs/09).

## 10. Working rules

- **Everything in English.** Every spec, every verdict, every code comment —
 this file included. No French, no mixed text, anywhere.
- **One commit per logical port.** Trace before/after per port (`LISA_TRACE`):
 the targeted phase must drop, nothing else may regress.
- **Validation = goldens ×2 + in-situ round-cost.** Golden run twice (flakes),
 round-cost interleaved in a live session. Isolated micro-benchmarks do not
 count as evidence (§7). A cache/key change additionally needs an interleaved
 e2e pair, because goldens cannot see a throughput regression.
- **`tools/check.sh` green before every commit**; a tail-ULP lib-test flake
 under GPU contention is green on isolated re-run.
- Machine safety per §2: sequential runs, one model at a time, memory check
 before every bench, kill residuals, cap = detected RAM − 8 GiB
 (`LISA_RAM_CAP_GB` to override), never hardcode this box's 120 GiB.

## 11. Optimization catalogue — dispositions

The full 44-item sweep (our earlier prototypes and experiments) is triaged in
`specs/10-campaign-dispositions.md`. Headline: **most of the catalogue was
already closed or dead on its premise** — the sweep's real yield was
robustness, not throughput. One line per standing verdict:

- Landed (goldens ×2 each): nax cache-key axis + law infra (O2.1/O2.2),
 `LISA_RAM_CAP_GB` + admission/503 (O3.1/O3.2), PLD width clamp ≤6 (O4.1),
 copy-draft guards + WidthRamp + CopyDraftGate (O4.2).
- Already in place: SSE/tool-calls contract, cross-turn resume, batched decode
 row-invariant, strict break-even.
- Falsified, do not reopen: AOT, ANE/GPU hybrid prefill on M5-class silicon
 (O10), tree verify (0.94× vs MTP), paged-KV decode, #8
 continue-from-correction (already implemented), MTP-head grafting.
- Set aside with reasons: queue-dependent TTFT schedulers (single-user serve),
 radix trie, quality-only re-samplers, measure-gated precision work.
- Open: O4 c3 (verify accept counter), #30 bench A/B, two M5 measurement
 anchors (long-context ratio, cold TTFT).
