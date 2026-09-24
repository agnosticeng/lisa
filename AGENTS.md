# AGENTS.md — lisa

Working knowledge for this repository. Read it before touching numerics, kernels,
or the serving paths. It records the facts that took days to establish; treat the
architecture choices here as load-bearing, not stylistic.

**Status.** Golden parity **256/256** (1024-token prompt → 256 greedy tokens),
serial and MTP `--depth 1..6`. Local M5 Max: serial decode ~68–72 tok/s, steady
prefill ~2200 tok/s, MTP d5 ~98–107 tok/s. Multi-turn chat, cohort/ragged/
continuous batching, an OpenAI-compatible server, and long-context chunked
prefill all work. §13 describes what is *not* done yet.

---

## 1. What this is

`lisa` is a from-scratch Rust inference engine for the `qwen4_exp` text tower
(Qwen 3.8 Flash-Next, 125B total / A6B active) on Apple Silicon. It runs on its
own objc2 Metal runtime, compiles its kernels at runtime from embedded Metal
source, and exposes an `lisa_mlx`-shaped op surface to the model code.

The model is a hybrid stack: full attention on every fourth layer, gated
DeltaNet (linear attention) on the rest, a hyper-connection mixer around each
block, a 512-expert sparse MoE, a per-layer n-gram embedding (PLE), and an
embedded multi-token-prediction (MTP) head.

There is a reference implementation of the same model under `reference/` used as
the correctness oracle. **Parity rule: match the oracle's arithmetic, not a
re-derived op chain.** The oracle is the ground truth for fused kernels; do not
trust generic op-chain dumps of a fused kernel.

The engine is built around two extension axes (§3.1, §3.2): **model
implementations**, dispatched on the checkpoint's structure, and **device
backends**, selected with `LISA_DEVICE`. Qwen 3.8 Flash-Next (language) and Laya
(typed decisions, §20) + Metal are implemented today.

---

## 2. Environment & paths

- macOS on Apple Silicon. Dev box: **M5 Max, 128 GiB**, macOS 27, Xcode 26.
  NAX / Metal 4 `MetalPerformancePrimitives` is available.
- **Model (raw checkpoint lisa loads):**
  `~/.cache/lisa-models/Qwen3.8-Flash-Next-MLX-4bit-MTP`
  (~105 GB on disk; ~76.6 GB of kept text-tower tensors materialize).
- **Golden:** `reference/correctness_prompts/public_longcopy_gate_english_1024_256.json`
  (`cases[0].prompt_tokens` 1024 → `expected_tokens` 256; oracle first token
  `6184`).
- Keep `reference/` clean: revert any instrumentation when done.

---

## 3. Workspace layout

Cargo workspace (virtual manifest, resolver 3). Four crates:

```
Cargo.toml                  virtual workspace
crates/lisa-mlx/            the tensor runtime: backend selection + Metal backend
                            (lib name `lisa_mlx`; deps: objc2*, block2, half)
  src/backend/mod.rs        `Backend` selection (`LISA_DEVICE`, default metal)
  src/backend/metal/        the Metal backend
    runtime.rs              Metal device/queue, command batching, buffer pool,
                            fences, pipeline cache
    array.rs                eager `Array` over MTLBuffer, Layout + views
    shim_api.rs             Array/Dtype/Stream + ops/fast/nn/transforms/memory
    mlx_rt.rs               generic kernel-source assembly + JIT pipelines
    ffi.rs                  MetalKernel / TemplateArg / OutputArg
    kernels/{common,nax}    generic Metal kernels (elementwise, gemm, rope, …)
    models/qwen4/           qwen4 host wrappers + kernels/ Metal sources
crates/lisa-engine/         the inference library
  src/models/mod.rs         `LanguageModel` trait + `load()` + `resolve_model_dir`
  src/models/qwen4/
    config.rs               ModelConfig (raw + flattened text_config)
    tower.rs                Tower: layers, forward, warmup, chunked prefill
    attention.rs            full attention + QSA consumer
    indexer.rs              QSA indexer: pooled blocks + block selection
    gdn.rs                  gated DeltaNet linear-attention layer
    hyper.rs                hyper-connection mixer + inject
    moe.rs                  SparseMoeBlock (router, 512 experts, shared expert)
    ple.rs                  PLE / n-gram layer (host hash + mmap shard gather)
    mtp.rs                  embedded MTP head
    speculate.rs            MTP helpers (argmax_id, concat_rows)
    smoke.rs, layerdiff.rs  qwen4 diagnostics
  src/core/
    loader.rs               mmap safetensors, name sanitize, TensorSource
    quant.rs                affine 4-bit group-32 QuantizedLinear/Embedding
    norm.rs                 RmsNorm, rms_row_exact, RmsNormGated, Rotary, rope_partial
    cache.rs                LayerCache + FullAttentionCache/GdnCache/IndexerTape
    generate.rs             serial prefill/decode loop + sampling
    session.rs              multi-turn Session + generate_mtp
    batch.rs                cohort/ragged Batch + ContinuousBatch
    sched.rs                continuous-batching scheduler
    sampler.rs              temperature/top-k/top-p/min-p/rep-penalty + seed
    tokenizer.rs, mem.rs    tokenizer; allocator diagnostics
crates/lisa-serve/          OpenAI-compatible HTTP server
crates/lisa-cli/            the `lisa` binary
NOTICE                      third-party kernel attribution
```

The engine depends on the kernels crate as
`lisa-mlx = { path = "../lisa-mlx" }`, so call sites read
`lisa_mlx::kernels::…`, `lisa_mlx::qsa::…`, `lisa_mlx::ffi::…`.

### 3.1 Models

A model is addressed by a **local directory or a Hugging Face repo id**.
`models::resolve_model_dir` maps either to a local dir (a path as-is; else
`$LISA_MODEL_DIR` / `~/.cache/lisa-models` by short name, then the HF hub cache,
then `hf download`), `models::model_type_of` reads the structure from
`config.json`, and `models::load` dispatches on it. The CLI's `--model` accepts
both forms (resolved at argument-parse time by `ModelDir`).

**Adding a model:** add `<name>/` under `src/models/` (a config, a tower
implementing [`LanguageModel`] or [`DecisionModel`], its kernels) and register
its `model_type` in `models::load_dir`. `models::Loaded` distinguishes the two
kinds: `Language` is driven by `core/` (generation, session, batching,
scheduler — already generic over `&mut dyn LanguageModel`); `Decision` is
non-generative and addressed through its own API (§20). Per-model kernels live
with the model; qwen4's are in the Metal backend under `models/qwen4/` and
re-exported at the crate root.

### 3.2 Devices

`backend::Backend` selects the device backend: `LISA_DEVICE` (`metal`/`cpu`)
when set, otherwise **Metal if a device is present, else CPU** (auto-fallback;
the CLI announces a CPU selection). `backend/metal/` supplies the runtime, the
eager `Array`/op surface and the kernels. Generic kernels are shared
(`kernels/{common,nax}`); per-model kernels live under `models/`.

`Backend::Cpu` selects a **host f32** path for models that provide one. Today
only Laya does (`models/laya/cpu.rs`, rayon-parallel matmuls; ~0.85 s/question
vs ~30 ms on Metal) — it reuses the same `prompt.rs` I/O and answers. Other
models require Metal (the loader rejects `cpu` for qwen4).

**Adding a language-model backend:** add a sibling `backend/<name>/` module and a
`Backend` variant implementing the generic `Array`/`Stream`/ops surface; the
crate root re-exports the backend's modules, so call sites (`crate::array`,
`lisa_mlx::kernels`, …) do not change.

**Caveat — the backend trait boundary is not drawn yet.** For the Metal-only
models, `lisa-engine` still consumes the Metal surface directly: both the generic
`ops` and the qwen4 custom kernels (`lisa_mlx::kernels`, `qsa`, `moe_decode`,
`prefill_indirect`). A real second backend must supply, behind a trait, (a) the
generic ops the model uses (matmul, sdpa, rope, rms-norm, softmax, quantized
gemm/gemv, elementwise) and (b) either CPU implementations of the custom kernels
or model paths that avoid them. The seam is the `Array`/`Stream`/ops surface in
`shim_api` plus the model-kernel wrappers; `Backend` and `models::load` are the
entry points.

---

## 4. Build, run, verify

```bash
cargo build --release

./target/release/lisa smoke                            # fast-op + GDN kernel parity

M=$HOME/.cache/lisa-models/Qwen3.8-Flash-Next-MLX-4bit-MTP
G=reference/correctness_prompts/public_longcopy_gate_english_1024_256.json
./target/release/lisa golden --model "$M" --golden "$G"            # 256/256
./target/release/lisa golden --model "$M" --golden "$G" --depth 5  # 256/256
./target/release/lisa prefill-bench --model "$M" --golden "$G"     # steady-state
./target/release/lisa session-check --model "$M" --golden "$G" --chunk 16
./target/release/lisa golden --model "$M" --golden "$G" --depth 5  # MTP
```

No MLX dependency: the only Metal toolchain the build needs is `xcrun metal` at
runtime. The **generic** op kernels (elementwise, reductions, softmax, rope,
sdpa, gemm/gemv) are assembled from a static preamble plus explicit
`[[host_name]]` template instantiations and JIT-compiled; the **custom** engine
kernels (`kernels.rs`, `moe_decode.rs`, `qsa.rs`) are `include_str!` sources.
`LISA_METALLIB=<path>` forces an AOT metallib.

**One model-holding command at a time.** The 125B model is RAM-resident
(~77 GB); two at once can exhaust memory. Check for orphaned `lisa` processes
after an abort.

---

## 5. Correctness method

### 5.1 Golden (primary regression)

`lisa golden` prefills 1024 tokens, greedily continues 256, and compares every
token. **Must stay 256/256** serial and `--depth 5` after any change.

### 5.2 Fast oracle (2-token)

Drive the reference worker with a 2-token prompt and compare its `top_logits`
(f32) against lisa's `LISA_KEEP=2 LISA_DUMP_TOKENS=2 … --max-tokens 1` with
`LISA_TOPK`. Use this for kernel-level questions; it is far cheaper than the
full golden.

### 5.3 Bisect structurally first

1. Check block in/out exact (is the divergence inside one layer?).
2. Then micro (ulps). 1 ulp can flip an argmax through 48 chaotic layers.
3. For fused kernels, cross-check the oracle taps, never a generic op-chain dump.

### 5.4 `session-check`

`lisa session-check --chunk N` proves that feeding a token sequence in chunks
through one `Session` yields the same argmax as one full prefill. Chunked
prefill relies on this invariant.

---

## 6. Target model & config (`config.json` → `text_config`)

```
model_type qwen4_exp          hidden_size 2560        num_hidden_layers 48
hc_count 4                    hc_lowrank 320          head_dim 256
num_attention_heads 24        num_key_value_heads 2   full_attention_interval 4
num_experts 512               num_experts_per_tok 10  moe_intermediate_size 640
shared_expert_intermediate_size 640
linear_num_key_heads 16 (Hk)  linear_num_value_heads 48 (Hv)
linear_key_head_dim 128 (Dk)  linear_value_head_dim 128 (Dv)
linear_conv_kernel_dim 4
ple_layer_ids [2] (1-based → layer index 1)   ngram_size 3   heads_per_ngram 8
ple_embed_dim 2560   ple_conv_kernel_size 4   split_ngram_parts 128
vocab_size 248320    rms_norm_eps 1e-6        rms_norm_weight_offset 0.0
eos_token_id [248046, 248044]   max_position_embeddings 262144 (256K)
quantization affine, group_size 32, bits 4
```

12 full-attention layers (indices 3,7,…,47) and 36 linear-attention (GDN)
layers. Only the 12 full layers carry a KV cache (24 KB/token).

Architecture in one line: embed → tile to `hc_count` streams → 48 decoder layers
(each: an attention block [GDN or full] with a hyper-connection mixer+inject,
then an MoE block with its own hyper-connection) → final mixer → lm_head. Layer 1
also injects the PLE (per-layer n-gram embedding).

### Loading

`loader.rs` reads the text-tower tensors directly from the raw checkpoint and
applies the per-tensor numeric transforms the kernels need: the `q_proj`
`[q|gate]` row reorder, the `conv1d` `(C,1,K) -> (C,K,1)` transpose, the
`in_proj_all` row concat, and the `switch_mlp` packing. Vision/audio/ngram
tensors are dropped at load. No separate weight-transform pass is required —
loading + 256/256 is the proof of the contract. `config.rs` reads whichever of
`quantization` / `quantization_config` the raw config carries.

---

## 7. Numerics architecture (per-op; do NOT "simplify")

Everything below was established by bit-comparison against the oracle.

- **Hyper-connection mixer** is the **mean form**: `normed` full norm;
  `w = sigmoid(up(silu(down(normed)/hc_count)))`;
  `input = (w * normed).mean(axis=-2)` over the hc axis. `mean` on bf16
  accumulates in bf16.
- **RMSNorm** (`rms_row_exact`) layout `[slices,32,4]`: 4 consecutive elements per
  lane, per-lane sequential f32 square-sum, 32-lane xor butterfly, zero-pad
  partials to 32, second butterfly, `precise::rsqrt(total/H + eps)`, then
  `bf16(x*inv) * scale` — the weight is applied **after** the bf16 rounding.
- **silu** is the fused `nn::silu`. **logaddexp0 / softplus** is the op chain
  where every step rounds through bf16. **GDN decay**
  `g = exp(-exp(A_log.f32) * softplus(a + dt_bias))` with the bf16 add and bf16
  softplus.
- **GDN recurrence** (`gated_delta_step`): sequential over T, Kahan-compensated
  f32 state, `simd_sum` reductions, state written f32. Source in
  `kernels/models/qwen4/gated_delta.metal`; the two-row and four-row forms
  (`gdn_lean_two_row.metal`, `gdn_rows.metal`) emit each row's operations in the
  one-row order, so the output is independent of rows-per-simdgroup.
- **MoE:** router `matmul(x_f32, gate_bf16^T)`;
  `indices = argpartition(-logits, top_k-1, -1)[.., ..<top_k]`;
  **`weights = softmax_axis(selected, -1, precise)`**; expert gather is a
  gather-sort by expert (`order = argsort(idx_flat)`, `token_idx = order/top_k`,
  `x_sorted = x_rows[token_idx]` — the **original** rows); expert matmuls use the
  3-D lhs `[m,1,K]` with `sorted_indices=true`; combine is bf16 products then the
  K=10 `col_reduce_small` association in f32; shared expert is
  `sigmoid(shared_gate(x)) * down(silu(gate)*up)`; output `routed + shared*gate`.
- **PLE / n-gram:** a host splitmix64 identical to the device hash; the seed
  ordinal is the **index into `ple_layer_ids`** (not the layer number). Gate =
  `sqrt(max(|gate|,1e-6)) * sign(gate)`; raw gate = `sum(key*query)/sqrt(2560)`
  over the `[hc,hidden]` reshape. Conv: causal depthwise, dilation `ngram_size`.
- **RoPE:** partial rotary; cos/sin computed on device from integer positions
  (f32 `exp`/`cos`, not libm).

---

## 8. The bugs that mattered

After the ulp work, **exactly two bugs** accounted for the fidelity failure
(2/256 → 256/256). Both in the MoE.

1. **`softmax` with no axis is a GLOBAL softmax.** The no-axis overload reduces
   over *all* axes, so MoE gate weights summed to 1 across the whole `[B,S,top_k]`
   instead of 1 per token. Fix: `softmax_axis(&selected, -1, true)`.
   **Rule: always pass an explicit axis.**
2. **Expert x-gather indexed the repeated array.** `x_sorted` must index the
   original rows, not the `top_k`-repeated copy. Symptom: bit-exact for token 0,
   wrong for every token > 0.

Two more found later:

3. **MTP head priming** — the head's attention must see the whole committed
   history, including the prompt (§9.3).
4. **MTP EOS** — the loop only rolled back when `emitted < a+1`, so an EOS on the
   last accepted position was skipped and generation continued past EOS; the EOS
   was also fed into the caches. Fixed: break on EOS at any position and exclude
   it from the caches (`keep = emitted-1`).

**Latent hazard:** warmup must **not** use synthetic zero tokens. An
all-token-id-0 warmup corrupts persistent state (a later MTP run diverges around
token 38 while serial stays clean). `Tower::warmup` draws its tokens from the
real prompt. Do not "simplify" back to `vec![0; s]`.

---

## 9. MTP speculative decoding

Lossless: every emitted token is the target's own, so all depths stay 256/256.
`Session::generate_mtp` is the production driver; `lisa golden --depth N` uses it.

### 9.1 The head

`MtpHead` is two norms + `fc_embedding`/`fc_hidden`, one full-attention decoder
layer (`mtp.layers.0`, with an indexer), and a mixer. It owns no embedding table
and no output head: it rides the target's `embed_tokens`/`lm_head` and consumes
the target's **pre-final-mixer `multi` stream** (`hc_count * hidden` wide), not
the collapsed hidden. Draft = `argmax(lm_head(head_hidden))`.

### 9.2 The round

Draft (head over history + carry, then `depth-1` chain steps) → one target
forward over `[carry, d1..ddepth]` with `capture=true` → accept the longest prefix
`a` where the target argmax matches → emit `mainTokens[0..=a]` → roll back the
target caches to the committed prefix (attention `trim(advance-n)`; GDN
`rollback_to(n, conv_kernel)` using the captured per-position SSM state and conv
input; the PLE conv is captured and rolled back too) → advance the head history.

### 9.3 Priming (read before touching the head)

The head's attention must see the whole committed history including the prompt.
Feeding only the carry made the head attend 1 row instead of 1024 and collapsed
acceptance (83/64/47%). Prime with `tokens = prompt[1..] ++ [first]`,
`multis = multi[0..n]`.

That priming must be fed in **2048-token windows** through the head's own cache,
not as one whole-history forward: at `offset == 0` the head's attention takes the
dense fallback and builds `[S, kv]` masks, which at 100K is tens of GB. The
windows keep the head's `cache_offset` advancing so the block-sparse path serves
every window after the first.

### 9.4 Numbers (local M5 Max, 1024 prompt, greedy)

| config | lisa decode | accept | oracle decode | accept |
|---|---|---|---|---|
| serial | ~68–72 | — | ~51 | — |
| d1 | 48.6 | 97.7% | 42.5 | 96% |
| d3 | 85.6 | 90.8% | 44.3 | 92% |
| d5 | **~98–107** | ~79–82% | — | — |
| d6 | 64–76 | 66.7% | 27.4 | 67% |

d5 is the optimum. The MTP head is 4-bit group-32; its `switch_mlp` experts alone
are ~1.26 GB, so 8-bit blows the 2 GiB head cap — head re-quantization is not a
lever.

### 9.5 MTP with batching

The drafter is speculative batch 1 (`draftStep` requires `B == 1`), so MTP is
per-request and the scheduler interleaves. `sched::run(..., depth>0)` drives each
request on its own `Session`; `lisa serve` routes `depth>0` to the per-stream path
and `depth==0` to the continuous batch.

### 9.6 MTP ≠ serial on free-form text

Inherent, not a bug: the verify is one `S=depth+1` forward while serial runs
`S=1`, so different kernels dispatch and near-tie tokens can flip. A copy-style
golden is robust either way.

The ruled semantics is **argmax agreement through the wide path**: a wide verify
commits what the wide forward says, and "the MTP leg is token-exact against
serial" is *not* a supported claim. A depth that diverges from the public serial
golden at a near-tie is expected — do not "fix" it by narrowing the verify.

**Corollary:** the 2..8 MoE window must reach the wide fused kernels. The inline
router used a reshape valid only for one row, so any 2..8 window fell through to
a generic MoE; the wide kernels were dead code. Route via `router_logits`
(S=1 GEMV, else matmul/NAX). The wide path's first real exercise is the MTP
verify, so treat a wide-path near-tie flip as semantics, not a kernel bug.

---

## 10. Batching

- **Cohort** (`Batch::prefill`): N equal-length prompts, one batched forward,
  lockstep decode.
- **Ragged** (`Batch::prefill_ragged`): each stream prefilled alone, then packed
  right-aligned (per-stream `next_pos` + per-step `attn_mask`); auto-selected
  when lengths differ.
- **Continuous** (`ContinuousBatch`): streams admitted/retired on the fly; packed
  caches compacted on membership change; a lone stream collapses to a contiguous
  left-aligned cache and takes the exact single-stream path.
  `lisa cbatch --max-batch N`.
- B>1 numerics differ from single-stream (kernel shape → accumulation order);
  batched streams agree with each other exactly. The scored metric is
  single-stream.

---

## 11. Sampling

`src/sampler.rs`: temperature, top-k, top-p, min-p, repetition penalty, seeded
SplitMix64. GPU filtering (`argpartition_axis` + `take_along_axis`), host draw.
**Greedy (`temperature <= 0`) is the exact argmax path** the golden relies on.

**Non-greedy sampling does not work.** `argpartition_axis`/`argsort` implement a
single-block sort (one 2048-element block); the vocabulary is 248320, so they
bail with `argsort: multi-block sort not implemented`. That breaks
`--temperature > 0`, `--top-k`, `--top-p`, `--min-p`, the serve API's sampling,
and the `LISA_TOPK` diagnostic. Greedy is unaffected. The fix is a block sort
plus merge (see §18).

---

## 12. Serving (`lisa serve`)

Dependency-free OpenAI-compatible HTTP server: `POST /v1/chat/completions` (JSON
or SSE), `GET /v1/models`, `GET /health`. No async runtime: an accept thread
parses HTTP and hands jobs to the worker thread that owns the `Tower` over an
`mpsc` channel.

`lisa serve` dispatches on the model kind: a language model gets the chat
endpoint above; a **decision model** (Laya, §20) gets `POST /v1/decisions`
instead, handled inline (no scheduler).

- **Admission queue + continuous batching**: the worker blocks for one request,
  drains up to `--max-batch`, serves stateless `depth==0` jobs through the
  continuous batch, streams tokens to each connection. `depth>0` jobs take the
  per-stream MTP path.
- **`session_id` prefix reuse**: a persistent `Session` compares the new prompt
  against its committed tokens and prefills only the suffix (resets on
  divergence). `usage` reports the suffix length.
- EOS is stripped from content; `finish_reason` is `stop`/`length`; malformed
  JSON or bodies >8 MiB → 400; unknown paths → 404.

---

## 13. Long context, attention and QSA

The model is 256K. KV is 24 KB/token (12 full layers × 2 kv heads × 256 × 2 ×2 B)
plus ~3 KB/token of indexer tape.

### 13.1 Chunked prefill (done)

`Tower::prefill` / `prefill_multi` feed the prompt in `PREFILL_CHUNK = 2048`
windows through the same caches, bounding MoE activation buffers
(`[S*top_k, hidden]`) and attention masks. Prompts ≤2048 take the original
single-forward path. Also used by `generate`, `Session`, ragged/continuous
prefill; `warmup` primes the 2048 shape.

### 13.2 Context cap (done)

`generate` / `Session::feed` reject `prompt + max_tokens > max_position_embeddings`.

### 13.3 QSA beyond 2048

All goldens are 1024, so the >2048 path had to be built and debugged from
scratch. It is now correct, and was broken in several independent ways that are
worth knowing:

- the indexer built a dense keep mask and applied RoPE with wrong axes;
- `mean_axis` rode an undefined op (the binary kernels have no i32 variant);
  `mean` now uses the sum kernel;
- i32 Metal ops the runtime lacks (`q_pos + 1`, `maximum`, `floor_divide`) are
  computed host-side as f32;
- `where_cond` has no i32 output and no scalar broadcast, and comparison does not
  broadcast — select/compare in f32 over materialised operands;
- the MoE counting sort required `S*top_k % 256 == 0`, so a chunked-prefill window
  that is not a multiple of 128 fell to the argsort fallback; the count/scatter
  now pad a partial last block with an out-of-range sentinel id.

`LISA_INDEXER_DEBUG` prints when the keep mask activates. `FullAttentionCache`
carries `indexer_ok`: true only for a fresh single-stream cache (whose tape is the
full history); false for packed/compacted caches, so packed batches stay exact
causal.

### 13.4 Block-sparse attention + selector

- **Block-sparse attention**: streams only the selected 4-token blocks per
  `(query row, kv head)` — no dense `[S,L]` mask — via `matmul2d`.
- **Fused selector**: one dispatch producing `block_ids` / `block_valid`
  `[S,512]`, with the tie-break `score - block_id*1e-12`.
- The dense-keep SDPA is the fallback (non-NAX / off-contract geometry).

Both engage at **every width** once the indexer activates (`offset >= 2048`),
not just past 32K: the kernel is O(budget), never O(kv). The indexer's pooled keys
are maintained incrementally and lazily — raw keys in a pre-allocated in-place
buffer, each block pooled + roped once when it completes.

### 13.5 Numerics vs the oracle

The oracle serves >2048 by delegating to a dense-mask fallback; that fallback is
the reference. The chunked production path (block-sparse from chunk 2 on) matches
its greedy continuation over 16 tokens at every length tried — 2348, 4096, 5000
(three chunks, partial last), 8192 — 16/16 each.

### 13.6 Where long context stands

Decode plateaus near 17 tok/s and prefill degrades far less than a dense path
would (1268 at 15K → 1018 at 250K). The selector was the big bottleneck and was
cut 2.87× by:

1. **Vectorized operand loads** (`qsa_unpack_trunc`): 8 bf16 per transaction
   instead of one scalar load per multiply-add (the tf32 operand truncation is a
   no-op for bf16, so nothing is lost). 45.3 → 24.8 ms/call at 61K.
2. **Threadgroup query staging**: the query row does not depend on the block, so
   device q traffic drops from once-per-block to once-per-row. 24.8 → 15.8
   ms/call.

The remaining floor is the in-kernel radix select + the deterministic bitonic
ordering network (~11 ms/call, mostly threadgroup barriers) plus the pooled
re-reads. See §18 for the next step. Note the FLOP floor: even 2048-budget
attention at 256K is minutes of GPU. The goal is correct, memory-bounded, and not
`O(S²)` in the avoidable parts.

---

## 14. Memory and the buffer pool

- **Loaded model:** ~76.6 GB active (only the kept text-tower tensors; the
  checkpoint's vision/ngram tensors are dropped at load; mmap pages are
  file-backed and reclaimable, so peak == active).
- **The pool is byte-accounted and swept past a cap.** `set_cache_limit` stores a
  limit, `clear_cache` sweeps for real, and `trim_cache` sweeps once the device
  allocation passes the post-load baseline + limit (**8 GiB**). Wired at
  prefill-chunk, MTP-round, and every 32 decode tokens; `LISA_MEM_TRACE` prints
  each sweep. Verified: 27 sweeps on a 100K MTP run, device bounded at ~94 GB.
- **Pool reuse is gated on command-buffer completion, not a full flush.** Gating
  on the flush *epoch* meant a buffer was only handed back after a
  `flush_and_wait`; a single big prefill never flushes, so every intermediate
  allocated a fresh `MTLBuffer` and prefill collapsed to ~208 tok/s. The correct
  gate keeps the `strong_count == 1` reuse but makes it sound for
  `HazardTrackingModeUntracked` buffers: each `Buffer` records the id of the
  command buffer that last bound it, and a contiguous completion watermark over
  command-buffer ids (advanced by per-command-buffer completion handlers,
  out-of-order-safe) says when reuse is safe. Prefill 208 → ~2200 tok/s, decode
  ~68, golden 256/256. The `LISA_PLAIN_REUSE` shortcut (ignore the watermark)
  GPU-hangs — the gate is load-bearing.
- **The MTP head is primed in 2048-token windows** (§9.3). 100K MTP d5: peak
  79 GB, ~123 ms/round (draft 33 + verify 88), ~40 tok/s vs 17 serial.
- `LISA_PROFILE` prints a read-only `[mlx-mem …]` line per serve wave;
  `lisa inspect` prints one too.

---

## 15. Performance

Baseline (local M5 Max, contended box, directional):

| | lisa |
|---|---|
| prefill (1024) | ~2200 tok/s |
| serial decode | ~68–72 tok/s |
| MTP d5 | ~98–107 tok/s |
| golden | 256/256 (serial, d1..6) |

- **First-call Metal JIT was the old "prefill gap."** `Tower::warmup` (shapes
  1, 9, 40, 2048 from real prompt tokens) plus removing a per-layer
  `sorted_idx.eval()` barrier moved prefill ~850–1100 → ~1150 → ~2250.
- **Decode is memory-bandwidth bound.** ~4.2 GB weight traffic/token ⇒ ~7.7 ms
  at 546 GB/s; measured ~12 ms ⇒ ~63% of the roofline. MoE ~4.6 ms/token
  (~280 GB/s), dense remainder ~7.6 ms. `LISA_PROFILE_DECODE` splits build
  (`~2 ms`) vs eval (`~12 ms`).
- `lisa indirect-bench`: the NAX indirect expert GEMMs run ~29 TFLOPS.
- The scored harness measures the reference tree, not lisa; a full local run is
  impossible (public goldens are rejected by the model-identity loader). Local
  proxy composite `prefill^0.25 * decode^0.75` ≈ 1.7.

---

## 16. Kernel inventory

All kernels reachable by the scored path are implemented. Generic Metal kernels
live in `crates/lisa-mlx/src/backend/metal/kernels/{common,nax}`; the qwen4
kernels (sources + host wrappers) live in
`crates/lisa-mlx/src/backend/metal/models/qwen4/` and are re-exported at the
crate root as `lisa_mlx::{kernels,moe_decode,prefill_indirect,qsa,…}`.

The four NAX kernels (`matmul_nax`, `qmm_nax`, `sdpa_full_nax`,
`gather_qmm_rhs_nax`) compile from flattened header closures
(`nax_{gemm,quant,attn}_header.metal`) — the reachable Metal header closure with
quoted `#include`s inlined in dependency order; the build prepends the right
closure and runs `xcrun metal` with no include path.

Covered: decode + wide MoE (`route`, `gate_up_reuse_2row`, `gate_up_act`,
`down_combine`), decode mixer (`down_inject`, `up_mix`), decode router
(`router_gemv`), hyper-connection (`inject_norm`, `hc_mix`, `silu_head`), GDN
(`gdn_prep_split`, `gated_rms`, `gdn_rows`, `lean_two_row`, `gated_delta_step`,
`gdn_decode_complete`), MoE prefill (`moe_sorted_combine`, `prefill_tile_table`,
`prefill_indirect_gate_up`/`down`, `router_bf16_storage`, `router_split_sum`, the
fused `route_block_counts`/`route_counting_scatter` counting sort), attention
(`attn_prep_split`, `attn_gate`), PLE (`ple_prod`, `ple_gated`, `ple_conv`,
`prepare_fuse2`, `convolution_fuse2`), `swiglu2`, and the QSA kernels.

Not implemented (dead / alternative / perf-only, not on the single-stream path):
non-`p12` kernel aliases, the S==1 MoE fallback, the B>1 `mixer_head`, the unused
`gdn_input_prefetch`, `prefill_mixer_silu_epilogue`.

Several kernels carry order-preserving optimizations that keep them
bit-identical: vectorized operand loads in the router GEMV, one simdgroup per
shared-expert row in the S=1 down-combine, a 5-way split-K fold in the S=1
mixer, and vectorized strided/`copy_gg` loads.

---

## 17. CLI & diagnostics

Commands: `smoke`, `inspect`, `run`, `golden`, `layer-diff`, `tok`, `logits`,
`cache-test`, `neg-slice`, `rounding`, `qmm-m`, `indirect-bench`,
`decode-moe-bench`, `prefill-bench`, `session-check`, `chat`, `batch`, `cbatch`,
`decide`, `serve`.

**Behavior-selecting gates were removed.** The exact path is the only path. The
remaining `LISA_*` env vars are read-only diagnostics (they print/eval, never
change the result): `LISA_PROFILE`, `LISA_PROFILE_DECODE`, `LISA_PROFILE_MOE`,
`LISA_DEBUG_MTP`, `LISA_DEBUG_SESSION`, `LISA_DEBUG_INDIRECT`, `LISA_DEBUG_ROWS`,
`LISA_DIAG`, `LISA_DUMP_DIR`, `LISA_DUMP_TOKENS`, `LISA_KEEP`, `LISA_TRUNC`,
`LISA_TOPK`, `LISA_SHAPE_DEBUG`, `LISA_INDEXER_DEBUG`, `LISA_QSA_PROF`,
`LISA_MEM_TRACE`, `LISA_LDTOKENS`, `LISA_FROM`/`LISA_TO`. Real config:
`LISA_DEVICE` (backend, default `metal`); `LISA_MODEL_DIR` (model cache root);
`LISA_METALLIB` (AOT metallib path); `LISA_METAL_COMPUTE_PER_BUFFER`.
`--model` is a local directory or a Hugging Face repo id.

---

## 18. Traps / anti-patterns

1. **`softmax` no-axis is a global softmax.** Always pass an explicit axis.
2. **`gather_qmm` M-count defect.** A 2-D lhs `[M,K]` with 1-D `rhs_indices`
   returns `[M,M,N]`. Use the 3-D lhs `[M,1,K]` with `sorted_indices=true`.
3. **Quantized matmul is not bit-identical across builds at M>1.** At M=1 it
   matches upstream.
4. **Wrong oracle.** Do not compare fused-kernel intermediates to generic
   op-chain dumps; cross-check the oracle taps.
5. **1 ulp can flip the argmax** (final RMSNorm weight up to 13.4 × 48 chaotic
   layers). Verify structurally first, then micro.
6. **Never warm up with token-id-0 inputs** (§8).
7. **Everything must dispatch through one command queue.** Metal does not order
   work across queues; custom kernels share buffers with the generic ops.
8. **Host readback must wait for the producing kernels.** The eager arrays have
   no graph retaining intermediates, so the CPU runs ahead of the GPU and an
   early `contents()` read sees pre-kernel data.
9. **Only the `_strided` kernel variants are dispatched** (the contiguous ones
   differ solely in the indexer).
10. **Compile options are per-kernel:** the generic MLX kernels need
    `MathModeSafe` + `Precise` + an explicit language version, while the vendored
    elementwise kernels need `MathModeFast` + `Fast`.
11. **One model-holding command at a time.**
12. **A partial last block breaks the MoE counting sort.** The counts kernel
    returns all-zero counts for any block that is not 256 rows; the zeroed block
    corrupts the scatter, and the tile table over it overflows its scratch (a GPU
    page fault). Pad the ids with the sentinel for the counts kernel and skip
    sentinels by value in the scatter. The fault is content/size dependent and
    vanishes under per-layer eval — treat a GPU address fault that a sync hides
    as a possible data bug, not a fence race.
13. **The O(capacity) append.** A `slice_assign` that pads to the full capacity
    and runs `where_cond` is O(capacity), not O(s). The KV cache and indexer tape
    use a row copy that writes only the appended rows.
14. **`argsort`/`argpartition` handle one 2048-element block** (§11).
15. **Crate boundaries:** CLI `lisa-cli`, server `lisa-serve`, library
    `lisa-engine`, kernels `lisa-mlx`; the root `Cargo.toml` is a virtual
    workspace.

---

## 19. Remaining work

1. **Non-greedy sampling** (§11): implement a block sort + merge for
   `argsort`/`argpartition` so the vocabulary-wide reduction works. Fixes
   `--temperature`/`--top-k`/`--top-p`/`--min-p`, the serve API's sampling, and
   `LISA_TOPK` in one change.
2. **QSA selector, round two** (§13.6): the rest is ~11 ms/call of radix select +
   the bitonic ordering network (mostly threadgroup barriers) plus per-row pooled
   re-reads. A `matmul2d` score sheet with row-tiling reuse plus a split select is
   the next step — keep the block selection bit-identical and re-validate against
   the oracle continuation.
3. **Decode MoE GEMVs**: ~2.3 ms/token of the ~12 ms decode is the gap to the
   roofline. A software-pipelined power-group load was implemented and measured
   **neutral** on the M5 Max (`moe.gate_up` 0.341 → 0.337 ms; A/B decode
   indistinguishable), so it was reverted — do not retry without a quiet box and
   a decode-only benchmark. A wholesale GEMV rewrite risks the bit-exact
   accumulation that keeps 256/256.
4. **Long-context MTP draft attention**: a sliding-window head-only draft would
   bound draft memory/cost as context grows; drafts are verified, so it changes
   acceptance rate, not correctness.
5. **State snapshots at prefix boundaries**: snapshot the f32 SSM state, conv
   carry and PLE conv state at fixed token boundaries so a new turn with a shared
   prefix prefills only the suffix, instead of all 36 GDN layers. Snapshot memory
   is bounded (LRU).
6. **QSA cache merge for batched decode** (there is no batched MTP).
7. **INT8 KV** and other KV re-quantization: only relevant past ~32K where KV
   stops being <1% of decode traffic, and it changes logits — acceptable only
   under argmax-agreement semantics. Last.
8. **A second model** (§3.1): add a `<name>/` under `models/` and a `model_type`
   arm. `core/` is already generic; the work is the model's config, tower and
   kernels.
9. **Draw the backend trait boundary** (§3.2): put the generic ops and the
   model-kernel wrappers the engine uses behind a trait, so a CPU backend can
   implement them. The seam is `backend/metal/shim_api` plus the qwen4 wrappers;
   `Backend`/`models::load` are the entry points.
10. **Laya forward** (§20): implement the ModernBERT encoder + decision head in
    `models/laya/`, using the generic ops (needs LayerNorm, GELU/GeGLU,
    bidirectional + sliding-window SDPA, standard RoPE). Then the typed-question
    I/O (`lisa decide` becomes real) and a serve endpoint.

### Verification checklist (after every change)

```bash
cargo build --release
M=$HOME/.cache/lisa-models/Qwen3.8-Flash-Next-MLX-4bit-MTP
G=reference/correctness_prompts/public_longcopy_gate_english_1024_256.json
./target/release/lisa smoke
./target/release/lisa golden --model "$M" --golden "$G"            # 256/256
./target/release/lisa golden --model "$M" --golden "$G" --depth 5  # 256/256
./target/release/lisa prefill-bench --model "$M" --golden "$G"
./target/release/lisa session-check --model "$M" --golden "$G" --chunk 16
```

- Decode: `LISA_PROFILE_DECODE`, `LISA_PROFILE_MOE`.
- Long context: the oracle continuation at 2348/4096/5000/8192 (16/16),
  `LISA_QSA_PROF`, `LISA_MEM_TRACE`.
- One model-holding command at a time. Measure on a quiet box; background GPU
  load swings numbers ~2×.

---

## 20. Laya (non-generative decision models)

Laya (`convaiinnovations/laya`, Apache-2.0) is a **non-autoregressive typed
decision model**: give it a *state* and *typed questions*, one forward pass
returns calibrated answers (~421M params). ModernBERT-large encoder + a 2-layer
decision head, an option-marker scorer, and an act head. It does not generate
text, so it is **not** driven by `core/`'s generation runtime; it is a separate
`DecisionModel` kind (`models::Loaded::Decision`) with its own CLI (`lisa
decide`) and, later, its own serve endpoint.

**Checkpoint layout (configless).** No root `config.json`:

```
model.safetensors              # original HF weights, fp16, one file
encoder/config.json            # ModernBERT config
rl_agent_config.json           # decision-head/agent config
tokenizer/tokenizer.json       # Metaspace pretokenizer
tokenizer/tokenizer_config.json
```

`multilingual/` and `typed-decisions/` are sub-checkpoints with the same shape.
`model_type_of` classifies a directory with `encoder/config.json` +
`rl_agent_config.json` as `"laya"`. We load the **original HF weights** (original
names, fp16) — not an MLX-converted checkpoint.

**Config (English checkpoint).** 28 layers, hidden 1024, 16 heads (head_dim 64),
intermediate 2624, vocab 50368; layers alternate 10 full / 18 sliding
(`local_attention` 128); RoPE base 160000 (global) / 10000 (local);
`norm/attention/mlp` biases false, `hidden_activation` gelu; head 2 layers,
`max_len` 512, `head_max_len` 192, `n_actions` 2; per-type temperatures plus
`temperature_by_options` buckets (choice/score/noul, clamped [0.5, 5] except
noul).

**Original tensor names.** `encoder.embeddings.tok_embeddings.weight`,
`encoder.embeddings.norm.weight`, `encoder.layers.{i}.{attn.Wqkv,attn.Wo,
mlp.Wi,mlp.Wo}.weight`, `encoder.layers.{i}.{attn,mlp}_norm.weight` (layer 0 has
no `attn_norm` — it reuses `embeddings.norm`), `encoder.final_norm.weight`,
`head.layers.{0,1}.*` (norm1/norm2, `self_attn.in_proj_weight`,
`self_attn.out_proj`, `linear1/linear2`, all with bias), `scorer.*`,
`act_head.*`, `type_emb.weight`, `temperature`. Linear layouts already match
(PyTorch `[out,in]` == MLX); the MLX port renames
`in_proj_weight -> in_proj.weight` and `scorer.N -> scorer.layers.N`.

**Architecture (to port).** Pre-norm ModernBERT: `x = x + attn(norm(x))`,
`x = x + mlp(norm(x))`; `Wqkv` → 3×(heads,head_dim), standard RoPE,
SDPA with boolean key masks (`full` = valid keys; `sliding` = `|i-j| <=
local_attention/2`, and padded queries may see valid keys so no row is
all-masked). MLP = `Wi` → split into (value, gate) → `Wo(gelu(value) * gate)`.
Head layer = `x + self_attn(norm1(x))` then `x + linear2(relu(linear1(norm2(x))))`
(**ReLU**, the PyTorch default, unlike the encoder's gelu). Decision forward:
`h = encoder(ids, mask) + type_emb(qtype)`; `h = head(h, key_mask)`;
gather `h` at marker positions; `logits = scorer(markers)`; mask → `-1e4`;
softmax; features `[top1, top1-top2, entropy, k/255]`; `pooled = [h[:,0],
features]`; `action = act_head(pooled)` where `act_head` is `Linear(dims+4,256)
→ gelu → Linear(256, n_actions)`.

**Status.** The full forward is implemented and verified against the reference:

- `LayaConfig` parse/validate (`encoder/config.json` + `rl_agent_config.json` +
  tokenizer), tokenizer load with special-id resolution, original-weight open +
  tensor verification, the `DecisionModel` registry arm, and
  `lisa decide --model <dir|hf-id> [--probe]`.
- **Encoder + head forward** (`models/laya/forward.rs`), composed from the
  generic ops: pre-norm ModernBERT (LayerNorm built from mean/var, exact GELU via
  the `Erf` unary, GeGLU MLP), attention via matmul/softmax with a boolean key
  mask, the 2-layer decision head (ReLU MLP), the option-marker scorer, and the
  act head. B=1, f32 compute (fp16 weights cast up).
- **RoPE note:** the generic `fast::rope` kernel does **not** reproduce MLX's
  `mx.fast.rope` on this layout, so Laya applies GPT-NeoX RoPE explicitly
  (`rope_neox`: host cos/sin, rotate pairs `(i, i+d/2)`).
- Verified on `--probe` (ids `[1000..6000, 7, 8]`, qtype 0, markers `[6,7]`):
  encoder matches the fp16 reference to f32 rounding (final `[-0.7324 …]` vs
  `[-0.7324 …]`), logits `[0.4396, 0.4287]` vs `[0.4451, 0.4329]`, action within
  ~0.2%. Reference harness: `laya-mlx/laya_mlx/model.py` run in Python MLX.

**Devices.** Laya runs on Metal by default and on **CPU** with `LISA_DEVICE=cpu`.
The host path (`models/laya/cpu.rs`) is a self-contained f32 forward (rayon over
rows) that mirrors the Metal path op for op and shares the `prompt.rs` I/O; both
devices agree to ~5 decimals. CPU is ~0.85 s/question (vs ~30 ms on Metal).

The typed-question I/O is implemented (`models/laya/prompt.rs`): `state` +
typed `questions` are rendered into the marker sequence (`[CLS] <type> question:
<instructions> [SEP] [MASK] opt0 … [SEP] <state> [SEP]`), the marker logits are
temperature-scaled (per-type, plus `temperature_by_options` buckets, clamped),
and answers are derived (`choice` label + probabilities; `score` expected value;
`noul` yes-probability; entropy `confidence`; `act_probability`). Run it with:

```
lisa decide --model convaiinnovations/laya \
  --state "…" \
  --questions '{"department":{"type":"choice","instructions":"…","criteria":{…}}}'
# or: --input request.json  ({"state":…,"questions":…})
```

`lisa decide` also exposes overrides that map onto `LayaConfig` (applied after
load, re-checking `4 < head_max_len < max_len <= max_position_embeddings`):
`--head-max-len N`, `--max-len N`, `--temperature c,s,n` (exactly 3),
`--temperature-by-options TYPE:SIZE=TEMP` (repeatable), `--device metal|cpu`,
and `--top-k K` (display: trims each answer's `probabilities` to the K largest).
Example: `lisa decide --model convaiinnovations/laya --input q.json
--head-max-len 512 --max-len 2048 --temperature-by-options choice:6-10=0.7
--top-k 3`.

**Temperature overrides.** Precedence is `--temperature-by-options` >
`--temperature` > shipped config: passing `--temperature` clears the shipped
buckets so it can win, and `--temperature-by-options` is inserted afterwards so
it wins over both. A `--temperature-by-options` key is validated — its size must
be one of the four buckets the lookup selects (`2`, `3-5`, `6-10`, `11+`);
anything else (e.g. `choice:21-40`) is rejected. Clamping is never silent: an
override outside `[0.5, 5.0]` (or non-positive/non-finite, which becomes 1.0)
prints a warning and the applied value is used (noul is exempt from the bounds).
Each answer reports its effective `temperature` and `bucket`.

Verified against the published model-card example (department→billing 0.933 vs
0.94; churn 0.879 vs 0.892; urgency 1.77 vs 1.84). `lisa serve --model
convaiinnovations/laya` dispatches to the decisions endpoint:
`POST /v1/decisions` (`{"state":…,"questions":…}` → answers), `GET /v1/models`,
`GET /health`; requests are handled inline on the accept thread (the model is a
pure function, no scheduling).

**Verification.** `cargo test -p lisa-engine --lib laya` parses the real configs
(fixtures in `crates/lisa-engine/tests/fixtures/`). End-to-end:
`hf download convaiinnovations/laya model.safetensors encoder/config.json
rl_agent_config.json tokenizer/tokenizer.json tokenizer/tokenizer_config.json
--local-dir DIR` then `lisa decide --model DIR`. Behavioral references: the
repo's own `rl_agent_*.py`, and the `laya-mlx` port (`laya_mlx/model.py`,
`laya_mlx/agent.py`), which also loads the original checkpoint.

---

## 21. Licensing

Third-party kernel sources and their licenses are recorded in `NOTICE`. Keep it
current when adding or adapting a kernel.