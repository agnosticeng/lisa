<p align="left">
  <img src="https://img.shields.io/badge/license-MIT-brightgreen" alt="License: MIT">
  <img src="https://img.shields.io/badge/language-Rust-orange" alt="Rust">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black" alt="Apple Silicon">
  <img src="https://img.shields.io/badge/version-0.1.0-blue" alt="version">
</p>

# lisa

**LLM Inference for Silicon Architecture**

`lisa` is a **pure-Rust, from-scratch inference engine** for large language
models on Apple Silicon — no Python, no MLX, no tensor framework. The backend
talks to the Metal C API directly through `objc2` and compiles its `.metal`
shader sources at startup, so the whole stack ships as one native binary.

It is built on two open extension axes — **models** and **device backends** — so
the first model it implements is not the last:

- **Models.** Today: `Qwen 3.8 Flash-Next` (125B total / A6B active), a hybrid
  gated-DeltaNet + full-attention decoder with a 512-expert sparse MoE, a
  per-layer n-gram embedding, and MTP speculative decoding on the M5's NAX
  tensor cores — and `Laya`, a non-generative typed-decision model. Bringing up
  another model is a `<name>/` module plus one `model_type` arm.
- **Devices.** Metal (GPU) and CPU, selected with `LISA_DEVICE` and defaulting
  to Metal when a device is present, CPU otherwise. Adding a backend is a
  `backend/<name>/` module plus one `Backend` variant.

On top of the engine: serial generation, multi-turn sessions, cohort / ragged /
continuous batching, and an **OpenAI-compatible HTTP server**
(`/v1/chat/completions`, plus `/v1/decisions` for decision models).

It stays exact where it matters: **256/256** on the public long-copy golden,
serial and MTP `--depth 1..6`.

## Install

Build from source on any Apple Silicon Mac (Xcode 26 / Metal 4 toolchain):

```bash
cargo build --release      # -> target/release/lisa
```

Model weights come from the Hugging Face cache; fetch them with the `hf` CLI:

```bash
brew install hf            # or: pip install -U huggingface_hub
```

## Run

`--model` takes a Hugging Face repo id or a local directory:

```bash
lisa serve --model <hf-repo-id|local-dir> --addr 127.0.0.1:8000
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Explain MTP speculation in one sentence."}],
    "max_tokens": 64
  }'
```

Any OpenAI-compatible client works: base URL `http://localhost:8000/v1`.

Typed decisions (Laya) use their own endpoint:

```bash
lisa decide --model convaiinnovations/laya \
  --state "We were billed twice; refund or we cancel." \
  --questions '{"department":{"type":"choice","instructions":"Which department?","criteria":{"billing":"payments","technical":"bugs","other":"misc"}}}'
```

`lisa decide` also takes `--device cpu`, `--head-max-len` / `--max-len`, and
temperature overrides (`--temperature`, `--temperature-by-options`). See
`AGENTS.md` §20.

## Architecture

```
crates/
  lisa-mlx/          tensor runtime: backend selection + the Metal backend
    src/backend/       Backend (metal|cpu); metal/ = runtime, Array/ops, JIT
    src/backend/metal/kernels/                 generic .metal (gemm, rope, sdpa, …)
    src/backend/metal/models/qwen4/kernels/    qwen4 .metal sources
  lisa-engine/       the inference library
    src/models/mod.rs  LanguageModel/DecisionModel traits + registry + resolution
    src/models/qwen4/  the Flash-Next implementation
    src/models/laya/   the Laya decision model (Metal + host forward)
    src/core/          loading, quantization, norms, caches, generation,
                       session, batching, scheduler, sampling, tokenizer
  lisa-serve/        OpenAI-compatible HTTP server (+ decisions endpoint)
  lisa-cli/          the `lisa` binary
```

### The hybrid decoder

`full_attention_interval = 4` → **full causal attention** at layers 3, 7, …, 47,
**gated DeltaNet** elsewhere (36 of 48 layers).

- **GDN** — short causal conv + SiLU, q/k RMSNorm, a gated-delta recurrence with
  a Kahan-compensated **f32** state `[48,128,128]`, gated output RMSNorm.
- **Attention** — 24 Q heads / 2 KV heads, head_dim 256, `q|gate` projections,
  partial RoPE (f32 cos/sin computed on device).
- **MoE** — fp32 router, top-10 of 512 experts, gather-sort by expert, bf16
  combine; a shared expert adds `sigmoid(gate) · down(silu(g) · u)`.
- **PLE / n-gram** — a host hash + mmap'd shard table, gated embedding and a
  dilated causal conv, injected at layer 1.
- **MTP** — the embedded head drafts; the target verifies in one wide forward
  and the caches roll back to the committed prefix.

### Zero-framework Metal

The runtime binds `MTLCreateSystemDefaultDevice` and the Objective-C runtime
directly through `objc2` — no `metal-rs`. Kernels are compiled from embedded
`.metal` source at startup; weight shards are mmap'd into shared,
hazard-untracked GPU buffers. The NAX kernels use flattened Metal header
closures (`kernels/nax/`).

## Models & devices

A model is addressed by a **local directory or a Hugging Face repo id**; its
structure is read from the config and dispatched to the matching implementation
(`qwen4_exp` or `laya`). A device backend is chosen with `LISA_DEVICE`
(`metal` or `cpu`); when unset, Metal is used if a device is present, otherwise
CPU. Today: `qwen4_exp` on Metal, and Laya on Metal **or** CPU.

Adding a model is a `<name>/` module plus one `model_type` arm; adding a backend
is a `backend/<name>/` module plus one `Backend` variant — see `AGENTS.md`
§3.1–3.2.

## Tests

```bash
cargo test                          # units (config/fixtures, kernels)
cargo build --release
./target/release/lisa smoke                            # fast-op + GDN parity
./target/release/lisa golden --model <dir> --golden <golden.json>          # 256/256
./target/release/lisa golden --model <dir> --golden <golden.json> --depth 5
```

## References

| Project | Role |
|---------|------|
| [candle](https://github.com/huggingface/candle) | concise Metal kernel cases (GEMM/GEMV, indexing) |
| [mlxfast-qwen38-125b-a6b-engine](https://github.com/Layr-Labs/mlxfast-qwen38-125b-a6b-engine) | correctness oracle and fused-kernel reference for this model (vendored under `reference/`) |
| [MTPLX](https://github.com/youssofal/MTPLX) | MTP speculative-decoding reference: block verify, context-copy, capture-commit rollback |
| [mlx-serve](https://github.com/ddalcu/mlx-serve) | native Apple-Silicon serving reference: selector split-select, shortlists, prefix cache |

Third-party kernel attribution is recorded in [NOTICE](NOTICE).

## License

MIT — see [LICENSE](LICENSE). Third-party kernel attribution is recorded in
[NOTICE](NOTICE); model weights and any upstream shaders are subject to their
own terms.