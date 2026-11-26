# Open work

The single list of unresolved gaps. Implemented behaviour belongs in the owning
spec; this file makes no completion or throughput estimate.

## Model acceptance

| Path | Remaining work |
| --- | --- |
| GLM GGUF | Attribute the layer-3 reference mismatch; packed wide projections, grouped expert prefill and wide verification; prove accepted, rejected and resumed MTP state |
| GLM MLX | State and serving coverage, independent same-weight reference identity, full-checkpoint acceptance, vision attention/embedding drift |
| DeepSeek | Broader real contexts, ring and DSML coverage, grouped expert prefill, long-context performance, native state spill |
| Qwen 27B GGUF | Small-memory hardware, long contexts, independent MTP-head comparison |
| Qwen 0.8B | Output, state and continuation on physical mobile devices |
| Qwen Flash-Next | Chunked versus full-prefill state drift; broader batching shapes; full-checkpoint vision |
| Inkling 2 | Per-layer reference parity; calibration rows measured on Apple7–9 and Apple11 hardware; task and image quality; contexts past 43,008 tokens |
| Olala | Per-layer reference parity, long contexts, wider grouped prefill |
| FLUX | Broader stage, image and size coverage; isolated performance acceptance |

## Engine and serving

- Byte-budgeted prefix and session retention with validated persistence
  identity; a versioned spill codec for native caches.
- Prompt/session compatibility and exact suffix-resume checks that preserve
  target and draft state together.
- Cancel active work at safe boundaries when a client disconnects.
- Shared speculative forwards across clients (independent masks, samplers and
  rollback per client).

## Backends

A ROCm provider needs kernels, import/completion and memory mechanisms plus
application registration, then per-model numerical and state checks on AMD
hardware. Plans, caches, sampling and speculation are already generic.

## Product

- Physical-device qualification of app lifecycle, memory warnings, background
  GPU builds, browser, voice, image generation and serving.
- Resolve the Simulator BF16 compiler-service failure without changing weights
  or arithmetic.
- Checkpoint-level parity commands for GLM, Inkling and FLUX that compare
  against reference outputs through `lisa-bench --model`.
- Move GLM and LFM3 prompt builders out of `core/tokenizer.rs` into their model
  bindings.
