# Architecture

LISA is a Rust inference engine with its own objc2 Metal runtime, JIT-compiled
kernels, CLI, HTTP server and native macOS UI. MLX and DS4 provide reference
implementations and kernel sources; runtime orchestration belongs to LISA.
Licenses and source attribution are recorded in [NOTICE](../NOTICE).

## Workspace boundaries

The workspace uses Rust edition 2024 and Cargo resolver 3.

| Crate | Owns |
| --- | --- |
| [lisa-mlx](../crates/lisa-mlx/src/) | Arrays, layouts, allocation, Metal dispatch, JIT and shaders |
| [lisa-gguf](../crates/lisa-gguf/src/) | GGUF parsing, shard discovery, mapped storage and CPU block decoding |
| [lisa-engine](../crates/lisa-engine/src/) | Models, loading, sampling, generation, sessions and caches |
| [lisa-serve](../crates/lisa-serve/src/) | HTTP parsing, API translation, response streams and request scheduling |
| [lisa-cli](../crates/lisa-cli/src/) | User command entry point |
| [lisa-bench](../crates/lisa-bench/src/) | Compiled diagnostics, numerical checks and benchmark instruments |
| [lisa-ui](../crates/lisa-ui/src/) | AppKit interface, curated downloads and saved chats |

Metal sources live under `lisa-mlx/src/shaders/`; Rust wrappers live under
`jit/` and `models/`. Shared architecture components such as GDN live at the
models root. Model-independent state and scheduling live in
`lisa-engine/src/core/`; `LanguageModel` supplies the model hooks.
`lisa-gguf` does not depend on the engine or Metal. The engine's
`core::gguf_resident` adapter owns GPU views and their resident reservations.

## Kernel packaging

The `lisa-mlx` build script writes a losslessly compressed `kernels.bin` beside
the Cargo executables. It contains the original Metal/header source bytes in
sorted path order; generated Rust ranges and a SHA-256 digest bind the package
to that build. The runtime reads, validates and decompresses it once before
creating the Metal device. Dispatch accesses immutable memory after startup.
Kernel arithmetic, source assembly and JIT compilation remain unchanged.

Ship the matching package beside a standalone executable, or in an app's
`Contents/Resources`. Library users must also distribute it with their host
executable. Missing, malformed or mismatched packages fail initialization;
there is no network or source-tree fallback. Cargo test executables also search
their parent profile directory.

## Checkpoint dispatch

[Model resolution](../crates/lisa-engine/src/models/checkpoint.rs) accepts a local
directory, a supported GGUF file, or a Hugging Face repository ID. Repository
IDs resolve through the hub cache and model-aware download selection. Config
and tensor geometry determine support, not the repository's display name.
`ModelKind` centralizes metadata aliases for the loader and desktop discovery.
CLI, UI and offline diagnostics share snapshot ordering: active `refs/main`
first, then directory modification time with a deterministic path tie-break.
Incomplete `.lisa-partial` snapshots are excluded. Presence of an entry point
does not replace weight/package validation when opening the checkpoint.

| Metadata | Accepted identity | Execution path |
| --- | --- | --- |
| `model_type` | `qwen4_exp`, `qwen4_exp_text` | Qwen GDN/attention + MoE |
| `model_type` | `qwen3_5`, `qwen3_5_text` | Qwen dense hybrid |
| `model_type` | `glm5_next`, `glm5_next_text` | Legacy MLX GLM |
| GGUF `general.architecture` | `glm5-next` | Native packed GLM |
| GGUF `general.architecture` | `deepseek41` | Native packed DeepSeek |
| `model_type` | `flux2-klein`, `flux3-klein` | Image generation pipeline |
| `model_type` | `laya` | Typed decision model |

Language-model and image inference require Metal. Laya also has a CPU path.
`LISA_DEVICE` selects an implemented backend; it does not provide missing CPU
implementations for the language models.

## Execution and ownership

Text requests pass through tokenization/template rendering, prefill and a
model-supported generation policy. Qwen uses its draft/verify controller;
GLM uses the target-MTP transaction; the current DeepSeek checkpoint decodes
serially. Batch support is a model capability, not inferred from tensor rank.

Sessions own committed token history, sampler state and model caches. Prefix
snapshots are reusable only under the same numerical law. Native cache types
can support RAM snapshots without supporting the generic SSD serialization
format. See [serving](serving.md) for the current reuse limits.

Safetensors loading maps original payloads. Legacy MLX GLM additionally needs
a byte-preserving aligned resident index prepared inside its checkpoint
directory. GGUF uses packed blocks directly, without that preparation step.
DeepSeek keeps its large Engram table on disk and reads selected rows.

Keep mappings and allocations alive until GPU users complete. Resident weight
reservations, private allocations and cache estimates are separate accounting
categories. One model is loaded at a time; a model switch drops the previous
model before loading its replacement. [Contracts](contracts.md) define the
allocation, reuse and teardown rules.

## Media paths

Image input adds preprocessing and a supported vision tower before text
generation. Image turns use a fresh serial session and full multimodal history.
They do not use text prefix snapshots or speculative decoding.

Image generation uses `ImagePipeline`, with staged text-encoder, denoiser and
VAE loading. It has its own HTTP worker and generation endpoint. It does not
run through language-model chat completion. See [media](media.md) for supported
formats, limits and evidence.
