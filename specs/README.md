# Specifications

These documents describe LISA as it is: responsibilities, interfaces,
execution, invariants, supported modes and known limits. They contain no
history, benchmark results or investigation notes. When a decision changes,
update the owning document.

| Document | Owns |
| --- | --- |
| [Architecture](architecture.md) | Crates and their boundaries, device API, plans, replay, kernel packaging |
| [Runtime](runtime.md) | Request flow, loading, memory admission, sessions, caches, speculation, scheduling |
| [Numerics](numerics.md) | Numerical laws, strict paths, the verify exception, state identity, validation |
| [Serving](serving.md) | HTTP surface, response semantics, history and request admission |
| [Apple app](apple-app.md) | SwiftUI/Rust split, chats, downloads, lifecycle and packaging |
| [Media](media.md) | Image input, voice and image generation |
| [Open work](roadmap.md) | Unresolved gaps and acceptance conditions |

Model pages: [Qwen](models/qwen.md) · [GLM-5.3](models/glm53.md) ·
[DeepSeek V4.1](models/deepseek41.md) · [Inkling 2](models/inkling2.md) ·
[Olala](models/olala.md) · [LFM3](models/lfm3.md)

## Capability matrix

All language and image models run on Metal. Laya also runs on the CPU.

| Model | Formats | Drafting | Batching | Images | Status |
| --- | --- | --- | --- | --- | --- |
| Qwen 3.8 27B | MLX, GGUF | MTP head (MLX `mtp/` sidecar, GGUF embedded) | MLX | MLX | Supported |
| Qwen 3.8 Flash-Next | MLX | MTP head | Yes | Yes | Supported |
| Qwen 3.5 (0.8B, 4B) | MLX | — | Yes | Yes | Supported |
| GLM-5.3 Flash | GGUF, MLX | Target MTP (embedded) | MLX | MLX | Experimental |
| DeepSeek V4.1 Flash Coder | GGUF | — | — | — | Experimental |
| Inkling 2 Mini | GGUF | DFlash2 drafter (tree rounds) | — | Yes | Experimental |
| Olala v1.0 mini | GGUF | — | — | — | Experimental |
| LFM3 1.7B | MLX | — | — | — | Supported |
| FLUX.2 / FLUX 3 Klein | MLX, BF16 | — | — | Generation | Supported |
| Laya | JSON + weights | — | — | — | Supported (CPU, Metal) |

Only draft heads whose weights ship with the model are supported. "Supported"
means wired end to end and covered by goldens or reference fixtures; it is not
a claim of parity at every context length or on every device. The app catalog
([`catalog.rs`](../crates/lisa-app/src/catalog.rs)) owns curated repositories.

## Rules for these documents

- State the current behaviour and its limits; put open work in the roadmap.
- Name external projects only in [NOTICE](../NOTICE). Reference revisions live
  in the provenance files beside the code that ports them.
- Keep fixture identities and licences beside their implementation and large
  captures outside the repository.
- An implemented path is not a parity or performance claim on every device.
