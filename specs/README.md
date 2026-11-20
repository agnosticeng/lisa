# Specifications

These documents describe current architecture decisions, behavior and constraints.
Implementation details belong beside source; this folder contains no migration
history, benchmark reports or audit diaries.

| Topic | Owns |
| --- | --- |
| [Architecture](architecture.md) | Crate boundaries, device API, plans, loading and backend extension |
| [Contracts](contracts.md) | Numerical, memory, state and response obligations |
| [Serving](serving.md) | HTTP surface, scheduling, sessions and admission |
| [Desktop](desktop.md) | SwiftUI/Rust ownership, local chats, downloads and packaging |
| [Media](media.md) | Vision, voice and image-generation paths |
| [Qwen](models/qwen.md) | Dense/MoE families, DSpine, MTP and GGUF policies |
| [LFM3](models/lfm3.md) | Compact hybrid text models, sliding state and reference scope |
| [GLM](models/glm53.md) | Native GGUF and legacy MLX execution contracts |
| [DeepSeek](models/deepseek41.md) | Mixed GGUF, Engram and native state |
| [Performance](performance.md) | Optimization decisions and validation protocol |
| [Open work](roadmap.md) | Unresolved architectural and acceptance gaps |

## Model status

Text and image generation use Metal. Laya also supports CPU. ROCm has no provider.
Qwen supports checkpoint-dependent batching, vision and drafting; capabilities
are not interchangeable between MLX and GGUF. GLM and DeepSeek remain
experimental against their DS4 reference. See each model's document for limits.
The [catalog](../crates/lisa-app/src/catalog.rs) owns curated repository selection.

Update the owning topic when a decision changes. Keep source licenses, reference
pins and required fixture identities beside the implementation. New test counts,
run logs and performance campaigns do not belong in these specs. An implemented
path is not a claim of numerical parity or performance on every device.
