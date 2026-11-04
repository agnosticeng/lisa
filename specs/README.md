# LISA specifications

Current implementation, requirements and recorded evidence. Model measurements
apply to their stated checkpoint, workload and comparison; they do not certify
subsequent implementation changes.

## Read by topic

| Document | Contents |
| --- | --- |
| [Architecture](architecture.md) | Crates, loading, runtime ownership and execution paths |
| [Contracts](contracts.md) | Numerical, memory, state and response requirements |
| [Qwen](models/qwen.md) | Hybrid towers, MTP, PLD and batching |
| [GLM-5.3](models/glm53.md) | Current GGUF port and separate legacy MLX path |
| [DeepSeek V4.1](models/deepseek41.md) | Mixed GGUF, disk-backed Engram and DS4 comparison |
| [Media](media.md) | Image input and native FLUX generation |
| [Serving](serving.md) | HTTP surfaces, scheduling, sessions and limits |
| [Desktop](desktop.md) | Model catalog, chats, image previews and navigation |
| [Performance](performance.md) | Scoped results, rejected claims and measurement rules |
| [Roadmap](roadmap.md) | Remaining work and acceptance conditions |

## Model status

“Implemented” means an execution path exists. “Validated” always names a
checkpoint, workload and comparison. “Experimental” means material numerical,
state or performance acceptance remains open. A model appearing in the catalog
does not certify every architecture feature or context length.

| Path | Implemented | Evidence and limits |
| --- | --- | --- |
| Qwen hybrid, MLX weights | Text, checkpoint-dependent MTP, PLD, native decode batching; compatible vision towers | Historical 27B 310-token and Flash-Next 256-token goldens; latest performance campaign rechecked 27B, not Flash-Next |
| GLM-5.3, DS4 Q2 GGUF | Text, embedded target MTP, CLI/server/UI | Experimental: 64 greedy IDs match DS4 in serial and MTP; measured logit difference remains; prefill/verify rows are serial |
| GLM-5.3, legacy MLX | Resident prepared weights, target MTP, wide schedules; compatible vision tower | Scoped same-weight DS4 adapter checks; complete state/performance acceptance remains open; vision is not numerically at parity |
| DeepSeek V4.1 Coder, mixed GGUF | Text, eight-row prefill chunks, Engram disk reads, DSML tools and session reuse | Experimental: short all-layer equality and selected 149/8,199-token comparisons; short decode timing only; no MTP head or vision |
| FLUX.2 Klein 4B/9B | Local text-to-image | Native pipeline and bounded numerical/image checks; no editing or broad quality/performance certification |
| FLUX 3 Image Klein | Local text-to-image and local caption expansion | Bounded stage checks; original BF16 package; local caption expansion differs from the official upsampler |
| Laya | Typed decisions on CPU or Metal | Separate CLI/server path; not covered by the language-model parity claims |

Exact curated repository IDs are maintained in
[the catalog](../crates/lisa-ui/src/catalog.rs) and listed in the
[user README](../README.md#supported-models). Model type strings are checkpoint
contracts; product names do not replace them.

## Documentation rules

- Change the relevant topic document when behavior changes. Keep open work in
  the roadmap, with a concrete condition for closing it.
- Record checkpoint, source/reference revision, mode, workload, comparison and
  timing denominator with each result. Preserve machine and compiler caveats.
- Keep provenance and licenses beside source. Specs summarize them and link to
  their authoritative files.
- Keep current behavior, scoped results and decisions in the owning topic
  document. Avoid campaign diaries and duplicated reports. Historical results
  remain historical after a kernel or schedule changes.
- Reproduction instructions use explicit input/output paths and retained
  capture identities. A skipped model test is not validation of that model.
- Do not express parity as a completion percentage. Report which numerical,
  state, serving and performance checks passed and which remain open.
