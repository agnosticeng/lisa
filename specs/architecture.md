# Architecture

`lisa-models` describes how inference is computed. `lisa-engine` imports weights,
binds model plans and executes them through `lisa-backend`. Applications select
and construct providers. Devices import tensors and execute individual kernels;
they never receive a model or choose its next operation.

## Workspace boundaries

| Crate | Responsibility |
| --- | --- |
| `lisa-backend` | Tensor import, storage views, kernel capabilities, prepared handles and completion contracts |
| `lisa-models` | Checkpoint schemas, model dataflow, numerical alternatives, shape gates and memory formulas |
| `lisa-mlx`, `lisa-gguf` | Format parsing, original encodings and immutable mapped host storage |
| `lisa-metal`, `lisa-cpu` | Device storage, kernels, queues and completion |
| `lisa-engine` | File resolution, portable weight binding, generic execution, host memory admission, sampling and session policy |
| `lisa-serve` | HTTP adapters, request admission and shared application composition |
| `lisa-cli`, `lisa-bench` | User commands and compiled diagnostic instruments |
| `lisa-app` | Shared Apple app worker, chat storage and C bridge; SwiftUI in `apps/Lisa` |

Models depend on backend contracts, not formats or device implementations. They
perform no filesystem, environment, networking or device-allocation work.
Formats depend on neither engine nor device. Devices depend on neither models
nor formats. Engine and device crates have no `src/models` namespace.
Metal shaders live under `lisa-metal/src/shaders`; launch wrappers live in `jit`
and operation-based `kernels` modules. Keep attribution and licenses beside
ported sources and in [NOTICE](../NOTICE).

## Plans and execution

Models supply symbolic inputs, weights, intermediates, dependencies and legal
alternatives. The engine's generic graph/plan/program/traversal executors validate
references, arities, shapes and encodings, prepare handles, bind storage and run
operations in order. They do not switch on model families or read activations
back to the host to decide model dataflow.

Required unsupported operations fail preparation. Optional implementations can
decline eligible shapes and use the model's fallback; compilation, allocation
and dispatch errors propagate. A backend cannot silently substitute a different
arithmetic law. Output/workspace effects and last-use information govern storage
reuse. Indirect resources retain all referenced allocations through completion.

`lisa-engine::bindings` connects model slots and state to generic device tensors.
Language, vision, DSpine and image bindings are parameterized by their provider;
there is no separate Metal model implementation. Laya executes the same plans on
CPU and Metal with explicitly selected arithmetic laws.

`DeviceContext<D>` retains the selected device, optional diagnostics and typed
prepared-state slots. Loaded models, caches and samplers receive that owner
explicitly. The engine has no active-context thread local, implicit default
device or native downcast. Dropping the owner releases its prepared state.
Native arithmetic comparators and their fixture harness live in `lisa-bench`;
applications do not depend on that harness.

## Device API

`Device` exposes capabilities, `pull`, `prepare`, `dispatch`, streams and `wait`.
Capabilities identify dtype, encoding, layout, geometry, numerical law and
workspace requirements. Preparation resolves context-bound operation handles;
shape-specific Metal pipeline compilation can remain lazy. Dispatch enqueues
one operation and returns outputs plus a completion handle.

`TensorAccess` supplies tensor views and explicit readback operations for shared
session algorithms. `ReadbackDevice` supplies synchronized f32 readback for
generic decision bindings. `HostAccessDevice` exposes raw readback, layout and
explicit enqueue/wait boundaries. These contracts contain no inference policy.

`DeviceHandle` erases concrete Rust provider types and checks the ownership of
tensors, prepared kernels, streams, completions and indirect resources before
forwarding operations. No native downcast is exposed to the engine. Applications
use it for CPU/Metal Laya loading; a second Laya runtime wrapper is unnecessary.
Indirect-capable handles retain the provider's resource leases. Unsupported
contracts remain unsupported, including different arithmetic laws.

`StorageDevice` adds the maximum allocation size and checked, byte-preserving
tensor views. `IndirectDevice` provides opaque retained resource sets for
indirect kernel operands. Neither interface knows filenames, layer numbers,
checkpoint types, tensor roles, sampling policy or model topology.

Copy imports preserve bytes and finish reading the source before returning.
`SharedRequired` retains immutable owned storage and never silently copies.
The engine chooses mapping windows, validates identities and ranges, performs
prefetch, and decides whether a separate copy is allowed. MLX shard/lazy stores,
prepared resident arenas and GGUF imports use these generic contracts.
`ResidencyPolicy::Mapped` requires shared storage; `Uploaded` copies original
bytes into private device allocations. GGUF admission reports host residency,
device residency and temporary host mapping needs separately before importing.
Prepared resident arenas expose the same explicit policy and footprint. Metal
selects mapped imports; no unsupported-sharing fallback silently copies a
resident checkpoint.

Host RAM detection, the OS reserve and `LISA_RAM_CAP_GB` admission policy live
in `lisa-engine::memory`. The environment cap only lowers detected physical RAM
minus 8 GiB; failed detection refuses admission. Device allocators receive explicit
byte budgets and retain their allocation/completion mechanisms.

Mappings and buffers outlive queued uses. Dropping a completion handle does not
wait or release in-flight resources. Never wait for completion while holding
an allocator lock. Partial writers need a clear; reuse is limited to the
ordered and completion-safe cases in [contracts](contracts.md).

## Sessions and scheduling

`LanguageModel<R>`, `Conversation<R>` and batching use runtime-associated tensor,
cache, draft-cache and sampler types. Shared PLD, independent MTP, target-MTP,
DSpine verification, scheduling and round-cost measurement are generic engine
policy. `SpeculativeRuntime` supplies engine tensor/sampling/rollback actions;
it is not a device API. The generic `language_runtime::Runtime<D, S>` supplies storage operations and diagnostic hooks. Cache packing,
survivor extraction, per-row sampling, checkpoints and verified rollback live
in the engine. Rollback uses the loaded cache's history extent.

Attention/indexer caches, recurrent capture/rollback bindings, per-layer
snapshots, the byte-preserving spill codec and prefix LRU policy live in
`lisa-engine`. Cache constructors receive an explicit `DeviceContext<D>`;
allocation, row copies and recurrent plans execute through backend contracts.
The prefix store receives its device and spill directory at construction and
its numerical law on each operation. Its policy tests run on CPU; Metal cache
oracles exercise compaction, batching and speculative rollback. Existing spill
tags remain readable, with checked payload sizes before tensor import.

`sampling::Sampler` owns filtering, acceptance and RNG state. Its execution
context supplies a device, an explicit native or ordered-host tensor law, and
an optional trace observer. Kernels expose individual reductions, selection and
keyed Gumbel operations; they receive neither sampler configuration nor token
history. Producer-bounded F32 readback preserves the GLM queue boundary, including
through an erased device handle. Shared C-reference sampling fixtures run on CPU
and Metal. The generic sampler receives the loaded model runtime on every draw.

Sessions own committed history and state. Restore only under the same numerical
law, including loaded arithmetic modes. Target and private draft state share a
commit boundary. Failed forward/head transactions invalidate the conversation.
API response history, chat storage and GPU prefix snapshots are separate owners;
see [serving](serving.md).

## Checkpoint dispatch

The engine resolves local directories, supported GGUF files and Hub repository
IDs. Config/tensor geometry determines support. `ModelKind` centralizes aliases;
product names never replace checkpoint identifiers.

| Identity | Model plan |
| --- | --- |
| `qwen4_exp`, `qwen4_exp_text` | Qwen Flash-Next GDN/attention + MoE |
| `qwen3_5`, `qwen3_5_text`, GGUF `qwen35` | Qwen dense hybrid |
| `glm5_next`, `glm5_next_text`, GGUF `glm5-next` | GLM KDA/sparse attention + MoE |
| GGUF `deepseek41` | DeepSeek mixed-format/Engram |
| `flux2-klein`, `flux3-klein` | Image generation |
| `laya` | Decision model |

`lisa-engine::loading::ModelProvider` admits a resolved checkpoint before weight
access and returns `Loaded<L, I>` without knowing concrete provider types.
`lisa-serve::application::device` resolves backend policy; application `LoadOptions` carries
explicit backend and drafter choices and binds through that shared interface. `LISA_DEVICE` selects implemented providers. Loading applies
checkpoint context limits before warmup; an explicit CLI capacity override still
undergoes model validation and memory admission.

MLX/GGUF payloads retain original bytes, scales, biases and quantization groups.
Legacy MLX GLM needs its prepared resident index; native GGUF does not. DeepSeek
Engram stays on disk. Only one model is loaded at a time; a switch releases the
previous model before loading the next.

## Adding a backend

Implement the required `lisa-backend` contracts in a device crate, including
kernel capabilities, original-byte tensor imports, views/readback, allocation,
completion and indirect resource ownership. `ManagedDevice` exposes mechanisms
for reservations, queue boundaries and reuse; `ResourceDevice` exposes opaque
residency sets, indirect tables and ordered full-write workspace. The engine
retains the policy deciding when to use those mechanisms.

Instantiate the existing engine bindings with the provider and register it at
the application composition boundary. No model, format, sampling, cache or
speculation implementation needs to be copied into the device crate. Current
language/image application aliases select Metal; CPU Laya uses `DeviceHandle`.
The erased handle is not a claim that every language/image capability is
implemented by every provider.

`lisa-engine` has no concrete device dependency or Metal feature, including with
all features enabled. CPU is a test dependency only. Linux CI checks dependency
direction and runs the portable format, model, backend, CPU and engine suites.
`lisa-runtime` no longer exists. Application CPU composition builds and is tested
without Metal using `cargo test -p lisa-serve --no-default-features --lib`. The
Metal HTTP adapters and application construction are enabled by the default
`metal` feature and tested in the Apple workspace suite.

ROCm remains unimplemented. A new crate must provide the required kernels and
storage/completion semantics, then be registered in applications and validated
on AMD hardware. Unsupported operations must fail explicitly. A successful
portable build alone does not establish backend or model acceptance.

## Kernel packaging

Cargo generates an authenticated, compressed `kernels.bin` matching the binary's
Metal sources. Ship it beside standalone executables or in app Resources.
Missing, malformed or mismatched packs fail initialization; there is no network
or source-tree fallback. The runtime validates/decompresses once, while Metal
compilation remains JIT. Preserve the matching dSYM outside the app bundle.
