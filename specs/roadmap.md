# Open work

Architecture and behavior already implemented belong in the owning specs.
This file lists unresolved work; it makes no throughput or completion estimate.

## Backend extension

ROCm needs device kernels, import/completion and device-memory mechanisms,
plus application registration. Model bindings, caches, batch packing,
sampling/RNG and speculative rollback are generic engine implementations.
Reuse the [existing plans and interfaces](architecture.md#adding-a-backend).
Admission requires numerical/state checks for each supported encoding/model on
actual AMD hardware; an enum case or portable build is insufficient.

## Model acceptance

| Path | Remaining work |
| --- | --- |
| GLM GGUF | Attribute the first layer-3 DS4 arithmetic mismatch; port packed wide projections, grouped expert prefill and wide target verification; prove accepted/rejected/resumed MTP state before performance claims |
| Legacy MLX GLM | Complete state/serving coverage, independent same-weight reference identity and full-checkpoint acceptance; resolve vision attention/embedding drift |
| DeepSeek | Broader real-context/ring/DSML coverage, grouped expert prefill and long-context performance; native state spill if needed |
| Qwen 27B GGUF | Real small-memory hardware coverage, long contexts and independent MTP-head comparison |
| Qwen 0.8B | Attribute the serial/PLD discrepancy without replacing the serial golden |
| Qwen Flash-Next | Attribute chunked/full-prefill state drift before claiming strict equivalence; broader batching shapes/contexts and full-checkpoint vision qualification |
| DSpine | Full sampled conditional-q/rejection and residual sampling, exact sparse target support, graph replay and multi-request ragged SPS; independent oracle/state checks for each |
| FLUX | Broader full-checkpoint stage/image/size coverage and isolated performance acceptance |

## Shared engine and serving

- Complete prompt/session compatibility and exact suffix-resume checks. Preserve
  target and draft state together and avoid redundant target replay only after
  proving equivalent committed state.
- Add byte-budgeted prefix/session retention and validated persistence identity.
  Finer reusable prefix boundaries need recurrent-state proof. Native cache
  spill needs a versioned codec with corrupt/partial-file handling.
- Propagate disconnect cancellation into active work at safe boundaries.
- Shared speculative forwards require independent masks, sampler/RNG ownership
  and rollback for every client; separate batch/MTP support is insufficient.
- Run live API conformance after HTTP/state changes. Structured output remains
  repair/validation with a retry; token grammar support is separate feature work.

## Product and validation

- Qualify physical iOS inference, suspension/reload, cancellation and saved-chat
  replay. Resolve Simulator BF16 compilation independently of model arithmetic.
  Bring SwiftUI feature coverage and distribution size to its shipping target.
- Maintain a durable external legacy GLM capture inventory: filenames, sizes,
  hashes and exact reference/generator identity. Do not restore large tracked
  capture payloads; enabled tests must fail when required data is absent.
- Optimization candidates require a concrete workload and the
  [validation protocol](performance.md#validation-protocol). Upstream source
  ideas or a proposed speed budget are not implementation or performance proof.
