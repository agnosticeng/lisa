# Remaining work

This is an acceptance list, not a percentage estimate or a promise of speedup.
New measurements close an item only for the exercised checkpoint, modes and
workloads.

## GLM GGUF parity — first priority

| Order | Work | Close when |
| --- | --- | --- |
| 1 | Attribute and fix the first layer-3 arithmetic mismatch | Identical inputs produce the required DS4 layer states and all logits, twice; the short greedy check still passes |
| 2 | Port packed-format tiny/wide projections and grouped routed prefill | Independent format/kernel checks pass; per-shape traces confirm the intended DS4 schedule without hidden copies/readbacks |
| 3 | Port native wide target verification and fused expert scheduling | Arithmetic and encoder/cache boundaries match the reference at every supported shape; serial and prefill remain valid |
| 4 | Complete MTP/state coverage | Accepted and rejected rounds, truncation, dense/sparse transitions, long private windows and resumed turns match committed reference state |
| 5 | Measure equivalent workloads | At least two controlled interleaved comparisons with matching weights, modes and denominators show no material regression; publish latency, throughput and memory together |

Use [the GLM document](models/glm53.md) for exact pins and the known mismatch.
Do not import a legacy MLX result as GGUF proof. Adding the separate vision
package is optional feature work, independent of text port acceptance.

## Other model acceptance

| Path | Remaining work | Acceptance |
| --- | --- | --- |
| DeepSeek GGUF | Wider real-context/prompt coverage; grouped expert prefill; long-context performance | Independent packed-format checks plus reference state/logits, repeated real boundary cases, matched end-to-end timings; retain adapter compiler caveat |
| Legacy MLX GLM | Complete state/serving matrix and consistent same-weight performance reference | Accepted/rejected/resumed cases under one recorded adapter identity; matching denominators and repeated interleaved timing |
| Qwen Flash-Next | Recheck latest runtime changes with the actual checkpoint | Its checkpoint golden twice, affected state checks and interleaved performance; earlier goldens alone do not close this |
| Qwen4 vision | Full-checkpoint exercise | Vision numerical comparison and end-to-end multimodal history checks |
| Legacy GLM vision | Find first attention divergence and resulting embedding drift | Operation-level attribution and agreed reference comparison; recognition smoke checks are insufficient |
| FLUX families | Broader stage shapes, image quality and controlled performance | Fixed-input reference checks plus larger prompt/size cases and isolated interleaved timings |

## Shared engine and serving

| Gap | Required outcome |
| --- | --- |
| Disconnect cancellation | Carry cancellation from the socket/reply channel through queues and active generation; release state at a safe boundary; measure work stopped and service recovery |
| Prefix RAM budget | Account retained bytes as well as entry counts, evict deterministically and preserve numerical-law checks |
| Native cache spill | Add a versioned serializer only for supported native state; reject incompatible identity and prove round-trip outputs/state, including partial/corrupt files |
| Prefix granularity | Evaluate reusable intermediate boundaries with exact recurrent/attention snapshots; prove continuation before increasing hit rate |
| Shared speculative forwards | Implement only with independent per-client masks, sampler state and rollback; measure aggregate throughput and per-client latency |
| Grammar-constrained output | If required, add supported grammar/token masking explicitly; current repair/retry is not a grammar engine |

These are separate capabilities. For example, cancellation recovery does not
prove cancellation propagation, and RAM snapshots do not prove disk spill.
Structured-output failure handling and bounded Responses history are implemented;
their changed HTTP behavior still needs a fresh conformance run. See
[serving evidence](serving.md#evidence-boundaries).

## Candidates requiring a measured case

Source reviews of TensorFold, MLX-Serve, Strata, Sushi and oMLX suggested the
following candidates. They are hypotheses, not comparative throughput evidence.
Refresh and pin the relevant source before porting an idea:

- External DFlash-style drafting: needs a compatible trained checkpoint,
  correct target-state taps, extra weight/state accounting and measured
  acceptance economics. It is not interchangeable with the current MTP head.
- Reusable dispatch plans: distinguish this from the existing pipeline cache;
  bind plans to shape, numerical law, storage identity and allocation lifetime.
- GLM automatic break-even control: price the actual packed-format verify path
  before choosing a depth from acceptance alone.
- Shared prefix blocks: investigate capacity/reuse separately from the earlier
  unsuccessful paged-KV decode experiment.

GGUF loading, DeepSeek disk-backed Engram and native media support are now
implemented. Earlier failed kernel ideas require a new premise; see
[performance dispositions](performance.md#historical-conclusions-to-preserve).

## Evidence and reproducibility

- Restore a **durable external capture inventory** for legacy GLM: filenames,
  sizes, hashes, generator/reference identity and exact enabled-test commands.
  The tracked fixture directories were intentionally removed. Enabled checks
  now load external captures at execution time, so all-feature compilation
  needs no capture payloads. Do not recreate large tracked payloads.
- Standardize benchmark receipts so future comparisons retain source/binary,
  weight, compiler, mode and denominator identities in durable records.
- Replace old ambiguous numbered-spec references when touching their owning
  code. Numbers were reused; resolve the actual historical subject before
  linking them to a current topic.

## Product upkeep

Keep [desktop](desktop.md) selection/hover invariants and the curated catalog
in sync with model capabilities. After UI changes, check row rebuild/scroll,
chat deletion, model reorder, resize/collapse behavior and image cards. Keep
the README preview representative of the actual app. This documentation review
does not certify a fresh UI interaction sweep.
