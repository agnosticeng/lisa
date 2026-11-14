# Qwen hybrid models

## Supported paths

| Checkpoint identity | Plan | Execution |
| --- | --- | --- |
| `qwen4_exp`, `qwen4_exp_text` | [qwen4](../../crates/lisa-models/src/qwen4/) | GDN/attention, MoE and checkpoint MTP |
| `qwen3_5`, `qwen3_5_text`, GGUF `qwen35` | [qwen3_5](../../crates/lisa-models/src/qwen3_5/) | Dense GDN/attention with format-specific arithmetic |

`Qwen35Tower` shares traversal, prefill and session integration across MLX and
GGUF. MLX retains BF16 casts and its attention/GDN law; GGUF retains original
GGML blocks and F32 state. Prefix snapshots cannot cross those laws. Compatible
dense MLX vision towers support [image input](../media.md#image-input).

## DSpine block drafter

Lisa UI defaults to `tencent/Qwen3.8-27B-DSpine-b16` for MLX dense targets with
64 layers, hidden size 5120 and vocabulary 248320. The companion is downloaded
and checked before target loading; complete copies are shared across target
quantizations. Target downloads omit the optional `mtp/` sidecar, and unused
MTP weights/coarse heads are not allocated. DSpine is not supported with GGUF.
CLI or explicit desktop selection uses `LISA_QWEN_DSPINE_MODEL` with a local
checkpoint or Hub ID.

DSpine consumes target layers `[1,16,31,46,61]`, the original embedding and output
head. Its trained block has 16 rows. Draft and verification widths are independent;
a matching hardware/checkpoint/schedule profile selects them. Unprofiled loads
use seven-row snapshots. Greedy UI requests use the profile's fixed width;
Auto adds request-local serial probes, cost-based fallback and re-entry.
`--depth 0` is serial. PLD's separate depth-six cap remains unchanged.

Only accepted target taps commit to draft context. Draft attention has an
8192-token window with absolute RoPE positions and bounded physical KV eviction.
A rewind outside retained history rebuilds context from the target prefix.
Auto's serial rounds use ordinary serial execution, not one-row verify capture.
Proposal acceptance uses ordinary target draws and committed sampling history;
the complete sampled proposal-q/p-q rejection path is not implemented.

Current native implementation choices:

- Parallel masked drafting, causal bridge, stable top-16 selection and integer
  chain traversal run on Metal. Bridge fusions preserve the composed MLX casts
  and reductions. M5 uses original-layout GEMV/wide GEMV/NAX; older hardware
  uses prepacked transposes.
- Eligible 5120-wide, 24-query/4-KV-head, dimension-256 target projections share
  a joined allocation for single-request widths 1..8. Prefill/batching keep
  their own kernels; originals become views rather than duplicate weights.
- GDN verify fusion preserves every FP32 state prefix and BF16 epilogue boundary.
  Compact replay is optional and preserves sequential recurrence. Prefill
  submission remains bounded at existing chunk boundaries.
- Unshared verification attention is the default. Paired shared-KV attention
  is opt-in through `LISA_QWEN_SHARED_KV_VERIFY=1`, with its own law/profile.
- Selected-row vocabulary projections and original-head retrieval are defaults.
  `LISA_DSPINE_TABLES=full` adds about 763 MB of immutable tables;
  `LISA_DSPINE_RETRIEVAL=coarse2` adds a derived 2-bit shortlist head when needed.
  Both trade startup/residency for possible workload-specific benefits.
- Profiles bind device, checkpoint and execution identities. Changing arithmetic
  or scheduling requires new admission checks and profile selection.

BF16 drafter weights occupy about 4.05 GB, roughly 3.8 GB more than the omitted
MTP sidecar. Optional affine 4-bit/group-64 conversion occupies about 1.56 GB:
36 linear matrices are quantized, while bridge/codebook/norm tensors stay BF16.
The [converter](../../crates/lisa-engine/tools/quantize_dspine.py) preserves the
source, refuses to overwrite output and writes `quantization.json`. BF16 remains
the default; smaller weights do not imply faster generation.

Reference identities: Tencent DSpine `321b0fc1388c4cdf03483ab73856975f7c0c0be7`,
companion checkpoint `5f54174d301dbca65d9aa4448dc47e56732d9a12`, and SGLang
`704681229f1cdda291dbd2261dabb2ae80a4a49d`. Source attribution is in
[NOTICE](../../NOTICE); independent fixture identities are beside the
[DSpine tests](../../crates/lisa-bench/tests/fixtures/dspine/).
This is a Metal adaptation, not complete SGLang/CUDA parity. General graph replay,
sampled conditional-q/rejection, sparse target sampling and BS8 ragged SPS remain
unimplemented.

## Unsloth 27B GGUF

`unsloth/Qwen3.8-27B-GGUF` selects `Qwen3.8-27B-UD-Q2_K_XL.gguf`: 64 trunk layers
and an embedded MTP block. Original mixed Q/IQ blocks and F32 parameters remain
packed. The experimental path is text-only and one stream per forward. Native
MTP and PLD are capped at two drafts; wider fixed requests fail explicitly.

Embeddings are read by row from disk; attention and recurrent state are F32.
Prefill/draft priming use 16-token windows. Memory admission includes mapped
weights, caches and scratch; file size and advertised context capacity are not
fit guarantees. Packed prefill/verify still use row-preserving matrix-vector
schedules. MLX projection kernels and numerical laws do not apply to these bytes.

The reference is llama.cpp `18ab62a8ec5144ae5d2f65e78374ba69652d75dd`, aligned to
its serial F32-cache execution. Keep model and capture identities in
[provenance](../../crates/lisa-engine/src/bindings/qwen35_gguf/provenance.json)
and packed-kernel provenance beside source. Wide reference schedules have a
different arithmetic law. Long-context and independent MTP-head acceptance remain
open; an imposed small RAM cap does not validate another physical Mac.

## Generation policies

MLX supports serial generation, chunked prefill and native decode batching with
independent row state. Automatic MTP needs a loaded drafter; unsupported fixed
MTP fails. The native MTP coarse head uses 2-bit/group-64 shortlisting and exact
trunk-head rescoring, without altering verification weights.

PLD uses a rolling 3-gram index and a depth cap of six. MTP copy proposals have
a separate default limit of seven. The optional router uses repeated 4-grams
and prompt length; it is not a universal performance rule. Cost-based decisions
use at least three samples per trusted cell, nearest trusted KV buckets and
per-depth acceptance estimates. Remaining output budget bounds the expected
commit count. Ordinary requests do not persist the round-cost table.

## Numerical and state requirements

Serial, prefill, batch and PLD are STRICT. Only scoped MTP verify split-K admits
the exception in [contracts](../contracts.md). Wide/serial near-ties require
explicit attribution. Rollback restores recurrent, attention, draft and absolute
position state together. Batched shared-expert dispatch must write every row;
attention masks must honor batch/head broadcasting, byte/Bool dtype and strides.

The small 0.8B model's serial/PLD discrepancy remains unattributed. Shared
multi-client speculative forwards, broader batching/vision qualification and
long-context performance remain open. Use checkpoint-specific goldens,
`session-check`, `round-cost` and explicit `layer-diff` inputs when changing these
paths; do not recapture a golden to hide a regression.
