# Qwen hybrid models

## Supported paths

| Checkpoint identity | Module | Architecture |
| --- | --- | --- |
| `qwen4_exp`, `qwen4_exp_text` | [qwen4](../../crates/lisa-engine/src/models/qwen4/) | GDN/attention, MoE, optional checkpoint MTP head |
| `qwen3_5`, `qwen3_5_text` | [qwen3_5](../../crates/lisa-engine/src/models/qwen3_5/) | Dense GDN/attention, optional checkpoint MTP head |

These metadata names remain unchanged even when a checkpoint is marketed as
Qwen 3.8. A family name alone does not establish tensor or quantization
compatibility. The curated 27B, Flash-Next, 4B and 0.8B packages are listed in
the [user README](../../README.md#supported-models).

Both implementations share GDN components and the engine's session machinery.
Compatible dense vision towers enable [image input](../media.md#image-input).
Quantized vision projections and DeepStack are not supported.

## Generation policies

- Serial generation, chunked prefill and native decode batching are implemented.
  Batch rows retain independent cache windows and sampling state.
- Automatic MTP requires an actual loaded drafter. A checkpoint without one
  uses serial generation; a fixed unsupported MTP request fails clearly.
- The 27B coarse draft head uses 2-bit/group-64 weights to shortlist 32 tokens,
  then exact trunk-head rows rescore them. This model-specific optimization
  does not change the verifier's weights.
- Plain prompt lookup uses a rolling 3-gram source index. The public depth cap
  is six. Copy proposals inside the MTP controller have a separate default
  limit of seven. Neither limit is a kernel tile size or a universal trained
  head depth.
- The explicit CLI router uses the prompt's repeated 4-gram score: plain PLD
  for score at least 0.80 or prompts shorter than 8,192 tokens; otherwise MTP.
  This is an optional policy, not proof that one mode wins for every prompt.

The controller combines measured round costs with per-depth acceptance EMAs.
Trusted cost cells require at least three samples; bucket lookup prefers the
nearest trusted lower bucket. Expected committed tokens are capped by the
remaining output budget. Ordinary request probes do not persist the cost table;
the `round-cost` instrument does. The implementation lives in
[round_cost.rs](../../crates/lisa-engine/src/core/round_cost.rs).

## Numerical and state requirements

Serial, prefill, native batch and PLD retain STRICT arithmetic. Only the scoped
MTP verify split-K operation admits the documented one-bf16-ULP/argmax
exception for supported M=2..8, 4-bit/group-64 geometry. Large output heads
stay strict. The [contracts](../contracts.md) define the complete rule.

Wide verification can resolve a genuine near-tie differently from serial.
That attribution must be demonstrated; token agreement or disagreement alone
does not prove a kernel correct or incorrect. Rollback includes GDN history,
attention state, draft history and absolute position.

## Recorded validation

Embedded [goldens](../../crates/lisa-bench/golden/) are checkpoint-specific:
historical runs recorded 310/310 tokens for the 27B and 256/256 for Flash-Next.
The November 3 performance campaign rechecked the 27B twice in serial,
depth-2 MTP and depth-4 PLD modes, including the coarse-head-disabled path.
Flash-Next weights were unavailable for that campaign, so its older results
do not certify those later changes.

The same campaign fixed layout offsets being passed as elements instead of
bytes in norm/RoPE binding. Performance comparisons use a baseline containing
that correctness fix. Its timing results did not establish an overall
throughput improvement. See [performance](../performance.md).

Earlier long-context, routing, TTFT and batching results do not establish
current universal rates. Retain the numerical requirements above and the
[performance dispositions](../performance.md#historical-conclusions-to-preserve)
when evaluating another optimization.

## Remaining scope

- Revalidate Flash-Next against the current runtime with its exact checkpoint.
- Shared speculative forwards across clients are absent; separate batch and
  MTP support does not imply batched MTP.
- Prefix caching still lacks a RAM byte budget and finer reusable boundaries.
- Revisit performance only with a defined workload and interleaved measurements.
  Old serial bandwidth-saturation claims were retracted; see the performance dispositions.

Use `lisa-bench golden`, `session-check` and `round-cost` for affected checks.
`layer-diff` takes explicit `--reference` (reference F32 arrays), `--golden`
(the matching prompt-token JSON) and `--output` (LISA arrays) paths in addition
to `--model`; it has no assumed capture directory.
Follow [validation and measurement rules](../performance.md#validation-protocol)
and the one-model rule before running them.
