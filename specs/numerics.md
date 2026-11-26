# Numerics

Every execution path runs under a named **numerical law**: the exact casts,
reductions, fusions, transcendental modes and shape gates it uses. Laws decide
what may change, what must be proven, and which saved state may be restored.
[AGENTS.md](../AGENTS.md) and explicit task instructions take precedence.

## Strict paths

Serial decode, prefill and B>1 batching preserve their reference's arithmetic,
casts and accumulation order **bit for bit**. Matching greedy tokens does not
establish parity. Reassociating a reduction, changing FMA behaviour or adding
a fusion needs proof against the actual reference operation; a speed gain
never relaxes this.

**The verify exception.** Qwen MTP verification may use
`affine_verify_qmm_splitk` for 2–8 rows, inside
`ops::enter_verify_splitk_scope`, provided every output is within one BF16 ULP
and the argmax equals the `qmv_wide` chain. Serial, prefill, batch and large
output heads stay strict. A wide-forward near-tie can change a greedy choice;
attribute it to this documented class before treating it as expected. The
exception does not extend to any other model.

**Captured verification.** Speculative verification captures per-row state for
rollback while preserving serial arithmetic for every accepted row
(`AffineSerialLinear`, `AttentionSerialCausal`, complete GDN history). It does
not enable the verify exception by itself.

## Accelerated laws

Some models offer an explicitly named faster law with its own precision and
reduction contract. Such a law needs independent kernel checks, full-vocabulary
logit comparisons, task-quality evidence and interleaved timings; bitwise
agreement is not its criterion. It has its own state identity and never shares
saved state with the reference law. The reference implementation and its
goldens stay intact for comparison.

| Model | Laws |
| --- | --- |
| Inkling 2 | The pinned reference's decode step (its tiles, splits and bf16 activations); see [Inkling](models/inkling2.md#the-step) |
| GLM-5.3 MLX | Paired Q8 gate/up projection is part of the law |

## Kernel proof

- Preserve per-operation BF16 casts, transcendental mode, FMA behaviour and
  reduction order. Equivalent source expressions can compile differently.
- `simd_sum` is not an XOR butterfly: match the reference operand layout and
  reduction, not an algebraically equal tree.
- Prove fused BF16 transcendental operations against the composed chain over
  all 65,536 inputs where that domain suffices, then exercise real activations.
- Prove a fusion against its actual oracle chain, including offsets, strides,
  empty inputs, tails and live aliases.

## State identity

Prefix entries, spill files, persistent sessions and timing profiles carry the
law they were produced under. Restoring requires the same law and the same
payload identity. Laws are available before lazy kernel compilation and include
loaded arithmetic modes.

| Identity | Scope |
| --- | --- |
| `lisa.gdn.complete-capture-history.v1` | Complete convolution history in both MLX Qwen families |
| `lisa.qwen35.serial-capture.v3` | Qwen dense MLX captured projection, attention and gate law |
| `-serial-capture-v3` | Qwen dense MLX MTP round-cost key |
| `lisa.qwen4.serial-capture.v1`, `-serial-capture-v1` | Flash-Next captured law and its cost key |
| `0x494e4b3253540000` | Inkling step law |
| `0x4f4c414c41000001` | Olala |

GLM and DeepSeek derive their laws from their reference revisions, numerical
identity constants and loaded projection modes. Models without an explicit law
use the runtime kernel fingerprint.

## Pipeline keys

The pipeline cache key encodes kernel name, math mode, NAX language target and
function constants. Source hashing and the same-name/different-source check run
only on cache misses, and a collision is fatal. Any
cache-key change needs an interleaved end-to-end pair: goldens cannot detect a
throughput regression.

## Goldens

Qwen goldens are embedded under
[`crates/lisa-bench/golden/`](../crates/lisa-bench/golden/) and are checkpoint
specific. Never recapture a golden to hide a regression or bless a new
reduction; a new checkpoint needs its own identified baseline. Affected goldens
run twice.

## Validation protocol

1. Identify source and binary, checkpoint, reference, compiler, layout,
   context, sampler, law and workload.
2. Read `sysctl hw.memsize` and `memory_pressure` first. Load one model at a
   time and run GPU work sequentially.
3. Run affected operation, model and state checks twice: accepted and rejected
   rounds, rollback, continuation, offsets and boundary shapes, using
   `lisa-bench` commands with `--model` for checkpoint-level checks. Replay
   paths need a second run on fresh request state.
4. Compare before/after Metal System Traces. Command-buffer intervals are not
   exclusive per-kernel counters.
5. Interleave baseline and change end to end with warmup. Report loading, time
   to first token, prefill and decode separately, with the token denominator.
6. Report variance and exercised routes. Captures, per-kernel probes
   (`--gpu-probe`) and contended runs are diagnostic only.

## Optimization admission

- Model plans own equations, chunking, speculative state and shape policy.
- Device operations own exact arithmetic, layouts and capability gates.
- Runtime changes keep alias lifetimes, import completion, error propagation
  and memory admission. Fewer launches alone is not an end-to-end gain.
- A new law needs independent numerical and state validation and its own state
  identity; existing goldens and tolerances are never loosened to admit it.
