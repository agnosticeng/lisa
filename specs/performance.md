# Performance and validation

## Current evidence

These are recorded measurements. Scope is part of each result. Different
checkpoints, formats, compiler settings, timing boundaries or draft policies cannot be
combined into a single “DS4 parity” percentage.

| Path / campaign | Recorded comparison | Admissible conclusion |
| --- | --- | --- |
| DeepSeek mixed GGUF, Nov 4 | LISA 37.5199 vs adapted DS4 36.1603 decode tok/s; 3 interleaved runs/arm, 63 timed forwards | Short tested decode is comparable; adapter `-O1` caveat, no matched prefill or general long-context verdict |
| GLM Q2 GGUF, Nov 4 | About 32 decode tok/s in a short run | Diagnostic only; numerical mismatch and serial verify remain; no controlled DS4 throughput verdict |
| GLM legacy MLX, Nov 3 | Final warmed HTTP serial 11.81→11.87, MTP 9.01→8.83 tok/s | No general throughput gain established; desktop activity/thermal drift limit attribution |
| Qwen 27B, Nov 3 | 37.15→36.55 decode tok/s | Correctness checks passed; no certified throughput improvement |
| FLUX 3 layout cache, Nov 3 | 219.67→83.53 seconds with identical PNG | Useful diagnostic; baseline overlapped a CPU build, so not a controlled speedup |

Model-specific scope: [DeepSeek](models/deepseek41.md),
[GLM](models/glm53.md), [Qwen](models/qwen.md) and [media](media.md).
Interleaved admission reduced a recorded
late-arrival TTFT from about 10 to 1.8 seconds; that is a queue-latency result,
not a per-stream decoding gain.

The November runtime changes include corrected norm/RoPE byte offsets, exact
top-32 selection, cached projection views, incremental detokenization/parser
work, cheaper prefix-spill lookup and bounded late admission. Numerical success
does not make each change a certified performance win. New comparisons must
retain source/binary identities and follow the validation protocol below.

## Distribution packaging, November 5

On Apple M5 Max with Rust 1.99.0, the signed `dist` app totals **9,965,431 bytes**:
the stripped engine/UI executable is 9,179,376 bytes and `kernels.bin` is 289,015
bytes; the remainder is the icon, metadata, notices and signature. Matching
dSYM data stays outside the app. No model implementation was removed. All 185
packaged shader/header files match the source bytes, and all nine PNG icon
representations retain identical decoded pixels. This is source compression
and host-link optimization; Metal compilation remains JIT.

The release workspace build, 340 library tests and smoke suite passed, with
25 explicitly ignored cases. The 118 passing tensor-runtime tests passed again
in isolation. Missing, truncated, malformed, corrupt and checksum-mismatched
packages were rejected. These checks do not certify ignored model fixtures.

A short A/B/A/B check used DeepSeek's six-shard IQ2_XXS package at Hub revision
`54229d38d588a66d1c6cff08af7b281b9326c656`, 1,024 context slots, greedy serial
generation, seed 0 and `--ignore-eos --max-tokens 64`. The prompt was
“Write a Python function that returns the nth Fibonacci number using a loop.”
(18 tokens). Physical RAM and memory pressure were checked before each run;
the detected 128 GiB machine allowed a 120 GiB cap. No other inference ran.

| Build | Decode tok/s, two runs | Mean |
| --- | --- | --- |
| `1b49003`, release | 38.2, 37.4 | 37.80 |
| External package, dist | 38.5, 36.2 | 37.35 |

The CLI reports 64 output tokens per decode timing. Generated text matched in
all four runs; timing/status lines were excluded from that comparison. The
mean was 1.2% lower in this small sample, so this establishes no throughput
improvement or broad performance equivalence. The first baseline process had
cold compilation/loading costs; its TTFT is not a matched comparison.

Binary SHA-256 identities:

- Baseline CLI: `c2e4d290cbcf32cdc5dc583f14b1ea40c15ec1354f2084cd4b425e8516a2a7b2`.
- Packaged CLI: `57b87fc2400c5899459c8834e63865babf92c4c90c57167c659b0cc7be02d9d2`.
- Decoded kernel sources: `82a2ca44433dbb74f0189cb16f02fbac6b2b4bfda6a6e8c8eb50c5aed150feb9`.

## Historical conclusions to preserve

| Investigation | Current disposition |
| --- | --- |
| Qwen serial bandwidth | The old 723 GB/s / 88%-of-peak saturation claim was retracted. The later census found about 417 GB/s effective and substantial thermal/attribution uncertainty. Do not declare all legal optimization impossible from the old denominator. |
| Serial split-K | Unchanged greedy goldens did not authorize reduction reassociation under STRICT. The measured candidate was a wash; wider partitioning regressed. No production admission. |
| Verify tile / occupancy | Tested alternatives lost or tied the shipped tile. Keep the proven shape gates; this is not a universal hardware lower-bound proof. |
| Width 8→9 | The measured Qwen verify cliff supports the seven-token MTP copy-proposal limit. Public plain-PLD depth stays six. |
| Tree verification / NAX m16 | Tested tree economics and m16 schedules lost. The MPP kernel eventually executed; the earlier “needs additive linkage” explanation was wrong. |
| Norm inside qmv | Tested fusion lost occupancy and performance. Reopening needs a materially different design and strict numerical proof. |
| AOT / paged-KV decode / ANE prefill / head grafting | Prior investigations rejected or set aside their tested premises. Reopening requires a new hypothesis and fresh correctness/performance evidence. |

Historical TTFT, routing and batching improvements remain scoped evidence,
not current cross-model promises.

## Validation protocol

1. **Identify the comparison.** Record source and binary hashes, checkpoint
   revision/hash, reference and adapter revisions, compiler flags, model
   format, context, prompt tokens, output budget, sampler/seed, draft policy
   and loaded numerical law. If helpers are shared, name the independent
   numerical oracle separately.
2. **Make the machine safe.** Detect physical RAM and free headroom before each
   run. Budget is RAM minus 8 GiB, or a lower `LISA_RAM_CAP_GB`. Stop residual
   processes from an aborted run. One model and one GPU workload at a time,
   including the desktop app and library tests. Prefer 64–256 output tokens.
3. **Prove numerics and state.** Run affected operation/model checks twice.
   Qwen uses its exact checkpoint goldens; DS4 ports need full logits and
   relevant layer/cache comparisons. Exercise accepted/rejected speculation,
   boundaries and continuation. Enabled tests must fail for missing inputs;
   ignored or skipped cases do not count.
4. **Attribute the change.** Compare before/after traces for the targeted phase.
   Per-kernel probe and capture modes are diagnostic. Command-buffer durations
   may overlap and are not exclusive hardware counters.
5. **Measure end to end.** Interleave baseline/candidate on matching workloads,
   preferably repeated ABBA order, with warmup and settled operating conditions.
   Use live-session `round-cost` where supported. Report TTFT, prefill, decode,
   accepted tokens/round, state memory and the exact token denominator. Separate
   cold process load, filesystem cache state and warmed execution.
6. **Decide within scope.** Show variability and regressions as well as gains.
   Reject a candidate that breaks the numerical/state contract. A cache-key
   change also needs the interleaved end-to-end pair: a per-dispatch source
   hash once halved decode while goldens stayed green.

Before committing implementation changes, the repository gate is a release
workspace build, sequential library suite and compiled smoke suite; see
[AGENTS.md](../AGENTS.md). This document does not turn documentation-only edits
into new runtime evidence.

## Evidence retention

Keep a compact result receipt with the identity fields above, workload, exact
commands, counts, timings, errors and hashes of external captures. Store
provenance beside source and summarize the verdict in the owning model doc.
The missing durable legacy GLM capture inventory is tracked in
[the roadmap](roadmap.md#evidence-and-reproducibility).
