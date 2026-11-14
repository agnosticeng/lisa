# Performance decisions

Optimization changes must preserve the [numerical and state contracts](contracts.md).
A smaller checkpoint, fewer dispatches or matching greedy tokens alone do not
establish a faster or numerically equivalent implementation.

## Current choices

- Keep serial, prefill, batch and PLD reductions STRICT. The scoped Qwen MTP
  exception does not extend to serial split-K or large output heads.
- Keep admitted verify shape gates. PLD depth six, MTP copy length seven,
  GLM's two drafts and DSpine's trained block size are separate constraints.
- DSpine defaults to original-head retrieval, selected projection tables and
  unshared verification attention. Full tables, coarse retrieval, paired KV
  and compact replay remain explicit alternatives with their own identities.
- Source hashes and expensive pipeline checks run on cache misses, never per
  dispatch. Numerical cache identity is available before lazy compilation.
- Packaging compresses source bytes and strips only the shipped binary. Metal
  arithmetic and JIT behavior are not changed to meet the desktop size limit.

## Optimization admission

| Candidate | Decision |
| --- | --- |
| Serial split-K | Not admitted under STRICT without independent arithmetic proof |
| Verify tile/occupancy changes | Retain proven shape gates until a controlled replacement passes |
| Tree verification / NAX m16 | Not enabled as a general scheduling policy |
| Norm inside qmv | Requires a new occupancy/design case and strict proof |
| AOT, paged-KV decode, ANE prefill, MTP-head grafting | Require a new premise and fresh end-to-end acceptance |
| Serial bandwidth saturation | Do not use a theoretical-bandwidth estimate to declare optimization exhausted |

## Validation protocol

1. Identify source/binary, checkpoint, reference/adapter, compiler, format,
   context, sampler, numerical law and exact workload. Shared helpers need a
   separate independent numerical oracle.
2. Detect physical RAM and memory pressure before each benchmark. Use RAM minus
   8 GiB or a lower `LISA_RAM_CAP_GB`; load one model and run GPU work sequentially.
   Stop residual processes after an abort. Prefer 64–256 output tokens.
3. Run affected operation/model/state checks twice. Cover accepted/rejected
   rounds, rollback, continuation and boundary shapes. Missing inputs must fail
   explicitly enabled tests; skipped tests do not count.
4. Compare before/after traces. Captures and per-kernel probes are diagnostic,
   not scored workloads. Command-buffer intervals are not exclusive GPU counters.
5. Interleave matching end-to-end baseline/candidate runs, with warmup and
   cooldown. Use in-session `round-cost` when supported. Separate load, TTFT,
   prefill and decode; name output-token denominators and memory owners.
6. Report variance and regressions. Cache-key changes also need an interleaved
   pair; numerical tests cannot detect per-dispatch overhead.

Use the release build, serialized library tests and smoke sequence in
[AGENTS.md](../AGENTS.md). Keep required fixture identities/provenance beside
source or with external captures. Run reports and benchmark history do not
belong in this folder.
