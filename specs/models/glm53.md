# GLM-5.3 Flash

## Status

The curated checkpoint is **DS4's original Q2 GGUF**. Legacy MLX checkpoints
remain loadable through a separate implementation. Numerical evidence,
schedules and throughput from one path do not certify the other.

| Property | Native GGUF | Legacy MLX |
| --- | --- | --- |
| Identity | GGUF `glm5-next` | `glm5_next`, `glm5_next_text` |
| Loading | Original packed blocks, no conversion | Original affine payloads through a prepared resident index |
| MTP | Embedded block 45; automatic by default | Target-MTP transaction; automatic when available |
| Prefill / target verify | Serial rows | Native shape-specific wide schedules |
| Multi-client native decode batch | Unsupported | Separate stream handling; no shared batched-MTP claim |
| Image input | Unsupported | Compatible dense vision tower; unresolved numerical gap |
| Full DS4 parity | Not certified; known arithmetic difference | Not certified; incomplete acceptance matrix |

Both paths implement 45 trunk layers: 34 KDA and 11 sparse-attention layers,
with HC mixing, routed MoE and a target-MTP block. The reference is executable
DS4 code at the pinned revision. Metadata descriptions are not an arithmetic
oracle. Preserve the [strict contracts](../contracts.md).

## Native GGUF checkpoint and reference

| Item | Pinned identity |
| --- | --- |
| Repository | `antirez/glm-5.3-flash-gguf` |
| Revision | `b2fa29d7a6b410db11221c904973967b80b760f5` |
| File | `GLM-5.3-Flash-Q2.gguf` |
| Size | 96,505,816,384 bytes / 89.88 GiB |
| SHA-256 | `e81fd6241c6e55a64e1e14e47a3eab61a173fa8d7e4b5c1d1848827119705b32` |
| DS4 revision | `b3f50180ec2c7324892be1f50bc83555ccb8d5e7` |

Authoritative implementation identities live in
[kernel provenance](../../crates/lisa-metal/src/shaders/glm53/gguf_provenance.json).
Refresh and record upstream identity before applying another port.

The loader validates all 1,412 tensors: 614 F32, 151 BF16, 450 Q8_0,
68 Q4_K, 86 IQ2_XXS and 43 Q2_K. The mixed formats are intentional:
Q8 dense/split K/V projections, Q4 KDA projections and IQ2/Q2 routed experts.
Embedding reads use original Q8 rows. Expert IDs stay on the GPU.

The shared resident adapter retains mapped owners; no second prepared
checkpoint or requantization is needed. The downloader selects the Q2 file
and notices. GGUF BPE selects DS4's GLM4 split rules and control markers.
CLI, server and desktop discovery recognize the architecture directly.

The MTP head uses shared session acceptance/rollback machinery. Its private
attention operation switches to the bounded wide-lora path above 2,051 selected
rows. This does not make the target's serial verification a wide matrix pass.
CLI and UI default context capacity comes from checkpoint metadata (1,048,576
for the curated GGUF), with an explicit CLI override through `--ctx-size`.
Allocation capacity is not long-context certification.

### Numerical limit

Full STRICT DS4 parity remains open. The first known layer-state mismatch is
at layer 3 and propagates to final logits despite matching short greedy output.
Attribute and fix the operation before claiming full equivalence. Automatic
MTP is implemented, but serial-row verification does not guarantee a speedup.

### Acceptance still required

1. Trace identical layer-3 inputs through DSA and routed FFN, eliminate the
   first arithmetic mismatch, and repeat complete layer/logit comparisons.
2. Port format-specific Q8/Q4 tiny and wide schedules, grouped routed prefill,
   paired expert gate/up and fused down/sum with independent kernel checks.
3. Implement native wide target verification, retaining DS4's shape gates,
   attention/reduction rules, cache writes and encoder boundaries.
4. Compare committed state across accepted/rejected MTP rounds, actual
   dense-to-sparse transitions, long private windows and resumed turns.
5. Run matched interleaved timing only after correctness and state acceptance.
   GGUF vision is a separate feature, not a prerequisite for text parity.

## Legacy MLX path

[Model plans](../../crates/lisa-models/src/glm53/) and engine bindings retain
original payloads, scales, biases and quantization groups. CPU-only
`lisa prepare --model /local/checkpoint` builds `.lisa-resident/index.json`;
`--plan-only` reports disk needs. Preparation is byte-preserving, validates
source identities and refuses to overwrite an existing output. The UI prepares
a missing index on first load. This step is specific to MLX GLM.

Resident experts retain stable addresses. Decode does not select experts on
the CPU or stage their weights per token. Native Q8, paired projections,
bounded grouped MoE and DSA shape gates are implemented. Projection modes
retained at load are part of numerical-law identity before lazy compilation.
`LISA_GLM53_Q8_GATE_UP=separate` selects the reference projection schedule.

Two target-MTP drafts can commit three tokens. Accepted/rejected transactions
must retain cache, parser, sampler and RNG state at the same commit boundary.
LISA's deterministic `seed: 0` and configured `ignore_eos` policy must be aligned
explicitly when comparing DS4; their public defaults are not interchangeable.

Engine and kernel checks load external captures through `LISA_GLM53_FIXTURE_DIR`,
which points to a directory containing the named capture files. All-feature compilation
requires no fixture payloads; running enabled checks requires their data and
fails if it is missing. A durable external capture inventory with hashes remains
required. Vision limits are in [media](../media.md).

For the KDA real-input diagnostic, `LISA_GLM53_DUMP` is an explicit capture
file path. Legacy GLM writes the layer-0 inputs there, replacing the file on
each captured forward. The ignored `kda_prepare_real_inputs` kernel check reads
the same path when explicitly enabled and fails if the data is unavailable.
Capture mode performs readbacks and is excluded from performance measurements.
