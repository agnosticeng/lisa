# Qwen

| Checkpoint identity | Plan | Model |
| --- | --- | --- |
| `qwen3_5`, `qwen3_5_text`, GGUF `qwen35` | [`qwen3_5`](../../crates/lisa-models/src/qwen3_5/) | Dense hybrid: GDN recurrence and full attention |
| `qwen4_exp`, `qwen4_exp_text` | [`qwen4`](../../crates/lisa-models/src/qwen4/) | Flash-Next: GDN, dense/sparse attention, MoE, hyper-connections, PLE |

`Qwen35Tower` shares traversal, prefill and session handling between the MLX
and GGUF dense paths. MLX keeps BF16 casts and its attention/GDN law; GGUF keeps
original packed blocks and F32 state. Their saved state never crosses laws.

## Dense MLX

The decoder alternates GDN and full attention as configured. For the hidden-5120
geometry (24 query heads, 4 KV heads, head dimension 256), Q/K/V share joined
storage for single-request widths 1–8, and the affine-4/group-64 MLP shares
gate/up storage for widths 2–8. Original projections remain views of the joined
allocation; other geometries use their declared paths.

**MTP head.** Shipped as the `mtp/weights.safetensors` sidecar in the same
repository and loaded automatically when present. Drafts use a derived
2-bit/group-64 shortlist of 32 candidates rescored with exact trunk-head rows;
verification uses the original head. Depth `auto` uses the expected-value
controller; fixed depths are 2–6, and context-copy hits may propose up to seven.

**Captured verification** preserves serial arithmetic for every accepted row:

- `AffineSerialLinear` uses the serial matrix-vector reduction on an
  independent row grid (composed single-row calls when unsupported). GDN,
  attention, MLP and the vocabulary head use it.
- `AttentionSerialCausal` gives each query its serial visible prefix, family
  and key partition; family and partition boundaries split the group, including
  the 1024-key single-/two-pass transition.
- GDN capture keeps all three convolution-history rows, every F32 recurrent
  prefix and the BF16 output boundary, with the serial gate's transcendental
  arithmetic.

## Flash-Next

Captured plans use serial projection laws for GDN, attention, PLE, the indexer
and the vocabulary head. MoE routing composes the original single-row
operations; eligible expert projections use the serial reduction on a row grid.
The hyper-connection mixer has no row grid and is composed per row. Sparse
attention reads each query's visible prefix through its original dense-mask or
block-attention operation, including captures crossing the 2048-token
block-attention threshold; captured sparse attention needs one stream. PLE keeps
wide convolution with complete history; rollback applies the model-owned
convolution prefix plan before trimming KV.

## Dense GGUF

The curated file is `Qwen3.8-27B-UD-Q2_K_XL.gguf` from `unsloth/Qwen3.8-27B-GGUF`:
64 trunk layers and an embedded MTP block. Mixed Q/IQ blocks and F32 parameters
stay packed; embeddings are read by row from disk; attention and recurrent state
are F32. Prefill and draft priming use 16-token windows. The path is text-only,
one stream per forward, and caps MTP at two drafts. Its execution reference is
pinned in [provenance](../../crates/lisa-bench/src/native/bindings/qwen35_gguf/provenance.json).

## State

Rollback restores recurrent, attention, PLE, indexer, draft and position state
at the same accepted boundary. The independent head keeps aligned token/hidden
pairs; snapshot restore trims them to the target boundary and restores the
matching last hidden row. When private draft tensors cannot be rewound, restore
records a cold head and primes it lazily from the retained rows. Laws and cost
keys are listed in [numerics](../numerics.md#state-identity).

## Validation

Kernel changes need exact comparisons against the operation chain, every
accepted prefix and exhaustive finite-BF16 gate checks where practical.
Attention checks cross family and partition boundaries with offset, strided KV.
`lisa-bench golden`, `session-check`, `round-cost` and `layer-diff` serve
distinct purposes.

## Limits

Flash-Next chunked versus full-prefill state equivalence, broader prompts and
long contexts, physical-mobile execution, broader batching and vision shapes,
GGUF long contexts and independent MTP-head comparison remain open.
