# GLM-5.3 Flash

A 45-layer trunk with 34 KDA layers, 11 sparse-attention (DSA) layers,
hyper-connection mixing, routed MoE and target-conditioned MTP. Two execution
paths with distinct laws, loading contracts and schedules:

| | Packed GGUF | Affine MLX |
| --- | --- | --- |
| Identity | `glm5-next` | `glm5_next`, `glm5_next_text` |
| Weights | Original mixed GGUF blocks | Original affine payloads plus a prepared resident index |
| MTP | Embedded block 45; automatic | Target MTP; automatic |
| Prefill | Layer-major blocks of up to eight rows | Shape-specific wide schedules |
| Verification | Serial rows | Shape-specific schedules |
| Images | Unsupported | Dense vision tower; numerical gap open |
| Reference parity | Not certified | Not certified |

## Packed checkpoint

| Item | Value |
| --- | --- |
| Repository | `antirez/glm-5.3-flash-gguf` |
| Revision | `b2fa29d7a6b410db11221c904973967b80b760f5` |
| File | `GLM-5.3-Flash-Q2.gguf`, 96,505,816,384 bytes |
| SHA-256 | `e81fd6241c6e55a64e1e14e47a3eab61a173fa8d7e4b5c1d1848827119705b32` |

The loader validates 1,412 tensors (614 F32, 151 BF16, 450 Q8_0, 68 Q4_K,
86 IQ2_XXS, 43 Q2_K). Dense and split K/V projections are Q8, KDA projections
Q4, routed experts IQ2/Q2; embeddings read original Q8 rows. Nothing is
requantized or prepared.

Prefill traverses each layer over at most eight rows: independent dense,
shared and routed FFN projections share dispatches while attention, KDA
recurrence, DSA boundaries and expert reduction order stay part of the
reference law. The queue drains after each block, bounding temporaries
independently of layer count. The routed FFN fuses clamped SwiGLU and route
weighting in one F32 dispatch with a volatile intermediate that keeps the
activation rounding boundary. MTP's private attention switches to the bounded
wide-lora operation above 2,051 selected rows. Tokenization uses the embedded
BPE, GLM4 splitting and checkpoint control markers.

## Affine checkpoint

MLX GLM needs `.lisa-resident/index.json` inside the checkpoint directory:
`lisa prepare --model <dir>` writes it (`--plan-only` reports disk needs), the
app prepares it on first load, and preparation preserves source bytes and
refuses to overwrite. Resident experts keep stable addresses; decode routes on
the GPU without CPU expert selection or staging. Q8 gate/up projections are
paired; the pairing is part of the law.

## MTP and sampling

Two drafts can commit three tokens. Accepted and rejected rounds preserve
target and draft caches, parser, sampling history and RNG at one commit
boundary. GLM uses the host-side reference sampler (top-k, top-p, min-p and its
own RNG; `uses_host_sampling`). `--mtp-exact-sampling` selects the reference's
exact stochastic acceptance and residual sampling.

## Reference and limits

The arithmetic oracle is the pinned GLM reference implementation; kernel
identities and adaptation scope are in
[provenance](../../crates/lisa-metal/src/shaders/sparse_attention/gguf_provenance.json)
and attribution in [NOTICE](../../NOTICE). Full strict parity is open: the
packed path has a known layer-3 state mismatch that reaches the final logits
despite matching short greedy output, and the affine acceptance matrix is
incomplete. Packed wide verification, grouped expert prefill and GGUF vision are
not implemented. The checkpoint advertises 1,048,576 positions; capacity does
not certify behaviour at dense/sparse transitions or long private windows.

`lisa-bench glm53-run --model <path>` produces token-ID JSON for comparison;
`--capture-logits` runs are diagnostic and excluded from timings.
