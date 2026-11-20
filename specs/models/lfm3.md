# LFM3 text models

The `lfm3` checkpoint identity selects the native Metal implementation. The
curated package is `LiquidAI/LFM3-1.7B-MLX-4bit`, with original affine 4-bit,
group-64 packed weights and BF16 scales/biases. It is text-only: the empty
`vision_config` does not enable an image encoder. The package is approximately
991 MB including its tokenizer and metadata; weights account for 973 MB.

## Execution and state

Portable model graphs live in `lisa-models/src/lfm3`; the engine adapter owns
file access, weight binding, cache handles and execution. The 24-layer tower
has 13 gated short convolutions, five sliding-attention layers and six full
attention layers. It retains sandwich RMS normalization, paired-half partial
RoPE, attention-head gates and tied embedding/readout weights.

Prefill uses at most eight rows per graph. Generation is single-stream;
B>1 batching, speculative verification, MTP and shared prefix reuse are not
advertised. Full-attention state grows with actual context. Sliding state
retains the reference's physical ring order during decode, including transitions
between chunked prefill and single-token generation. Snapshots share immutable
tensors; subsequent graph execution produces replacement state. Cache cursor
and absolute position are included in snapshots.

The checkpoint advertises 262,144 positions. This is a configuration limit,
not certification of that context length on a phone. Weight admission follows
the host budget; actual context and transient allocations still consume memory.
Physical iOS devices retain the existing per-app allowance and 512 MiB reserve.

Text turns use the checkpoint's BOS and exact empty thinking block. Tool
requests render its supplied Jinja template with tokenizer-config special
tokens and object-valued tool arguments. The app preserves its existing default
model; LFM3 is an additional selectable model, pinned to each conversation.

## Reference and scope

The arithmetic oracle is MLX 0.32.4 with MLX-VLM 0.7.11. The LFM3 reference
sources are pinned at `a22a364af9d7ee5e7e063881923cf18a3f42533d`; checkpoint
revision is `91f34088d6946482cbcae11b85d4657093618c34`. Source attribution and
Apache-2.0 terms are retained in `NOTICE` and beside the model plans. Downloaded
weights retain LiquidAI's separate LFM Open License v1.0.

Exact intermediate/logit digests, inputs, reproduction commands and validation
scope are kept in [the reference harness](../../crates/lisa-bench/reference/lfm3/README.md).
Physical iPhone 15 Pro Max checks cover downloads, generation, conversation
recall, cancellation, saved-chat replay and direct lifecycle/memory-handler
invocation. They do not certify OS suspension, every iPhone, long contexts or
image input. The existing simulator BF16 compiler limitation still applies.
