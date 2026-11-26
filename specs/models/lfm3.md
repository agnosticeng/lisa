# LFM3

The `lfm3` identity selects a compact hybrid text model. The curated package is
`LiquidAI/LFM3-1.7B-MLX-4bit` (revision `91f34088d6946482cbcae11b85d4657093618c34`):
affine 4-bit, group-64 weights with BF16 scales and biases, about 991 MB. An
empty `vision_config` does not enable images.

## Model and execution

24 layers: 13 gated short convolutions, five sliding-attention layers and six
full-attention layers, with sandwich RMS normalization, paired-half partial
RoPE, attention-head gates and tied embedding/readout weights. Prefill runs at
most eight rows per graph; generation is single-stream with no batching,
verification or MTP. RoPE frequencies are computed once on the device; each
forward computes one BF16 cosine/sine pair per attention kind and shares it
across layers. Warmup covers row widths eight through one.

## State

Full-attention state grows with the context; sliding attention keeps physical
ring order through chunked prefill, decode and window crossings. Prefix
snapshots are reusable at multiples of eight tokens, keep the final prefill
chunk intact and include cursor and absolute position. Extension, continuation
and replay must reproduce fresh-prefill logits bit for bit. Text turns use the
checkpoint's BOS and empty thinking block; tool requests render the supplied
template. The checkpoint advertises 262,144 positions.

## Reference and limits

The numerical oracle is the pinned reference recorded with the
[reference harness](../../crates/lisa-bench/reference/lfm3/README.md), which
checks intermediate and logit digests for serial, chunked and mixed execution,
256-token window crossings and snapshot replay. Attribution is in
[NOTICE](../../NOTICE); the weights keep their own licence. Long contexts,
suspension on iOS and image input are not certified.
