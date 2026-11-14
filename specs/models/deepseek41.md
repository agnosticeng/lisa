# DeepSeek V4.1 Flash Coder

## Status and identity

Experimental native Rust/Metal text inference using original mixed GGUF blocks.
CLI, HTTP and desktop integration are implemented. The
checkpoint has **no draft head**: automatic depth selects serial generation,
and an unsupported fixed MTP depth errors. Vision and native multi-client
decode batching are unsupported.

| Item | Pinned identity |
| --- | --- |
| Repository | `ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF` |
| Revision | `54229d38d588a66d1c6cff08af7b281b9326c656` |
| GGUF architecture | `deepseek41` |
| DS4 revision | `b3f50180ec2c7324892be1f50bc83555ccb8d5e7` |
| Package | Five transformer shards plus `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` |

Full identities and adaptation scope live in
[provenance](../../crates/lisa-models/src/deepseek41/provenance.json).
All six files are required at the snapshot root. The downloader preserves
notices and expert-selection metadata. No MLX conversion or prepared index is
needed.

## Implemented execution

The loader validates 1,043 tensors and keeps approximately 67.18 GiB of resident
weight storage. Dense projections use Q5_K/Q8_0/BF16/F32; routed experts use
IQ1_M/IQ2_XS/IQ2_XXS. The model selects six of 192 experts per routed layer.

The 103 GiB Engram table uses IQ4_NL and remains on disk. Positional reads fetch
selected rows, with bounded asynchronous readers joined during teardown.
The precombined F32 gate applies its factor of two once. Disk size must not be
treated as resident RAM or ignored when checking download space.

The 40-layer graph implements HC, MLA, compressed/raw-ring attention, FP4
cache handling, index selection, MoE and Engram. Native layer-major prefill
processes 1..8 rows per chunk; dense/expert projections share launches while
attention remains ordered causally by row. A wide grouped expert GEMM remains
absent.

Current schedules include grouped output projections, paired Q8 gate/up,
fused RoPE/cache writes, per-position coefficients, Engram split gates and
wide index selection. Context gates include 8,192, 10,240, 65,536 and 98,304;
their existence does not certify those real-context workloads.

CLI and UI default to the checkpoint metadata limit of 1,048,576 tokens.
`--ctx-size` can override it in the CLI; allocation remains subject to memory
admission. Advertised capacity does not certify numerical parity at every context length.

The native cache supports copy-on-write RAM snapshots bound to model/file and
numerical identity. Admission accounts for configured cache capacity. The generic
SSD spill format cannot serialize this native state.

Embedded BPE, DS4 chat rendering and native DSML tool parsing support HTTP
chat and streaming. Persistent continuation compares exact token prefixes,
including the final EOS token when it has not yet been fed into the model.
Desktop chats can resume text history after reopening or app restart.

## Reference contract

Stock DS4 does not load this exact mixed-format package. The pinned
[reference adapter](../../crates/lisa-bench/reference/deepseek41/README.md)
adds its formats while retaining DS4 execution. Some Metal format helpers are
shared with LISA. Therefore adapter agreement, independent CPU/GPU format
validation and stock upstream performance are distinct claims.

The reference adapter uses `-O1` for its large C unit because Clang 21 fails
at `-O3`; other units retain `-O3` and original Metal. Its scalar prompt-logit
prefill is unsuitable as a matched prefill performance baseline. Preserve these
adapter limits when making a future parity or throughput claim.

## Remaining acceptance

- Broader prompts and real contexts beyond the first wide index gate, including
  state continuation, errors and DSML edge cases.
- Native grouped expert prefill and matched long-context throughput measurements.
  Original mixed IQ formats need their own kernels; enabling unrelated DS4
  MXFP4 gates is not a port.
- Durable native-state spill support if required, with round-trip equality and
  correct memory accounting; RAM snapshots alone do not provide this.
- Preserve the reference compiler caveat and independent format checks in any
  future performance or parity verdict.

## Reproduction entry points

Use `lisa-bench deepseek41-run --help` for current options. It accepts the first
transformer shard and exposes `--snapshot-check`, `--capture`,
`--capture-layers`, `--tokenize-only`, `--prefill-chunk` (1..8) and `--output`.
The model-dependent tests use `LISA_DEEPSEEK41_GGUF` and must be explicitly
enabled. Follow the reference adapter's build/compare instructions and
[measurement protocol](../performance.md#validation-protocol).

Synthetic zero-filled cache checks at larger capacities do not establish
language-model quality there. Reproduction requires the pinned checkpoint and
reference adapter with explicitly selected capture inputs and output paths.
