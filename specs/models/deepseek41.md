# DeepSeek V4.1 Flash Coder

Experimental native text inference over original mixed GGUF weights, available
in the CLI, server and app.

| Property | Value |
| --- | --- |
| Identity | GGUF `general.architecture = deepseek41` |
| Repository | `ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-GGUF` |
| Revision | `54229d38d588a66d1c6cff08af7b281b9326c656` |
| Package | Five transformer shards under `IQ2_XXS/` and `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` |
| Drafting | None; `auto` is serial, fixed depths fail |
| Images, batching | Unsupported |

`checkpoint::deepseek41::open_package` accepts the published layout, in which
the shared Engram shard sits beside or one level above the quantized shards.
Identities and adaptation scope are in
[provenance](../../crates/lisa-models/src/deepseek41/provenance.json).

## Weights and execution

Adapters validate 1,043 tensors. Resident weights take about 67.2 GiB: dense
projections in Q5_K, Q8_0, BF16 and F32, routed experts in IQ1_M, IQ2_XS and
IQ2_XXS, six of 192 experts per routed layer. The ~103 GiB IQ4_NL Engram table
stays on disk behind bounded asynchronous row readers that teardown joins. The
precombined F32 gate applies its factor of two once.

The 40-layer graph covers hyper-connections, MLA, compressed and raw-ring
attention, FP4 cache handling, index selection, MoE and Engram. Prefill is
layer-major over one to eight rows, sharing dense and expert projection
launches while attention stays causal per row. Wide grouped expert GEMM is not
implemented.

## State

The native cache supports copy-on-write RAM snapshots bound to model, file and
law; it has no spill codec. Chat uses the embedded BPE, the model's chat format
and DSML tool parsing. Continuation compares exact token prefixes, including an
emitted but unfed EOS. The checkpoint advertises 1,048,576 positions.

## Reference and limits

The [same-weight reference adapter](../../crates/lisa-bench/reference/deepseek41/README.md)
adds this package's formats to the pinned reference so it can load them. Some
packed-format helpers are shared with LISA, so adapter agreement and
independent format validation are separate claims. Full context, throughput and
language quality are not certified. `lisa-bench deepseek41-run --model <gguf>`
runs the native graph end to end.
