# DSpine independent oracle

`provenance.json` pins Tencent's implementation, the unmodified draft weights,
the inspected SGLang source and the MLX Metal kernels. `inventory.json` records
the size and SHA-256 of every external capture. Binary captures are not tracked.
The test checks the inventory before comparing values and fails when explicitly
enabled without the checkpoint or captures.

The November 10 cleanup preserved all 431 inventoried captures outside the
repository, with every size and hash verified before and after copying. This
machine's cache root is
`~/.cache/lisa/fixtures/dspine-321b0fc1388c-mlx-0.32.4/`: `oracle` contains the
BF16 seven-row fixture, `oracle-b16` the sixteen-row fixture, and `oracle-4bit`
the independently quantized fixture. Set the corresponding fixture environment
variables to these absolute directories. Other machines must generate or copy
the same inventoried data; the tests do not silently download or skip it.
Temporary adapters, logs, model-weight symlinks and benchmark binaries are not
part of the retained captures. The generator and provenance remain here.

Generate on Apple M5 with a separate environment containing `mlx==0.32.4`,
`mlx-lm==0.32.2` and `numpy==2.4.6`:

```sh
python capture.py --upstream /path/to/pinned/DSpine \
  --model /path/to/Qwen3.8-27B-DSpine-b16 --out /absolute/fixture/directory
```

The upstream MLX loader supports full attention only. This short-context
adapter changes attention metadata for 8 context + 7 proposal rows, where
causal full attention has exactly the same visible keys as the checkpoint's
8192-token window. Original safetensors are symlinked, never rewritten.
The staged diagnostics are checked against the unmodified upstream forward.
This adapter does not certify long-context upstream MLX support for Qwen27B.

The fixture covers the shared feature projection, all five draft layers and
bridge writes, final raw/packed/RMS states, and the original refinement tail
with a deterministic 512-entry synthetic target head. Separate feature
projections cover 1, 7, 16, 32 and 64 rows. The Rust test requires exact values;
it does not accept token equality as a substitute for hidden-state agreement.
The candidate and proposal checks do not establish full-vocabulary target-head
or CUDA equivalence. Separate GPU tests cover attention past the window edge.

Run sequentially, after the repository's RAM and memory-pressure checks:

```sh
LISA_QWEN_DSPINE_MODEL=/path/to/Qwen3.8-27B-DSpine-b16 \
LISA_DSPINE_FIXTURE_DIR=/absolute/fixture/directory \
cargo test --release -p lisa-bench --lib model_draft_matches_mlx_fixture \
  -- --ignored --nocapture --test-threads=1
cargo test --release -p lisa-metal --lib \
  dspine_projections_match_exact_dyadic_oracle -- --test-threads=1
```

The independent CPU projection oracle exercises offset bindings, tail tiles
and dispatch boundaries at M=1/2/5/7/15/16/32/63/64, K=128/260, N=96/37.
It uses dyadic inputs whose FP32 partial sums are exact. The NAX-specific test
does not certify older Apple GPU projection paths.

## Affine 4-bit reference

`capture.py --four-bit` independently quantizes the original upstream model's
36 attention/MLP/fc projections with MLX 0.32.4, group size 64. Bridge,
codebook and norms stay BF16. This is separate from loading the checkpoint
produced by [the converter](../../../tools/quantize_dspine.py).
`inventory-4bit.json` pins these external captures. It additionally covers
fc and all five context K/V projections at 1/2/3/4/5/6/7/8/12/13/16/32/36/64
rows, including single-row and split-K dispatch boundaries.

```sh
LISA_QWEN_DSPINE_4BIT_MODEL=/path/to/derived-4bit-drafter \
LISA_DSPINE_4BIT_FIXTURE_DIR=/absolute/4bit-fixture/directory \
cargo test --release -p lisa-bench --lib quantized_draft_matches_mlx_fixture \
  -- --ignored --nocapture --test-threads=1
```

The first check exposed old QMV BF16 input sums and BF16 split-K partials at
some context shapes. The dedicated DSpine affine path now retains MLX 0.32.4's
promoted sums and F32 partials with sequential final accumulation. Target
projection kernels remain unchanged. Both final expanded checks require and
achieve exact equality; no tolerance was relaxed.

## Full trained block and compact state history

`capture.py --block-size 16` captures the original full trained forward;
`inventory-b16.json` is separate from the retained seven-row fixture. Enable
`full_block_draft_matches_mlx_fixture` with `LISA_DSPINE_B16_FIXTURE_DIR` and
the original `LISA_QWEN_DSPINE_MODEL`. Missing or changed captures fail.
The short-context metadata adapter remains below the native 8192-token window.

`compact_replay_matches_every_snapshot_bitwise` compares every output and every
accepted-prefix FP32 state against the existing snapshot kernel at widths
2/3/7/8/12/16 with key/value heads 4/16 and the target's 16/48, including repeated commits in reverse order and the zero/full
boundaries. `dspine_replay_commits_conv_and_survives_snapshot_restore` covers
the legacy three-dimensional live-state layout, convolution alignment and
snapshot invalidation of speculative history. These are exact bitwise checks;
no target arithmetic tolerance was introduced.
