# LFM3 reference checks

Oracle: MLX 0.32.4 and MLX-VLM 0.7.11. The installed LFM3 `language.py`
matches MLX-VLM revision `a22a364af9d7ee5e7e063881923cf18a3f42533d` byte for byte
(SHA-256 `3a6cf68a5d7478683ac8a891c53f16d362b8d26ff8cf615f38aab175f0dc7907`).
Checkpoint: `LiquidAI/LFM3-1.7B-MLX-4bit`, revision
`91f34088d6946482cbcae11b85d4657093618c34`.
`model.safetensors`: 973,083,386 bytes, SHA-256
`1be5bf672013cd91aa62148f77c5eea8d2784a95845d5fe826e08c7b84823815`.

`goldens.json` contains fixed token sequences, chunk sizes and digests of
reference F32 views of BF16 tensors. Each digest covers sorted tensor names,
shapes and exact values: embeddings, all 24 layer outputs, convolution state
and final logits. The 29-token cases cover serial and eight-row prefill; the
270-token cases additionally cross the 256-token sliding window, in both
serial and chunked execution. `window_mixed` uses eight-row prefill followed
by eight single-token calls (`8:8`), checking the physical ring transition. The compiled capture also restores the last
chunk's snapshot and requires identical replay logits. An absent checkpoint
or capture fails; it is never treated as a passed comparison.

Build the diagnostic and select a case:

```sh
cargo build --release -p lisa-bench --example lfm3-capture
python3 - <<'PY'
import json
from pathlib import Path
case = json.loads(Path('crates/lisa-bench/reference/lfm3/goldens.json').read_text())['cases']['window_chunk8']
Path('/tmp/lfm3-tokens.json').write_text(json.dumps(case['tokens']))
PY
sysctl hw.memsize
memory_pressure
./target/release/examples/lfm3-capture "$MODEL" /tmp/lfm3-tokens.json 8 /tmp/lfm3-native.npz
python crates/lisa-bench/reference/lfm3/compare.py window_chunk8 /tmp/lfm3-native.npz
```

`MODEL` must point to the pinned local snapshot. Use the reference environment's
Python (with NumPy) for comparison. To independently reproduce the oracle:

```sh
uv venv /tmp/lfm3-reference
uv pip install --python /tmp/lfm3-reference/bin/python mlx==0.32.4 mlx-vlm==0.7.11
/tmp/lfm3-reference/bin/python crates/lisa-bench/reference/lfm3/capture.py \
  "$MODEL" 8 /tmp/lfm3-mlx.npz /tmp/lfm3-tokens.json
/tmp/lfm3-reference/bin/python crates/lisa-bench/reference/lfm3/compare.py \
  window_chunk8 /tmp/lfm3-mlx.npz
```

Run GPU work sequentially and read RAM/pressure before benchmark runs. Captures
retain extra tensors and are excluded from performance evidence. Do not update
digests to hide arithmetic changes. These checks establish the listed shapes
on the tested Mac, not reference equivalence on every Apple GPU or at all
context lengths. Physical iPhone checks exercise the actual app/Metal path
but are behavioral, not a second MLX numerical oracle.

The newly admitted kernel shapes are Q4/group-64 projection `[M,2048]` by
`[16,2048]` for M=1..8, and masked BF16 vector attention with D=128, 16 query
heads, four KV heads and S<=8. Existing projection shapes retain their prior
dispatch. LFM3 uses the existing depthwise arithmetic as explicit ordered F32
products/adds, with one final BF16 cast. No new convolution shader, model
requantization or speculative arithmetic exception is introduced.

## Validation record

On 2026-11-19, both repeated passes on an Apple M5 Max matched all stored
reference digests exactly: 1,103 serial tensors, 153 short chunked tensors,
10,261 long serial tensors, 1,293 long chunked tensors and 1,559 mixed
prefill/decode tensors. Every case also passed exact snapshot replay.
Diagnostic Chrome traces were collected for both passes; their command-buffer
intervals are not exclusive per-kernel hardware timings.

A signed Debug build on an iPhone 15 Pro Max, iOS 27.2.1, passed the real app's
download, Metal load, reply, multi-turn recall, cancellation, persisted-chat
replay, foreground reload and direct memory-warning handler checks. The phone
reported 8,027,389,952 physical bytes and a 2,972,612,672-byte admission budget
at the start of that run; these are observations, never hardcoded capacities.
