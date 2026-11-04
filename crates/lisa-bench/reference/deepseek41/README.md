# Same-weight DeepSeek V4.1 reference

This harness builds the pinned DS4 C/Metal graph against the original
ISTA IQ2_XXS package. Generated sources, binaries, mappings and captures belong
in an external output directory. No model payload is rewritten.

```sh
DS4_SOURCE=/path/to/ds4
GGML_SOURCE=/path/to/llama.cpp
MODEL_FILE=/path/to/IQ2_XXS/DeepSeek-V4.1-Flash-1017-GSQ-RCO-Coder-IQ2_XXS-00001-of-00006.gguf
REFERENCE_OUTPUT=/path/to/reference-output
CAPTURE_ROOT=/path/to/captures
mkdir -p "$CAPTURE_ROOT"

python3 crates/lisa-bench/reference/deepseek41/build.py \
  --ds4 "$DS4_SOURCE" --ggml "$GGML_SOURCE" \
  --model "$MODEL_FILE" --out "$REFERENCE_OUTPUT"
(
  cd "$DS4_SOURCE"
  "$REFERENCE_OUTPUT/reference" "$CAPTURE_ROOT/stages" 0 128803 23166 128804 128822 23166 3 1730
  "$REFERENCE_OUTPUT/reference" --logits "$CAPTURE_ROOT/logits" 0 128803 23166 128804 128822 23166 3 1730
  "$REFERENCE_OUTPUT/reference" --benchmark 64 0 128803 23166 128804 128822
)
```

Replace the example paths with absolute paths to your inputs and outputs.
The working directory must be the pinned DS4 checkout for Metal source loading.
Build rejects modified or different upstream revisions and existing output
directories. Capture refuses existing directories. `build.json` records source
hashes, checkpoint file identities and the exact compiler commands. Run only
one model process at a time and check `sysctl hw.memsize` and `memory_pressure`
before each run. The reference checks resident bytes plus DS4 scratch against
detected RAM minus 8 GiB.

`--logits` executes DS4's original `ds41_graph_step`, including its queue cadence,
RoPE table and reusable asynchronous Engram reader. Full stage capture runs the
same original layer operations with explicit command completion before reads;
it is diagnostic and must not be timed. `--benchmark` generates a fixed number
of tokens, with N-1 timed decode forwards, including disk reads and CPU argmax.
It uses scalar prefill and currently computes prompt logits at every position;
its prefill time is not comparable to LISA's bounded prefill. Initialization
and shader compilation boundaries also differ. Decode timings compare the
adapted same-weight engines, not stock DS4 running its own checkpoint.

Compare with a LISA `deepseek41-run --capture` NPZ (requires NumPy):

```sh
LISA_CAPTURE=/path/to/lisa-capture.npz
python3 crates/lisa-bench/reference/deepseek41/compare.py \
  "$LISA_CAPTURE" "$CAPTURE_ROOT/stages" --output "$CAPTURE_ROOT/comparison.json"
```

Use `--logits-only` for the original decode-loop capture. Missing arrays,
nonfinite values, shape differences and any changed float bit fail the check.
Reference captures can contain extra arrays because LISA normally computes
logits only for the final prompt token. Force all prompt IDs followed by all
but the last generated ID when comparing a generated sequence.

## Independence and adapters

The C graph, router, HC, attention, cache algorithms, original supported Metal
kernels and asynchronous reader orchestration come from DS4. The six-file map,
192-expert shape, metadata aliases and combined Engram gate are adapted.
Original GGML CPU routines decode the Q5 embedding and IQ4_NL Engram rows.
DS4's Engram reader changes only row width and payload decoding.

DS4 cannot execute this package unmodified: its dense and expert fast paths
expect different formats. The Q5_K, IQ1_M and IQ2_XS extensions use LISA's
byte-pinned GGML/DS4 Metal helper extraction; IQ2_XXS is dispatched through that
same extension. They are **shared kernels**, so whole-graph agreement does not
independently prove those kernels. Separate GGML CPU decoder and exact-input
kernel checks cover their addressing and decoding. BF16 weights use original
DS4 BF16 kernels. The adapter preserves these limitations explicitly.

The full DS4 C translation unit uses `-O1` following an observed Clang 21 crash
at `-O3`. Other units use `-O3`; the original Metal compile options are
unchanged. Account for this host-build limitation in performance comparisons.

Generated upstream portions retain their upstream licenses. DS4 attribution
and license are also kept under `crates/lisa-mlx/src/shaders/deepseek41/`;
GGML/llama.cpp attribution is recorded in the GGUF and shader provenance files.
