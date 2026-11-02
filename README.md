# lisa

Local inference for Apple Silicon, with a Rust engine, a native Metal runtime,
JIT-compiled kernels, a command-line interface, an HTTP server and a macOS app.

![lisa](docs/images/screenshot.png)

## Setup

Use macOS on Apple Silicon, a Rust toolchain supporting edition 2024, and Apple
command line developer tools. Language-model inference requires Metal. Model
weights are separate; RAM must accommodate weights, caches and working buffers.

```bash
cargo build --release --workspace
./target/release/lisa run --model agnosticeng/Qwen3.8-27B-4bit \
  --prompt "Explain multi-token prediction in two sentences." --max-tokens 64
```

`--model` accepts a local checkpoint directory or a Hugging Face repository ID.
Repository IDs use the local hub cache or download the checkpoint when missing.

Start the server, including streaming `/v1/chat/completions`:

```bash
./target/release/lisa serve --model agnosticeng/Qwen3.8-27B-4bit \
  --addr 127.0.0.1:8000
```

Launch the desktop app with `./target/release/lisa-ui`. Chat and the app's HTTP
server use automatic MTP when the loaded checkpoint supports it. Use
`./target/release/lisa --help` and each subcommand's `--help` for options.
`chat` and cohort `batch` currently support Qwen4; use `run` or `serve` for
the other language-model families.

## Models

The loader uses `model_type` to select a family for `run` and `serve`:

| Family | Accepted `model_type` values |
| --- | --- |
| Qwen 3.8 Flash-Next | `qwen4_exp`, `qwen4_exp_text` |
| Qwen 3.8 27B | `qwen3_5`, `qwen3_5_text` |
| GLM-5.3 Flash | `glm5_next`, `glm5_next_text` |

Compatible tensor names, shapes and quantization are also required. Laya
(`laya`) is a separate typed-decision model, exposed through `decide` and
`/v1/decisions`, with CPU and Metal backends.

### GLM-5.3 checkpoints

GLM loads compatible MLX-format quantized weights through a resident index at
`<model>/.lisa-resident/index.json`, with its validated backing files. The index
must match the checkpoint; downloading ordinary weight shards alone is insufficient.

Prepare a local checkpoint once with the native CPU command below. It copies
the original tensor bytes into aligned storage, verifies SHA-256 readback and
publishes the index atomically. Allow approximately one extra checkpoint's
worth of free disk space. Existing prepared storage is never overwritten.

```bash
./target/release/lisa prepare --model /path/to/glm53 --plan-only
./target/release/lisa prepare --model /path/to/glm53
./target/release/lisa run --model /path/to/glm53 \
  --ctx-size 32768 --temperature 0 --max-tokens 64 --prompt "Hello"
```

`prepare` accepts a local directory only. `--plan-only` reports storage needs
without copying weights. Loading rejects a missing or stale index.

GLM's default context capacity is 32,768 tokens, capped by the checkpoint limit.
`run` and `serve` accept `--ctx-size` to set that capacity.

## Validation and development

Run GPU checks sequentially, with other model processes stopped and at least
8 GiB reserved for the host:

```bash
cargo test --release --workspace --lib -- --test-threads=1
./target/release/lisa-bench smoke
./target/release/lisa-bench golden --model agnosticeng/Qwen3.8-27B-4bit
```

Model checks require the corresponding checkpoint. Numerical and performance
results apply to the checkpoint, configuration and execution path measured.
GLM integration tests also require the external captures selected through
`LISA_GLM53_FIXTURE_DIR`; see the
[fixture inventory](crates/lisa-engine/src/models/glm53/fixtures/README.md).
For a compilation-only check, use `cargo check --workspace --all-targets`.

Read [AGENTS.md](AGENTS.md) for development and numerical rules. The
[GLM status](specs/11-glm53-flash.md) records implemented paths, measured cases
and remaining acceptance gaps.

## License and provenance

lisa is [MIT licensed](LICENSE). Sources include adaptations from MLX, DS4 and
other projects; see [NOTICE](NOTICE). The GLM port follows DS4 execution with
MLX-format weights. Model weights retain their own licenses.
