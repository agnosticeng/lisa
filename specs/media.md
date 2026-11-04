# Image input and generation

All inference runs inside the native Rust/Metal engine. Python reference
programs are validation tools, not runtime dependencies. Image pipelines have
their own measured numerical bounds; they do not relax text-model STRICT
requirements.

## Image input

| Text path | Vision support | Recorded evidence |
| --- | --- | --- |
| Qwen3.5 hybrid | Dense Qwen3-VL tower, merger and text M-RoPE | 0.8B encoder fixture: 65,536/65,536 values exact against MLX, twice |
| Qwen4 hybrid | Same integration family | Full Qwen4 checkpoint not exercised |
| Legacy MLX GLM | RMSNorm/QK-norm tower, 2D RoPE, downsample and merger | Integration works; final embedding cosine 0.98229, RMS 0.002855; parity open |
| GLM / DeepSeek GGUF | Unsupported | No GGUF vision package/runtime |

Vision weights must be present alongside language weights. Dense projections
are supported; quantized vision and DeepStack fail explicitly. GLM inserts
merged embeddings before HC expansion. Qwen preserves image positions across
prefill chunk boundaries.

PNG, JPEG and WebP decoding applies EXIF orientation and composites alpha on
white. Resizing uses Pillow-compatible fixed-point bicubic interpolation.
Qwen bounds resized inputs to 1,003,520 pixels; GLM bounds merged tokens to
1,024. Inputs are additionally limited to eight images per full history,
20 MiB encoded per image, 32 megapixels decoded in aggregate and a 32 MiB HTTP
request body.

CLI `run --image`, desktop Attach, OpenAI chat image blocks, Responses
`input_image` and Anthropic base64 sources use shared processing. HTTP accepts
inline base64 only; remote URLs and file IDs are unsupported.

Each image turn rebuilds a serial session from complete multimodal history.
MTP, PLD, wave batching and token-only prefix reuse are bypassed; `session_id`
and standalone image token-count requests are rejected. Clear image embeddings
on success and error. The UI retains history and attachment copies locally.

The GLM vision comparison first diverged in attention: 14/409,600 bf16 values,
maximum 0.001953125, with errors accumulating through 24 blocks. Correct color
or object answers do not establish numerical parity. The image-input reference
was MLX-Serve `6c4306145b7331f6b962b97e145fd9ac46e9e14a`.

## Native text-to-image models

| Checkpoint | Original weights | Default steps | Encoder / transformer |
| --- | --- | ---: | --- |
| `mlx-community/flux2-klein-4b-4bit` | MLX 4-bit, group 64 | 4 | Qwen3; 5 double + 20 single blocks |
| `mlx-community/flux2-klein-9b-4bit` | MLX 4-bit, group 64 | 4 | Qwen3-8B; 8 double + 24 single blocks |
| `black-forest-labs/flux-3-image-klein` | BF16 | 16 | Qwen3-VL-4B text encoder; 5 double + 28 joint blocks |

The [FLUX.2](../crates/lisa-engine/src/models/flux2/) and
[FLUX 3](../crates/lisa-engine/src/models/flux3/) implementations load encoder,
denoiser and VAE stages sequentially. Selecting an image model releases the
chat model. Admission includes stage weights/layout copies and conservative
8 GiB scratch; actual allocations still obey the runtime budget. FLUX 3 reserves
twice the transformer file size for mapped originals and cached linear layouts.

Shared surface: CLI `image`, desktop Generate and
`POST /v1/images/generations`. One PNG per request, default 512×512, each side
64..1024 and a multiple of 32. FLUX.2 accepts 1..50 steps and truncates encoding
to 512 tokens; prompt input is bounded to 16,384 bytes. HTTP returns `b64_json`.
See [generation options](../crates/lisa-engine/src/models/image_generation.rs)
for model-specific validation.

SplitMix64/Box–Muller noise makes a LISA seed reproducible within this engine;
the same integer seed does not reproduce MLX/PyTorch noise. Attention uses
bounded f32 score/softmax/PV work, and VAE convolutions use bounded im2col.
These operations are not certified bit-exact to upstream fused implementations.
No image editing, multi-reference input, layout boxes or LoRA is exposed.

### FLUX 3 execution details

The original BF16 download is 20,320,953,966 bytes and excludes duplicate
Diffusers, FP8 and NVFP4 weights. Gated model access and a local HF token are
required. Model licenses remain separate from inference-source licenses.

The image-conditioning encoder concatenates layers 4/8/12/16/20/24/28/32 and
pads toward a multiple of 80, capped at 4,096 tokens. The distilled flow-map
schedule uses alpha 16 and no CFG, following executable reference code; checkpoint metadata's
alpha 6.93 and text limit 8,192 are not used. Noise is bf16-rounded, recurrent
latents are f32 and velocity×dt rounds to bf16 before the f32 update. VAE
execution is f32 with clamp/truncate conversion to image bytes.

Short prompts are expanded locally by the bundled instruction encoder, bounded
to 8,192 input tokens and 512 generated caption tokens. This is not the separate
official BFL 9B upsampler. `--prompt-is-caption` or HTTP
`prompt_is_caption: true` bypasses it.

## Pinned references

| Package / source | Revision |
| --- | --- |
| Klein 4B checkpoint | `860e87183ceb29e39627c0612ebd66d8ea66e68c` |
| Klein 9B checkpoint | `f0c62655b9da432932ba24d4bfce8a96800cf0bd` |
| FLUX.2 source | `50fe5162777813d869182b139e83b10743caef15` |
| FLUX 3 checkpoint | `09561231f44e41381c461caf53d5452005a60ac9` |
| FLUX 3 execution source | `bfc8889a4497f846416f253a3972a1941ac698b9` |

## Validation scope

Recorded November 3, with independent fixed inputs at each stage. Encoder error
is not carried into the denoiser comparison. BF16 stage acceptance was relative
RMSE ≤0.03; f32 decoder ≤0.001. Both external suites passed twice.

| Model / stage | Relative RMSE | Workload |
| --- | ---: | --- |
| Klein 9B encoder | 0.00758361 | 512 text tokens |
| Klein 9B denoiser | 0.01481519 | 24 image tokens |
| FLUX 3 encoder | 0.00267177 | 80 text tokens |
| FLUX 3 denoiser | 0.02010682 | 24 image tokens |
| FLUX 3 decoder | 0.00000332 | 64×96 output |

Fixed-seed image reproducibility, HTTP parameter rejection, native UI generation
and image-card display were exercised. Qwen 27B text goldens passed twice.
Small stage fixtures and a few recognizable images do not certify broad image
quality or end-to-end numerical parity. Controlled cross-engine throughput
remains unmeasured.

The dense layout cache removes repeated transposed-weight materialization
without changing values. A reported FLUX 3 219.67→83.53-second observation had
a competing CPU build in the baseline; it is diagnostic, not a speedup verdict.

External stage tests use `LISA_FLUX2_MODEL`, `LISA_FLUX2_FIXTURES`,
`LISA_FLUX3_MODEL` and `LISA_FLUX3_FIXTURES`. The `reference.py` beside each
implementation generates inputs; explicitly enabled tests fail when data is
missing. Run sequentially under the [validation protocol](performance.md#validation-protocol).
Retain fixture hashes, environment versions and exact measurements with each
external capture set before using it as a reproducible comparison.
