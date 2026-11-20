# Image input and generation

All inference runs inside the native Rust/Metal engine. Python reference
programs are validation tools, not runtime dependencies. Image pipelines have
their own numerical bounds; they do not relax text-model STRICT
requirements.

## Image input

| Text path | Vision support / limit |
| --- | --- |
| Qwen dense hybrid | Dense Qwen3-VL tower, merger and text M-RoPE |
| Qwen Flash-Next | Integrated; full-checkpoint vision qualification remains open |
| Legacy MLX GLM | RMSNorm/QK-norm tower, 2D RoPE, downsample and merger; numerical parity remains open |
| GLM / DeepSeek GGUF | Unsupported |

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

GLM vision has an unresolved attention/embedding discrepancy; successful
recognition is not numerical parity. Its execution reference is MLX-Serve
`6c4306145b7331f6b962b97e145fd9ac46e9e14a`.

## Desktop voice chat

The native macOS composer uses Apple's Speech framework for microphone
dictation and AVFAudio speech synthesis for spoken replies. Dictation updates
the composer with partial and final text; the user reviews and sends it as an
ordinary text turn. The **Read replies** control speaks completed assistant
answers. Chat history stores the transcript and answer text; captured audio is
not saved.

Recognition requires microphone and speech-recognition permission and is
restricted to Apple's on-device mode. If the current system language is not
available for on-device recognition, voice input reports that it cannot start
instead of sending audio to a network service. Voice is currently a desktop UI
feature; the CLI and HTTP APIs continue to accept text and image input.

## Native text-to-image models

| Checkpoint | Original weights | Default steps | Encoder / transformer |
| --- | --- | ---: | --- |
| `mlx-community/flux2-klein-4b-4bit` | MLX 4-bit, group 64 | 4 | Qwen3; 5 double + 20 single blocks |
| `mlx-community/flux2-klein-9b-4bit` | MLX 4-bit, group 64 | 4 | Qwen3-8B; 8 double + 24 single blocks |
| `black-forest-labs/flux-3-image-klein` | BF16 | 16 | Qwen3-VL-4B text encoder; 5 double + 28 joint blocks |

The [FLUX.2](../crates/lisa-engine/src/bindings/flux2/) and
[FLUX 3](../crates/lisa-engine/src/bindings/flux3/) implementations load encoder,
denoiser and VAE stages sequentially. Selecting an image model releases the
chat model. Admission includes stage weights/layout copies and conservative
8 GiB scratch; actual allocations still obey the runtime budget. FLUX 3 reserves
twice the transformer file size for mapped originals and cached linear layouts.

Shared surface: CLI `image`, desktop Generate and
`POST /v1/images/generations`. One PNG per request, default 512×512, each side
64..1024 and a multiple of 32. FLUX.2 accepts 1..50 steps and truncates encoding
to 512 tokens; prompt input is bounded to 16,384 bytes. HTTP returns `b64_json`.
See [generation options](../crates/lisa-engine/src/image_generation.rs)
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

## Validation requirements

Use independent fixed inputs for each stage, so encoder errors do not contaminate
denoiser comparisons. Current stage tolerances are relative RMSE <=0.03 for BF16
and <=0.001 for the F32 decoder; these are not text-model STRICT exceptions.
Full-checkpoint quality, larger shapes and controlled cross-engine performance
remain separate acceptance work.

External stage tests use `LISA_FLUX2_MODEL`, `LISA_FLUX2_FIXTURES`,
`LISA_FLUX3_MODEL` and `LISA_FLUX3_FIXTURES`. The `reference.py` beside each
binding generates inputs. Enabled tests fail if required data is missing.
Follow the [validation protocol](performance.md#validation-protocol).

### Bonsai Image Binary 4B

The native FLUX.2 family also accepts
`prism-ml/bonsai-image-binary-4B-mlx-1bit`. This is the image checkpoint, not
the similarly named Bonsai language model. It retains original U32 1-bit
transformer payloads, BF16 affine scales/biases (groups of 128), dense skip
layers, and the original 4-bit Qwen3 text encoder. Diffusers VAE convolution
layouts and checkpoint names are adapted without requantization.

- Checkpoint revision: `d1b3ac11a7f1ba61d84b277339daeeed4a98e0e2`.
- PrismML MLX kernel reference: `bbc151c6360ee79a7412fa23cfc5c5fc63ced1ca`.
- PrismML mflux reference: `bcd13e83b7fcfd76186c98ef322dd9cf28e996c1`.
- Swift fork inspected: `2a673fa8324ca9900f37445db1f66f4400e4af15`, which pins
  core `a10ad3bf31d7700d464a993e90636ca4446c6639`. No Swift MLX runtime is linked.

The 3.43 GB package includes a 965 MB transformer, 2.26 GB encoder and 168 MB
VAE. Tensor hashes were verified against the published manifest; its README
hash is stale at this revision. Four denoising steps use the executable
mflux empirical FLUX.2 schedule, rather than interpreting the scheduler's
exported `shift: 3` as a different fixed schedule. Binary Q/K normalization
uses the reference's epsilon of 1e-6; existing FLUX.2 retains 1e-5.

Binary affine projections use the portable Metal GEMM path (including on M5),
with on-the-fly bit decoding; packed weights are never expanded permanently.
Only the first 27 encoder layers are loaded and executed, since the requested
hidden states are layers 9, 18 and 27. Later layers cannot contribute to those
outputs. The encoder and denoiser introduce completion barriers every 24 graph calls
and sweep unleased buffers to bound queued temporaries without changing arithmetic. Each stage releases
its weights before the next loads. VAE decoding uses 128px tiles, 64px
overlap and cosine blending, matching the reference tile policy.

Admission requires 2.6 GiB of available model budget, including scratch.
Supported output sizes are multiples of 32 between 64 and 512 pixels.
The model is hidden when this budget does not fit. The iOS per-app memory
allowance and 512 MiB reserve remain in force.

Validation on 2026-11-19: binary affine unpacking and GEMM matched an exact
integer oracle twice in BF16 and FP32, including partial row/output tiles;
rectangular tile blending passed coverage/constant-image checks. A native
512×512 four-step generation produced a valid PNG on Mac, with 2,697,775,936
bytes peak physical footprint reported by `time -l` (maximum RSS
3,678,306,304 bytes). This is functional and bounded-memory evidence, not
full numerical parity with PrismML's fused transformer or a speed comparison.

The full-encoder, compact-encoder and compact-encoder/cache-sweep generations
produced byte-identical PNGs at seed 42 for the same 512px prompt
(SHA-256 `6cbcfbb6acf9d159c6cbfd20dc0c1a985c1003917bca769e2f27ff7a309c64ca`).
The 2.6 GiB admission allowance is above the measured peak footprint and is
checked against the live iOS app allowance after the separate 512 MiB reserve.

The existing FLUX.2 Klein 4B path also produced byte-identical 512px seed-42
PNGs before/after this change; traces retained 30,478 dispatches. No existing
FLUX arithmetic or scheduling path was changed. The workspace release build,
574 library tests (32 explicitly ignored) and smoke suite passed. The denoiser
plan test compares bounded and ordinary execution against the retained oracle.

Generated images in the Apple app are written beneath the explicit Application
Support/Lisa/generated-images directory supplied at initialization. The worker
creates this directory before starting inference; it does not derive iOS
output paths from `HOME`. Saved chat paths are relocated when iOS changes the
container UUID during updates. The first physical-device generation completed
the compute stages but exposed the former HOME/Pictures permission error; the
subsequent end-to-end save check is recorded in the desktop validation notes.
