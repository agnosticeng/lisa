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
