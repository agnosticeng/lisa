# Media

Image input, voice and text-to-image generation. All inference runs in the
native engine; reference scripts are validation tools only. Image pipelines
have their own tolerances and do not relax text-model strict paths.

## Image input

| Text model | Vision |
| --- | --- |
| Qwen dense (MLX) | Dense ViT tower, merger and multimodal RoPE |
| Qwen Flash-Next | Integrated; full-checkpoint qualification open |
| GLM-5.3 MLX | Tower with 2D RoPE, downsample and merger; numerical parity open |
| Inkling 2 | GGUF still-image encoder ([details](models/inkling2.md#images)) |
| GLM / DeepSeek / Olala GGUF, LFM3 | Unsupported |

Vision weights ship with the language weights. Quantized vision towers fail
explicitly. Image positions survive prefill chunk boundaries.

**Decoding.** PNG, JPEG and WebP; EXIF orientation applied; alpha composited
on white; fixed-point bicubic resizing that matches the reference
preprocessor. Qwen bounds resized input to 1,003,520 pixels, GLM to 1,024
merged tokens. Every request is limited to eight images, 20 MiB encoded per
image, 32 megapixels decoded in total and the 32 MiB HTTP body.

**Surfaces.** CLI `run --image`, app Attach, Chat image blocks, Responses
`input_image` and Messages base64 sources share one path. HTTP accepts inline
base64 only.

**State.** Each image turn rebuilds a serial session from the full multimodal
history: speculation, batching and token-only prefix reuse are bypassed, and
`session_id` is rejected. Image embeddings are cleared on success and error.

## Voice (macOS)

The composer uses on-device speech recognition for dictation and speech
synthesis for "Read replies". Dictation fills the composer for review; audio is
never stored. If on-device recognition is unavailable for the system language,
voice input reports that it cannot start rather than using a network service.

## Image generation

| Checkpoint | Weights | Steps | Encoder / transformer |
| --- | --- | ---: | --- |
| `mlx-community/flux2-klein-4b-4bit` | 4-bit, group 64 | 4 | 4B text encoder; 5 double + 20 single blocks |
| `mlx-community/flux2-klein-9b-4bit` | 4-bit, group 64 | 4 | 8B text encoder; 8 double + 24 single blocks |
| `black-forest-labs/flux-3-image-klein` | BF16 | 16 | 4B vision-language text encoder; 5 double + 28 joint blocks |
| `prism-ml/bonsai-image-binary-4B-mlx-1bit` | 1-bit, group 128 | 4 | FLUX.2 family with binary transformer |

Weight adapters ([FLUX.2](../crates/lisa-engine/src/weights/flux2.rs),
[FLUX 3](../crates/lisa-engine/src/weights/flux3.rs)) load encoder, denoiser and
VAE as sequential stages, each released before the next. Selecting an image
model releases the chat model. Admission covers stage weights, layout copies
and an 8 GiB scratch margin (2.6 GiB total for Bonsai).

**Surfaces.** CLI `image`, app Generate, `POST /v1/images/generations`. One PNG
per request, default 512×512, sides 64–1024 in multiples of 32 (Bonsai up to
512). Prompts are limited to 16,384 bytes. Sizes are validated by the model
plan ([`flux/pipeline.rs`](../crates/lisa-models/src/flux/pipeline.rs)).

**Determinism.** SplitMix64/Box–Muller noise makes seeds reproducible within
LISA only. Attention uses bounded F32 score/softmax/PV work and the VAE uses
bounded im2col; neither is certified bit-exact to fused reference kernels.

**FLUX 3.** The 20,320,953,966-byte BF16 download excludes duplicate weight
formats and needs gated access. The conditioning encoder concatenates layers
4–32 in steps of four, padded towards a multiple of 80 and capped at 4,096
tokens. The distilled schedule uses alpha 16 without CFG. Noise is
BF16-rounded, latents F32, and velocity×dt rounds to BF16 before the update.
Short prompts are expanded locally by the bundled encoder (8,192 input, 512
caption tokens); `--prompt-is-caption` or `prompt_is_caption: true` skips it.

**Bonsai.** Binary transformer payloads stay packed and are decoded on the fly
by the portable GEMM path. Only the first 27 encoder layers run, since the
requested hidden states are layers 9, 18 and 27. Stages insert completion
barriers every 24 graph calls to bound temporaries. The VAE decodes 128-pixel
tiles with 64-pixel overlap and cosine blending.

## Validation

Each stage is checked on independent fixed inputs so errors do not compound:
relative RMSE ≤ 0.03 for BF16 stages and ≤ 0.001 for the F32 decoder. Binary
affine kernels match the integer oracle exactly. Stage plans are checked against
their original operation chains by self-contained tests; checkpoint-level
image quality is not certified.
