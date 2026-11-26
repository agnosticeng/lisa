# Olala v1.0 mini 7A1B

Experimental native execution of the publisher's Q5_K_M GGUF in the CLI,
server and app.

| Item | Value |
| --- | --- |
| Identity | GGUF `general.architecture = olala` |
| Repository | `OVHaiLLM/Olala-v1.0-mini-7A1B-GGUF`, revision `b9f5d20adf2ac0aec05e9fa00b057cf26df263fc` |
| File | `Olala-v1.0-mini-7A1B-q5_k_m.gguf`, 5,990,551,040 bytes |
| SHA-256 | `ed353d2632c406be9ca2022b35e4c13049d6f8f7390b1c1bb61d6a7d6b971860` |
| Law | `0x4f4c414c41000001` |

Repository resolution downloads only that file and the publisher's notices.
Mixed Q5_K, Q6_K, Q8_0, BF16 and F32 payloads stay packed; embedding rows are
decoded from disk and selected experts are addressed directly on the GPU.

## Model

29 Mamba3-MIMO layers and seven differential tensor-product attention layers,
hidden width 1,536, latent MoE with six of 256 experts. The loader rejects
incompatible geometry, metadata, shapes, encodings and EOS settings. Metadata
advertises 100,000 positions; the publisher describes a 32,768-token window, so
use `--ctx-size 32768` or lower.

## Execution

Prefill groups up to eight rows per layer and shares dense and routed
projection dispatches; each row keeps the serial matrix-vector reduction and
causal and recurrent operations keep their order. Decode is serial. MIMO uses
the pinned serial recurrence; attention uses explicit F32 projections, token
shift, softcapping and full KV history; MoE uses GPU-indexed packed experts,
squared ReLU and an ordered six-term weighted sum, all six experts sharing one
up and one down dispatch.

Dedicated F32 fusions cover routing, squared ReLU, ordered accumulation, paired
token shifts and differential SiLU gating. The router picks six finite
sigmoid-plus-bias scores with ascending expert ID as tie-break; products and
additions stay separate F32 operations (no contraction), and the expert sum is
left-associated. Attention products for two to eight queries use an explicit
F32 8×32 tile schedule with BK=16.

Serial execution submits every four layers; grouped prefill waits at block
boundaries to bound recurrent storage. A step commits its cache only after all
layers succeed. There is no draft head, batching or cache spill.

## Chat and prompt reuse

The embedded template supplies channel framing: the app and CLI show the final
channel; HTTP separates reasoning, final text and tool calls. Thinking maps to
`medium`, disabled thinking to `disabled`. Warmup evaluates two tokens. The app
reuses exact prompt prefixes from its two snapshots, filling in 128-token
chunks and leaving the final token for generation.

## Validation and limits

Fusion checks compare each operation with its standalone chain (every BF16
pattern for ReLU, cancellation and subnormals for sums, tied and clamped
routing, tile boundaries). Checkpoint checks compare serial and grouped
prefill, uneven tails, prefix forks and resumed logits through `lisa-bench`
commands with `--model`. Bitwise parity with the reference's fused operations, long
contexts, sampled generation and tool quality are not certified.
