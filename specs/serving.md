# Serving

`lisa serve` exposes the loaded model over HTTP. The model kind selects the
worker; an endpoint existing does not mean every model implements every
capability.

## Routes

| Worker | Routes |
| --- | --- |
| Language model | `GET /health`, `/healthz`, `/v1/models` |
| | `POST /v1/chat/completions`, `/v1/completions` |
| | `POST /v1/messages`, `/v1/messages/count_tokens` |
| | `POST /v1/responses`, `GET /v1/responses/{id}` |
| | `POST /v1/embeddings` |
| Image generation | `GET /health`, `/v1/models`; `POST /v1/images/generations` |
| Laya | `GET /health`, `/healthz`, `/v1/models`; `POST /v1/decisions` |

Modules: [server](../crates/lisa-serve/src/server.rs),
[HTTP framing](../crates/lisa-serve/src/http.rs),
[responses and streams](../crates/lisa-serve/src/responses.rs),
[per-server response state](../crates/lisa-serve/src/api_state.rs),
[image worker](../crates/lisa-serve/src/images.rs). Chat, Responses and
Messages adapters normalize requests into internal jobs. There is no
`/metrics` endpoint and no blanket compatibility claim.

The parser limits headers to 64 KiB and bodies to 32 MiB, reads a body only
through `Content-Length` (absent means empty), and rejects conflicting
lengths, truncation and transfer-encoded bodies. `n` is 1–16 and only non-streaming Chat/Completions
return several choices.

## Response semantics

- **Determinism.** Identical requests and seeds (including `seed: 0`) reproduce
  under the same law. Streamed and collected responses agree in content,
  finish reason and usage.
- **Stops.** Stop strings never leak through a partial streamed suffix.
  `ignore_eos` applies to generation and to the serving callback.
- **Identity.** `/v1/models` and payloads name the loaded checkpoint.
- **Channels.** Reasoning, content and tool calls stay separate. Tool calls
  carry valid arguments and IDs; tool-result turns keep the template's
  semantics. DeepSeek uses its native DSML format and parser.
- **Structured output.** The JSON-schema subset is enforced by instruction,
  extraction, repair, validation and one corrective retry, not by a token
  grammar. Exhausted retries return HTTP 500, or an error event on an open
  stream. These requests bypass waves and reject `session_id`.
- **Embeddings.** Mean-pooled, L2-normalized trunk states; the dimension is
  model dependent (5,120 for the 27B model).
- **Sampling.** Greedy sampling still applies the repetition penalty to each
  distinct ID in the history window. Raw argmax replaces the sampler only for
  greedy with penalty 1. Batched selection keeps each stream's configuration.

## Speculation and scheduling

Automatic depth uses the loaded MTP head (independent, target-conditioned or
block); a model without one runs serial. A fixed unsupported depth fails.
Image turns always run serial.

Four or more eligible requests may run through continuous batching. Stops,
tools, logprobs, structured output and persistent sessions constrain
eligibility. All GGUF towers and LFM3 decline batched decode;
Qwen MLX, Flash-Next and GLM MLX batch. Separate-client
speculation is not a shared speculative forward: Qwen speculative requests run
individually, and GLM interleaves independent sessions within its waves.

## State

**Response history.** Each server keeps at most 256 resources and 64 MiB of
serialized resource JSON, evicting the oldest. Background requests expose a
queued resource and later the result or an error. An unavailable resource
returns 404; continuing from an unavailable, unfinished or failed
`previous_response_id` returns 400.

**Model state.** Persistent sessions keep history, positions, caches and
sampling configuration; continuations feed only the unseen suffix under a
compatible law. Message-based reuse compares every message field, tool
definition and thinking setting; editing history restores only an exact
message-boundary snapshot or prefills the whole prompt. The serial stateless
path consults the [prefix cache](runtime.md#sessions-and-prefix-reuse); batch
waves report zero reused tokens. Image requests bypass token reuse and reject
`session_id`.

## Admission

`--ram-cap-gb` enables request admission. A request whose estimated round
does not fit is refused with HTTP 503 and the budget before its forward; the
server stays usable. This is separate from resident weight reservations.

Disconnects do not yet cancel active inference
([roadmap](roadmap.md#engine-and-serving)).
