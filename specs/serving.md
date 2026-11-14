# Serving and sessions

The loaded model selects the server path. Language-model, image-generation
and Laya decision workers expose different endpoints. An endpoint existing
does not imply that every loaded model implements all of its capabilities.

## HTTP surface

| Worker | Routes |
| --- | --- |
| Language model | `GET /health`, `/healthz`, `/v1/models` |
| Language model | `POST /v1/chat/completions`, `/v1/completions` |
| Language model | `POST /v1/messages`, `/v1/messages/count_tokens` |
| Language model | `POST /v1/responses`, `GET /v1/responses/{id}` |
| Language model | `POST /v1/embeddings` |
| Image generation | `GET /health`, `/v1/models`; `POST /v1/images/generations` |
| Laya | `GET /health`, `/healthz`, `/v1/models`; `POST /v1/decisions` |

Route handling and scheduling are in [the server](../crates/lisa-serve/src/lib.rs)
and [image worker](../crates/lisa-serve/src/images.rs).
[HTTP framing](../crates/lisa-serve/src/http.rs),
[response rendering/streams](../crates/lisa-serve/src/responses.rs) and
[per-server response state](../crates/lisa-serve/src/api_state.rs) have separate
modules. The OpenAI, Responses and Anthropic adapters normalize requests into
supported internal jobs. There is no `/metrics` endpoint or universal
compatibility claim for those API families.

The shared HTTP parser limits headers to 64 KiB and bodies to 32 MiB. It rejects
invalid/conflicting content lengths, truncated input and transfer-encoded request
bodies; clients must send `Content-Length`. Language-model requests require a
JSON object. `n` must be an integer from 1 to 16; multiple choices are supported
only by non-streaming Chat/Completions. The server does not advertise synthetic
rate-limit quotas.

Chat supports streaming, seeds, stop strings, logprobs, tools, separate reasoning
and content, and the implemented structured-output subset. `/v1/models` reports
the loaded ID, including after a server restart with a different model. DeepSeek
uses its native DSML formatting/parser. Embeddings
mean-pool and L2-normalize trunk states, with model-dependent dimensions.

Structured output uses instructions, extraction/repair, schema validation and
one corrective retry. It does not apply a token grammar. Do not present this
mechanism as an unconditional schema guarantee. See the response obligations
in [contracts](contracts.md#2-serving-surface).

If repair/retry still fails validation, generation returns an engine error:
HTTP 500 for a collected response, or an error event after streaming headers
have been sent. Invalid content is not emitted as a validated completion.
These requests bypass scheduler waves, whose reply path has no schema validator.
Structured output with `session_id` is rejected with 400; send full message
history for a stateless validated completion.

## Scheduling and speculation

Automatic depth uses a loaded Qwen drafter or GLM target-MTP head. A model
without either falls back to serial. A fixed unsupported depth fails instead
of pretending to speculate. Image turns always use serial generation.

Qwen 3.8 27B can select the external DSpine block drafter with
`LISA_QWEN_DSPINE_MODEL`; it supports compatible MLX targets only and
supports up to fifteen proposals per verify round. HTTP preserves explicit
DSpine depth and lets the engine validate it against the loaded drafter; the
six-token PLD clamp is independent. Automatic depth reads the
matching local SPS profile (seven-row snapshots when unprofiled). DSpine's draft KV is rebuilt from
the current session's committed prefix when the loaded draft prefix does not
match, so interleaved sessions do not share draft context. A prefix rebuild
adds a target prefill pass; the path has not been performance-qualified.

The server may route four or more eligible requests through continuous native
batching. Request features such as stops, tools, logprobs, structured output and persistent sessions
also constrain eligibility. A model that declines batching uses its supported
fallback; both native GGUF towers currently decline native decode batching.

Separate-client speculative service is not a shared GPU speculative forward.
Qwen speculative requests are processed individually; GLM interleaves independent
sessions within its supported waves. The multi-session GLM wave can admit late
requests up to its configured bound; an already-running single-session MTP
request does not thereby become joinable.

Chunked prefill and decode interleave are scheduler policies. Latency gains
from queue admission are separate from single-stream throughput. Preserve
per-stream sampler state, exact retirement/repacking and cache padding limits.

## State and prefix reuse

### Responses history

Each language server owns its response history and model identity. The store
retains at most **256 resources** and **64 MiB of serialized resource/history
JSON**, evicting the oldest entries on insertion. This byte accounting bounds
serialized content, not exact allocator usage. An oversized entry is not
retained. Resources and their continuation messages are stored together.

Background requests expose a queued resource and retain the completed result
with its conversation, or a failed resource with an error. History expires on
eviction or server restart. Fetching an unavailable resource returns 404;
continuing an unavailable, unfinished or failed `previous_response_id` returns
400 rather than silently dropping its context. This API history is separate
from GPU sessions and the prefix cache below.

### Model state

Persistent sessions retain committed history, positions, caches and sampling
configuration. A continuation must feed only the unseen suffix under a compatible
numerical law. Native DeepSeek state supports RAM snapshots with copy-on-write
ownership; its current format has no SSD spill codec.

Message-based reuse compares all message fields, tool definitions and thinking
configuration. Editing history restores only an exact message-boundary snapshot;
if that snapshot is unavailable, the server prefills the complete prompt.
A string suffix must never be paired with an earlier or empty restored state.

The serial stateless path consults the prefix cache; ordinary batch-wave
requests report zero reused prefix tokens. The
[prefix cache](../crates/lisa-engine/src/conversation_cache.rs) currently has:

- A minimum stored prefix of 128 tokens, at complete prompt boundaries.
- Exact token matching and a **strict extension** requirement: an identical
  prompt cannot reuse its final boundary as if a next-step logit were stored.
- A configurable RAM entry-count LRU. There is no RAM byte-budget eviction yet.
- Up to eight generic spill files under `~/.lisa/prefix_spill/`, keyed by
  tokens and numerical law, for serializable state types.
- Header/token-prefix reads before full spill loading; successful RAM hits
  promote the entry, and incompatible numerical laws cannot be restored.

This is not a radix tree or a shared paged-prefix store. Finer prefix boundaries
need state/numerical proof for recurrent hybrid models. Disk serialization
support must be established for each native cache type.

Image requests bypass token-only reuse and reject `session_id`. Clients must
send complete image history; see [media](media.md#image-input).

## Admission and failures

`LISA_RAM_CAP_GB` enables request-level memory admission. A request that cannot
fit its estimated state/work is refused with HTTP 503 before its forward.
This policy is distinct from resident weight/private allocation reservations.
The default computed budget is detected RAM minus 8 GiB; never hardcode a
particular Mac's capacity.

Reply-channel failures do not consistently propagate cancellation into active
inference. Immediate
release of compute after disconnect is an open requirement. See
[roadmap](roadmap.md#shared-engine-and-serving).

## Evidence boundaries

Endpoint availability and compilation do not certify every model/mode. Changes
to HTTP/state handling require live stream/non-stream, seed, stop, tool, schema,
continuation, admission and failure checks. Model numerical/state acceptance is
separate from API conformance.
