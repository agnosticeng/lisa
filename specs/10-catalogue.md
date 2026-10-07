# specs/10-catalogue — Optimization-catalogue dispositions without a topical spec

Thecampaign triaged the full optimization catalogue (our own
earlier prototypes and experiments) item by item: landed, already in place,
falsified, or set aside with a reason. Items whose topic is owned by another
spec live there (serving/admission/law → 00, batching/wave/EV/parity → 01,
long-context → 02/04, draft width/quality → 03, TTFT/prefill → 05,
serial/bandwidth → 07, tree → 08, depth controller → 09). This file keeps
only the items with no topical home. Headline of the sweep: most of the
catalogue was already closed or dead on its premise — the real yield was
robustness, not throughput.

## Set aside (measured or documented reason — do not reopen without new evidence)

- **Confidence calibration**: premise absent — our MTP head is *trained*;
 the pathology that idea treats does not exist here.
- **Layered-logit decoding (dynamic logit-lens variants)**: quality-only,
 opt-in, 0 tok/s.
- **Entropy temperature / power sampling**: would change the sampling
 semantics of the keyed Gumbel-max (00-contracts §4).
- **Second perturbed pass on hard tokens**: compute on hard tokens; the
 top-2 margin is already the signal (specs/09 §6).
- **Rep-penalty "fix"**: no defect — `sampler.rs` is the standard
 implementation, off by default.
- **Self-consistency on idle slots**: product feature, N× compute.
- **Tail replay (re-scheduling the rejected tail)**: invariant —
 `kv_len_i == next_pos[i]` + supersequence/LCP rejection (00-contracts §4).
- **Terminal-state sharing · aging GDN precision · state-delta skip ·
 snapshot compression**: measure-gated — nothing to implement without data
 (needs a per-layer readback instrument).
- **KV int4 · dual-path precision · asymmetric hot window**: no quantized KV
 path exists; major kernel work, out of campaign.
- **HBM↔device split KV cache**: premise broken — no HBM↔PCIe split on
 Apple Silicon (unified memory).
- **Block top-k sparse attention**: needs continue-training; our full
 attention is on the certified path.
- **int8 event-driven sparsity**: measure-gated — count real activation
 zeros first.
- **Draft-guided expert prefetch**: premise broken — unified memory means
 every expert of the shipped 4-bit expert classes is resident.
- **`/metrics` endpoint · tool-parser coverage · suffix decoding**:
 comfort / already covered by the serving surface (00-contracts §2).

## Falsified (do not reopen)

- **Paged KV for decode.**
- **AOT compile**: our JIT pipeline IS the AOT path — compiled once, cached,
 twin-checked (00-contracts §1); a separate AOT artifact buys nothing
 measured.
- **Grafting the 27B's MTP head onto the 9B**: hidden 5120 vs 4096 + no MTP
 weights.
- **Native MTP for the 9B**: needs training.
- **Rollout cache (own idea)**: rejected — frozen outputs, reckless.
- **Keepalive LRU (own idea)**: withdrawn — subsumed by cross-turn resume
 (00-contracts §4).

## Blocked / open

- **9B → 27B oracle A/B**: blocked — needs two models resident, which
 machine safety forbids (00-contracts §3); owner decision.
- **O4 c3 / #30**: see specs/03 and specs/05 respectively (the two items
 with topical homes).
