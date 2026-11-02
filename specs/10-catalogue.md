# specs/10-catalogue — Optimization-catalogue dispositions without a topical spec

Historical Qwen campaign; current requirements are in [00](00-contracts.md).
The campaign triaged the full optimization catalogue (our own
earlier prototypes and experiments) item by item: landed, already in place,
falsified, or set aside with a reason. Items whose topic is owned by another
spec live there (serving/admission/law → 00, batching/wave/EV/parity → 01,
long-context → 02/04, draft width/quality → 03, TTFT/prefill → 05,
serial/bandwidth → 07, tree → 08, depth controller → 09). This file keeps
only the items with no topical home. Headline of the sweep: most of the
catalogue was already closed or dead on its premise — the real yield was
robustness, not throughput.

## Set aside (measured or documented reason — do not reopen without new evidence)

- **Confidence calibration**: set aside because this campaign did not establish
 a calibration defect in the trained MTP head. Training alone is not proof of
 calibration; reopen only with a measured error and a concrete target.
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
- **Replacing the 27B full-attention pattern with block top-k sparsity**:
 requires model-quality validation beyond a faithful kernel port. Existing
 Qwen4 and GLM sparse attention are different checkpoint contracts.
- **int8 event-driven sparsity**: measure-gated — count real activation
 zeros first.
- **Draft-guided expert prefetch**: not needed for the resident expert paths measured in this campaign. Unified
 memory alone does not guarantee residency; GLM has a separate explicit owner.
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

- **Joint 9B → 27B oracle experiment**: not run under the one-resident-model
 constraint. Ordinary comparisons can run sequentially; a joint experiment
 needs an explicit design that also respects the memory contract.
- **O4 c3 / #30**: see specs/03 and specs/05 respectively (the two items
 with topical homes).
