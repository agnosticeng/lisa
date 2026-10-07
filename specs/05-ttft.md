# specs/05 — TTFT / Prefill decomposition and cold-start ports

Baseline (clean tree, goldens green). Model `agnosticeng/Qwen3.8-27B-4bit`,
serial `lisa run --max-tokens 1 --depth 0 --seed 0`, raw prompts, 3 reps, medians.
Instrumentation: one `[ttft] +<s>` stderr milestone per phase (`lisa_engine::ttft_mark`,
anchored at `main`); per-shape warmup and per-chunk prefill marks. Artifacts:
`bench/data/ttft/{baseline_serial,after_serial}.txt`, `/tmp/ttft_prof.json`.

## Phase 1 — where the first-token time goes (process → token 1)

| phase (s, warm) | 1k | 4.1k | 8k | 16k |
|---|---|---|---|---|
| shard read (mmap + host dequant) | 2.1-2.3 | 2.1-2.3 | 2.3 | 2.3 |
| tower build (of which draft-head requant) | 5.9 (5.2) | 5.9-6.3 (5.2-5.7) | 6.3 (5.6) | 6.3 (5.6) |
| tokenizer load + encode | 0.18 | 0.18 | 0.19 | 0.19 |
| warmup shapes 1/9/40 | 0.5 | 0.5 | 0.5 | 0.5 |
| warmup shape 2048 (BEFORE port) | 2.2-2.7 | 2.2-2.7 | 2.7 | 2.7 |
| prefill (chunked, 2048) | 0.6 | 5.5-6.1 | 9.5-10.4 | ~26 |
| first decode step | 0.3-0.7 | 0.05-0.15 | 2.3 | 3.0 |
| **TOTAL process → first token (BEFORE)** | **12.0-12.6** | **17.1-17.5** | **23.8-24.7** | — |

Key findings:
1. **The MTP draft-head requant was 5.2-5.7 s of EVERY process start** (GPU dequant of the
 [248320, 5120] trunk head + host 2-bit requant), cold and warm alike.
2. **warmup s=2048 was 2.2-2.7 s of every serial `generate`**: a throwaway 2048-token
 forward it ran "to pre-empt JIT" — but traced JIT compile is cheap
 (`jit.compile` ≈ 0 ms across 65 compiles/run; `generate.rs`'s own comment says ~0.45 s).
 The warmup's GPU work was ~5× the compile it avoided.
3. **JIT compile is NOT a TTFT cost** (falsified as hypothesis): with the disk-JIT question
 open, `LISA_TRACE=1` shows 65 compiles ≈ 0 traced ms; dropping the s=2048 warmup did
 NOT slow the first real 2048-chunk (1.04 s vs 1.19 s before — no compile penalty).
4. **The "slow tail chunk" is an attribution artifact**: the s=4 chunk's own forward is
 ~33 ms; its 2.2 s wall is `memory::trim_cache` synchronizing the PREVIOUS chunk's
 still-executing GPU work (lazy eval drains at the next sync point — visible as an
 unattributed gap in the LISA_TRACE_JSON timeline).
5. Prefill chunk cost grows with KV (1.0 s @kv0 → 2.7 s @kv2048 → 3.5 s @kv4096 per
 2048-chunk): the 16 full-attention layers' sdpa at growing kL. That is the prefill
 tok/s class already known (668-772 tok/s @16k); nothing new hiding beyond it. At 4.1k
 the measured prefill (5.5-6.1 s ≈ 680-740 tok/s) MATCHES the class rate — the "3-4 s
 missing" at 4.1k was the draft-head rebuild (5.5 s) + warmup (2.7 s) paid per process,
 not prefill inefficiency.
6. Tokenizer: negligible (0.18 s load, <0.01 s encode of 16k).
7. COLD vs WARM: `purge` needs sudo (not available) — page-cache-cold disk reads not
 measurable; the shards-read phase (2.1-2.3 s warm) would absorb the 15.5 GB cold read.
 All other phases are process-cold in every rep (fresh process, in-memory caches empty).

## Phase 2 — ports landed

### Port 1: draft-head disk cache (`draft_head_2bit_gs64.lsdh`)
`Qwen35Tower::load` now passes the checkpoint dir to `build_draft_lm_head_cached`: first
run builds and saves the packed 2-bit/gs64 tensors (atomic tmp+rename, geometry-guarded
magic/gs/bits/V/K header — the trunk head is itself packed, so logical-vs-logical width
comparison); later runs load verbatim (bit-identical bytes → identical tokens).
**load 8.3 s → 2.8 s (−5.5 s) on every process start after the first.**
Traps hit: packed-dim math twice (words/row = K·bits/32; logical K = words·32/bits; the
trunk reference's own width must be unpacked with ITS bits before comparing).

### Port 2: warmup s=2048 dropped (shapes 1/9/40 remain)
`Qwen35Tower::warmup` no longer runs the throwaway 2048-token forward; the first real
prefill chunk pays the ~0.45 s JIT instead. **−2.3 s per serial generate** at every
length; serve startup also gets cheaper (first big-prompt request pays the JIT once).

### TTFT BEFORE → AFTER (process → first token, medians, same session)

| prompt | before | after | Δ |
|---|---|---|---|
| 1k | 12.3 s | **4.5 s** | −63 % |
| 4.1k | 17.2 s | **9.6 s** | −44 % |
| 8k | 24.2 s | **16.8 s** | −31 % |
| 16k | — | 32.3 s | (new) |

Prefill throughput unchanged within thermal drift (4.1k: 647-651 vs 725-733 tok/s across
sessions; per-chunk walls identical 1.0/2.7/3.5). The wins are pure load+warmup removal.
Serve-resident TTFT (warm model, e.g. 4.1k) is prefill-dominated (~6 s); the ports move
the per-PROCESS costs, which is what cold start and every CLI/server restart pay.

### Validation
golden 310/310 ×2, `lisa-bench smoke` green, `lisa-engine` lib tests 22/22 (--release).
RAM: single model resident, peak unchanged (~31.8 GB class); +0.4 GB cache file read.

## Remaining
1. **Prefix-cache resume stall (specs §9)**: `restore_session` restores full KV snapshots
 (O(n) memcpy, ~0.5 GB @8k ≈ 0.1 s — should NOT stall seconds). Suspects beyond the
 copy: a `trim_cache` sync at resume, drafter offset replay, or snapshot capture during
 prefill. Needs a reproduction run before touching.
2. Prefill KV-growth cost (sdpa at growing kL) is the whole remaining TTFT slope at 8-16k;
 the only lever class is the batched/fused attention arm (specs/07 item 2 territory).
3. First-decode step @8-16k pays ~2-3 s: lazy drain of the last prefill chunk + first
 decode JIT at novel kL — partially foldable by evaluating the last chunk before head.
4. Page-cache-cold (true COLD) measurement requires `sudo purge` — not available to the
 agent; numbers here are process-cold.
## Later measured additions

- **Chunked prefill** exists and respects decode; the named decode-share
 credit is absent (partial) — multi-chunk prefill + decode-share + shrink:
 the decode-share part remains unimplemented.
- **`shortest_validated_tail` scheduling — set aside**: its −60 % TTFT comes
 from a *queue* under load; our serve is effectively single-user. Reopen
 only with real concurrency (cf. prefix-homogeneous batching, specs/01 §5).
- **ANE/GPU hybrid prefill — FALSIFIED for this box (M5 Max).** The engine
 uses zero ANE (repo-wide 0 occurrences; an earlier "ANE is in our code"
 note was wrong), but ANE prefill only pays on older silicon (+35 % M1 Pro,
 +32 % M4, +19 % M4 Max, +20 % M3 Ultra at 16k) and gives nothing on
 M5-class Macs, where the GPU is already faster on its own. **Do not port
 it for this box.** Reopen only if a future ANE generation changes the
 balance — measure first, on the same box.
- **TensorOps / NA prefill on M5 — set aside**: uncertain value (the NAX
 `mpp` path is already engaged; prefill at parity with the composed path).
 The one purely-bench item — re-measure the declined `msv_attn_p256` causal
 arm with an interleaved round-cost A/B.
- **Open anchors**: cold (page-cache-cold) TTFT; long-context TTFT ratio.
