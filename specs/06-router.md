# 06 — Prompt-lookup / MTP routing

Historical Qwen campaign at base `69d2f60`. Measurements below describe that
source and machine, not a current guarantee for every model. Current numerical
and safety requirements are in [00](00-contracts.md); GLM's separate MTP path
is described in [11](11-glm53-flash.md).

## Existing mechanisms and routing boundary

The campaign found no automatic `ngram-score 0.010` gate. Plain PLD was an
explicit greedy opt-in. The Qwen MTP loop already combined copy proposals with
its draft chain within each round: copy hits supply covered positions and the
chain fills the remainder. A full-copy round can skip chain construction while
preserving the draft-head restart.

Both mechanisms use one loaded model. PLD adds no weights; MTP owns the draft
head. Switching proposal sources inside the MTP loop was exercised by its auto
runs. Switching a persistent session from plain PLD to MTP was not admitted:
the plain path did not populate all token/multi-state history needed to prime
the MTP head. A future transition requires explicit history/state proof; the
campaign's request-level router did not establish it.

## Baseline decision table

Model: `agnosticeng/Qwen3.8-27B-4bit`; about 2K KV, 128 output tokens, greedy,
forced-length decoding. Runs were sequential, one model resident, with memory
preflight and two interleaved repetitions in alternating order. Rates are
median tokens/s.

| Content | Serial | Plain PLD | MTP auto |
| --- | ---: | ---: | ---: |
| Echo | 22.6 | 33.5 | 17.3 |
| Prose | 22.9 | 22.8 | 21.6 |
| Mixed | 23.5 | 24.8 | 20.8 |

A PLD miss falls back to a serial step, and every proposal is verified. This
explains its small overhead in these cases; it is not a universal proof that
PLD can never regress. MTP auto lost on echo here because full-copy rounds
still paid head restart while those rounds were excluded from EV pricing.
Long-context prose had a different balance; [03](03-draft-long.md) records
41.6 versus 33.6 tokens/s at 12.5K after the controller fix.

## Implemented router

`echo_score` in `core/copy_draft.rs` measures the fraction of prompt 4-gram
positions that occurred earlier. Actual subword repeats gave echo/prose/mixed
scores of 0.857/0.707/0.594, so the score separated content less cleanly than
synthetic tests suggested.

The CLI router selected PLD when the score met the configured threshold
(default 0.80) or the prompt was shorter than 8,192 tokens. Otherwise it used
the selected depth policy. The trace reported the score and selected engine.
The threshold therefore governed the long-context branch; the short-context
branch selected PLD independently.

## Interleaved before/after result

Three alternating pairs compared `--depth auto` with `--router` in the same
measurement session. Rates are median tokens/s.

| Content | Auto baseline | Router | Observed change |
| --- | ---: | ---: | --- |
| Echo | 19.5 | 34.4 | +76%; positive in all three pairs |
| Prose | 16.7 | 18.7 | +12%; positive in all three pairs |
| Mixed | 22.9 | 21.7 | -5% raw; pair deltas +0.1/-1.4/+1.7 within session drift |

Router and forced PLD emitted identical text on the tested echo and mixed
cases. Baseline prose drifted from 21.6 in the decision table to 16.7 in the
later pairs; cross-session rates were not used as causal evidence.

Validation at that revision: Qwen35 golden 310/310 twice and 24 engine tests,
including two score tests. Flash-Next could not run because the local checkpoint
lacked `ngram.safetensors`; it was not certified by this campaign. Minimum
observed free memory was about 66 GB; the run obeyed its detected-RAM budget.

## Unresolved by this campaign

- Per-request serving integration with persistent-session history preserved.
- A dedicated long-context interleaved comparison before changing the threshold.
- Safe plain-PLD-to-MTP transitions with complete multi-state history.

Recheck current implementation and reproduce the relevant workload before
reopening any item; historical open status is not proof that it is still absent.
