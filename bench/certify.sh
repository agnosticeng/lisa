#!/bin/bash
# THE bench entry point. One script, one result file — no other harness belongs
# in bench/.
#
# Everything here goes through `lisa-bench`, the in-repo instrument: `golden` for
# correctness (bit-exact goldens) and `serial-audit` for the decode step, which
# is the number perf work is judged on. `round-cost` is the interleaved arbiter
# when you need an A/B; drive it by hand rather than growing this script.
#
# Machine safety (HARD CAP 120 GB on this box): models run ONE AT A TIME and the
# loop never overlaps them. Add a model id to run it too — but not two at once.
#
# Usage:
#   bench/certify.sh                       # the 27B
#   bench/certify.sh <hf-id> [<hf-id> ...] # each in turn, sequentially
#
# Appends one record per model to bench/results.json.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
BENCH="$ROOT/target/release/lisa-bench"
RESULTS="$ROOT/bench/results.json"
KV=1024
STEPS=24

[ -x "$BENCH" ] || { echo "build first: cargo build --release --bin lisa-bench" >&2; exit 1; }

MODELS=("$@")
[ ${#MODELS[@]} -eq 0 ] && MODELS=("agnosticeng/Qwen3.8-27B-4bit")

pkill -9 -x lisa 2>/dev/null
sleep 2
[ -f "$RESULTS" ] || echo "[]" > "$RESULTS"

for M in "${MODELS[@]}"; do
  echo "===== $M ====="
  memory_pressure | head -1

  GOLD="$("$BENCH" golden --model "$M" 2>&1)"
  echo "  goldens : $(echo "$GOLD" | tail -1)"

  STEP="$("$BENCH" serial-audit --model "$M" --kv $KV --steps $STEPS 2>&1 \
          | grep -oE 'wall ms/step: median [0-9.]+' | head -1)"
  echo "  step    : $STEP"

  python3 - "$RESULTS" "$M" "$GOLD" "$STEP" "$KV" <<'PY'
import json, sys, re, datetime
path, model, gold, step, kv = sys.argv[1:6]
rec = {
    "ts": datetime.datetime.now().isoformat(timespec="seconds"),
    "model": model,
    # qwen3_5 paths print a "TOTAL: x/y ... OK" line; the generic path prints a
    # "matches m/n" counter followed by a rate line — parse both, across the
    # whole output, since the rate line may come last.
    "goldens": (m := re.search(r"TOTAL:\s*(\d+/\d+)", gold)) and m.group(1)
    or ((m := re.search(r"matches\s+(\d+)/(\d+)", gold))
        and f"{m.group(1)}/{m.group(2)}" or gold.strip().splitlines()[-1]),
    "goldens_ok": "OK" in gold
    or bool((m := re.search(r"matches\s+(\d+)/(\d+)", gold))
            and m.group(1) == m.group(2)),
    "serial_step_ms": float(re.search(r"median ([0-9.]+)", step).group(1)) if re.search(r"median ([0-9.]+)", step) else None,
    "kv": int(kv),
}
rows = json.load(open(path))
rows.append(rec)
json.dump(rows, open(path, "w"), indent=2)
print(f"  -> bench/results.json ({len(rows)} records)")
PY
  sleep 5
done

pkill -9 -x lisa 2>/dev/null
echo "===== done ====="
