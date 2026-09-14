#!/usr/bin/env bash
# Wait for the GPUs to free, then run Stage M1 + M2 of the Phase 1.6 extended
# evaluation. Nothing here trains anything.
#
#   bash scripts/run_phase1_6_extended_when_free.sh [MIN_FREE_MIB] [DEVICE]
#
# Why a waiter rather than just launching: as of 2026-08-27 all eight H100s were
# held by another user's job (`graph_world_model`, ~49 GiB per card), and this
# evaluation was explicitly deferred until that finished. It polls rather than
# queues because there is no scheduler on this box.
#
# GPU 0-3 are off limits on this machine by standing instruction; the default
# device is cuda:4.

set -euo pipefail

MIN_FREE_MIB="${1:-20000}"
DEVICE="${2:-cuda:4}"
GPU_INDEX="${DEVICE##*:}"
POLL_SECONDS="${POLL_SECONDS:-120}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-172800}"   # 48 h

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/ubuntu/miniforge3/envs/md/bin/python}"
OUT_DIR="$ROOT/runs/phase1_6_extended_metrics_seed0"
LOG="$ROOT/runs/phase1_6_extended_metrics_seed0.log"

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== waiting for GPU $GPU_INDEX to have >= ${MIN_FREE_MIB} MiB free ==="
waited=0
while true; do
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
                 --id="$GPU_INDEX" | tr -d ' ')"
    if [ "$free_mib" -ge "$MIN_FREE_MIB" ]; then
        echo "$(date -Is) GPU $GPU_INDEX has ${free_mib} MiB free — starting."
        break
    fi
    if [ "$waited" -ge "$MAX_WAIT_SECONDS" ]; then
        echo "$(date -Is) gave up after ${waited}s; GPU $GPU_INDEX still has only" \
             "${free_mib} MiB free. Nothing was run."
        exit 75
    fi
    echo "$(date -Is) GPU $GPU_INDEX: ${free_mib} MiB free, waiting ${POLL_SECONDS}s"
    sleep "$POLL_SECONDS"
    waited=$((waited + POLL_SECONDS))
done

echo
echo "=== Stage M0: metric and analysis tests ==="
cd "$ROOT"
"$PYTHON" -m pytest tests/test_extended_metrics.py tests/test_extended_analysis.py \
    tests/test_extended_evaluation.py -q

echo
echo "=== Stage M1: re-score the frozen checkpoints (no training) ==="
"$PYTHON" scripts/evaluate_phase1_6_extended.py \
    --run runs/phase1_6_bounded_seed0 \
    --config configs/phase1_6_bounded.yaml \
    --out "$OUT_DIR" \
    --device "$DEVICE"

echo
echo "=== Stage M2: aggregate and report ==="
"$PYTHON" scripts/analyze_phase1_6_extended.py \
    --records "$OUT_DIR/records.jsonl" \
    --out docs/phase1_6_results_extended.md \
    --plots docs/figures

echo
echo "=== done. No weights were changed. ==="
