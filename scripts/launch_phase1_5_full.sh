#!/usr/bin/env bash
# Phase 1.5 full experiment: 3 seeds x 5 arms, one seed per GPU.
#
# GPU policy on this machine: long training uses 4-7 only, never 0-3.
# Each seed is one process so the ablation runner's fairness assertion (same
# manifest, same Phase 1 checkpoint, same seed, same step budget across arms)
# covers every arm it compares.
set -u
cd "$(dirname "$0")/.."

for seed in 0 1 2; do
  gpu=$((4 + seed))
  out="runs/phase1_5_full_seed${seed}"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8 \
    nohup python scripts/run_phase1_5_ablation.py \
      --config configs/phase1_5_full.yaml --seed "$seed" \
      > "$out/train.log" 2>&1 &
  echo "seed $seed -> GPU $gpu, pid $!, log $out/train.log"
done
wait
