#!/usr/bin/env bash
# Phase 1.6 Stage C: 3-seed confirmation, one seed per GPU.
#
#   bash scripts/launch_phase1_6_full.sh                 # confirmation arm set
#   bash scripts/launch_phase1_6_full.sh P1_pair_physics_frozen   # override arms
#
# GPU policy on this machine (2026-08-25): **4,5,6,7 only**, for every job.
# One seed per GPU, three seeds, so 7 is left free.
#
# This script is not run by anything. Stage C starts only when the screening
# result justifies it and the user asks for it; the plan says so and the runner
# will not start a long run on its own.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${CONFIG:-configs/phase1_6_full.yaml}
# Which arms Stage C confirms: the two controls, the node-physics reference, the
# best pair arm from screening, and the oracle. Pass arms as arguments to change
# it -- but change it deliberately, because an arm added here was not part of the
# screening that justified the run.
ARMS=("$@")
if [ ${#ARMS[@]} -eq 0 ]; then
  ARMS=(S0_structure_history S2_node_physics_152d P0_pair_geometry_control \
        P2_pair_physics_moments O_current_gt_force_oracle)
fi

echo "config : $CONFIG"
echo "arms   : ${ARMS[*]}"
echo "seeds  : 0 1 2  -> GPUs 4 5 6"
echo

for seed in 0 1 2; do
  gpu=$((4 + seed))
  out="runs/phase1_6_full_seed${seed}"
  mkdir -p "$out"
  CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8 \
    nohup python scripts/run_phase1_6_ablation.py \
      --config "$CONFIG" --seed "$seed" --out-dir "$out" \
      --arms "${ARMS[@]}" \
      > "$out/train.log" 2>&1 &
  echo "seed $seed -> GPU $gpu, pid $!, log $out/train.log"
done
wait

echo
echo "all seeds finished. Analyse with:"
echo "  python scripts/analyze_phase1_6.py --runs runs/phase1_6_full_seed0 \\"
echo "      runs/phase1_6_full_seed1 runs/phase1_6_full_seed2 \\"
echo "      --out docs/phase1_6_report.md"
