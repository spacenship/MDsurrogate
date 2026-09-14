#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

# The esm3 environment contains the installed ESM-C backend. With GPUs 6 and 7
# visible, LOCAL_RANK 0/1 maps to physical GPU 6/7. Opt in to DDP with
# NPROC_PER_NODE=2; preserve single-GPU behavior for existing commands.
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/miniforge3/envs/esm3/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export PYTHONUNBUFFERED=1
# Keep allocator fragmentation from recreating the old peak-memory failure.
# This is complementary to the model-side edge chunking/checkpointing.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
if [[ "$NPROC_PER_NODE" != 1 && "$NPROC_PER_NODE" != 2 ]]; then
  echo "NPROC_PER_NODE must be 1 or 2" >&2
  exit 2
fi
IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
for GPU_ID in "${GPU_IDS[@]}"; do
  if [[ "$GPU_ID" != 6 && "$GPU_ID" != 7 ]]; then
    echo "This launcher is restricted to physical GPUs 6 and 7" >&2
    exit 2
  fi
done
if (( ${#GPU_IDS[@]} < NPROC_PER_NODE )) || [[ "$NPROC_PER_NODE" == 2 && "${GPU_IDS[0]}" == "${GPU_IDS[1]}" ]]; then
  echo "DDP needs two distinct visible GPUs" >&2
  exit 2
fi
LAUNCH=("$PYTHON_BIN")
DEFAULT_OUTPUT=outputs/heavy_flow/stage3_v2/chunk_rotation_latest.pt
if [[ "$NPROC_PER_NODE" == 2 ]]; then
  LAUNCH+=(-m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2)
  DEFAULT_OUTPUT=outputs/heavy_flow/stage3_v2/chunk_rotation_ddp_latest.pt
fi

exec "${LAUNCH[@]}" experiments/heavy_flow/train_physics_chunks.py \
  --config "${CONFIG:-configs/heavy_flow/stage3.yaml}" \
  --data-dir "${DATA_DIR:-data}" \
  --esm2-cache-dir "${ESM2_CACHE_DIR:-esm2_cache}" \
  --quarantine-path "${QUARANTINE_PATH:-mdcath_force_quarantine.json}" \
  --coord-quarantine-path "${COORD_QUARANTINE_PATH:-mdcath_coord_quarantine.json}" \
  --chunk-size "${CHUNK_SIZE:-500}" \
  --output "${OUTPUT:-$DEFAULT_OUTPUT}" \
  --normalizer-path "${NORMALIZER_PATH:-outputs/heavy_flow/stage3_v2/normalizer.json}" \
  --reuse-normalizer "${REUSE_NORMALIZER_PATH:-outputs/heavy_flow/stage3/chunk_rotation_normalizer.json}" \
  --device "${DEVICE:-cuda:0}" \
  --prefetch "${PREFETCH:-1}" \
  --checkpoint-every "${CHECKPOINT_EVERY:-500}" \
  --checkpoint-seconds "${CHECKPOINT_SECONDS:-0}" \
  "$@"
