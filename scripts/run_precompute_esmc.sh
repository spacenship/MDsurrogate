#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/miniforge3/envs/esm3/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-6,7}"
export PYTHONUNBUFFERED=1

exec "$PYTHON_BIN" scripts/precompute_esmc.py \
  --config "${CONFIG:-configs/heavy_flow/stage3.yaml}" \
  --data-dir "${DATA_DIR:-data}" \
  --cache-dir "${CACHE_DIR:-outputs/heavy_flow/esmc_cache}" \
  --device "${DEVICE:-cuda:0}" \
  "$@"
