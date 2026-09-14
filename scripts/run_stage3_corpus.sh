#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
exec "${PYTHON_BIN:-/home/ubuntu/miniforge3/envs/esm3/bin/python}" scripts/run_stage3_corpus.py "$@"
