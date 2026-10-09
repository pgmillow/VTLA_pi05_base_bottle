#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/fastslow_env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
exec "$PYTHON_BIN" scripts/train.py "${FASTSLOW_CONFIG:-pi05_righthand_ee_fastslow}" "$@"
