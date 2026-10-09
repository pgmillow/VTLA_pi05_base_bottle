#!/usr/bin/env bash
# pi05 JAX fine-tune on /data/0727fulldata (action 14 / state 26)
set -euo pipefail

ROOT="/home/xudi_ge/duibi/openpi"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/home/xudi_ge/.cache/huggingface/lerobot}"

mkdir -p "${HF_LEROBOT_HOME}/local"
ln -sfn /data/0727fulldata "${HF_LEROBOT_HOME}/local/0727fulldata"

exec env \
  PYTHONPATH="$ROOT/src" \
  HF_LEROBOT_HOME="$HF_LEROBOT_HOME" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
  NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
  XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}" \
  /home/xudi_ge/openpi/.venv/bin/python \
  "$ROOT/scripts/train.py" \
  pi05_0727fulldata \
  "$@"
