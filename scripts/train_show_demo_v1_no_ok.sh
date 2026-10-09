#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/xudi_ge/duibi/openpi"

exec env \
  PYTHONPATH="$ROOT/src" \
  HF_LEROBOT_HOME="/home/xudi_ge/.cache/huggingface/lerobot" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
  NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
  XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}" \
  /home/xudi_ge/openpi/.venv/bin/python \
  "$ROOT/scripts/train.py" \
  pi05_show_demo_v1_no_ok \
  "$@"
