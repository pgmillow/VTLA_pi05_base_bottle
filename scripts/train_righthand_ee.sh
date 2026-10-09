#!/usr/bin/env bash
# pi05 JAX fine-tune: right-arm/hand loss + right state (arm/hand/EE) in prompt.
# Data: /data/V1show_lerobot/merged (517944 frames, 583 eps)
# Schedule: 46000 steps (~5.68 epochs; ~Thu night from Mon 03:00)
set -euo pipefail

ROOT="/home/xudi_ge/duibi/openpi"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/home/xudi_ge/.cache/huggingface/lerobot}"

DATA_DIR="${DATA_DIR:-/data/V1show_lerobot/merged}"

mkdir -p "${HF_LEROBOT_HOME}/local"
ln -sfn "${DATA_DIR}" "${HF_LEROBOT_HOME}/local/V1show_merged"

exec env \
  PYTHONPATH="$ROOT/src" \
  HF_LEROBOT_HOME="$HF_LEROBOT_HOME" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
  NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
  XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}" \
  /home/xudi_ge/openpi/.venv/bin/python \
  "$ROOT/scripts/train.py" \
  pi05_righthand_ee \
  "$@"
