#!/usr/bin/env bash
# pi05 继续训练：从原目录最新完整 checkpoint（如 32368）resume
# （恢复 params + AdamW + step；不要只用 weight_loader 重置 optimizer）
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
  pi05_righthand_ee_continue \
  "$@"
