#!/usr/bin/env bash
# Train pi05_righthand_ee_tactile from pi05_base.
# RGB/LLM weights load from pi05_base; tactile encoder + Linear(256->2048)
# + scalar tanh(gate) are randomly initialized (gate starts at 0).
set -euo pipefail

ROOT="/home/xudi_ge/Ttest_V1/openpi"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/home/xudi_ge/.cache/huggingface/lerobot}"

DATA_DIR="${DATA_DIR:-/data/V1show_lerobot/merged_tactile}"

mkdir -p "${HF_LEROBOT_HOME}/local"
ln -sfn "${DATA_DIR}" "${HF_LEROBOT_HOME}/local/V1show_merged_tactile"

exec env \
  PYTHONPATH="$ROOT/src" \
  HF_LEROBOT_HOME="$HF_LEROBOT_HOME" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
  NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
  XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}" \
  /home/xudi_ge/openpi/.venv/bin/python \
  "$ROOT/scripts/train.py" \
  pi05_righthand_ee_tactile \
  "$@"
