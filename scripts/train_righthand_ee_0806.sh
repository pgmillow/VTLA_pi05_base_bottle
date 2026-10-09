#!/usr/bin/env bash
# Resume pi05_righthand_ee from step 24276 into a new exp: pi05_righthand_ee_0806.
# Copies the full checkpoint (params + train_state/AdamW + step) once, then resumes.
set -euo pipefail

ROOT="/home/xudi_ge/duibi/openpi"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/home/xudi_ge/.cache/huggingface/lerobot}"

DATA_DIR="${DATA_DIR:-/data/V1show_lerobot/merged}"
SRC_CKPT="${SRC_CKPT:-/data/final_double_hands_openpi/pi05_righthand_ee/pi05_righthand_ee/24276}"
DST_DIR="${DST_DIR:-/data/final_double_hands_openpi/pi05_righthand_ee_0806/pi05_righthand_ee_0806}"

mkdir -p "${HF_LEROBOT_HOME}/local"
ln -sfn "${DATA_DIR}" "${HF_LEROBOT_HOME}/local/V1show_merged"

mkdir -p "${DST_DIR}"
# Bootstrap 24276 only if this exp dir has no numeric checkpoints yet.
shopt -s nullglob
existing_steps=( "${DST_DIR}"/[0-9]* )
shopt -u nullglob
latest_step=""
if ((${#existing_steps[@]} > 0)); then
  latest_step=$(basename "$(printf '%s\n' "${existing_steps[@]}" | sort -n | tail -1)")
  echo "[0806] Found checkpoint(s) in ${DST_DIR}; latest=${latest_step} (resume will load this)."
elif [[ ! -d "${DST_DIR}/24276" ]]; then
  if [[ ! -d "${SRC_CKPT}" ]]; then
    echo "ERROR: source checkpoint not found: ${SRC_CKPT}" >&2
    exit 1
  fi
  echo "[0806] No checkpoints yet; copying 24276 -> ${DST_DIR}/24276 (one-time)..."
  cp --reflink=auto -a "${SRC_CKPT}" "${DST_DIR}/24276"
  latest_step=24276
  echo "[0806] Copy done."
fi

echo "[0806] Starting pi05_righthand_ee_0806 (resume=True, expect restore step=${latest_step:-latest} -> 46000)"
exec env \
  PYTHONPATH="$ROOT/src" \
  HF_LEROBOT_HOME="$HF_LEROBOT_HOME" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
  NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
  XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}" \
  /home/xudi_ge/openpi/.venv/bin/python \
  "$ROOT/scripts/train.py" \
  pi05_righthand_ee_0806 \
  "$@"
