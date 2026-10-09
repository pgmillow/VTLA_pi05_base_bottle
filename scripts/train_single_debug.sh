#!/usr/bin/env bash
# Debug: resume from 27000, train on a single data sample for 700 steps.
# Copies ckpt 27000 from pi05_righthand_ee_0806 experiment, creates a 1-episode
# dataset from V1show_merged, then launches pi05_righthand_ee_single_debug.
set -euo pipefail

ROOT="/home/xudi_ge/duibi/openpi"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/home/xudi_ge/.cache/huggingface/lerobot}"

# ── 1. Prepare single-sample dataset ──────────────────────────────────────────
FULL_DATA="/data/V1show_lerobot/merged"
MINI_DATA="/data/V1show_lerobot/debug_single"
MINI_LINK="${HF_LEROBOT_HOME}/local/V1show_debug_single"

if [[ ! -d "${MINI_DATA}" ]]; then
  echo "[debug] Creating single-sample dataset from episode 0..."
  mkdir -p "${MINI_DATA}/data/chunk-000" "${MINI_DATA}/meta"

  # Copy only the first episode's parquet.
  cp --reflink=auto "${FULL_DATA}/data/chunk-000/episode_000000.parquet" \
     "${MINI_DATA}/data/chunk-000/episode_000000.parquet"

  # Copy full tasks.jsonl and info.json (info will show total_episodes=1 after fixup).
  cp "${FULL_DATA}/meta/tasks.jsonl" "${MINI_DATA}/meta/tasks.jsonl"

  # Write single-episode episodes.jsonl.
  head -1 "${FULL_DATA}/meta/episodes.jsonl" > "${MINI_DATA}/meta/episodes.jsonl"

  # Write single-episode episode_sources.jsonl.
  if [[ -f "${FULL_DATA}/meta/episode_sources.jsonl" ]]; then
    head -1 "${FULL_DATA}/meta/episode_sources.jsonl" > "${MINI_DATA}/meta/episode_sources.jsonl"
  fi

  # Patch info.json: total_episodes=1, total_frames=906 (from episode 0).
  python3 -c "
import json
with open('${FULL_DATA}/meta/info.json') as f:
    info = json.load(f)
info['total_episodes'] = 1
info['total_frames'] = 906
info['splits'] = {'train': '0:1'}
with open('${MINI_DATA}/meta/info.json', 'w') as f:
    json.dump(info, f, indent=2)
"
  echo "[debug] Single-sample dataset created at ${MINI_DATA}"
else
  echo "[debug] Single-sample dataset already exists at ${MINI_DATA}"
fi

# Symlink so HF_LEROBOT_HOME can find it.
mkdir -p "$(dirname "${MINI_LINK}")"
ln -sfn "${MINI_DATA}" "${MINI_LINK}"

# ── 2. Copy checkpoint 27000 ─────────────────────────────────────────────────
SRC_CKPT="/data/final_double_hands_openpi/pi05_righthand_ee_0806/pi05_righthand_ee_0806/27000"
DST_DIR="/data/final_double_hands_openpi/pi05_righthand_ee_single_debug/pi05_righthand_ee_single_debug"

mkdir -p "${DST_DIR}"

# Only copy 27000 if no numeric checkpoints exist yet in the dest dir.
shopt -s nullglob
existing_steps=( "${DST_DIR}"/[0-9]* )
shopt -u nullglob
if ((${#existing_steps[@]} > 0)); then
  latest_step=$(basename "$(printf '%s\n' "${existing_steps[@]}" | sort -n | tail -1)")
  echo "[debug] Found existing checkpoint(s) in ${DST_DIR}; latest=${latest_step} (resume will load this)."
elif [[ ! -d "${DST_DIR}/27000" ]]; then
  if [[ ! -d "${SRC_CKPT}" ]]; then
    echo "ERROR: source checkpoint not found: ${SRC_CKPT}" >&2
    exit 1
  fi
  echo "[debug] No checkpoints yet; copying 27000 -> ${DST_DIR}/27000 (one-time)..."
  cp --reflink=auto -a "${SRC_CKPT}" "${DST_DIR}/27000"
  echo "[debug] Copy done."
fi

# ── 3. Launch training ────────────────────────────────────────────────────────
echo "[debug] Starting pi05_righthand_ee_single_debug (resume=True, from step ~27000, 700 steps)"
exec env \
  PYTHONPATH="$ROOT/src" \
  HF_LEROBOT_HOME="$HF_LEROBOT_HOME" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
  NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" \
  XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}" \
  /home/xudi_ge/openpi/.venv/bin/python \
  "$ROOT/scripts/train.py" \
  pi05_righthand_ee_single_debug \
  "$@"
