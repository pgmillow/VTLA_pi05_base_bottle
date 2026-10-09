#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
case "${1:-}" in
  visual) CONFIG=vtla_bottle_stage1_visual ;;
  tactile) CONFIG=vtla_bottle_stage2_tactile ;;
  joint) CONFIG=vtla_bottle_stage3_joint ;;
  *) echo 'Usage: bash scripts/train_vtla_staged.sh {visual|tactile|joint} {preflight|train}'; exit 2 ;;
esac
MODE="${2:-preflight}"
if [[ "$MODE" != preflight && "$MODE" != train ]] || [[ $# -gt 2 ]]; then
  echo 'Select preflight or train; each stage uses its recorded configuration.'; exit 2
fi
export PYTHONPATH="$ROOT/src:$ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
# NCCL settings do not govern PJRT's implicit device-to-device copies.
# RNGs/control scalars are uploaded separately from CPU; reject other implicit
# copies instead of silently corrupting memory on this server's PCIe path.
export JAX_TRANSFER_GUARD_DEVICE_TO_DEVICE=disallow
PYTHON=/home/xudi_ge/openpi/.venv/bin/python
cd "$ROOT"
test -f /data/open_bottle_cap_vtla_20261006/lerobot/frame_cache/manifest.json
test -f "$ROOT/assets/bottle/open_bottle_cap/norm_stats.json"
test -f "$ROOT/artifacts/data_ready.json"
if [[ "$MODE" == preflight ]]; then
  exec "$PYTHON" scripts/benchmark_vtla.py --config "$CONFIG" --steps 30 --diagnose
fi
"$PYTHON" - "$CONFIG" <<'PY'
import json
import os
import sys
from pathlib import Path
from openpi.training.config import get_config

cfg = get_config(sys.argv[1])
path = Path(f'artifacts/benchmark_optimized_{cfg.name}.json')
if not path.exists():
    raise SystemExit('This stage has not passed its full-batch optimizer preflight.')
report = json.loads(path.read_text())
assert report['config'] == cfg.name and report['mode'] == 'optimized'
assert report['batch_size'] == cfg.batch_size
assert report['accumulation'] == cfg.gradient_accumulation_steps
assert report['fsdp_devices'] == cfg.fsdp_devices
assert report['initialization'] == cfg.weight_loader.params_path
assert report['training_objective'] == cfg.model.training_objective
assert report.get('xla_flags','') == os.environ.get('XLA_FLAGS','')
assert report.get('seed') == cfg.seed
assert report.get('control_inputs_host_replicated') is True
# The runtime guard may be stricter than during a numerical preflight: it
# rejects extra copies, without changing math or relaxing numerical checks.
assert report.get('device_transfer_guard') in (None, 'disallow')
assert len(report['iterations']) >= 31 and all(x['skipped_update'] == 0 for x in report['iterations'])
assert not report['initial_parameter_audit']['bad']
assert not report['fresh_adam_audit']['nonzero_paths']
PY
exec "$PYTHON" scripts/train.py "$CONFIG"
