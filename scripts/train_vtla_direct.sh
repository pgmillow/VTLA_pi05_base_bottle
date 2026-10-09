#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:-preflight}"
if [[ "$MODE" != preflight && "$MODE" != train ]] || [[ $# -gt 1 ]]; then
  echo 'Usage: bash scripts/train_vtla_direct.sh {preflight|train}'; exit 2
fi
export PYTHONPATH="$ROOT/src:$ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export JAX_TRANSFER_GUARD_DEVICE_TO_DEVICE=disallow
PYTHON=/home/xudi_ge/openpi/.venv/bin/python
cd "$ROOT"
test -f /data/open_bottle_cap_vtla_20261006/lerobot/frame_cache/manifest.json
test -f "$ROOT/assets/bottle/open_bottle_cap/norm_stats.json"
test -f "$ROOT/artifacts/data_ready.json"
if [[ "$MODE" == preflight ]]; then
  exec "$PYTHON" scripts/benchmark_vtla.py --config vtla_bottle_direct_full --steps 99 --diagnose
fi
"$PYTHON" - <<'PY'
import dataclasses
import json
import math
import os
from pathlib import Path
from openpi.training.config import get_config

cfg = get_config('vtla_bottle_direct_full')
path = Path(f'artifacts/benchmark_optimized_{cfg.name}.json')
if not path.exists():
    raise SystemExit('Direct joint training requires its own successful 100-update preflight.')
r = json.loads(path.read_text())
assert r['config'] == cfg.name and r['mode'] == 'optimized'
assert r['batch_size'] == cfg.batch_size and r['accumulation'] == cfg.gradient_accumulation_steps
assert r['fsdp_devices'] == cfg.fsdp_devices and r['seed'] == cfg.seed
assert r['split_seed'] == cfg.model.split_seed
assert r['initialization'] == cfg.weight_loader.params_path
assert r['training_objective'] == cfg.model.training_objective == 'paired'
assert r['num_train_steps'] == cfg.num_train_steps
for key, actual in [('lr_schedule', cfg.lr_schedule), ('optimizer', cfg.optimizer), ('model_config', cfg.model)]:
    assert r[key] == json.loads(json.dumps(dataclasses.asdict(actual))), key
assert r['parameter_count'] == r['trainable_parameter_count'] > 0
assert r['xla_flags'] == os.environ.get('XLA_FLAGS', '')
assert r['control_inputs_host_replicated'] is True
assert r['device_transfer_guard'] == 'disallow'
assert not r['initial_parameter_audit']['bad'] and not r['fresh_adam_audit']['nonzero_paths']
rows = r['iterations']
assert len(rows) >= 100
for row in rows:
    assert row['skipped_update'] == 0 and row['update_norm'] > 0
    assert row['param_max_abs'] < 1e6
    assert all(math.isfinite(row[k]) for k in ['loss', 'flow_loss', 'fast_loss', 'contact_loss',
                                              'grad_norm', 'param_norm', 'param_max_abs', 'update_norm'])
print('Direct joint full-finetuning preflight passed: 100 healthy updates.', flush=True)
PY
exec "$PYTHON" scripts/train.py vtla_bottle_direct_full
