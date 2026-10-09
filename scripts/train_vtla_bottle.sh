#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src:$ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.95}"
export XLA_PYTHON_CLIENT_PREALLOCATE=true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# This host's historical launcher disables peer-to-peer communication.
# The native four-GPU smoke test stalled with P2P enabled on its PCIe topology.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
cd "$ROOT"
test -f /data/open_bottle_cap_vtla_20261006/lerobot/frame_cache/manifest.json
test -f "$ROOT/assets/bottle/open_bottle_cap/norm_stats.json"
test -f "$ROOT/artifacts/data_ready.json"
/home/xudi_ge/openpi/.venv/bin/python - <<'PY'
import json
from pathlib import Path
path = Path('artifacts/benchmark_optimized.json')
if not path.exists():
    raise SystemExit('Full-batch preflight has not passed; inspect logs/benchmark_optimized_separate.log')
report = json.loads(path.read_text())
assert len(report['iterations']) >= 4 and all(x['skipped_update'] == 0 for x in report['iterations']), 'Incomplete or failed preflight'
assert not report['initial_parameter_audit']['bad'], 'Invalid initial weights'
PY
exec /home/xudi_ge/openpi/.venv/bin/python scripts/train.py vtla_pi05_base_bottle "$@"
