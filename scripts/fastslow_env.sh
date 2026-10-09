#!/usr/bin/env bash
set -euo pipefail
FASTSLOW_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/xudi_ge/openpi/.venv/bin/python}"
export PYTHONPATH="$FASTSLOW_ROOT/src:$FASTSLOW_ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
cd "$FASTSLOW_ROOT"
