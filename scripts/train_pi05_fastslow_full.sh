#!/usr/bin/env bash
set -euo pipefail
export FASTSLOW_CONFIG=pi05_righthand_ee_fastslow_full
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/train_pi05_fastslow.sh" "$@"
