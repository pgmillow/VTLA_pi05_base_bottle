#!/usr/bin/env bash
set -euo pipefail
export FASTSLOW_CONFIG=pi05_righthand_ee_fastslow_27k_b64
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/train_pi05_fastslow_46k_b64.sh" "$@"
