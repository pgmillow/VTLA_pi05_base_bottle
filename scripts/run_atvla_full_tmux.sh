#!/usr/bin/env bash
set -uo pipefail
cd /home/xudi_ge/pi05_atvla
train_stamp=$(date +%Y%m%d_%H%M%S)
train_log="/home/xudi_ge/pi05_atvla/logs/full_from_10000_30k_${train_stamp}.log"
printf '%s\n' "$train_log" > logs/current_full_train_log.txt
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/train_pi05_atvla_full.sh 2>&1 | tee "$train_log"
train_exit=${PIPESTATUS[0]}
printf '\nTraining exited with code %s\n' "$train_exit" | tee -a "$train_log"
printf '%s\n' "$train_exit" > logs/full_from_10000_30k.exit_code
exit "$train_exit"
