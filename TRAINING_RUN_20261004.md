# 2026-10-04: pi05 fast/slow, 27000 policy + 26000 tactile

Status: stopped at step 27 on 2026-10-04 02:16:37 CST. The failure was confirmed on
2026-10-05: 27 VLM gradient leaf norms overflowed despite finite gradient elements.
An unlimited gradient threshold incorrectly allowed the update through because
`inf <= inf`; the later fatal check stopped the process. No checkpoint was saved.
See `TRAINING_RUN_20261005.md` for the guarded restart.

Earlier startup verification: PID 935939 completed four optimizer iterations by
2026-10-04 01:57:43 CST, with no skipped-update warnings or fatal numerical checks.
Steady intervals observed after compilation were 57.074 and 56.996 seconds per
update: roughly 20 days for 30000 updates at this throughput (an initial estimate).

The user approved replacing the originally requested 45999 policy checkpoint with
27000. The retained 30000–45999 checkpoints have extreme finite values in the VLM
MLP. Full CPU validation of the selected 27000 policy and 26000 tactile parameters
passed, including shape compatibility and identical normalization statistics.

## Configuration

- Config: `pi05_righthand_ee_fastslow_27k_b64`
- Experiment: `from_27000_touch26000_b64_30k_20261004`
- Full fine-tuning; no frozen parameter filter; no EMA.
- Effective global batch 64 = microbatch 8 × 8 accumulated gradients.
- 30000 additional optimizer steps, starting with a fresh optimizer and schedule.
- LR: 200-step warmup from 2e-6 to 5e-5, cosine decay to 2e-6 at step 30000.
- AdamW: beta1 0.9, beta2 0.95, eps 1e-8, weight decay 1e-10, gradient norm clip 1.
- Checkpoint every 1000 steps; keep every 2000 steps.
- Four GPUs, FSDP across four devices.

Policy source:
`/data/final_double_hands_openpi/pi05_righthand_ee_0806/pi05_righthand_ee_0806/27000/params`

Only `tactile_encoder/` and `tactile_adapter/` come from:
`/data/final_double_hands_openpi_fastslow/pi05_righthand_ee_fastslow_full/full_from_10000_30k_min2e6_20261002/26000/params`

The loader requires every target parameter in its designated source. No fallback
to random weights and no optimizer state from either source.

## Runtime

- Host: `hit100`
- Project: `/home/xudi_ge/pi05_fastslow`
- tmux: `train:pi05_fs27k_b64`
- Launcher: `bash scripts/train_pi05_fastslow_27k_b64.sh`
- Log: `logs/from_27000_touch26000_b64_30k_20261004.log`
- Checkpoints: `/data/final_double_hands_openpi_fastslow/pi05_righthand_ee_fastslow_27k_b64/from_27000_touch26000_b64_30k_20261004`
- TensorBoard: remote and local port 16011; old port 16010 preserved.
- TensorBoard scalars are recorded every 10 optimizer steps (roughly 9.5 minutes).
- Dated JSON configuration snapshots in `logs/` record exact settings, source
  audits, and source-file hashes for each launch.

During startup retries `--resume` is supplied to preserve the existing directory.
If there is no numeric checkpoint, the checkpoint manager explicitly starts fresh.
Once numeric checkpoints exist, `--resume` restores the saved training state.

## Numerical diagnosis

Three initial attempts stopped at optimizer step 0 because parameter/update norms
overflowed. No numeric checkpoint was saved from these attempts. The second
attempt's actual GPU initial parameters passed a full readback audit; first loss
was 0.42653495, flow loss 0.00418650, fast loss 0.42232943, gradient norm 2.14844465.
Its updated attention output-projection matrix contained finite values up to
8.12e29. These are failed diagnostic measurements, not accepted training results.

The third attempt disabled buffer donation and verified all 280 fresh Adam state
arrays were zero. It still produced extreme finite updates after a 15-minute
monolithic compilation (report `failed_update_step0_928624.json`). Disabling
donation alone is therefore not a demonstrated fix.

The successful attempt uses separate GPU executables for each microbatch gradient,
gradient summation/averaging, and the single AdamW update. Values stay on GPU;
the host only dispatches these smaller computations. Tests compare loss, RNG
behavior, gradients, metrics and the resulting state against the scan version.
Buffer donation remains disabled. Startup checks inspect every parameter and
require all fresh Adam moments and counters to be zero. Fatal norm checks remain
enabled. No learning-rate, loss, data or effective-batch change is introduced.

The first accepted optimizer update (logged at step 0) had:

| Metric | Value |
| --- | ---: |
| loss | 0.31034693 |
| flow_loss | 0.00145492 |
| slow_loss | 0.00145533 |
| fast_loss | 0.30888891 |
| lr | 2e-6 |
| grad_norm | 1.59633660 |
| update_norm | 0.07488996 |
| param_norm | 1824.10815 |
| skipped_update | 0 |

This is a first-batch measurement, not a convergence or generalization result.
Fifteen focused tests passed for the deployed implementation. Splitting execution
has removed the observed failure in the verified updates; the underlying GPU
compiler/runtime cause has not been conclusively identified.

Relevant evidence: `artifacts/initialization_27000_touch26000_audit.json`,
`artifacts/run_20261004/failed_update_step0_915996.json`, and the run directory's
`initial_device_parameters_*.json`, `initial_adam_state_*.json`, and failure reports.
