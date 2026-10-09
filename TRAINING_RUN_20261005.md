# 2026-10-05: numerical guard fix and restart

Status: running as PID 1071667 in `train:pi05_fs27k_b64`. Verified at
2026-10-05 22:40:38 CST: 32 iterations completed, with no skipped-update or fatal
diagnostics. This exceeds the previous failure at iteration 27.

## Failure and fix

The previous process (935939) stopped at step 27 on October 4, 02:16:37 CST.
Its diagnostic reported finite individual gradient elements, but 27 VLM gradient
leaf norms were infinite. Both magnitude thresholds were unlimited, so `inf <=
inf` incorrectly passed. A post-update check terminated the process. All 140
inspected parameter arrays were finite and below the weight safety threshold.
No checkpoint was saved. The source checkpoints remain unchanged.

The repair in `scripts/train.py` requires finite loss, global gradient norm and
leaf norms before dispatching AdamW. It rejects nonfinite candidate updates,
parameter norms and optimizer state, even when optional magnitude thresholds
are disabled. A rejection preserves parameters, Adam moments/counters and EMA.
NaN/Inf metrics are no longer disguised as zero. Each rejection saves sample
identifiers, affected parameter paths and gradient maximum magnitudes. Ten
consecutive rejected updates stop the run for diagnosis.

This repairs update rejection and recovery. It does not establish the underlying
cause of the extreme gradients; live observations are recorded below.

## Restart configuration

- Config: `pi05_righthand_ee_fastslow_27k_b64`.
- Experiment: `from_27000_touch26000_b64_30k_guardfix_20261005`.
- Original 27000 backbone/action expert plus 26000 tactile encoder/adapter.
- Full fine-tuning; effective batch 64 (8 microbatches of 8); four GPUs.
- 30000 training iterations. Rejected iterations do not advance Adam's counter.
- LR: 200 accepted updates warm up from 2e-6 to 5e-5, cosine to 2e-6.
- Checkpoint interval reduced from 1000 to 100; keep period remains 2000.
- Separate experiment directory; previous logs and diagnostics are preserved.
- tmux target: `train:pi05_fs27k_b64`.
- TensorBoard remains on local/remote port 16011.

Launch command (from `/home/xudi_ge/pi05_fastslow`):

```bash
bash scripts/train_pi05_fastslow_27k_b64.sh \
  --exp-name from_27000_touch26000_b64_30k_guardfix_20261005 \
  --save-interval 100
```

## Validation

CPU: 23 regression tests passed in 58.71 seconds.
GPU (A100): all 15 numerical tests passed in 17.14 seconds.
The initial GPU audit passed for all 140 parameter arrays; all 280 fresh Adam
state arrays were exactly zero. Live training passed the previous failure at
step 27; 32 iterations were complete when checked. No skipped-update diagnostics
or fatal failure reports existed. The first checkpoint is scheduled at step 100.

TensorBoard at step 30 (averaged over the logging interval):

- Total loss: 0.17841248
- Flow loss: 0.00506826
- Fast loss: 0.17330854
- Gradient norm: 0.72373259; finite flag: 1
- Parameter norm: 1824.10974
- Skipped-update fraction: 0
- Learning rate: 9.2e-6

Evidence: `artifacts/run_20261005/verified_restart.json`.
Local TensorBoard: http://127.0.0.1:16011/?darkMode=true#scalars .
The forwarding connection uses SSH control socket
`/Users/rosepenguin/.ssh/pi05_tb_16011.sock` to reuse an established connection.
The original abnormal gradients did not recur during these observed iterations;
this is not proof that the underlying gradient-spike cause is resolved.
