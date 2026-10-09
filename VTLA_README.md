# Pi05 base / O10 tactile fast–slow bottle experiment

Latest completed run: `vtla_bottle_direct_full` resumed checkpoint 19000 for
5000 additional updates and finished at checkpoint **24000** on 2026-10-09.
The 46,000-step recipe below records the original plan. See
[README_VTLA_BOTTLE.md](README_VTLA_BOTTLE.md) for the final run and Git-managed source snapshot.

This is an isolated copy of `pi05_fastslow`. Existing training code and checkpoints are not overwritten.
Server project: `/home/xudi_ge/VTLA_pi05_base_bottle`.

## Active recipe: direct tactile full fine-tuning

The user switched from the three-stage curriculum to direct joint training on
2026-10-06. `vtla_bottle_direct_full` initializes the policy from official
pi05_base and initializes the O10 encoder, contact Gate and action Adapter anew.
All 3,354,066,833 parameters are trainable, including 632,961 tactile parameters.
Only the adapter's residual output projection is zero-initialized; its contact
gate remains a trainable sigmoid. The paired fast/slow objective is enabled from
the start, retaining six slow denoising steps and four fast steps at inference.

Settings: batch 64, accumulation 1, FSDP across four GPUs, eight data workers,
46,000 optimizer steps, seed 43, unchanged episode split seed 42. The schedule
warms up for 200 steps to 5e-5, then decays to 2e-6. Adam starts fresh. This does
not load the abandoned visual-stage weights or the historical 46k checkpoint.

```bash
bash scripts/train_vtla_direct.sh preflight  # 100 disposable real joint updates
bash scripts/train_vtla_direct.sh train      # requires matching successful report
```

The launched tmux window `train:bottle_joint46k` runs these sequentially; any
preflight failure stops the pipeline. CPU uploads for RNG/control inputs and the
active implicit device-transfer guard remain enabled. Preflight checks the
configuration, model, optimizer, learning-rate schedule, full trainable count,
finite losses/gradients/weights and positive updates with zero skips.

Output: `/data/checkpoints_vtla_bottle/vtla_bottle_direct_full/base_touch_joint_46k_seed43`.
Live phase/log paths are recorded in `artifacts/direct_full_startup.json`;
configuration and counts are in `artifacts/direct_full_config_audit.json`.
The interrupted visual run is preserved under `/data/checkpoints_vtla_bottle_staged`.
The staged recipes and earlier diagnostic recipe below remain for reference.

The joint preflight passed all 100 updates with zero skips and the device-transfer
guard active. Total loss moved from 2.182118 to 0.352197, flow loss from 0.301949
to 0.053592, and fast loss from 1.870546 to 0.295608. Final contact loss was
0.149994. All checked weights, gradients and updates remained finite. Median
device work was 9.472 seconds/update, excluding data loading. The pipeline then
started the formal 46k process; its observed phase and PID are recorded in
`artifacts/direct_full_startup.json`. The disposable preflight saved no checkpoint,
so the formal run starts from base again with fresh optimizer state.

## Staged alternative prepared on 2026-10-06

The requested curriculum is prepared in three separate configurations. Pending
any different choice for stages 2/3, their defaults preserve 46,000 total updates:

| Stage | Steps | Trainable parameters | Objective / initialization |
|---|---:|---:|---|
| `vtla_bottle_stage1_visual` | 20,000 | 3,353,433,872 | Pure pi05 flow; official pi05_base; no tactile parameters or tactile input |
| `vtla_bottle_stage2_tactile` | 6,000 | 632,961 | Flow + 0.01 contact BCE; freeze the entire policy, train encoder/gate/adapter; stage 1 final weights |
| `vtla_bottle_stage3_joint` | 20,000 | 3,354,066,833 | Full fine-tuning, slow/fast paired objectives; stage 2 final weights |

Each stage starts a fresh optimizer and uses batch 64, four GPUs, 200-step warmup
to 5e-5 and cosine decay to 2e-6. Stages share normalization, episode split,
frame selection and terminal action masks. No trajectory is moved from validation
to training between stages. Fast/slow rollouts are introduced only in stage 3.
The new tactile output projection starts at zero when stage 2 loads stage 1.

A GPU peer-copy integrity failure was reproduced without a model or optimizer.
The exact bytes of the training RNG were found in corrupted GPU weights. RNGs
and microbatch indices now upload independently from CPU to each GPU, and the
staged launcher rejects other implicit device-to-device copies. Stage 1 uses
training seed 43 for the requested control; episode split/normalization seed stays 42.
Each stage must pass its own optimizer preflight before launch:

```bash
bash scripts/train_vtla_staged.sh visual preflight
bash scripts/train_vtla_staged.sh visual train
# After stage 1 finishes and its checkpoint is audited:
bash scripts/train_vtla_staged.sh tactile preflight
bash scripts/train_vtla_staged.sh tactile train
# After stage 2 finishes and its checkpoint is audited:
bash scripts/train_vtla_staged.sh joint preflight
bash scripts/train_vtla_staged.sh joint train
```

Outputs live in `/data/checkpoints_vtla_bottle_staged`. Final checkpoint directory
indices are 19999 / 5999 / 19999 because training logs use zero-based steps.
The stages do not auto-chain. The first four-update stage-1 check passed, but
production subsequently rejected nonfinite updates at steps 6, 8 and 12 and was
interrupted without a model checkpoint. Its output is preserved as
`visual_20k_failed_20261006_182458`. Disabling Triton GEMM also failed on update 5.
The default preflight requires 31 healthy updates. The repaired stage-1 path
passed the requested 100-update validation with zero skipped updates: loss
0.304745 to 0.050226 (last-ten mean 0.052420), median device time 7.487 s/update.
The learning-rate schedule was unchanged. This disposable preflight saved no
checkpoint; production starts again from base in `train:bottle_visual20k`.
Its startup/observation record is `artifacts/stage1_hostrng_startup.json`.
JAX 0.5.3 did not apply the device-transfer-guard environment variable during
that preflight, so its actual guard metadata remains null. The mesh initializer
now applies it explicitly; 16 four-GPU RNG checks passed with `disallow` active,
and the production launcher requests this guard. See
`artifacts/gpu_rng_corruption_diagnosis.json`. The single-stage configuration below is retained
for comparison and diagnostic reproducibility.

## What is being trained

- Official pi05 base initializes the image encoder, language backbone and action expert. Fresh Adam state and step zero; no 27000/46000 task checkpoint is loaded.
- A new native O10 encoder reads all 268 pressure channels for 16 causal frames. A per-frame MLP and causal temporal convolution produce 16 tokens. No Inspire 11-patch geometry is fabricated.
- A fresh contact MLP predicts a sigmoid gate. The action adapter's final residual projection starts at zero, preserving the base action path initially; the gate itself is not hard-zeroed.
- The slow system uses cameras, prompt, discrete state and old tactile context. It performs six of ten flow integration steps. The fast system reuses its prefix cache and refines the future action portion with fresh pressure. Training uses randomly delayed touch, 0–8 frames, within the same episode.
- The action target is provisionally the recorded **17 absolute joint targets**: right arm 7 plus O10 hand 10, in radians. State has 23 values: measured arm 7, hand 10 and EE xyz/rpy 6. The model pads to 32 and supervises dimensions 0–16. Measured EE positions are not substituted for commanded EE targets.
- The old robot used right EE6 plus one hand scalar, so these loss values cannot be compared directly as an identical physical objective.

## Data preparation

Input: `/data/拧瓶盖/open_bottle_cap.zip`, SHA256 `15a65675cc84e298490115ef9aea6d0718fd2afee644df4326a95a5a3ea61fe4`.
Working data: `/data/open_bottle_cap_vtla_20261006/`.

60 MCAP files contain 54,419 top-camera frames before boundary trimming. The first canonical classifier cannot parse this robot and its parse-error reports are retained for traceability. The native classifier finds no nonfinite values, isolated arm/EE spikes or long dynamic hand gaps; all 60 need timestamp repair because the hand command publisher repeats header stamps. Conversion stably sorts by header time and retains the last log-order command at duplicate times. Counts and conflicting duplicate values are recorded per episode.

The canonical conversion strategy is preserved where applicable: top-camera anchors; continuous linear numeric interpolation; unwrap RPY first; remove brief leave-and-return hand glitches (0.15 rad L2, 60 ms); nearest wrist frames; 224×224 letterboxing; JPEG quality 90. Only unmatched numeric boundaries are trimmed. Persistent arm command jumps and return-home transitions remain flagged for review, not automatically deleted. No raw MCAP is modified.

Pressure is resampled causally using the latest packet at or before the image anchor, valid for at most 50 ms. Native channels and source timestamps are retained. 26 episodes contain only zero pressure and are retained for action learning with their tactile branch/supervision masked; a zero-only recording is not reliable negative contact evidence. The other 34 episodes contain some nonzero pressure, which by itself does not certify sensor health.

Calibration follows the user's first-16-frames-no-contact assumption: median channel baseline and a global q99.5 absolute noise scale, minimum one raw count. Before calibration completes the tactile branch is invalid. Contact weak labels use the average of the four largest noise-relative channel deviations: ≤1 negative, ≥3 positive, otherwise unknown. Using the mean of all 268 channels would suppress localized contacts in these recordings. These are heuristic weak labels, not measured contact ground truth or force units.

Normalization uses only the training episodes (seed 42; 5% entire episodes held out). Exact q01/q99 statistics are used. Action statistics include the same 30-frame chunks and terminal padding as training. Original JPEG bytes and numeric/tactile arrays are cached with mmap; source size/mtime fingerprints prevent silently using an outdated cache.

## Training settings aligned with pi05_righthand_ee_0806

| Setting | New configuration |
|---|---|
| Global batch | 64, one batch per optimizer update, no accumulation |
| Training length | 46,000 optimizer steps |
| Learning rate | 200-step warmup; peak 5e-5; cosine toward 2e-6 at step 46,000 |
| Initial warmup LR | 5e-5 / 201, matching historical schedule; the 2e-6 endpoint is not a warmup floor |
| Optimizer | AdamW β1=.9, β2=.95, epsilon=1e-8, weight decay=1e-10, gradient clip=1 |
| Parameters | Full fine-tune, no EMA |
| Devices / workers | FSDP across 4 GPUs / 8 loader workers |
| Horizon / prompt length | 30 / 200 |
| Save / keep / log | Every 1000 / 3000 / 10 steps |
| Image augmentation | Same historical mild color augmentation, crop disabled |

The historical 46k run resumed from step 24276. This experiment intentionally starts from official pi05 base and a fresh tactile encoder/gate, as requested. Numeric guards remain enabled; finite gradients alone are insufficient, so proposed optimizer state and weights are checked before accepting an update.

## Why the preceding run was slow

The last old run stopped at iteration 590 after 10 rejected optimizer updates, at about 53.5 seconds/iteration. It used eight sequential microbatches, repeated image/prefix encoding for the six-step teacher rollout, and random indexing into a one-episode parquet cache. The four GPUs have PCIe/SYS links, with no NVLink shown by `nvidia-smi topo -m`. A P2P-enabled smoke test stalled at four-GPU communication; the launcher therefore retains this host's historical `NCCL_P2P_DISABLE=1` / `NCCL_IB_DISABLE=1` settings.

This version removes the duplicate image/prefix forward, shares the same augmented observation between the flow loss and teacher rollout, and uses the mmap frame cache. The real six-step teacher rollout and both losses remain. An apples-to-apples ablation is `reuse_training_prefix=False`; an old-execution benchmark additionally uses eight microbatches. Speed and batch64 memory feasibility must be measured after compilation, not inferred from code changes.

## Run and validation artifacts

Prepare data with `data_pipeline/convert_o10.py`, then `scripts/prepare_vtla_bottle.py`. The latter generates `artifacts/data_audit.json`, representative plots, train-only normalization, and `artifacts/data_ready.json`.

The first monolithic/donated full-batch update failed the parameter-norm guard despite valid initial weights. The new configuration uses separate gradient/optimizer executables with no buffer donation, preserving batch64, one optimizer update and the same schedule.

**Historical diagnostics before the host-transfer fix:** the original joint
batch64 training executable returned infinite action losses and non-finite
gradients before Adam despite finite independent forwards. The action AdaRMS
Dense kernel and its Adam states remain replicated, with other large parameters
using FSDP. Subsequent bitwise transfer checks and RNG bytes in actual weights
identified a GPU peer-copy integrity failure. Stage 1 now passes 100 real updates
after the host-transfer workaround. Stages 2/3 and the single-stage joint recipe
have not yet passed their own repaired full-batch preflight; stage-1 success is
not evidence of their validation. See `实验说明.md` and the diagnostic artifacts.

A bounded disposable benchmark is `scripts/benchmark_vtla.py --mode optimized --steps 3` (or `--mode legacy`). It loads official weights, audits fresh Adam state, runs real training batches and writes `artifacts/benchmark_*.json`; it does not save a model checkpoint. Iteration zero includes compilation; steady timing excludes it and loader wait. The benchmark uses synchronous data loading (0 workers) to avoid repeated process startup; production retains 8 workers.

`--diagnose` records the first gradient and averaged-gradient boundary; `--no-prefix-reuse` disables only training prefix reuse; `--accumulation N` tests the same global batch with N microbatches. Ablations have distinct report filenames and do not authorize the production launcher. The launcher requires four accepted updates in the production configuration's successful benchmark report.

After validation, the production launcher is:

```bash
cd /home/xudi_ge/VTLA_pi05_base_bottle
bash scripts/train_vtla_bottle.sh
```

Production output is `/data/checkpoints_vtla_bottle/vtla_pi05_base_bottle/base_fresh_gate_46k`.
No production run is started by data preparation or the benchmark. This directory provides the new training/model implementation; the old Inspire-only robot deployment client still needs an O10 input/action adapter before real robot use.

## Changed and added files

- `data_pipeline/o10_mcap.py`, `classify_o10_quality.py`, `convert_o10.py`: native parsing, quality review and conversion.
- `src/openpi/vtla/model.py`, `tactile.py`, `pressure.py`, `cache.py`: native tactile model, calibration, prefix reuse and frame cache.
- `src/openpi/models/pi0.py`: expose flow context and select native tactile encoder.
- `src/openpi/fastslow/model.py`, `data.py`: factor the paired loss, permit native O10 and select cache reader.
- `src/openpi/training/config.py`: `vtla_pi05_base_bottle` configuration.
- `scripts/prepare_vtla_bottle.py`, `benchmark_vtla.py`, `train_vtla_bottle.sh`: audited preparation, benchmark and launcher.
- `tests/vtla/test_o10.py`, `tests/fastslow/test_fastslow.py`: causal alignment, sparse contacts, missing masks, cache episode boundaries, fresh residual, prefix loss/gradient equivalence and historical hyperparameter checks.
