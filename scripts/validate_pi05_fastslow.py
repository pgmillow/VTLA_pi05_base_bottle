"""Recorded-data checkpoint validation. No robot connection or long training."""
import argparse
import dataclasses
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from openpi.fastslow.data import FastSlowDataset
from openpi.fastslow.model import FastSlowObservation
from openpi.fastslow.policy import load_policy
from openpi.fastslow.serve import DEFAULT_CHECKPOINT
from openpi.shared.nnx_utils import module_jit
from openpi.training.config import get_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", default="artifacts/validation_checkpoint_29999.json")
    parser.add_argument("--train-step", action="store_true", help="Also run one in-memory optimizer update")
    parser.add_argument("--compute-dtype", choices=("float32", "bfloat16"), default="bfloat16")
    args = parser.parse_args()
    cfg = get_config("pi05_righthand_ee_fastslow")
    data_cfg = cfg.data.create(cfg.assets_dirs, cfg.model)
    train = FastSlowDataset(data_cfg, cfg.model)
    validation = FastSlowDataset(data_cfg, cfg.model, split="validation")
    assert set(train.episode_ids).isdisjoint(validation.episode_ids)
    print(f"Data split: {len(train.episode_ids)} train, {len(validation.episode_ids)} validation episodes", flush=True)
    policy = load_policy(args.checkpoint, compute_dtype=args.compute_dtype)
    base = train.base
    ep = train.episode_ids[0]
    frame = base._load_episode(ep)
    session = policy.session_id
    for i in range(16):
        row = frame.iloc[i]
        policy.feed_pressure(session, base._parse_tactile_patch(row[base._tactile_key]),
                             np.asarray(row[base._tactile_valid_key], bool), i/policy.fps)
    row = frame.iloc[15]
    observation = {"images": {k: base._parse_image(row[f"observation.images.{k}"])
                              for k in ("top", "left_wrist", "right_wrist")},
                   "state": np.asarray(row["observation.state"], np.float32),
                   "prompt": base.tasks[int(row["task_index"])]}
    noise = np.asarray(jax.random.normal(jax.random.key(7), (1, 30, 32)))
    slow = policy.slow(session, observation, 15/policy.fps, noise=noise)
    baseline = module_jit(policy.model.sample_actions)
    original = baseline(jax.random.key(0), policy.observation, num_steps=10, noise=jnp.asarray(noise))
    split, _ = policy.fast_fn(policy.snapshot, policy.observation, jnp.zeros(1, jnp.int32))
    difference = float(jnp.max(jnp.abs(original-split)))
    mean_difference = float(jnp.mean(jnp.abs(original-split)))
    # Separate compiled graphs may choose different BF16 fusion/rounding paths.
    # Keep strict FP32 comparison and explicitly report the BF16 error, rather
    # than claiming bitwise equivalence for low-precision execution.
    tolerance = 2e-4 if args.compute_dtype == "float32" else 1e-2
    np.testing.assert_allclose(split, original, atol=tolerance, rtol=0)
    print(f"Split comparison ({args.compute_dtype}): max={difference:.8g}, mean={mean_difference:.8g}", flush=True)
    outputs = [policy.fast(session, slow["chunk_id"], 15/policy.fps, 0)]
    for i in range(16, 24):
        row = frame.iloc[i]
        policy.feed_pressure(session, base._parse_tactile_patch(row[base._tactile_key]),
                             np.asarray(row[base._tactile_valid_key], bool), i/policy.fps)
        outputs.append(policy.fast(session, slow["chunk_id"], i/policy.fps, i-15))
    # Warm slow timing with exactly the same graph, excluding compile latency.
    warm_slow = policy.slow(session, observation, 23/policy.fps, noise=noise)
    warm_fast = np.asarray([x["fast_ms"] for x in outputs[1:]])
    report = {"checkpoint": args.checkpoint, "backend": str(jax.devices()), "episode": ep,
              "parameter_structure": "unchanged pi05 + original tactile adapter",
              "same_touch_split_max_abs_error": difference,
              "same_touch_split_mean_abs_error": mean_difference, "compute_dtype": args.compute_dtype,
              "split_comparison_atol": tolerance,
              "slow_ms_first_call": slow["slow_ms"], "slow_ms_warm": warm_slow["slow_ms"],
              "fast_ms_median_warm": float(np.median(warm_fast)),
              "fast_ms_max_warm": float(warm_fast.max()), "revisions": len(outputs),
              "action_shape": list(np.asarray(outputs[-1]["actions"]).shape),
              "valid_from": [o["valid_from"] for o in outputs], "gates": [o["gate"] for o in outputs],
              "train_episodes": train.episode_ids, "validation_episodes": validation.episode_ids,
              "robot_tested": False}
    sample = train[0]
    batch = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], sample)
    obs = FastSlowObservation.from_dict(batch)
    loss_fn = module_jit(policy.model.compute_loss_and_metrics, static_argnames=("train",))
    loss, metrics = loss_fn(jax.random.key(11), obs, batch["actions"], train=True)
    report["paired_training_loss"] = float(jnp.mean(loss))
    report["paired_metrics"] = {k: float(v) for k, v in metrics.items()}
    assert np.isfinite(report["paired_training_loss"])
    if args.train_step:
        from flax import nnx
        import optax
        from openpi.training.utils import TrainState
        import train as train_script
        graph, params = nnx.split(policy.model)
        tx = optax.adam(1e-5)
        state = TrainState(step=jnp.asarray(0), params=params, model_def=graph,
                          opt_state=tx.init(params.filter(cfg.trainable_filter)), tx=tx, ema_decay=None)
        step = jax.jit(lambda s: train_script.train_step(cfg, jax.random.key(12), s, (obs, batch["actions"])))
        started = time.monotonic()
        updated, info = step(state)
        jax.block_until_ready(updated)
        report["optimizer_step_seconds_including_compile"] = time.monotonic()-started
        report["optimizer_metrics"] = {k: np.asarray(v).tolist() for k, v in info.items()}
        assert int(updated.step) == 1 and float(info["skipped_update"]) == 0
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("train_episodes", "validation_episodes", "optimizer_metrics")}, indent=2), flush=True)
    print(f"Saved {dest}", flush=True)


if __name__ == "__main__":
    main()
