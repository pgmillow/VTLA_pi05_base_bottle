"""Resume the existing full-finetuning run for an exact, short cosine tail."""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path


@dataclasses.dataclass(frozen=True)
class FinishCosineSchedule:
    original: object
    start_step: int
    additional_steps: int
    end_lr: float

    def create(self):
        import jax.numpy as jnp
        import optax

        if self.start_step < 0 or self.additional_steps < 2 or self.end_lr <= 0:
            raise ValueError("Invalid continuation schedule")
        old = self.original.create()
        start_lr = float(old(self.start_step))
        if self.end_lr > start_lr:
            raise ValueError("The continuation must decay the learning rate")
        # Counts are restored with Adam. The final *executed* update reaches end_lr.
        tail = optax.cosine_decay_schedule(
            start_lr, self.additional_steps - 1, alpha=self.end_lr / start_lr)
        return lambda count: jnp.where(
            count < self.start_step, old(count), tail(jnp.maximum(count - self.start_step, 0)))


def checkpoint_ready(directory: Path) -> bool:
    """Orbax's committed directory and marker must both be present."""
    try:
        meta = json.loads((directory / "_CHECKPOINT_METADATA").read_text())
        return bool(meta.get("commit_timestamp_nsecs")) and all(
            (directory / item).is_dir() for item in ("params", "train_state", "assets")
        ) and all((directory / item / "_METADATA").is_file() for item in ("params", "train_state"))
    except (OSError, ValueError):
        return False


def checkpoint_counts(directory: Path) -> dict[str, int]:
    """Read three tiny OCDBT tensors on the CPU, without loading the weights."""
    import tensorstore as ts

    if not checkpoint_ready(directory):
        raise ValueError(f"Checkpoint is not committed: {directory}")
    kv = {"driver": "ocdbt", "base": "file://" + str(directory / "train_state") + "/"}
    names = {"step": "step", "adam_count": "opt_state.1.0.count",
             "schedule_count": "opt_state.1.2.count"}
    return {name: int(ts.open({"driver": "zarr", "kvstore": kv, "path": key}, open=True)
                      .result().read().result()) for name, key in names.items()}


def make_config(config_name: str, checkpoint_step: int, additional_steps: int, end_lr: float):
    from openpi.training.config import get_config

    base = get_config(config_name)
    start = checkpoint_step + 1  # train.py saves the already-incremented TrainState.
    schedule = FinishCosineSchedule(base.lr_schedule, start, additional_steps, end_lr)
    cfg = dataclasses.replace(base, resume=True, overwrite=False,
                              num_train_steps=start + additional_steps,
                              keep_period=1000, lr_schedule=schedule)
    lr = schedule.create()
    plan = {"config": cfg.name, "exp_name": cfg.exp_name,
            "checkpoint_dir": str(cfg.checkpoint_dir), "source_checkpoint": checkpoint_step,
            "start_step": start, "additional_steps": additional_steps,
            "final_checkpoint": cfg.num_train_steps - 1,
            "start_lr": float(lr(start)), "end_lr": float(lr(cfg.num_train_steps - 1)),
            "batch_size": cfg.batch_size, "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "fsdp_devices": cfg.fsdp_devices, "resume_optimizer": True,
            "warmup_restarted": False, "preserve_checkpoints_every": cfg.keep_period}
    return cfg, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="vtla_bottle_direct_full")
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--additional-steps", type=int, default=5000)
    parser.add_argument("--end-lr", type=float, default=1e-5)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    cfg, plan = make_config(args.config, args.checkpoint_step, args.additional_steps, args.end_lr)
    print(json.dumps(plan, indent=2), flush=True)
    if args.plan_only:
        return

    import jax
    import train as training
    from openpi.training import checkpoints

    source = cfg.checkpoint_dir / str(args.checkpoint_step)
    counts = checkpoint_counts(source)
    if set(counts.values()) != {plan["start_step"]}:
        raise ValueError(f"Source checkpoint/optimizer counters do not agree: {counts}")
    (cfg.checkpoint_dir / "finish_5k_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    original_restore = checkpoints.restore_state

    def restore_exact(manager, state, loader, step=None):
        restored = original_restore(manager, state, loader, step=args.checkpoint_step)
        actual_step = int(jax.device_get(restored.step))
        counter_values = {}
        for path, value in jax.tree_util.tree_flatten_with_path(restored.opt_state)[0]:
            if path and getattr(path[-1], "name", None) == "count":
                counter_values[jax.tree_util.keystr(path)] = int(jax.device_get(value))
        if actual_step != plan["start_step"] or len(counter_values) != 2 or any(
            value != actual_step for value in counter_values.values()
        ):
            raise ValueError(f"Restored optimizer mismatch: {actual_step}, {counter_values}")
        logging.info("FINISH_RESUME_VERIFIED step=%d optimizer_counters=%s", actual_step, counter_values)
        return restored

    def tensorboard(config, *, enabled=True):
        if not enabled:
            return None
        from torch.utils.tensorboard import SummaryWriter
        # Remove only old-process scalar points beyond the restored state.
        return SummaryWriter(str(config.checkpoint_dir / "tensorboard"), purge_step=plan["start_step"])

    checkpoints.restore_state = restore_exact
    training.init_tensorboard = tensorboard
    training.main(cfg)
    final = cfg.checkpoint_dir / str(plan["final_checkpoint"])
    counts = checkpoint_counts(final)
    expected = plan["start_step"] + args.additional_steps
    if set(counts.values()) != {expected}:
        raise ValueError(f"Final checkpoint/optimizer counts do not agree: {counts}")
    print("FINISH_COMPLETE " + json.dumps({**plan, "final_counts": counts}), flush=True)


if __name__ == "__main__":
    main()
