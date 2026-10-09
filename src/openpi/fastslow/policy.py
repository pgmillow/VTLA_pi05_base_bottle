"""Single-session pi05 runtime. Sensor ingestion never waits for model inference."""
import dataclasses
import math
import threading
import time
import uuid

import jax
import jax.numpy as jnp
import numpy as np

from openpi.fastslow.buffer import ActionUpdate
from openpi.fastslow.model import Pi05FastSlow
from openpi.fastslow.pressure import PressureStream
from openpi.models.model import Observation
from openpi.shared.nnx_utils import module_jit


class FastSlowPolicy:
    """All timestamps are from one client monotonic clock, not the server clock.

    pi05's state is discretized in the prefix, so it refreshes on slow() only.
    The fast API deliberately has no state/image/prompt arguments.
    """
    def __init__(self, trained_policy, *, fps=29.994, max_delay_frames=8, max_sensor_age_s=0.12, seed=0):
        if not isinstance(trained_policy._model, Pi05FastSlow):
            raise TypeError("Load a pi05 fastslow config with the existing JAX checkpoint")
        self.model = trained_policy._model
        self.model.eval()
        if not np.isfinite(fps) or fps <= 0 or not 0 <= max_delay_frames < self.model.action_horizon:
            raise ValueError("Invalid action rate or cache lifetime")
        if not np.isfinite(max_sensor_age_s) or max_sensor_age_s <= 0:
            raise ValueError("Sensor lifetime must be positive")
        self.fps, self.max_delay_frames, self.max_sensor_age_s = fps, max_delay_frames, max_sensor_age_s
        self.input_transform = trained_policy._input_transform
        self.output_transform = trained_policy._output_transform
        self.slow_fn = module_jit(self.model.slow_snapshot)
        self.fast_fn = module_jit(self.model.fast_refine)
        self.rng = jax.random.key(seed)
        self.infer_lock, self.sensor_lock = threading.RLock(), threading.RLock()
        self.stream = PressureStream(history=16, baseline_frames=16, fps=fps)
        self.reset()

    def reset(self):
        with self.infer_lock, self.sensor_lock:
            self.stream.reset()
            self.session_id = uuid.uuid4().hex
            self.snapshot = self.observation = None
            self.chunk_id = self.revision = 0
            self.observation_time = self.last_fast_time = -math.inf
            self.last_valid_from = -1
            return self.session_id

    def _session(self, session_id):
        if session_id != self.session_id:
            raise ValueError("Stale session; reset client action buffer")

    def feed_pressure(self, session_id, raw, valid, timestamp):
        with self.sensor_lock:
            self._session(session_id)
            self.stream.update(raw, valid, timestamp)

    def _touch(self, at_time):
        with self.sensor_lock:
            fields, stamp = self.stream.snapshot(at_time)
        if not -1e-6 <= at_time-stamp <= self.max_sensor_age_s:
            raise ValueError("Pressure is stale; feed a fresh frame")
        if not fields["tactile_valid"].any():
            raise ValueError("No reliable current tactile sensors")
        return fields, stamp

    def slow(self, session_id, observation, observation_time, *, noise=None):
        with self.infer_lock:
            self._session(session_id)
            stamp = float(observation_time)
            if not np.isfinite(stamp) or stamp <= self.observation_time:
                raise ValueError("Slow observation timestamps must strictly increase")
            touch, _ = self._touch(stamp)
            inputs = self.input_transform({**observation, **touch})
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
            obs = Observation.from_dict(inputs)
            rng, next_rng = jax.random.split(self.rng)
            if noise is not None:
                noise = jnp.asarray(noise, jnp.float32)
                if noise.ndim == 2:
                    noise = noise[None]
                if noise.shape != (1, self.model.action_horizon, self.model.action_dim):
                    raise ValueError("Incorrect noise shape")
            start = time.monotonic()
            snapshot = self.slow_fn(rng, obs, noise=noise)
            jax.block_until_ready(snapshot)
            self.snapshot, self.observation = snapshot, obs
            self.rng, self.observation_time = next_rng, stamp
            self.chunk_id += 1
            self.revision = 0
            self.last_fast_time, self.last_valid_from = -math.inf, -1
            return {"session_id": self.session_id, "chunk_id": self.chunk_id, "observation_time": stamp,
                    "fps": self.fps, "slow_ms": (time.monotonic()-start)*1000}

    def fast(self, session_id, chunk_id, observation_time, executed_steps=0):
        with self.infer_lock:
            self._session(session_id)
            if self.snapshot is None or chunk_id != self.chunk_id:
                raise ValueError("Stale chunk or no slow context")
            stamp = float(observation_time)
            if not np.isfinite(stamp) or stamp < self.observation_time or stamp <= self.last_fast_time:
                raise ValueError("Fast observation timestamps must strictly increase")
            progress = int(executed_steps)
            if progress != executed_steps or not 0 <= progress < self.model.action_horizon:
                raise ValueError("Invalid executed_steps")
            if progress < self.last_valid_from:
                raise ValueError("Execution progress cannot move backwards")
            age = stamp-self.observation_time
            valid_from = max(progress, math.ceil(age*self.fps-1e-6))
            if valid_from > self.max_delay_frames or age*self.fps > self.max_delay_frames+1e-3:
                raise ValueError("Slow context expired; refresh images/state with slow()")
            touch, sensor_time = self._touch(stamp)
            obs = dataclasses.replace(self.observation, **{k: jnp.asarray(v)[None] for k, v in touch.items()})
            start = time.monotonic()
            actions, gate = self.fast_fn(self.snapshot, obs, jnp.asarray([valid_from], jnp.int32))
            actions, gate = np.asarray(actions[0]), float(gate[0])
            outputs = self.output_transform({"actions": actions, "state": np.asarray(self.observation.state[0])})
            # This checkpoint supervises only the right-hand 7 dimensions. Never
            # expose the other padded/untrained model outputs as robot commands.
            indices = self.model.loss_action_dims
            if indices is None:
                raise ValueError("Specify trained action dimensions before deployment")
            physical = np.asarray(outputs["actions"])[..., list(indices)]
            if not np.isfinite(physical).all() or not np.isfinite(gate):
                raise ValueError("Model emitted a non-finite action or gate")
            self.revision += 1
            self.last_fast_time, self.last_valid_from = stamp, valid_from
            update = ActionUpdate(self.session_id, self.chunk_id, self.revision, self.observation_time,
                                  sensor_time, physical, valid_from, self.fps)
            return {**update.as_dict(), "gate": gate, "action_dims": list(indices),
                    "fast_ms": (time.monotonic()-start)*1000}


def load_policy(checkpoint, config_name="pi05_righthand_ee_fastslow", *, compute_dtype=None, **kwargs):
    import json
    from pathlib import Path
    from openpi.policies.policy_config import create_trained_policy
    from openpi.training.config import get_config
    config = get_config(config_name)
    if compute_dtype is not None:
        if compute_dtype not in ("float32", "bfloat16"):
            raise ValueError("Unknown compute dtype")
        config = dataclasses.replace(config, model=dataclasses.replace(config.model, dtype=compute_dtype))
    path = Path(checkpoint) / "assets" / "fastslow.json"
    if path.exists():
        metadata = json.loads(path.read_text())
        if metadata.get("version") != 1 or metadata.get("state_refresh") != "slow":
            raise ValueError("Unsupported fast/slow checkpoint metadata")
        for key in ("action_horizon", "action_dim", "loss_action_dims"):
            expected = config.model.runtime_metadata()[key]
            if metadata.get(key) != expected:
                raise ValueError(f"Checkpoint action layout mismatch: {key}")
        fields = ("num_flow_steps", "slow_steps", "max_delay_frames", "validation_fraction", "split_seed", "fps")
        config = dataclasses.replace(config, model=dataclasses.replace(config.model, **{k: metadata[k] for k in fields}))
    requested_fps = kwargs.pop("fps", None)
    if requested_fps is not None and (not np.isfinite(requested_fps) or abs(requested_fps-config.model.fps) > .02):
        raise ValueError("Serving fps must match the training/checkpoint fps")
    kwargs["fps"] = config.model.fps
    policy = create_trained_policy(config, checkpoint)
    return FastSlowPolicy(policy, max_delay_frames=config.model.max_delay_frames, **kwargs)
