"""Split pi05 flow sampling without adding or renaming checkpoint parameters.

The slow snapshot contains only immutable prefix KV and a partially denoised
action chunk. Every fast request restarts from that snapshot with fresh touch;
it never feeds a previous revision back into the next refinement.
"""
import dataclasses
import math
from typing import Any

from flax import nnx, struct
import jax
import jax.numpy as jnp

from openpi.models import model as base_model
from openpi.models.pi0 import Pi0, make_attn_mask
from openpi.models.pi0_config import Pi0Config


@struct.dataclass
class SlowSnapshot:
    kv_cache: Any
    prefix_mask: Any
    x_split: Any
    time: Any


@struct.dataclass
class FastSlowObservation(base_model.Observation):
    fast_tactile: dict | None = None
    action_valid: Any = None
    fast_start: Any = None

    @classmethod
    def from_dict(cls, data):
        slow = base_model.Observation.from_dict(data)
        return cls(**{f.name: getattr(slow, f.name) for f in dataclasses.fields(slow)},
                   fast_tactile=data.get("fast_tactile"), action_valid=data.get("action_valid"),
                   fast_start=data.get("fast_start"))


@dataclasses.dataclass(frozen=True)
class Pi05FastSlowConfig(Pi0Config):
    pi05: bool = True
    tactile_enabled: bool = True
    tactile_fusion: str = "action_adapter"
    num_flow_steps: int = 10
    slow_steps: int = 6
    max_delay_frames: int = 8
    slow_loss_weight: float = 1.0
    fast_loss_weight: float = 1.0
    validation_fraction: float = 0.05
    split_seed: int = 42
    fps: float = 29.994

    def __post_init__(self):
        super().__post_init__()
        if not self.pi05 or not self.tactile_enabled or self.tactile_fusion != "action_adapter":
            raise ValueError("Fast/slow requires pi05 with the tactile action adapter")
        if self.continuous_state_input or not self.discrete_state_input:
            raise ValueError("This checkpoint-compatible version refreshes discrete state in the slow branch")
        if not 0 < self.slow_steps < self.num_flow_steps:
            raise ValueError("Require 0 < slow_steps < num_flow_steps")
        if not 0 <= self.max_delay_frames < self.action_horizon:
            raise ValueError("Delay must fit inside the action chunk")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("An episode-level validation split is required")
        if self.slow_loss_weight <= 0 or self.fast_loss_weight <= 0:
            raise ValueError("Both branches need a positive training weight")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Training fps must be finite and positive")
        native_o10 = getattr(self, 'tactile_layout', None) == 'o10_268' and self.tactile_num_sensors == 1
        if self.tactile_num_frames != 16 or (self.tactile_num_sensors != 11 and not native_o10):
            raise ValueError("Expected 16 frames and either native O10 or the legacy 11-sensor layout")

    def create(self, rng):
        return Pi05FastSlow(self, nnx.Rngs(rng))

    def runtime_metadata(self):
        return {"version": 1, "num_flow_steps": self.num_flow_steps, "slow_steps": self.slow_steps,
                "max_delay_frames": self.max_delay_frames, "action_horizon": self.action_horizon,
                "action_dim": self.action_dim, "loss_action_dims": list(self.loss_action_dims or ()),
                "validation_fraction": self.validation_fraction, "split_seed": self.split_seed,
                "state_refresh": "slow", "fps": self.fps}


class Pi05FastSlow(Pi0):
    def __init__(self, config, rngs):
        super().__init__(config, rngs)
        # Static metadata only: the NNX parameter tree is exactly the old pi05 tree.
        self.flow_steps = config.num_flow_steps
        self.slow_steps = config.slow_steps
        self.slow_loss_weight = config.slow_loss_weight
        self.fast_loss_weight = config.fast_loss_weight

    def cached_velocity(self, snapshot, observation, x_t, time, condition):
        batch_size = x_t.shape[0]
        tokens, suffix_mask, ar_mask, adarms = self.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, (batch_size,)), condition)
        prefix_attn = jnp.broadcast_to(snapshot.prefix_mask[:, None, :],
                                     (batch_size, tokens.shape[1], snapshot.prefix_mask.shape[1]))
        mask = jnp.concatenate([prefix_attn, make_attn_mask(suffix_mask, ar_mask)], axis=-1)
        positions = jnp.sum(snapshot.prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, output), _ = self.PaliGemma.llm(
            [None, tokens], mask=mask, positions=positions, kv_cache=snapshot.kv_cache,
            adarms_cond=[None, adarms])
        return self.action_out_proj(output[:, -self.action_horizon:])

    def _denoise(self, snapshot, observation, condition, steps, progress=None):
        dt = -1.0 / self.flow_steps

        def step(_, carry):
            x_t, time = carry
            velocity = self.cached_velocity(snapshot, observation, x_t, time, condition)
            next_x = x_t + dt * velocity
            if progress is not None:
                future = jnp.arange(self.action_horizon)[None, :] >= progress[:, None]
                next_x = jnp.where(future[..., None], next_x, snapshot.x_split)
            return next_x, time + dt

        return jax.lax.fori_loop(0, steps, step, (snapshot.x_split, snapshot.time))

    def slow_snapshot(self, rng, observation, *, noise=None):
        observation = base_model.preprocess_observation(None, observation, train=False)
        if noise is None:
            noise = jax.random.normal(rng, (observation.state.shape[0], self.action_horizon, self.action_dim))
        tokens, mask, ar_mask = self.embed_prefix(observation)
        _, cache = self.PaliGemma.llm([tokens, None], mask=make_attn_mask(mask, ar_mask),
                                    positions=jnp.cumsum(mask, axis=1)-1)
        snapshot = SlowSnapshot(cache, mask, noise, jnp.asarray(1.0, jnp.float32))
        condition = self.encode_tactile_condition(observation)
        x_split, time = self._denoise(snapshot, observation, condition, self.slow_steps)
        return snapshot.replace(x_split=x_split, time=time)

    def fast_refine(self, snapshot, observation, progress=None):
        # No image encoder, language tokenization or prefix forward here.
        condition = self.encode_tactile_condition(observation)
        actions, _ = self._denoise(snapshot, observation, condition, self.flow_steps-self.slow_steps, progress)
        gate = condition[3] if condition is not None else jnp.zeros(actions.shape[0])
        return actions, gate

    def compute_loss_and_metrics(self, rng, observation, actions, *, train=False):
        if not isinstance(observation, FastSlowObservation) or observation.fast_tactile is None:
            if train:
                raise ValueError("Fast/slow training requires paired delayed tactile observations")
            return super().compute_loss_and_metrics(rng, observation, actions, train=False)
        slow_rng, rollout_rng, time_rng = jax.random.split(rng, 3)
        slow_loss, metrics = super().compute_loss_and_metrics(slow_rng, observation, actions, train=train)
        snapshot = jax.tree.map(jax.lax.stop_gradient, self.slow_snapshot(rollout_rng, observation))
        return self._paired_loss(slow_loss, metrics, snapshot, time_rng, observation, actions)

    def _paired_loss(self, slow_loss, metrics, snapshot, time_rng, observation, actions):
        valid = observation.action_valid
        horizon = actions.shape[1]
        # Keep padding out of the slow loss too; each sample contributes equally.
        slow_loss = jnp.where(valid, slow_loss, 0) * horizon / jnp.maximum(valid.sum(-1, keepdims=True), 1)
        fast_obs = dataclasses.replace(observation, **observation.fast_tactile)
        condition = self.encode_tactile_condition(fast_obs)
        # A flow bridge from the actual slow rollout endpoint to the demonstrated
        # chunk, using the remaining flow time. No teacher-forced slow cache.
        alpha = jax.random.uniform(time_rng, (actions.shape[0],), minval=0.001, maxval=1.0)
        if self.loss_action_dims is None:
            dims = jnp.arange(self.action_dim) < self.loss_action_dim
        else:
            dims = jnp.isin(jnp.arange(self.action_dim), jnp.array(self.loss_action_dims))
        # Inactive dimensions have no target in this checkpoint. Keep them at the
        # rollout endpoint instead of injecting unavailable demonstration values.
        future = valid & (jnp.arange(horizon)[None, :] >= observation.fast_start[:, None])
        target_actions = jnp.where(dims & future[..., None], actions, snapshot.x_split)
        x_t = alpha[:, None, None]*snapshot.x_split + (1-alpha[:, None, None])*target_actions
        target = (snapshot.x_split-target_actions) / snapshot.time
        velocity = self.cached_velocity(snapshot, fast_obs, x_t, alpha*snapshot.time, condition)
        errors = jnp.where(dims, (velocity.astype(jnp.float32)-target)**2, 0).sum(-1) / dims.sum()
        fast_loss = jnp.where(future, errors, 0) * horizon / jnp.maximum(future.sum(-1, keepdims=True), 1)
        _, tactile_valid, logits, gate = condition
        known = fast_obs.tactile_contact_valid & jnp.any(tactile_valid, axis=-1)
        bce = jax.nn.softplus(logits)-logits*fast_obs.tactile_contact
        contact = jnp.where(known, bce, 0).sum() / jnp.maximum(known.sum(), 1)
        metrics = {**metrics, "slow_loss": slow_loss.mean(), "fast_loss": fast_loss.mean(),
                   "fast_contact_loss": contact, "fast_gate": gate.mean(),
                   "fast_start": observation.fast_start.astype(jnp.float32).mean(),
                   "future_fraction": future.astype(jnp.float32).mean()}
        loss = self.slow_loss_weight*slow_loss + self.fast_loss_weight*(fast_loss+self.contact_loss_weight*contact)
        return loss, metrics
