"""Reuse pi05's visual/language prefix for fast/slow training.

The real six-step slow rollout and delayed-touch bridge objective are preserved.
Only the redundant image encoder/prefix forward and old-touch encoding are
removed. With augmentation enabled, both branches now share the same augmented
slow observation. Deployment still uses the existing six/four split sampler.
"""
import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp

from openpi.fastslow.model import FastSlowObservation, Pi05FastSlow, Pi05FastSlowConfig, SlowSnapshot
from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class Pi05VTLAConfig(Pi05FastSlowConfig):
    use_frame_cache: bool = True
    tactile_layout: str = 'inspire11'
    reuse_training_prefix: bool = True
    training_objective: str = 'paired'

    def __post_init__(self):
        super().__post_init__()
        if self.training_objective not in ('flow', 'paired'):
            raise ValueError('Unknown VTLA training objective')
        if self.tactile_layout not in ('inspire11','o10_268'):
            raise ValueError('Unknown tactile layout')
        if self.tactile_layout == 'o10_268' and self.tactile_num_sensors != 1:
            raise ValueError('Native O10 must use one 268-channel vector')

    def inputs_spec(self, *, batch_size=1):
        observation, actions = super().inputs_spec(batch_size=batch_size)
        if self.tactile_layout == 'o10_268':
            # Abstract initialization uses shape structs rather than arrays,
            # matching Pi0Config.inputs_spec's typecheck scope.
            with at.disable_typechecking():
                observation = dataclasses.replace(observation,
                    tactile=jax.ShapeDtypeStruct((batch_size,1,16,1,268),jnp.float32),
                    tactile_baseline=jax.ShapeDtypeStruct((batch_size,1,1,268),jnp.float32))
        return observation, actions

    def runtime_metadata(self):
        return {**super().runtime_metadata(), 'tactile_layout':self.tactile_layout,
                'training_objective':self.training_objective}

    def create(self, rng):
        return Pi05VTLA(self, nnx.Rngs(rng))


class Pi05VTLA(Pi05FastSlow):
    def __init__(self,config,rngs):
        super().__init__(config,rngs)
        self.tactile_layout = config.tactile_layout
        self.reuse_training_prefix = config.reuse_training_prefix
        self.training_objective = config.training_objective

    def encode_tactile_condition(self,obs):
        if self.tactile_layout != 'o10_268':
            return super().encode_tactile_condition(obs)
        if obs.tactile is None:
            raise ValueError('Native O10 pressure history is required')
        valid = obs.tactile_sample_valid[:,0,:] & obs.tactile_valid[:,0,None]
        tokens = self.tactile_encoder(obs.tactile,valid,obs.tactile_baseline,obs.tactile_scale)
        scale = jnp.maximum(obs.tactile_scale[:,0,None],1.)
        current = (obs.tactile[:,0,-1,0,:]-obs.tactile_baseline[:,0,0,:])/scale
        current = jnp.where(valid[:,-1,None],current,0.)
        delta = (obs.tactile[:,0,-1,0,:]-obs.tactile[:,0,-2,0,:])/scale
        delta = jnp.where((valid[:,-1]&valid[:,-2])[:,None],delta,0.)
        magnitude = jnp.abs(current)
        stats = jnp.stack([jnp.log1p(magnitude.mean(-1)),jnp.log1p(magnitude.max(-1)),
                           (magnitude>=3).mean(-1),jnp.log1p(jnp.abs(delta).mean(-1))],axis=-1)
        logits,gate = self.tactile_adapter.contact(tokens,valid,stats)
        return tokens,valid,logits,gate

    def compute_loss_and_metrics(self, rng, observation, actions, *, train=False):
        if self.training_objective == 'flow':
            from openpi.vtla.stages import masked_flow_loss
            return masked_flow_loss(self, rng, observation, actions, train=train)
        if not self.reuse_training_prefix:
            return super().compute_loss_and_metrics(rng,observation,actions,train=train)
        if not isinstance(observation, FastSlowObservation) or observation.fast_tactile is None:
            return super().compute_loss_and_metrics(rng, observation, actions, train=train)
        slow_rng, rollout_rng, time_rng = jax.random.split(rng, 3)
        slow_loss, metrics, context = self._flow_loss_with_context(slow_rng, observation, actions, train=train)
        prepared, prefix_mask, cache, condition = context
        # Joint pi05 attention returns prefix AND action KV. Only prefix KV may
        # be reused; suffix KV would leak the supervised noisy action into fast.
        prefix_length = prefix_mask.shape[1]
        cache = jax.tree.map(lambda x: jax.lax.stop_gradient(x[:, :, :prefix_length]), cache)
        condition = jax.tree.map(jax.lax.stop_gradient, condition)
        noise = jax.random.normal(rollout_rng, actions.shape)
        snapshot = SlowSnapshot(cache, prefix_mask, noise, jnp.asarray(1., jnp.float32))
        endpoint, split_time = self._denoise(snapshot, prepared, condition, self.slow_steps)
        snapshot = jax.tree.map(jax.lax.stop_gradient, snapshot.replace(x_split=endpoint, time=split_time))
        return self._paired_loss(slow_loss, metrics, snapshot, time_rng, observation, actions)
