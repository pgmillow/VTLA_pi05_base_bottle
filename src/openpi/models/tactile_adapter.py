"""Contact-supervised, action-side residual adapter (AT-VLA-inspired, not a port)."""

import flax.nnx as nnx
import jax
import jax.numpy as jnp


class TactileActionAdapter(nnx.Module):
    def __init__(self, tactile_dim, action_dim, rngs, *, hidden_dim=128, num_heads=4):
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.contact_in = nnx.Linear(tactile_dim + 4, hidden_dim, rngs=rngs)
        self.contact_out = nnx.Linear(hidden_dim, 1, rngs=rngs)
        self.query = nnx.Linear(action_dim, hidden_dim, rngs=rngs)
        self.key = nnx.Linear(tactile_dim, hidden_dim, rngs=rngs)
        self.value = nnx.Linear(tactile_dim, hidden_dim, rngs=rngs)
        # Zero only the residual output; do not also zero the multiplicative gate.
        self.output = nnx.Linear(hidden_dim, action_dim, kernel_init=jax.nn.initializers.zeros,
                                 bias_init=jax.nn.initializers.zeros, rngs=rngs)

    def contact(self, tokens, valid, pressure_stats):
        weight = valid[..., None].astype(tokens.dtype)
        pooled = jnp.sum(tokens * weight, axis=1) / jnp.maximum(jnp.sum(weight, axis=1), 1)
        features = jnp.concatenate([pooled, pressure_stats], axis=-1)
        logits = self.contact_out(jax.nn.gelu(self.contact_in(features)))[..., 0]
        gate = jax.nn.sigmoid(logits) * jnp.any(valid, axis=-1)
        return logits, gate

    def __call__(self, actions, tokens, valid, gate):
        b, horizon, _ = actions.shape
        sensors = tokens.shape[1]
        h, d = self.num_heads, self.head_dim
        q = self.query(actions).reshape(b, horizon, h, d).transpose(0, 2, 1, 3)
        k = self.key(tokens).reshape(b, sensors, h, d).transpose(0, 2, 1, 3)
        v = self.value(tokens).reshape(b, sensors, h, d).transpose(0, 2, 1, 3)
        scores = jnp.einsum("bhqd,bhkd->bhqk", q, k) * d**-0.5
        scores = jnp.where(valid[:, None, None, :], scores, -1e9)
        weights = jax.nn.softmax(scores.astype(jnp.float32), axis=-1)
        weights = weights * valid[:, None, None, :]
        weights = weights / jnp.maximum(weights.sum(axis=-1, keepdims=True), 1e-9)
        residual = jnp.einsum("bhqk,bhkd->bhqd", weights, v)
        residual = residual.transpose(0, 2, 1, 3).reshape(b, horizon, h * d)
        gain = (gate * jnp.any(valid, axis=-1))[:, None, None]
        return actions + (gain * self.output(residual)).astype(actions.dtype)
