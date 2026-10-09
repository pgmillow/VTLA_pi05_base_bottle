import importlib.util
import dataclasses
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import nnx

from openpi.models import model as base_model
from openpi.training import config
from openpi.training import weight_loaders
from openpi.training import sharding
from openpi.training.utils import TrainState


class ToyModel(base_model.BaseModel):
    def __init__(self):
        super().__init__(2, 1, 1)
        self.w = nnx.Param(jnp.array([[0.2, 0.4], [-0.3, 0.1]]))

    def compute_loss(self, rng, observation, actions, *, train=False):
        del rng, train
        prediction = observation.state @ self.w.value
        return jnp.mean((prediction[:, None, :] - actions) ** 2, axis=-1)

    def sample_actions(self, rng, observation, **kwargs):
        del rng, kwargs
        return (observation.state @ self.w.value)[:, None, :]


class NoisyToyModel(ToyModel):
    def compute_loss(self, rng, observation, actions, *, train=False):
        del train
        prediction = (observation.state @ self.w.value)[:, None, :]
        return ((prediction + 0.1 * jax.random.normal(rng, actions.shape) - actions) ** 2).mean(-1)


@pytest.fixture(scope="module")
def train_module():
    path = Path(os.environ.get("FASTSLOW_TEST_TRAIN_SCRIPT", Path(__file__).parents[2] / "scripts/train.py"))
    spec = importlib.util.spec_from_file_location("train_accumulation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_accumulation_matches_full_batch_and_one_clipped_adam_update(train_module):
    weights = {"w": jnp.array([0.3, -0.8])}
    obs = {"x": jnp.arange(128, dtype=jnp.float32).reshape(64, 2) / 20, "optional": None}
    targets = jnp.sin(jnp.arange(64, dtype=jnp.float32))

    def loss_fn(params, data, target):
        mse = jnp.mean((data["x"] @ params["w"] - target) ** 2)
        return mse, {"mse": mse}

    def grad_fn(key, data, target):
        del key
        return jax.value_and_grad(loss_fn, has_aux=True)(weights, data, target)

    full = grad_fn(None, obs, targets)
    accum = jax.jit(lambda x, y: train_module._accumulate_gradients(grad_fn, jax.random.key(1), x, y, 8))(obs, targets)
    for x, y in zip(jax.tree.leaves(full), jax.tree.leaves(accum), strict=True):
        np.testing.assert_allclose(x, y, rtol=2e-6, atol=1e-6)
    tx = optax.chain(optax.clip_by_global_norm(1), optax.adamw(1e-3))
    state = tx.init(weights)
    for x, y in zip(jax.tree.leaves(tx.update(full[1], state, weights)),
                    jax.tree.leaves(tx.update(accum[1], state, weights)), strict=True):
        np.testing.assert_allclose(x, y, rtol=2e-6, atol=1e-6)


def test_nan_in_any_microbatch_is_not_hidden(train_module):
    def grad_fn(key, obs, actions):
        del key, obs
        val = actions.mean()
        return (val, {"metric": val}), {"w": val}

    values = jnp.ones(64).at[33].set(jnp.nan)
    result = train_module._accumulate_gradients(grad_fn, jax.random.key(0), values, values, 8)
    assert all(np.isnan(x) for x in jax.tree.leaves(result))


def test_nnx_train_step_accumulates_then_updates_once(train_module):
    model = ToyModel()
    params = nnx.state(model)
    tx = optax.chain(optax.clip_by_global_norm(1), optax.adamw(1e-3))
    state = TrainState(step=jnp.array(0), params=params, model_def=nnx.graphdef(model),
                       opt_state=tx.init(params), tx=tx, ema_decay=None)
    obs = base_model.Observation(images={}, image_masks={}, state=jnp.arange(16, dtype=jnp.float32).reshape(8, 2))
    batch = obs, jnp.zeros((8, 1, 2))
    cfg = dataclasses.replace(config.get_config("debug"), batch_size=8,
                              skip_update_max_loss=float("inf"), skip_update_max_grad_norm=float("inf"))
    results = []
    for steps in [1, 4]:
        c = dataclasses.replace(cfg, gradient_accumulation_steps=steps)
        results.append(jax.jit(lambda s, b: train_module.train_step(c, jax.random.key(0), s, b))(state, batch))
    for x, y in zip(jax.tree.leaves(results[0]), jax.tree.leaves(results[1]), strict=True):
        np.testing.assert_allclose(x, y, rtol=3e-6, atol=1e-6)
    assert int(results[1][0].step) == 1


def test_single_microbatch_keeps_rng_and_values(train_module):
    def grad_fn(key, obs, acts):
        v = jax.random.normal(key) + obs.mean() + acts.mean()
        return (v, {}), {"w": v}

    key, data = jax.random.key(0), jnp.ones(8)
    expected = grad_fn(key, data, data)
    actual = train_module._accumulate_gradients(grad_fn, key, data, data, 1)
    for a, b in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("steps", [1, 4])
def test_separate_gpu_executables_match_scan_rng_and_update(train_module, steps):
    model = NoisyToyModel()
    params = nnx.state(model)
    tx = optax.chain(optax.clip_by_global_norm(1), optax.adamw(1e-3))
    state = TrainState(step=jnp.array(7), params=params, model_def=nnx.graphdef(model),
                       opt_state=tx.init(params), tx=tx, ema_decay=None)
    obs = base_model.Observation(images={}, image_masks={}, state=jnp.arange(16, dtype=jnp.float32).reshape(8, 2))
    batch = obs, jnp.ones((8, 1, 2))
    cfg = dataclasses.replace(config.get_config("debug"), batch_size=8,
                              gradient_accumulation_steps=steps, donate_train_buffers=False,
                              skip_update_max_loss=float("inf"), skip_update_max_grad_norm=float("inf"))
    mesh = sharding.make_mesh(1)
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    state_sharding = sharding.fsdp_sharding(state, mesh)
    key = jax.random.key(31)
    with sharding.set_mesh(mesh):
        expected = jax.jit(lambda s, b: train_module.train_step(cfg, key, s, b))(state, batch)
        actual = train_module.make_accumulated_train_step(cfg, state_sharding, data, replicated)(key, state, batch)
    for x, y in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
        np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-6)
    assert int(actual[0].step) == 8
    assert float(actual[1]["skipped_update"]) == 0


def test_mixed_loader_uses_only_designated_sources(monkeypatch):
    old = {"PaliGemma": {"w": np.array([1.], np.float32)}, "action_out_proj": {"w": np.array([2.], np.float32)}}
    touch = {"PaliGemma": {"w": np.array([999.], np.float32)},
             "tactile_encoder": {"w": np.array([3.], np.float32), "bias": None},
             "tactile_adapter": {"w": np.array([4.], np.float32)}}
    ref = {**old, "tactile_encoder": dict(touch["tactile_encoder"]), "tactile_adapter": dict(touch["tactile_adapter"])}
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda p: p)
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda p, **kw: {"old": old, "touch": touch}[p])
    loader = weight_loaders.BackboneTactileWeightLoader("old", "touch")
    merged = loader.load(ref)
    assert merged["PaliGemma"]["w"][0] == 1
    assert merged["action_out_proj"]["w"][0] == 2
    assert merged["tactile_encoder"]["w"][0] == 3
    assert merged["tactile_encoder"]["bias"] is None
    del touch["tactile_adapter"]["w"]
    with pytest.raises(ValueError, match="Missing tactile parameter"):
        loader.load(ref)


def test_experiment_counts_effective_batches_and_optimizer_steps():
    cfg = config.get_config("pi05_righthand_ee_fastslow_46k_b64")
    assert cfg.batch_size == 64 and cfg.gradient_accumulation_steps == 8
    assert cfg.num_train_steps == 30000 and not cfg.resume and not cfg.overwrite
    assert cfg.lr_schedule.decay_steps == 30000
    assert cfg.lr_schedule.decay_lr == 2e-6
