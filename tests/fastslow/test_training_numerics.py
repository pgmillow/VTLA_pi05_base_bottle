import importlib.util
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import optax
from flax import nnx

from openpi.training import config
from openpi.training.utils import TrainState


@pytest.fixture(scope="module")
def train_module():
    path = Path(__file__).parents[2] / "scripts/train.py"
    spec = importlib.util.spec_from_file_location("train_numerics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("value", [np.nan, np.inf, 1e35])
def test_rejects_unsafe_checkpoint_before_initialization(train_module, value):
    loader = SimpleNamespace(load=lambda _: {"w": np.array([value], np.float32)})
    with pytest.raises(ValueError, match="Unsafe initial checkpoint parameter: w"):
        train_module._load_weights_and_validate(loader, {"w": jax.ShapeDtypeStruct((1,), np.float32)})


def test_healthy_checkpoint_and_optional_parameters(train_module):
    weights = {"w": np.array([0.25], np.float32), "bias": None,
               "new": jax.ShapeDtypeStruct((2,), np.float32)}
    loader = SimpleNamespace(load=lambda _: weights)
    result = train_module._load_weights_and_validate(loader, weights)
    assert list(result) == ["w"]
    np.testing.assert_array_equal(result["w"], weights["w"])


def test_detects_overflow_hidden_by_zero_logged_update_norm(train_module):
    info = {"param_norm": 100., "skipped_update": 0., "update_norm": 0.,
            "_diag_grad_leaf_norms": np.array([1.]), "_diag_update_leaf_norms": np.array([np.inf])}
    with pytest.raises(FloatingPointError, match="accepted update"):
        train_module._check_step_numerics(info)
    info["_diag_update_leaf_norms"] = np.array([0.01])
    train_module._check_step_numerics(info)
    info["param_norm"] = np.inf
    with pytest.raises(FloatingPointError, match="Parameter norm overflow"):
        train_module._check_step_numerics(info)


def test_fresh_adam_audit_rejects_nonzero_and_nonfinite_moments(train_module):
    zero = {"count": jnp.asarray(0), "mu": jnp.zeros((2, 3)), "nu": jnp.zeros((2, 3))}
    assert train_module._audit_fresh_adam_state(zero)["nonzero_paths"] == []
    bad = {**zero, "mu": zero["mu"].at[1, 2].set(1e-20), "nu": zero["nu"].at[0, 0].set(jnp.nan)}
    paths = train_module._audit_fresh_adam_state(bad)["nonzero_paths"]
    assert paths == ["['mu']", "['nu']"]


@pytest.mark.parametrize("skipped", [0., 1.])
def test_finite_corrupted_weights_stop_even_when_update_was_rejected(train_module, skipped):
    info = {"param_norm": 2.98e8, "param_max_abs": 1e8, "skipped_update": skipped,
            "_diag_grad_leaf_norms": np.array([1.]), "_diag_update_leaf_norms": np.array([0.])}
    with pytest.raises(FloatingPointError, match="Unsafe parameter magnitude"):
        train_module._check_step_numerics(info)


def test_device_audit_detects_large_finite_and_nonfinite_weights(train_module):
    params = SimpleNamespace(to_pure_dict=lambda: {
        "good": jnp.array([0.25]), "huge": jnp.array([1e35]),
        "bad": jnp.array([jnp.nan]), "optional": None})
    report = train_module._audit_device_parameters(params)
    assert report["arrays"] == 3
    assert {x["name"] for x in report["bad"]} == {"bad", "huge"}


def _small_state(tx=None):
    model = nnx.Linear(2, 2, rngs=nnx.Rngs(42))
    params = nnx.state(model)
    tx = tx or optax.chain(optax.clip_by_global_norm(1), optax.adamw(1e-3))
    return TrainState(step=jnp.asarray(0), params=params, model_def=nnx.graphdef(model),
                      opt_state=tx.init(params), tx=tx, ema_decay=0.9, ema_params=params)


def _assert_same_tree(a, b):
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1e30])
@pytest.mark.parametrize("thresholds_enabled", [True, False])
def test_bad_gradient_rejected_atomically_then_training_recovers(train_module, value, thresholds_enabled):
    cfg = dataclasses.replace(config.get_config("debug"),
                              skip_update_guard_enabled=thresholds_enabled,
                              skip_update_max_loss=float("inf"), skip_update_max_grad_norm=float("inf"))
    state = _small_state()
    apply = jax.jit(lambda s, g: train_module._apply_gradients(cfg, s, (jnp.asarray(0.2), {}), g))
    healthy = jax.tree.map(lambda x: jnp.full_like(x, 0.1), state.params)
    state, _ = apply(state, healthy)  # Nonzero Adam moments must also survive a rejection.
    bad = jax.tree.map(lambda x: jnp.full_like(x, value), state.params)
    rejected, info = apply(state, bad)
    assert float(info["skipped_update"]) == 1
    assert float(info["skip_nonfinite"]) == 1
    assert float(info["grad_norm_finite"]) == 0
    assert not np.isfinite(float(info["grad_norm"]))  # No misleading logged zero.
    for field in ("params", "opt_state", "ema_params"):
        _assert_same_tree(getattr(rejected, field), getattr(state, field))
    train_module._check_step_numerics(jax.device_get(info))  # Rejection is recoverable.
    # Attempt counters advance, while Adam's counter and LR do not advance on a skip.
    recovered, recovery_info = apply(rejected, healthy)
    reference, _ = apply(state, healthy)
    for field in ("params", "opt_state", "ema_params"):
        _assert_same_tree(getattr(recovered, field), getattr(reference, field))
    assert float(recovery_info["skipped_update"]) == 0


@pytest.mark.parametrize("value", [float("inf"), 1e30, 1e8])
def test_bad_optimizer_candidate_rejected_even_without_thresholds(train_module, value):
    def update(grads, opt_state, params=None):
        del params
        return jax.tree.map(lambda x: jnp.full_like(x, value), grads), opt_state + 1

    tx = optax.GradientTransformation(lambda _: jnp.asarray(0), update)
    state = _small_state(tx)
    cfg = dataclasses.replace(config.get_config("debug"), skip_update_guard_enabled=False)
    grads = jax.tree.map(lambda x: jnp.ones_like(x), state.params)
    result, info = jax.jit(lambda s, g: train_module._apply_gradients(cfg, s, (jnp.asarray(0.2), {}), g))(state, grads)
    assert float(info["skipped_update"]) == 1
    assert float(info["skip_bad_candidate"]) == 1
    for field in ("params", "opt_state", "ema_params"):
        _assert_same_tree(getattr(result, field), getattr(state, field))
    train_module._check_step_numerics(jax.device_get(info))
