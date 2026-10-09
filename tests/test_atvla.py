import dataclasses

import flax.linen as nn
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import pi0, pi0_config
from openpi.models.tactile_adapter import TactileActionAdapter
from openpi.models.tactile_contact import TactileContactPreprocessor, calibrate_baseline, contact_fields, spatial_mask
from openpi.models.tactile_encoder import TactileEncoder, TactileEncoderConfig


def test_causal_calibration_sustained_contact_and_missing_data():
    processor = TactileContactPreprocessor()
    baseline = np.full((11, 14, 14), 100, np.float32)
    valid = np.ones(11, bool)
    for _ in range(15):
        result = processor.update(baseline, valid)
        assert not result["tactile_valid"].any()
        assert not result["tactile_contact_valid"]
        offline = contact_fields(np.zeros((11, 16, 14, 14)), np.zeros((11, 16), bool),
                                 baseline, np.full(11, 5.0), valid, calibrated=False)
        for key in result:
            np.testing.assert_array_equal(result[key], offline[key])
    result = processor.update(baseline, valid)
    assert result["tactile_contact_valid"] and result["tactile_contact"] == 0
    for _ in range(32):
        result = processor.update(baseline + 10 * spatial_mask(), valid)
    assert result["tactile_contact_valid"] and result["tactile_contact"] == 1
    # Sustained contact must not become the new baseline.
    np.testing.assert_array_equal(result["tactile_baseline"], baseline * spatial_mask())
    result = processor.update(baseline, np.zeros(11, bool))
    assert not result["tactile_valid"].any()
    assert not result["tactile_contact_valid"]
    processor.reset()
    assert not processor.update(baseline, valid)["tactile_valid"].any()


def test_calibration_ignores_padding_and_nonfinite_samples():
    mask = spatial_mask()
    frames = np.full((16, 11, 14, 14), 20, np.float32)
    frames[:, ~mask] = np.nan
    valid = np.ones((16, 11), bool)
    baseline, scale, usable = calibrate_baseline(frames, valid)
    assert usable.all() and np.isfinite(baseline).all()
    assert np.all(scale == 1)
    history = np.swapaxes(frames, 0, 1)
    history[0, -1, 0, 0] = np.nan
    result = contact_fields(history, valid.T, baseline, scale, usable)
    assert not result["tactile_valid"][0]
    assert not result["tactile_contact_valid"]  # incomplete data is not a negative label
    assert np.isfinite(result["tactile"]).all()


def test_fixed_baseline_keeps_pressure_and_masks_invalid_history():
    encoder = TactileEncoder(TactileEncoderConfig(preprocessing="episode_baseline"), nnx.Rngs(0))
    raw = jnp.full((1, 11, 16, 14, 14), 110.0)
    baseline = jnp.full((1, 11, 14, 14), 100.0)
    valid = jnp.ones((1, 11, 16), bool).at[:, :, 0].set(False)
    normalized = encoder._preprocess(raw, baseline, jnp.ones((1, 11)), valid)
    assert float(normalized[0, 0, -1, 0, 0]) > 2
    assert float(jnp.max(jnp.abs(normalized[:, :, 0]))) == 0
    assert float(jnp.max(jnp.abs(normalized[:, :, :, :, 8:]))) == 0


def test_adapter_zero_gate_missing_sensors_and_trainable_initialization():
    adapter = TactileActionAdapter(32, 64, nnx.Rngs(1))
    actions = jax.random.normal(jax.random.key(2), (2, 3, 64))
    tokens = jax.random.normal(jax.random.key(3), (2, 11, 32))
    valid = jnp.ones((2, 11), bool)
    logits, gate = adapter.contact(tokens, valid, jnp.zeros((2, 4)))
    np.testing.assert_array_equal(adapter(actions, tokens, valid, gate), actions)

    def loss(m):
        logits, g = m.contact(tokens, valid, jnp.zeros((2, 4)))
        return jnp.mean((m(actions, tokens, valid, g) - 1) ** 2) + jnp.mean(jax.nn.softplus(-logits))

    grads = nnx.grad(loss)(adapter)
    assert float(jnp.linalg.norm(grads.output.kernel.value)) > 0
    assert float(jnp.linalg.norm(grads.contact_out.kernel.value)) > 0
    # Simulate a learned adapter; disabled/missing tactile must still be exact identity.
    adapter.output.kernel.value = jnp.ones_like(adapter.output.kernel.value) * 0.01
    np.testing.assert_array_equal(adapter(actions, tokens, valid, jnp.zeros(2)), actions)
    np.testing.assert_array_equal(adapter(actions, tokens, ~valid, jnp.ones(2)), actions)
    assert not np.allclose(adapter(actions, tokens, valid, jnp.ones(2)), actions)


class TinyVision(nn.Module):
    @nn.compact
    def __call__(self, images, train=False):
        return jnp.zeros((images.shape[0], 2, 64)), None


def make_tiny_model(monkeypatch):
    # Real Gemma/flow-matching code, tiny experts; avoid downloading or allocating the big vision backbone.
    monkeypatch.setattr(pi0._siglip, "Module", lambda **kwargs: TinyVision())
    cfg = pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy",
                              dtype="float32", action_horizon=3, tactile_enabled=True,
                              tactile_embed_dim=32, tactile_fusion="action_adapter",
                              image_aug_crop_enabled=False, image_aug_prob=0.0)
    model = cfg.create(jax.random.key(0))
    obs = cfg.fake_obs(batch_size=1)
    obs = dataclasses.replace(obs,
        tactile=jnp.full((1, 11, 16, 14, 14), 10.0),
        tactile_valid=jnp.ones((1, 11), bool), tactile_sample_valid=jnp.ones((1, 11, 16), bool),
        tactile_frame_valid=jnp.ones((1, 16), bool), tactile_baseline=jnp.zeros((1, 11, 14, 14)),
        tactile_scale=jnp.ones((1, 11)), tactile_contact=jnp.ones(1), tactile_contact_valid=jnp.ones(1, bool))
    return model, obs


def test_pi05_loss_aux_gradients_and_sampling(monkeypatch):
    model, obs = make_tiny_model(monkeypatch)
    actions = jnp.zeros((1, 3, 32))
    loss, metrics = model.compute_loss_and_metrics(jax.random.key(3), obs, actions)
    assert loss.shape == (1, 3) and np.isfinite(loss).all()
    assert float(metrics["contact_loss"]) > 0
    assert float(metrics["contact_label_fraction"]) == 1
    absent = dataclasses.replace(obs, tactile=None)
    # No added tokens/positions in the visual-language prefix.
    for left, right in zip(model.embed_prefix(obs), model.embed_prefix(absent), strict=True):
        np.testing.assert_array_equal(left, right)
    condition = model.encode_tactile_condition(obs)
    off_condition = (*condition[:3], jnp.zeros_like(condition[3]))
    base_suffix = model.embed_suffix(obs, actions, jnp.ones(1))
    gated_suffix = model.embed_suffix(obs, actions, jnp.ones(1), off_condition)
    for left, right in zip(base_suffix, gated_suffix, strict=True):
        np.testing.assert_array_equal(left, right)
    # Gate supervision masks must not silently turn missing data into negative labels.
    unknown = dataclasses.replace(obs, tactile_contact_valid=jnp.zeros(1, bool))
    _, unknown_metrics = model.compute_loss_and_metrics(jax.random.key(3), unknown, actions)
    assert float(unknown_metrics["contact_loss"]) == 0
    sample = model.sample_actions(jax.random.key(4), obs, num_steps=2)
    assert sample.shape == actions.shape and np.isfinite(sample).all()


def test_new_training_config_isolated_and_protects_existing_checkpoints():
    from openpi.training.config import get_config
    new = get_config("pi05_righthand_ee_atvla")
    old = get_config("pi05_righthand_ee_tactile")
    assert new.model.tactile_fusion == "action_adapter"
    assert old.model.tactile_fusion == "prefix"
    assert new.data.contact_supervision and new.data.baseline_frames == 16
    assert not new.overwrite and not new.resume and new.skip_update_guard_enabled
    assert new.checkpoint_base_dir != old.checkpoint_base_dir
    assert new.trainable_filter(("tactile_adapter", "query", "kernel"), nnx.Param(jnp.ones(1)))
    assert not new.trainable_filter(("PaliGemma", "llm", "kernel"), nnx.Param(jnp.ones(1)))


def test_training_step_updates_adapter_and_preserves_frozen_backbone(monkeypatch):
    import importlib.util
    from pathlib import Path
    import optax
    from openpi.training.config import get_config
    from openpi.training.utils import TrainState

    spec = importlib.util.spec_from_file_location("atvla_train", Path(__file__).parents[1] / "scripts/train.py")
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    model, obs = make_tiny_model(monkeypatch)
    config = get_config("pi05_righthand_ee_atvla")
    graph, params = nnx.split(model)
    tx = optax.adam(1e-4)
    state = TrainState(step=jnp.asarray(0), params=params, model_def=graph,
                       opt_state=tx.init(params.filter(config.trainable_filter)), tx=tx, ema_decay=None)
    updated, info = train.train_step(config, jax.random.key(7), state, (obs, jnp.zeros((1, 3, 32))))
    assert int(updated.step) == 1 and float(info["skipped_update"]) == 0
    assert float(info["grad_norm"]) > 0 and float(info["contact_loss"]) > 0
    old_frozen = params.filter(config.freeze_filter)
    new_frozen = updated.params.filter(config.freeze_filter)
    for old, new in zip(jax.tree.leaves(old_frozen), jax.tree.leaves(new_frozen), strict=True):
        np.testing.assert_array_equal(old, new)
    assert not np.array_equal(params["tactile_adapter"]["output"]["kernel"].value,
                              updated.params["tactile_adapter"]["output"]["kernel"].value)
