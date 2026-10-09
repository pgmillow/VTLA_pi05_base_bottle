"""Check stage boundaries, gradient routing, and visual-only isolation."""
import dataclasses
import math
from types import SimpleNamespace

from flax import linen as nn, nnx, traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from openpi.fastslow.data import FastSlowDataset
from openpi.fastslow.model import FastSlowObservation
from openpi.models import pi0
from openpi.training.config import get_config
from openpi.training.weight_loaders import _merge_params
from openpi.vtla.model import Pi05VTLAConfig
from openpi.vtla.stages import Pi05VisualAdaptConfig, masked_flow_loss


def test_full_size_parameter_counts_and_stage_initialization():
    configs = [get_config(f'vtla_bottle_stage{i}_{name}')
               for i, name in [(1, 'visual'), (2, 'tactile'), (3, 'joint')]]
    counts = []
    for cfg in configs:
        state = nnx.state(nnx.eval_shape(cfg.model.create, jax.random.key(0)), nnx.Param)
        counts.append(sum(math.prod(x.shape) for x in jax.tree.leaves(state.filter(cfg.trainable_filter))))
        assert cfg.batch_size == 64 and cfg.gradient_accumulation_steps == 1
        assert not cfg.resume and not cfg.overwrite
        assert cfg.lr_schedule.decay_steps == cfg.num_train_steps
        assert cfg.lr_schedule.decay_lr == 2e-6
    assert counts == [3353433872, 632961, 3354066833]
    visual, touch, joint = configs
    assert not visual.model.tactile_enabled and not visual.data.include_tactile
    assert touch.model.training_objective == 'flow' and joint.model.training_objective == 'paired'
    assert touch.weight_loader.params_path == str(visual.checkpoint_dir / '19999' / 'params')
    assert joint.weight_loader.params_path == str(touch.checkpoint_dir / '5999' / 'params')
    assert sum(c.num_train_steps for c in configs) == 46000


class TinyVision(nn.Module):
    def __call__(self, images, train=False):
        return jnp.broadcast_to(images.mean((1, 2, 3))[:, None, None], (images.shape[0], 2, 64)), None


def test_visual_to_touch_weights_and_frozen_gradient_routing(monkeypatch):
    monkeypatch.setattr(pi0._siglip, 'Module', lambda **kwargs: TinyVision())
    args = dict(paligemma_variant='dummy', action_expert_variant='dummy', dtype='float32',
                action_horizon=4, max_token_len=8, image_aug_prob=0., image_aug_crop_enabled=False,
                loss_action_dims=tuple(range(17)))
    visual = Pi05VisualAdaptConfig(**args).create(jax.random.key(1))
    source = nnx.state(visual, nnx.Param).to_pure_dict()
    assert not any(k.startswith('tactile') for k in source)
    touch_cfg = Pi05VTLAConfig(**args, tactile_layout='o10_268', tactile_num_sensors=1,
                              tactile_embed_dim=32, training_objective='flow', max_delay_frames=2)
    touch = touch_cfg.create(jax.random.key(2))
    initial = nnx.state(touch, nnx.Param).to_pure_dict()
    merge = _merge_params(source, initial, missing_regex=get_config('vtla_bottle_stage2_tactile').weight_loader.missing_regex)
    touch = touch_cfg.load(merge)
    for path, value in traverse_util.flatten_dict(source).items():
        np.testing.assert_array_equal(value, traverse_util.flatten_dict(merge)[path])
    np.testing.assert_array_equal(touch.tactile_adapter.output.kernel.value, 0)
    obs = touch_cfg.fake_obs()
    obs = dataclasses.replace(obs, tactile_baseline=jnp.zeros_like(obs.tactile_baseline))
    obs = FastSlowObservation(**{f.name:getattr(obs,f.name) for f in dataclasses.fields(obs)},
                              action_valid=jnp.array([[True, True, True, False]]))
    monkeypatch.setattr(type(touch), '_paired_loss', lambda *a, **k: (_ for _ in ()).throw(AssertionError('Unexpected fast loss')))
    def objective(model):
        loss, metrics = model.compute_loss_and_metrics(jax.random.key(3), obs, jnp.zeros((1,4,32)), train=True)
        assert 'fast_loss' not in metrics
        return loss.mean()
    cfg = get_config('vtla_bottle_stage2_tactile')
    loss, grads = nnx.value_and_grad(objective, argnums=nnx.DiffState(0, cfg.trainable_filter))(touch)
    assert np.isfinite(float(loss))
    paths = traverse_util.flatten_dict(grads.to_pure_dict())
    assert paths and all(p[0] in ('tactile_encoder', 'tactile_adapter') for p in paths)
    assert any(np.any(np.asarray(g) != 0) for g in jax.tree.leaves(grads))
    joint_cfg = dataclasses.replace(touch_cfg, training_objective='paired')
    joint = joint_cfg.load(nnx.state(touch, nnx.Param).to_pure_dict())
    assert nnx.state(joint, nnx.Param).to_pure_dict().keys() == merge.keys()


def test_visual_dataset_never_reads_tactile_and_preserves_terminal_mask():
    class VisualOnlyBase:
        def __getitem__(self, index):
            return {'sample_frame_in_episode': 18, 'actions': np.zeros((4,17))}
        def _tactile_window(self, *a):
            raise AssertionError('Visual-only data accessed tactile values')
    ds = object.__new__(FastSlowDataset)
    ds.paired = False
    ds.base = VisualOnlyBase()
    ds.model_config = SimpleNamespace(action_horizon=4)
    ds.transform = lambda x:x
    ds.pair_indices = lambda index, delay: (18,18,20)
    sample = ds[0]
    assert 'fast_tactile' not in sample and 'fast_start' not in sample
    np.testing.assert_array_equal(sample['action_valid'], [True,True,False,False])


def test_terminal_mask_keeps_contact_loss_scale(monkeypatch):
    monkeypatch.setattr(pi0.Pi0, 'compute_loss_and_metrics',
                        lambda *a, **k:(jnp.array([[2.5,4.5,100.5,100.5]]), {'contact_loss':jnp.array(5.)}))
    obs = SimpleNamespace(action_valid=jnp.array([[True,True,False,False]]))
    loss, metrics = masked_flow_loss(SimpleNamespace(contact_loss_weight=.1), None, obs, jnp.zeros((1,4,32)), train=True)
    np.testing.assert_allclose(loss.mean(), 3.5)
    np.testing.assert_allclose(metrics['flow_loss'], 3.)
