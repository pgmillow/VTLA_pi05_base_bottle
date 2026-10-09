import dataclasses
import io
import json
from types import SimpleNamespace

from flax import linen as nn, nnx, traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import pytest

from openpi.fastslow.buffer import ActionBuffer, ActionUpdate
from openpi.fastslow.model import FastSlowObservation, Pi05FastSlowConfig
from openpi.fastslow.pressure import PressureStream
from openpi.models import pi0, pi0_config
from openpi.models.tactile_contact import spatial_mask
from openpi.shared.nnx_utils import module_jit


def test_vtla_shared_prefix_loss_and_gradients_match_reference(tiny, monkeypatch):
    from openpi.vtla.model import Pi05VTLA, Pi05VTLAConfig
    cfg, reference, obs = tiny
    optimized = Pi05VTLAConfig(**dataclasses.asdict(cfg)).load(nnx.state(reference).to_pure_dict())
    touch = {f.name:getattr(obs,f.name) for f in dataclasses.fields(obs) if f.name.startswith('tactile')}
    touch['tactile'] = touch['tactile']*2
    paired = FastSlowObservation(**{f.name:getattr(obs,f.name) for f in dataclasses.fields(obs)},
        fast_tactile=touch, action_valid=jnp.array([[True,True,True,False]]), fast_start=jnp.array([2]))
    actions = jnp.zeros((1,4,32))
    rng = jax.random.key(101)
    def objective(model):
        loss, metrics = model.compute_loss_and_metrics(rng,paired,actions,train=True)
        return loss.mean(), metrics
    reference_result, reference_grad = nnx.value_and_grad(objective,has_aux=True)(reference)
    calls = []
    original = Pi05VTLA.embed_prefix
    def counted(model, observation):
        calls.append(1)
        return original(model, observation)
    monkeypatch.setattr(Pi05VTLA,'embed_prefix',counted)
    optimized_result, optimized_grad = nnx.value_and_grad(objective,has_aux=True)(optimized)
    assert len(calls) == 1
    assert traverse_util.flatten_dict(nnx.state(reference).to_pure_dict()).keys() == traverse_util.flatten_dict(nnx.state(optimized).to_pure_dict()).keys()
    for a,b in zip(jax.tree.leaves(reference_result),jax.tree.leaves(optimized_result),strict=True):
        np.testing.assert_allclose(a,b,rtol=3e-4,atol=3e-5)
    for a,b in zip(jax.tree.leaves(reference_grad),jax.tree.leaves(optimized_grad),strict=True):
        np.testing.assert_allclose(a,b,rtol=3e-3,atol=3e-5)


def test_vtla_initialization_and_hyperparameters():
    from openpi.training.config import get_config
    new, old = get_config('vtla_pi05_base_bottle'), get_config('pi05_righthand_ee_0806')
    for field in ('batch_size','num_workers','num_train_steps','log_interval','save_interval','keep_period','fsdp_devices','ema_decay'):
        assert getattr(new,field) == getattr(old,field)
    assert new.lr_schedule == old.lr_schedule and new.optimizer == old.optimizer
    assert new.gradient_accumulation_steps == 1 and not new.resume
    assert new.separate_gradient_update and not new.donate_train_buffers
    assert new.weight_loader.params_path.endswith('/pi05_base/params')
    assert new.weight_loader.missing_regex == r'(tactile_encoder|tactile_adapter)/.*'


class TinyVision(nn.Module):
    def __call__(self, images, train=False):
        return jnp.broadcast_to(images.mean((1, 2, 3))[:, None, None], (images.shape[0], 2, 64)), None


@pytest.fixture
def tiny(monkeypatch):
    monkeypatch.setattr(pi0._siglip, "Module", lambda **kwargs: TinyVision())
    cfg = Pi05FastSlowConfig(paligemma_variant="dummy", action_expert_variant="dummy", dtype="float32",
                            action_horizon=4, max_token_len=8, max_delay_frames=2, num_flow_steps=4, slow_steps=2,
                            tactile_embed_dim=32, image_aug_prob=0., image_aug_crop_enabled=False,
                            loss_action_dims=(6, 7, 8, 9, 10, 11, 13))
    model = cfg.create(jax.random.key(0))
    # The trained adapter has a nonzero residual projection; make that explicit.
    model.tactile_adapter.output.kernel.value = jnp.full_like(model.tactile_adapter.output.kernel.value, 0.03)
    obs = cfg.fake_obs()
    obs = dataclasses.replace(obs, tactile=obs.tactile*10, tactile_baseline=jnp.zeros_like(obs.tactile_baseline),
                              tokenized_prompt_mask=obs.tokenized_prompt_mask.at[:, -2:].set(False))
    return cfg, model, obs


def test_parameter_compatibility_split_equivalence_and_fresh_touch(tiny, monkeypatch):
    cfg, model, obs = tiny
    original_cfg = pi0_config.Pi0Config(**{f.name: getattr(cfg, f.name) for f in dataclasses.fields(pi0_config.Pi0Config)})
    original = original_cfg.create(jax.random.key(1))
    params = nnx.state(model).to_pure_dict()
    assert traverse_util.flatten_dict(params).keys() == traverse_util.flatten_dict(nnx.state(original).to_pure_dict()).keys()
    # Use the original load path, validating all shapes, not just key names.
    original = original_cfg.load(params)
    noise = jax.random.normal(jax.random.key(3), (1, 4, 32))
    reference = module_jit(original.sample_actions)(jax.random.key(2), obs, num_steps=4, noise=noise)
    snapshot = module_jit(model.slow_snapshot)(jax.random.key(2), obs, noise=noise)
    fast = module_jit(model.fast_refine)
    split, _ = fast(snapshot, obs, jnp.array([0]))
    np.testing.assert_allclose(split, reference, rtol=2e-5, atol=2e-5)
    cached_before = jax.tree.map(np.asarray, snapshot)
    changed = dataclasses.replace(obs, tactile=obs.tactile*0)
    refined, _ = fast(snapshot, changed, jnp.array([0]))
    assert np.max(np.abs(np.asarray(refined-split))) > 1e-5
    again, _ = fast(snapshot, obs, jnp.array([0]))
    np.testing.assert_array_equal(again, split)
    for before, after in zip(jax.tree.leaves(cached_before), jax.tree.leaves(snapshot), strict=True):
        np.testing.assert_array_equal(before, after)
    # Fast sampling cannot run vision or the language prefix again.
    monkeypatch.setattr(type(model), "embed_prefix", lambda *a: pytest.fail("fast branch rebuilt prefix"))
    fast_no_prefix = module_jit(model.fast_refine)
    late, _ = fast_no_prefix(snapshot, changed, jnp.array([2]))
    np.testing.assert_array_equal(late[:, :2], snapshot.x_split[:, :2])
    # Masked prefix cache slots cannot affect action predictions.
    padding = ~snapshot.prefix_mask[None, :, :, None, None]
    corrupt_cache = jax.tree.map(lambda x: jnp.where(padding, 12345., x), snapshot.kv_cache)
    masked, _ = fast_no_prefix(snapshot.replace(kv_cache=corrupt_cache), changed, jnp.array([2]))
    np.testing.assert_allclose(masked, late, rtol=1e-5, atol=1e-5)


def test_delayed_loss_gradients_and_training_update(tiny):
    import importlib.util
    from pathlib import Path
    import optax
    from openpi.training.config import get_config
    from openpi.training.utils import TrainState
    cfg, model, obs = tiny
    touch = {f.name: getattr(obs, f.name) for f in dataclasses.fields(obs) if f.name.startswith("tactile")}
    touch["tactile"] = touch["tactile"]*2
    batch_obs = FastSlowObservation(**{f.name: getattr(obs, f.name) for f in dataclasses.fields(obs)},
                                    fast_tactile=touch, action_valid=jnp.array([[True, True, True, False]]),
                                    fast_start=jnp.array([2]))
    config = dataclasses.replace(get_config("pi05_righthand_ee_fastslow"), model=cfg)
    path = Path(__file__).parents[2]/"scripts/train.py"
    spec = importlib.util.spec_from_file_location("fastslow_train", path)
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)
    graph, params = nnx.split(model)
    tx = optax.adam(1e-4)
    state = TrainState(step=jnp.asarray(0), params=params, model_def=graph,
                       opt_state=tx.init(params.filter(config.trainable_filter)), tx=tx, ema_decay=None)
    step = jax.jit(lambda s: train.train_step(config, jax.random.key(8), s, (batch_obs, jnp.zeros((1, 4, 32)))))
    updated, info = step(state)
    assert int(updated.step) == 1 and float(info["skipped_update"]) == 0
    for key in ("slow_loss", "fast_loss", "fast_contact_loss", "grad_norm"):
        assert np.isfinite(float(info[key]))
    assert float(info["fast_loss"]) > 0 and float(info["grad_norm"]) > 0
    assert float(info["future_fraction"]) == .25
    for old, new in zip(jax.tree.leaves(params.filter(config.freeze_filter)),
                        jax.tree.leaves(updated.params.filter(config.freeze_filter)), strict=True):
        np.testing.assert_array_equal(old, new)
    flat_before = traverse_util.flatten_dict(params.to_pure_dict())
    flat_after = traverse_util.flatten_dict(updated.params.to_pure_dict())
    assert any(not np.array_equal(v, flat_after[k]) for k, v in flat_before.items() if "_1" in "/".join(k))
    assert not np.array_equal(flat_before[("tactile_adapter", "output", "kernel")],
                              flat_after[("tactile_adapter", "output", "kernel")])


def test_pressure_resampling_calibration_and_no_future():
    stream = PressureStream(fps=30)
    raw = np.full((11, 14, 14), 100., np.float32)
    for i in range(61):
        stream.update(raw, np.ones(11, bool), i/120)
    fields, stamp = stream.snapshot(.5)
    assert stamp == .5 and fields["tactile_valid"].all()
    np.testing.assert_allclose(fields["tactile_baseline"], raw*spatial_mask())
    stream.update(raw+20*spatial_mask(), np.ones(11, bool), .51)
    old, stamp = stream.snapshot(.5)
    np.testing.assert_array_equal(fields["tactile"], old["tactile"])
    assert stamp == .5
    current, _ = stream.snapshot(.51)
    assert current["tactile_contact"] == 1
    stream.update(raw, np.zeros(11, bool), .7)
    missing, _ = stream.snapshot(.7)
    assert not missing["tactile_valid"].any()
    with pytest.raises(ValueError):
        stream.update(raw, np.ones(11, bool), .6)
    stream.reset()
    with pytest.raises(ValueError):
        stream.snapshot(1.)


def test_action_buffer_expiry_revisions_and_rate_limits():
    buf = ActionBuffer(7, np.full(7, .1), np.zeros(7), blend_steps=2)
    buf.reset("session", np.zeros(7))
    update = ActionUpdate("session", 1, 1, 0., .1, np.ones((30, 7)), 3, 30)
    assert buf.apply(update, .11)
    result = buf.next_action(4/30)
    assert result[1] == 4 and np.allclose(result[2], .1)
    assert buf.next_action(4/30) is None
    assert not buf.apply(update, .15)
    revision = dataclasses.replace(update, revision=2, sensor_time=.16, actions=np.full((30, 7), 10.))
    assert buf.apply(revision, .17)
    assert np.allclose(buf.next_action(.2)[2], .2)
    assert not buf.apply(dataclasses.replace(revision, revision=3, session_id="other"), .21)
    assert not buf.apply(dataclasses.replace(revision, revision=3), 2.)
    assert buf.next_action(2.) is None


def test_dataset_delay_masks_episode_split_and_causal_inputs(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from openpi.fastslow.data import FastSlowDataset
    from openpi import transforms
    (tmp_path/"meta").mkdir()
    (tmp_path/"data/chunk-000").mkdir(parents=True)
    (tmp_path/"meta/episodes.jsonl").write_text('\n'.join(json.dumps({"episode_index": e, "length": 24}) for e in range(4)))
    (tmp_path/"meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "pour"}))
    image = io.BytesIO()
    Image.fromarray(np.zeros((2, 2, 3), np.uint8)).save(image, format="PNG")
    for ep in range(4):
        rows = [{"observation.state": [float(t)]*26, "action": [float(t)]*14, "task_index": 0,
                 "observation.tactile.right": (np.full((11, 14, 14), t)*spatial_mask()).reshape(-1).tolist(),
                 "observation.tactile.right_valid": [True]*11,
                 **{f"observation.images.{k}": {"bytes": image.getvalue()} for k in ("top", "left_wrist", "right_wrist")}}
                for t in range(24)]
        pq.write_table(pa.Table.from_pylist(rows), tmp_path/f"data/chunk-000/episode_{ep:06d}.parquet")
    monkeypatch.setattr(transforms, "Normalize", lambda *a, **k: lambda x: x)
    dc = SimpleNamespace(repo_id=str(tmp_path), tactile_num_frames=16, tactile_key="observation.tactile.right",
        tactile_valid_key="observation.tactile.right_valid", tactile_contact_supervision=True,
        tactile_baseline_frames=16, tactile_contact_off=1., tactile_contact_on=3., norm_stats={},
        use_quantile_norm=True, repack_transforms=transforms.Group(), data_transforms=transforms.Group(),
        model_transforms=transforms.Group())
    mc = Pi05FastSlowConfig(action_horizon=30)
    train, validation = FastSlowDataset(dc, mc), FastSlowDataset(dc, mc, split="validation")
    assert set(train.episode_ids).isdisjoint(validation.episode_ids)
    assert len(train.episode_ids)+len(validation.episode_ids) == 4
    monkeypatch.setattr(np.random, "randint", lambda limit: limit-1)
    sample = train[0]
    assert int(sample["fast_start"]) == 8
    assert sample["observation.state"][0] == 15
    assert sample["tactile"][0, -1, 0, 0] == 15
    assert sample["fast_tactile"]["tactile"][0, -1, 0, 0] == 23
    assert sample["action_valid"].sum() == 9
    np.testing.assert_array_equal(sample["tactile_baseline"], sample["fast_tactile"]["tactile_baseline"])
    tail = train[8]
    assert int(tail["fast_start"]) == 0 and tail["action_valid"].sum() == 1
    with pytest.raises(ValueError):
        train.pair_indices(8, delay=1)


def test_config_old_checkpoint_and_isolation():
    from openpi.training.config import get_config
    config = get_config("pi05_righthand_ee_fastslow")
    assert isinstance(config.model, Pi05FastSlowConfig)
    assert config.model.num_flow_steps == 10 and config.model.slow_steps == 6
    assert config.num_train_steps == 30_000 and not config.overwrite and not config.resume
    assert "fastslow" in str(config.checkpoint_base_dir)
    assert config.weight_loader.params_path.endswith("/29999/params")
    assert config.trainable_filter(("PaliGemma", "llm", "layers", "attn", "q_einsum_1", "w"), nnx.Param(jnp.ones(1)))
    assert not config.trainable_filter(("PaliGemma", "llm", "layers", "attn", "q_einsum", "w"), nnx.Param(jnp.ones(1)))
    assert not isinstance(get_config("pi05_righthand_ee_atvla_full").model, Pi05FastSlowConfig)
    with pytest.raises(ValueError):
        Pi05FastSlowConfig(slow_steps=10)


def test_runtime_session_freshness_and_right_hand_output(tiny):
    from openpi.fastslow.policy import FastSlowPolicy
    _, model, obs = tiny
    base_policy = SimpleNamespace(_model=model, _input_transform=lambda x: x,
                                 _output_transform=lambda x: x)
    policy = FastSlowPolicy(base_policy, fps=30, max_delay_frames=2)
    session = policy.session_id
    pressure = np.full((11, 14, 14), 100., np.float32)
    for i in range(16):
        policy.feed_pressure(session, pressure, np.ones(11, bool), i/30)
    inputs = jax.tree.map(lambda x: np.asarray(x[0]), obs.to_dict())
    slow = policy.slow(session, inputs, .5)
    cached = policy.snapshot
    result = policy.fast(session, slow["chunk_id"], .5)
    assert np.asarray(result["actions"]).shape == (4, 7)
    assert result["action_dims"] == [6, 7, 8, 9, 10, 11, 13]
    assert np.isfinite(result["gate"]) and result["valid_from"] == 0
    with pytest.raises(ValueError, match="strictly increase"):
        policy.fast(session, slow["chunk_id"], .5)
    policy.feed_pressure(session, pressure+10*spatial_mask(), np.ones(11, bool), 16/30)
    result2 = policy.fast(session, slow["chunk_id"], 16/30, 1)
    assert result2["revision"] == 2 and result2["valid_from"] == 1
    assert policy.snapshot is cached
    with pytest.raises(ValueError, match="expired"):
        policy.fast(session, slow["chunk_id"], .61, 2)
    with pytest.raises(ValueError, match="Stale chunk"):
        policy.fast(session, 999, .55, 1)
    policy.reset()
    with pytest.raises(ValueError, match="Stale session"):
        policy.feed_pressure(session, pressure, np.ones(11, bool), 1.)


def test_http_client_contract_and_errors():
    import threading
    from openpi.fastslow.serve import make_server
    from openpi.fastslow.client import Pi05FastSlowClient

    class StubPolicy:
        session_id = "session"
        chunk_id = 0
        model = SimpleNamespace(slow_steps=6, flow_steps=10)
        stream = SimpleNamespace(calibration=None)

        def reset(self):
            return self.session_id

        def feed_pressure(self, session, raw, valid, stamp):
            assert session == "session" and np.asarray(raw).shape == (11, 14, 14)

        def slow(self, session, observation, stamp):
            assert observation["images"]["top"].dtype == np.uint8
            return {"chunk_id": 1}

        def fast(self, session, chunk, stamp, progress):
            if chunk != 1:
                raise ValueError("Stale chunk")
            return {**ActionUpdate(session, chunk, 1, 1., stamp, np.zeros((30, 7)), progress, 30).as_dict(),
                    "gate": .6, "fast_ms": 20.}

    server = make_server(StubPolicy(), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = Pi05FastSlowClient(f"http://127.0.0.1:{server.server_port}")
        client.reset()
        client.feed_pressure(np.zeros((11, 14, 14)), np.ones(11, bool), 1.)
        slow = client.slow({"top": np.zeros((4, 4, 3), np.uint8)}, "pour", np.zeros(26), 1.)
        update, meta = client.fast(slow["chunk_id"], 1.1, 3)
        assert update.actions.shape == (30, 7) and meta["gate"] == .6
        with pytest.raises(RuntimeError, match="Stale chunk"):
            client.fast(2, 1.1, 3)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_checkpoint_metadata_roundtrip_and_resume_guard(tmp_path):
    import optax
    from openpi.training import checkpoints
    from openpi.training.utils import TrainState
    config = Pi05FastSlowConfig(action_horizon=30)
    model = nnx.Linear(2, 2, rngs=nnx.Rngs(1))
    graph, params = nnx.split(model)
    tx = optax.sgd(.01)
    state = TrainState(step=jnp.asarray(1), params=params, model_def=graph,
                       opt_state=tx.init(params), tx=tx, ema_decay=None)
    loader = SimpleNamespace(fastslow_metadata=config.runtime_metadata(),
                             data_config=lambda: SimpleNamespace(norm_stats=None, asset_id=None))
    manager, _ = checkpoints.initialize_checkpoint_dir(tmp_path/"checkpoint", keep_period=1,
                                                       overwrite=False, resume=False)
    try:
        checkpoints.save_state(manager, state, loader, 1)
        manager.wait_until_finished()
        restored = checkpoints.restore_state(manager, state, loader, 1)
        for left, right in zip(jax.tree.leaves(state.params), jax.tree.leaves(restored.params), strict=True):
            np.testing.assert_array_equal(left, right)
        path = tmp_path/"checkpoint/1/assets/fastslow.json"
        assert json.loads(path.read_text())["slow_steps"] == 6
        loader.fastslow_metadata["slow_steps"] = 7
        with pytest.raises(ValueError, match="metadata mismatch"):
            checkpoints.restore_state(manager, state, loader, 1)
    finally:
        manager.close()


def test_native_o10_fresh_adapter_and_prefix_reuse(tiny):
    from openpi.vtla.model import Pi05VTLAConfig
    cfg,_,_=tiny
    cfg=Pi05VTLAConfig(**{**dataclasses.asdict(cfg),'tactile_layout':'o10_268','tactile_num_sensors':1,'loss_action_dims':tuple(range(17))})
    optimized=cfg.create(jax.random.key(55))
    reference=dataclasses.replace(cfg,reuse_training_prefix=False).load(nnx.state(optimized).to_pure_dict())
    obs=cfg.fake_obs()
    assert obs.tactile.shape==(1,1,16,1,268)
    changed=dataclasses.replace(obs,tactile=obs.tactile.at[:,:,:,:,0:4].set(20.))
    tokens,valid,logit,gate=optimized.encode_tactile_condition(changed)
    assert tokens.shape==(1,16,32) and np.isfinite(gate).all()
    a=jnp.ones((1,4,64))
    np.testing.assert_array_equal(optimized.tactile_adapter(a,tokens,valid,gate),a)
    missing=dataclasses.replace(obs,tactile_valid=jnp.zeros_like(obs.tactile_valid))
    assert np.all(optimized.encode_tactile_condition(missing)[-1]==0)
    touch={f.name:getattr(changed,f.name) for f in dataclasses.fields(changed) if f.name.startswith('tactile')}
    paired=FastSlowObservation(**{f.name:getattr(obs,f.name) for f in dataclasses.fields(obs)},fast_tactile=touch,
                              action_valid=jnp.ones((1,4),bool),fast_start=jnp.array([1]))
    def objective(model):
        loss,metrics=model.compute_loss_and_metrics(jax.random.key(21),paired,jnp.zeros((1,4,32)),train=True)
        return loss.mean(),metrics
    a,ga=nnx.value_and_grad(objective,has_aux=True)(optimized)
    b,gb=nnx.value_and_grad(objective,has_aux=True)(reference)
    for x,y in zip(jax.tree.leaves((a,ga)),jax.tree.leaves((b,gb)),strict=True):
        np.testing.assert_allclose(x,y,rtol=3e-3,atol=3e-5)
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(ga))
