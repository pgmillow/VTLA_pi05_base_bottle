import dataclasses
import functools
import logging
import json
import os
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def init_tensorboard(config: _config.TrainConfig, *, enabled: bool = True):
    """Create a TensorBoard SummaryWriter under checkpoint_dir/tensorboard."""
    if not enabled:
        return None

    from torch.utils.tensorboard import SummaryWriter

    log_dir = config.checkpoint_dir / "tensorboard"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.info("TensorBoard logging to %s", log_dir)
    return SummaryWriter(log_dir=str(log_dir))


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Keep only arrays that actually came from the checkpoint. ShapeDtypeStruct
    # placeholders are new modules (e.g. tactile encoder). None is nnx.Conv(use_bias=False)
    # storing Param(None); writing that back via replace_by_pure_dict fails.
    arrays = {
        k: v
        for k, v in traverse_util.flatten_dict(loaded_params).items()
        if not isinstance(v, jax.ShapeDtypeStruct) and v is not None
    }
    for path, value in arrays.items():
        array = np.asarray(value)
        # Network weights of this scale indicate a damaged/diverged checkpoint,
        # even when every individual float is finite (e.g. the old 29999 run).
        if not np.all(np.isfinite(array)) or (array.size and float(np.max(np.abs(array))) > 1e6):
            raise ValueError(f"Unsafe initial checkpoint parameter: {'/'.join(path)}")
    return traverse_util.unflatten_dict(arrays)


def _check_step_numerics(info: dict[str, Any]) -> None:
    """Fail before saving when finite elements still overflow norm reductions."""
    if not np.isfinite(float(info["param_norm"])):
        raise FloatingPointError("Parameter norm overflow; stopping before checkpoint save")
    if "param_max_abs" in info:
        maximum = float(info["param_max_abs"])
        if not np.isfinite(maximum) or maximum > 1e6:
            raise FloatingPointError("Unsafe parameter magnitude; stopping before checkpoint save")
    if float(info["skipped_update"]) <= 0.5:
        for key in ("_diag_grad_leaf_norms", "_diag_update_leaf_norms"):
            if not np.all(np.isfinite(np.asarray(info[key]))):
                raise FloatingPointError(f"{key} overflow in an accepted update; stopping before checkpoint save")


def _audit_device_parameters(params):
    """Inspect every device weight, transferring only scalar summaries to CPU."""
    flat = {p: x for p, x in traverse_util.flatten_dict(params.to_pure_dict(), sep="/").items() if x is not None}
    if not flat:
        return {"arrays": 0, "bad": [], "largest": []}
    def summarize(*values):
        return jnp.stack([jnp.stack((jnp.all(jnp.isfinite(x)).astype(jnp.float32),
                                   jnp.max(jnp.abs(x)).astype(jnp.float32) if x.size else jnp.asarray(0.)))
                          for x in values])
    stats = np.asarray(jax.device_get(jax.jit(summarize)(*flat.values())))
    entries = [{"name": path, "finite": bool(row[0]), "max_abs": float(row[1])}
               for path, row in zip(flat, stats, strict=True)]
    return {"arrays": len(entries),
            "bad": [x for x in entries if not x["finite"] or x["max_abs"] > 1e6],
            "largest": sorted(entries, key=lambda x: x["max_abs"], reverse=True)[:12]}


def _audit_fresh_adam_state(opt_state):
    """Every fresh Adam moment and counter must be exactly zero on device."""
    paths = _tree_leaf_paths(opt_state)
    if not paths:
        return {"arrays": 0, "nonzero_paths": []}
    checks = jax.jit(lambda state: jnp.stack([jnp.all(x == 0) for x in jax.tree.leaves(state)]))(opt_state)
    zero = np.asarray(jax.device_get(checks), dtype=bool)
    return {"arrays": len(paths), "nonzero_paths": [p for p, ok in zip(paths, zero, strict=True) if not ok]}


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True,
                                           replicated_paths=config.fsdp_replicated_paths)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    init_rng = sharding.replicate_via_host(init_rng, replicated_sharding)

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,) if config.donate_train_buffers else (),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


def _microbatch_gradients(config, rng, state, observation, actions):
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        if hasattr(model, "compute_loss_and_metrics"):
            chunked_loss, metrics = model.compute_loss_and_metrics(rng, observation, actions, train=True)
        else:
            chunked_loss = model.compute_loss(rng, observation, actions, train=True)
            metrics = {}
        return jnp.mean(chunked_loss), metrics

    diff_state = nnx.DiffState(0, config.trainable_filter)
    return nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model, rng, observation, actions)


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    def microbatch_grad(key, obs, acts):
        return _microbatch_gradients(config, key, state, obs, acts)

    loss_info, grads = _accumulate_gradients(
        microbatch_grad, train_rng, observation, actions, config.gradient_accumulation_steps)
    return _apply_gradients(config, state, loss_info, grads)


def _apply_gradients(config, state, loss_info, grads):
    loss, loss_metrics = loss_info
    model = nnx.merge(state.model_def, state.params)
    model.train()

    params = state.params.filter(config.trainable_filter)
    grad_norm = optax.global_norm(grads)

    def leaf_finite(tree):
        return jnp.stack([jnp.all(jnp.isfinite(x)) for x in jax.tree.leaves(tree)])

    def leaf_norms(tree):
        return jnp.stack(
            [jnp.linalg.norm(jnp.asarray(x, dtype=jnp.float32)) for x in jax.tree.leaves(tree)]
        )

    grad_leaf_finite = leaf_finite(grads)
    grad_leaf_norms = leaf_norms(grads)
    # Finite elements can still overflow the squared-sum norm. In particular,
    # inf <= an unlimited (inf) threshold is True, so thresholds are not a
    # substitute for a finite check. Never send such gradients through Adam.
    finite = (
        jnp.isfinite(loss) & jnp.isfinite(grad_norm)
        & jnp.all(grad_leaf_finite) & jnp.all(jnp.isfinite(grad_leaf_norms))
    )
    within = (
        (loss <= config.skip_update_max_loss) & (grad_norm <= config.skip_update_max_grad_norm)
        if config.skip_update_guard_enabled else jnp.asarray(True)
    )
    input_good = finite & within
    updates, candidate_opt_state = jax.lax.cond(
        input_good,
        lambda g: state.tx.update(g, state.opt_state, params),
        lambda g: (jax.tree.map(jnp.zeros_like, g), state.opt_state),
        grads,
    )
    candidate_params = optax.apply_updates(params, updates)
    update_norm = optax.global_norm(updates)
    update_leaf_norms = leaf_norms(updates)
    update_leaf_finite = leaf_finite(updates)
    candidate_param_leaf_finite = leaf_finite(candidate_params)
    candidate_opt_leaf_finite = leaf_finite(candidate_opt_state)

    # Reject both bad inputs and bad candidate updates. Candidate validation is essential:
    # finite loss/grads alone do not guarantee that AdamW produced finite params/state.
    candidate_finite = jnp.logical_and(
        jnp.all(update_leaf_finite),
        jnp.logical_and(jnp.all(candidate_param_leaf_finite), jnp.all(candidate_opt_leaf_finite)),
    )
    candidate_finite &= (
        jnp.isfinite(update_norm) & jnp.all(jnp.isfinite(update_leaf_norms))
        & jnp.isfinite(optax.global_norm(candidate_params))
        & jnp.all(jnp.stack([jnp.max(jnp.abs(x)) <= 1e6
                            for x in jax.tree.leaves(candidate_params) if x.size]))
    )
    # Disabling optional magnitude thresholds must never disable numerical safety.
    good = input_good & candidate_finite

    # Commit the candidate atomically, or return the exact old params/optimizer state.
    new_params, new_opt_state = jax.lax.cond(
        good,
        lambda _: (candidate_params, candidate_opt_state),
        lambda _: (params, state.opt_state),
        operand=None,
    )

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: jnp.where(good, state.ema_decay * old + (1 - state.ema_decay) * new, old),
                state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        # Preserve NaN/Inf instead of displaying a misleading zero. The explicit
        # flags remain visible in TensorBoard even when a norm is not plottable.
        "loss": loss,
        "grad_norm": grad_norm,
        "update_norm": update_norm,
        "grad_norm_finite": jnp.isfinite(grad_norm).astype(jnp.float32),
        "param_norm": optax.global_norm(kernel_params),
        # Detect finite corruption too, including in a state returned after a
        # rejected update. Match the checkpoint initialization audit bound.
        "param_max_abs": jnp.max(jnp.stack([jnp.max(jnp.abs(x)).astype(jnp.float32)
                                           for x in jax.tree.leaves(new_params) if x.size])),
        "skipped_update": jnp.where(good, 0.0, 1.0),
        "skip_nonfinite": jnp.where(finite, 0.0, 1.0),
        "skip_over_threshold": jnp.where(jnp.logical_and(finite, jnp.logical_not(within)), 1.0, 0.0),
        "skip_bad_candidate": jnp.where(candidate_finite, 0.0, 1.0),
        # Diagnostic arrays are removed before interval/TensorBoard aggregation.
        "_diag_loss_finite": jnp.isfinite(loss),
        "_diag_grad_finite": grad_leaf_finite,
        "_diag_update_finite": update_leaf_finite,
        "_diag_candidate_param_finite": candidate_param_leaf_finite,
        "_diag_candidate_opt_finite": candidate_opt_leaf_finite,
        "_diag_grad_leaf_norms": grad_leaf_norms,
        "_diag_grad_leaf_max_abs": jnp.stack([jnp.max(jnp.abs(x)).astype(jnp.float32)
                                              for x in jax.tree.leaves(grads)]),
        "_diag_update_leaf_norms": update_leaf_norms,
    }
    info.update(loss_metrics)
    if hasattr(model, "tactile_gate"):
        info["tactile_gate"] = jnp.tanh(jnp.reshape(model.tactile_gate.value, ()))
    return new_state, info


def _accumulate_gradients(grad_fn, rng, observation, actions, accumulation_steps):
    """Mean microbatch gradients; clipping and AdamW run once on this mean.

    All microbatches have equal size. A non-finite value in any microbatch is
    retained in the mean so the existing atomic update guard rejects the step.
    """
    if accumulation_steps == 1:
        return grad_fn(rng, observation, actions)
    if accumulation_steps < 1 or actions.shape[0] % accumulation_steps:
        raise ValueError("Batch size must divide evenly into positive accumulation steps")
    micro_size = actions.shape[0] // accumulation_steps

    def one(index):
        micro_obs, micro_actions = jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(x, index * micro_size, micro_size, axis=0),
            (observation, actions))
        return grad_fn(jax.random.fold_in(rng, index), micro_obs, micro_actions)

    initial = one(0)

    def body(total, index):
        result = one(index)
        return jax.tree.map(lambda x, y: x + y, total, result), None

    total, _ = jax.lax.scan(body, initial, jnp.arange(1, accumulation_steps))
    return jax.tree.map(lambda x: x / accumulation_steps, total)


def make_accumulated_train_step(config, state_sharding, data_sharding, replicated_sharding, *, first_stage_callback=None, audit_every_step=False):
    """Use separate GPU executables for microbatch gradients and AdamW.

    The host only dispatches operations. Gradients and accumulation remain on
    device, with exactly the same RNGs, summation order and one optimizer update
    as the scan implementation. Avoid buffer donation across these boundaries.
    """
    steps = config.gradient_accumulation_steps
    micro_size = config.batch_size // steps
    grad_sharding = state_sharding.params.filter(config.trainable_filter)
    result_sharding = (replicated_sharding, grad_sharding)
    pgrad = jax.jit(
        lambda key, state, batch: _microbatch_gradients(config, key, state, *batch),
        in_shardings=(replicated_sharding, state_sharding, data_sharding),
        out_shardings=result_sharding,
    )
    pslice = jax.jit(
        lambda batch, index: jax.tree.map(
            lambda x: jax.lax.dynamic_slice_in_dim(x, index * micro_size, micro_size, axis=0), batch),
        in_shardings=(data_sharding, replicated_sharding), out_shardings=data_sharding,
    )
    padd = jax.jit(lambda a, b: jax.tree.map(lambda x, y: x + y, a, b),
                   in_shardings=(result_sharding, result_sharding), out_shardings=result_sharding)
    pmean = jax.jit(lambda total: jax.tree.map(lambda x: x / steps, total),
                    in_shardings=(result_sharding,), out_shardings=result_sharding)
    pupdate = jax.jit(functools.partial(_apply_gradients, config),
                     in_shardings=(state_sharding, replicated_sharding, grad_sharding),
                     out_shardings=(state_sharding, replicated_sharding))

    first_update = True

    def step(rng, state, batch):
        nonlocal first_update
        rng = sharding.replicate_via_host(rng, replicated_sharding)
        key = jax.random.fold_in(rng, state.step)
        total = None
        for index in range(steps):
            if first_update:
                logging.info("First update: computing microbatch %d/%d gradients", index + 1, steps)
            if steps > 1:
                index_array = sharding.replicate_via_host(np.asarray(index, np.int32), replicated_sharding)
                micro_key = jax.random.fold_in(key, index_array)
                micro_batch = pslice(batch, index_array)
            else:
                micro_key, micro_batch = key, batch
            result = pgrad(micro_key, state, micro_batch)
            if first_update or audit_every_step:
                jax.block_until_ready(result)
                if first_stage_callback is not None:
                    first_stage_callback(f"gradient_{index}", state, result)
            total = result if total is None else padd(total, result)
        loss_info, gradients = pmean(total)
        if first_update or audit_every_step:
            if first_stage_callback is not None:
                jax.block_until_ready((loss_info, gradients))
                first_stage_callback("averaged_gradient", state, (loss_info, gradients))
        if first_update:
            logging.info("First update: applying AdamW once to the averaged gradients")
        first_update = False
        return pupdate(state, loss_info, gradients)

    return step


def _np_meta(value: Any) -> np.ndarray:
    return np.asarray(jax.device_get(value))


def _format_batch_sample_ids(meta: dict[str, Any], *, max_items: int = 16) -> str:
    if not meta:
        return "(no sample ids in batch meta)"
    g = _np_meta(meta["sample_global_index"]).reshape(-1)
    e = _np_meta(meta["sample_episode_index"]).reshape(-1)
    f = _np_meta(meta["sample_frame_in_episode"]).reshape(-1)
    n = int(g.shape[0])
    lines = [f"batch_size={n}"]
    show = min(n, max_items)
    for i in range(show):
        lines.append(f"  [{i}] global_frame={int(g[i])} episode_index={int(e[i])} frame_in_episode={int(f[i])}")
    if n > show:
        lines.append(f"  ... ({n - show} more)")
    # Compact unique episode list for quick scanning.
    uniq, counts = np.unique(e, return_counts=True)
    top = sorted(zip(uniq.tolist(), counts.tolist()), key=lambda x: -x[1])[:12]
    lines.append("episode_counts=" + ", ".join(f"{int(ep)}:{int(c)}" for ep, c in top))
    return "\n".join(lines)


def _tree_leaf_paths(tree: Any) -> list[str]:
    return [jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_flatten_with_path(tree)[0]]


def _format_bad_leaf_paths(finite_mask: Any, paths: list[str], *, max_items: int = 16) -> str:
    mask = np.asarray(finite_mask, dtype=bool).reshape(-1)
    bad = np.flatnonzero(~mask)
    if not len(bad):
        return "(none)"
    shown = [paths[int(i)] if int(i) < len(paths) else f"<leaf {int(i)}>" for i in bad[:max_items]]
    suffix = f", ... ({len(bad) - max_items} more)" if len(bad) > max_items else ""
    return ", ".join(shown) + suffix


def _format_top_leaf_norms(values: Any, paths: list[str], *, max_items: int = 8) -> str:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(np.nan_to_num(values, nan=np.inf))[::-1][:max_items]
    entries = []
    for i in order:
        index = int(i)
        path = paths[index] if index < len(paths) else f"<leaf {index}>"
        entries.append(f"{path}={values[index]:.6g}")
    return ", ".join(entries)


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )
    micro_size = config.batch_size // config.gradient_accumulation_steps
    if micro_size % jax.device_count():
        raise ValueError("Each global microbatch must be divisible by the number of devices")
    logging.info("Effective global batch=%d; microbatch=%d; accumulation=%d; optimizer steps=%d",
                 config.batch_size, micro_size, config.gradient_accumulation_steps, config.num_train_steps)

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_rng = sharding.replicate_via_host(train_rng, replicated_sharding)

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    tb_writer = init_tensorboard(config, enabled=config.tensorboard_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    num_views = min(5, len(next(iter(batch[0].images.values()))))
    # Index on the host: indexing a sharded JAX image uploads a GPU0 index to
    # the other GPUs, reintroducing the faulty peer-copy path before training.
    preview_images = [np.asarray(img) for img in batch[0].images.values()]
    camera_arrays = [
        np.concatenate([img[i] for img in preview_images], axis=1) for i in range(num_views)
    ]
    if config.wandb_enabled:
        wandb.log({"camera_views": [wandb.Image(arr) for arr in camera_arrays]}, step=0)
    if tb_writer is not None:
        for i, arr in enumerate(camera_arrays):
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            tb_writer.add_image(f"camera_views/{i}", arr, global_step=0, dataformats="HWC")

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    device_audit = _audit_device_parameters(train_state.params)
    device_audit_path = config.checkpoint_dir / f"initial_device_parameters_{os.getpid()}.json"
    device_audit_path.write_text(json.dumps(device_audit, indent=2) + "\n")
    logging.info("Initial device parameter audit: %d arrays, bad=%s", device_audit["arrays"], device_audit["bad"])
    if device_audit["bad"]:
        raise FloatingPointError("Initial device weights failed audit before any optimizer update")
    if not resuming and isinstance(config.optimizer, _optimizer.AdamW):
        opt_audit = _audit_fresh_adam_state(train_state.opt_state)
        (config.checkpoint_dir / f"initial_adam_state_{os.getpid()}.json").write_text(
            json.dumps(opt_audit, indent=2) + "\n")
        logging.info("Fresh Adam state audit: %s", opt_audit)
        if opt_audit["nonzero_paths"]:
            raise FloatingPointError("Fresh Adam state is not zero before any optimizer update")

    param_leaf_paths = _tree_leaf_paths(train_state.params.filter(config.trainable_filter))
    opt_leaf_paths = _tree_leaf_paths(train_state.opt_state)

    if config.separate_gradient_update or (config.gradient_accumulation_steps > 1 and not config.donate_train_buffers):
        logging.info("Using separate GPU microbatch-gradient and optimizer executables; no buffer donation")
        ptrain_step = make_accumulated_train_step(config, train_state_sharding, data_sharding, replicated_sharding)
    else:
        ptrain_step = jax.jit(
            functools.partial(train_step, config),
            in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
            out_shardings=(train_state_sharding, replicated_sharding),
            donate_argnums=(1,) if config.donate_train_buffers else (),
        )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    lr_schedule = config.lr_schedule.create()
    if config.skip_update_guard_enabled:
        logging.info(
            "Skip-update guards: max_loss=%s max_grad_norm=%s (also skips NaN/Inf)",
            config.skip_update_max_loss,
            config.skip_update_max_grad_norm,
        )
    else:
        logging.info("Optional magnitude thresholds are disabled; numerical safety checks remain enabled")

    infos = []
    consecutive_skips = 0
    for step in pbar:
        batch_meta = getattr(data_loader, "last_batch_meta", {})
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        info_host = jax.device_get(info)
        try:
            _check_step_numerics(info_host)
        except FloatingPointError:
            failure = {"step": step, "metrics": {k: np.asarray(v).tolist() for k, v in info_host.items()},
                       "parameter_paths": param_leaf_paths,
                       "device_parameters": _audit_device_parameters(train_state.params)}
            failure_path = config.checkpoint_dir / f"failed_update_step{step}_{os.getpid()}.json"
            failure_path.write_text(json.dumps(failure, indent=2) + "\n")
            logging.error("Fatal update diagnostic: %s; loss=%s grad_norm=%s update_norm=%s bad_weights=%s",
                          failure_path, info_host["loss"], info_host["grad_norm"], info_host["update_norm"],
                          failure["device_parameters"]["bad"])
            raise
        metrics = {key: value for key, value in info.items() if not key.startswith("_diag_")}
        if float(info_host["skipped_update"]) > 0.5:
            consecutive_skips += 1
            reason = []
            if float(info_host["skip_nonfinite"]) > 0.5:
                reason.append("nonfinite")
            if float(info_host["skip_over_threshold"]) > 0.5:
                reason.append(
                    f"over_threshold(loss>{config.skip_update_max_loss} or "
                    f"grad_norm>{config.skip_update_max_grad_norm})"
                )
            if float(info_host["skip_bad_candidate"]) > 0.5:
                reason.append("bad_candidate_update")
            diagnostics = (
                f"loss_finite={bool(info_host['_diag_loss_finite'])}\n"
                f"bad_grad_paths={_format_bad_leaf_paths(info_host['_diag_grad_finite'], param_leaf_paths)}\n"
                f"bad_update_paths={_format_bad_leaf_paths(info_host['_diag_update_finite'], param_leaf_paths)}\n"
                f"bad_candidate_param_paths="
                f"{_format_bad_leaf_paths(info_host['_diag_candidate_param_finite'], param_leaf_paths)}\n"
                f"bad_candidate_opt_paths="
                f"{_format_bad_leaf_paths(info_host['_diag_candidate_opt_finite'], opt_leaf_paths)}\n"
                f"top_grad_leaf_norms={_format_top_leaf_norms(info_host['_diag_grad_leaf_norms'], param_leaf_paths)}\n"
                f"top_grad_max_abs={_format_top_leaf_norms(info_host['_diag_grad_leaf_max_abs'], param_leaf_paths)}\n"
                f"top_update_leaf_norms={_format_top_leaf_norms(info_host['_diag_update_leaf_norms'], param_leaf_paths)}"
            )
            pbar.write(
                f"[SKIP update] step={step} loss={float(info_host['loss']):.6g} "
                f"grad_norm={float(info_host['grad_norm']):.6g} "
                f"update_norm={float(info_host['update_norm']):.6g} "
                f"param_norm={float(info_host['param_norm']):.6g} "
                f"skipped_update={float(info_host['skipped_update']):.0f} "
                f"skip_nonfinite={float(info_host['skip_nonfinite']):.0f} "
                f"skip_over_threshold={float(info_host['skip_over_threshold']):.0f} "
                f"skip_bad_candidate={float(info_host['skip_bad_candidate']):.0f} "
                f"reason={'+'.join(reason) or 'unknown'}\n"
                f"{diagnostics}\n"
                f"{_format_batch_sample_ids(batch_meta)}"
            )
            logging.warning(
                "Skipped optimizer update at step=%s loss=%s grad_norm=%s reason=%s",
                step,
                float(info_host["loss"]),
                float(info_host["grad_norm"]),
                "+".join(reason) or "unknown",
            )
            logging.warning("Skipped update diagnostics:\n%s", diagnostics)
            sample_ids = _format_batch_sample_ids(batch_meta)
            logging.warning("Skipped batch sample ids:\n%s", sample_ids)
            diagnostic_path = config.checkpoint_dir / f"skipped_update_step{step}_{os.getpid()}.json"
            diagnostic_path.write_text(json.dumps({
                "step": step, "reason": reason, "sample_ids": sample_ids,
                "metrics": {k: np.asarray(v).tolist() for k, v in info_host.items()},
                "parameter_paths": param_leaf_paths,
            }, indent=2) + "\n")
            if tb_writer is not None:
                tb_writer.add_text(
                    "diagnostics/skipped_update",
                    f"step={step}\n\nreason={'+'.join(reason) or 'unknown'}\n\n"
                    f"```\n{diagnostics}\n{sample_ids}\n```",
                    global_step=step,
                )
            if consecutive_skips >= 10:
                if tb_writer is not None:
                    tb_writer.flush()
                raise FloatingPointError("10 consecutive updates rejected; stopping for diagnosis")
        else:
            consecutive_skips = 0

        infos.append(metrics)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            # LR for the step that was just trained (schedule is indexed by step count).
            lr = float(np.asarray(lr_schedule(step)))
            reduced_info["lr"] = lr
            # Keep routine terminal/file output compact. Detailed optimizer diagnostics
            # are printed immediately only when an update is skipped; TensorBoard still
            # receives every scalar below.
            info_str = f"loss={reduced_info['loss']:.4f}, lr={lr:.4e}"
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            if tb_writer is not None:
                for k, v in reduced_info.items():
                    tb_writer.add_scalar(k, float(v), global_step=step)
                tb_writer.flush()
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()


if __name__ == "__main__":
    main(_config.cli())
