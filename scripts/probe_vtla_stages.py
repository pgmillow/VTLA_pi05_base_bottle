"""Read-only diagnostics for the first non-finite pi05 forward stage."""
import argparse
import dataclasses
import json
import logging
from pathlib import Path

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.models.pi0 import make_attn_mask
from openpi.training import config as configs, data_loader, sharding
import train


def stats(x):
    value = x.astype(jnp.float32)
    return {"finite": jnp.all(jnp.isfinite(value)), "max_abs": jnp.max(jnp.abs(value))}


def trace_llm(module, embedded, positions, mask, adarms_cond):
    embedded = jax.tree.map(lambda x: x.astype(module.embed_dtype), embedded)
    embedded, _ = module.layers(embedded, None, positions, mask[:, None], adarms_cond, True)
    report = {"prefix_before_final_norm": stats(embedded[0]),
              "action_before_final_norm": stats(embedded[1])}
    weight = module.variables["params"]["final_norm_1"]["Dense_0"]
    modulation = (adarms_cond[1].astype(embedded[1].dtype)
                  @ weight["kernel"].astype(embedded[1].dtype)
                  + weight["bias"].astype(embedded[1].dtype))
    for key, value in zip(("final_scale", "final_shift", "final_gate"), jnp.split(modulation, 3, -1)):
        report[key] = stats(value)
    suffix = module.final_norms[1](embedded[1], adarms_cond[1])[0]
    report["action_after_final_norm"] = stats(suffix)
    return suffix, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--no-touch", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--benchmark-rng", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, force=True)
    jax.config.update("jax_compilation_cache_dir", str(Path.home()/".cache/jax"))
    cfg = configs.get_config("vtla_pi05_base_bottle")
    mesh = sharding.make_mesh(cfg.fsdp_devices)
    ds = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    batch = next(iter(data_loader.create_data_loader(dataclasses.replace(cfg, num_workers=0), sharding=ds, shuffle=True)))
    batch = jax.device_put(jax.tree.map(lambda x: np.asarray(x)[:args.batch], batch), ds)
    input_stats = {jax.tree_util.keystr(p): {"shape": list(x.shape), **{k: float(v) for k, v in stats(x).items()}}
                   for p, x in jax.tree_util.tree_flatten_with_path(batch)[0]}
    train_rng, init_rng = jax.random.split(jax.random.key(cfg.seed))
    loss_rng = jax.random.split(jax.random.fold_in(train_rng, 0), 3)[0] if args.benchmark_rng else jax.random.key(23)
    state, ss = train.init_train_state(cfg, init_rng if args.benchmark_rng else jax.random.key(7), mesh, resume=False)
    jax.block_until_ready(state)
    print("STAGE_INIT_COMPLETE", flush=True)

    def forward(params, rng, batch):
        model = nnx.merge(state.model_def, params)
        observation, actions = batch
        pre_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        obs = model_lib.preprocess_observation(pre_rng, observation, train=args.train,
                image_aug_crop_enabled=model.image_aug_crop_enabled,
                image_aug_crop_ratio=model.image_aug_crop_ratio,
                image_aug_rotate_deg=model.image_aug_rotate_deg,
                image_aug_brightness=model.image_aug_brightness,
                image_aug_contrast=model.image_aug_contrast,
                image_aug_saturation=model.image_aug_saturation,
                image_aug_hue=model.image_aug_hue,image_aug_prob=model.image_aug_prob)
        active = jnp.isin(jnp.arange(model.action_dim), jnp.asarray(model.loss_action_dims))
        acts = jnp.where(active, actions, 0.)
        noise = jnp.where(active, jax.random.normal(noise_rng, actions.shape), 0.)
        t = jax.random.beta(time_rng, 1.5, 1, actions.shape[:-2]) * .999 + .001
        x = t[:, None, None]*noise+(1-t[:, None, None])*acts
        prefix, pmask, par = model.embed_prefix(obs)
        touch = None if args.no_touch else model.encode_tactile_condition(obs)
        suffix, smask, sar, cond = model.embed_suffix(obs, x, t, touch)
        mask = jnp.concatenate([pmask, smask], 1)
        attn = make_attn_mask(mask, jnp.concatenate([par, sar]))
        output, report = model.PaliGemma.llm([prefix, suffix], positions=jnp.cumsum(mask, 1)-1,
                                           mask=attn, adarms_cond=[None, cond], method=trace_llm)
        predicted = model.action_out_proj(output[:, -model.action_horizon:])
        target = noise-acts
        report.update(prefix_embedding=stats(prefix), action_embedding=stats(suffix),
                      time_condition=stats(cond), predicted_velocity=stats(predicted),
                      target_velocity=stats(target), noisy_action=stats(x), noise=stats(noise),
                      flow_time=stats(t))
        if touch is not None:
            report.update(touch_tokens=stats(touch[0]), touch_logits=stats(touch[2]), touch_gate=stats(touch[3]))
        loss = jnp.mean(jnp.sum(jnp.where(active, jnp.square(predicted-target), 0.), -1)/jnp.sum(active))
        return loss, report

    rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    fn = jax.jit(forward, in_shardings=(ss.params, rep, ds))
    with sharding.set_mesh(mesh):
        loss, report = fn(state.params, loss_rng, batch)
    jax.block_until_ready((loss, report))
    result = {"batch": args.batch, "no_touch": args.no_touch, "loss": float(loss),
              "train": args.train, "benchmark_rng": args.benchmark_rng,
              "inputs": input_stats,
              "stages": {k: {a: float(b) for a, b in v.items()} for k, v in report.items()},
              "parameter_audit": train._audit_device_parameters(state.params)}
    out = Path(f"artifacts/stage_probe_b{args.batch}_{'no_touch' if args.no_touch else 'touch'}.json")
    out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
