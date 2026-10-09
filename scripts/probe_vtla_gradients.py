"""Compare forward, pure JAX differentiation and NNX differentiation, no Adam."""
import argparse
import dataclasses
import gc
import hashlib
import json
import logging
from pathlib import Path
import time

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from openpi.models.pi0 import Pi0
from openpi.training import config as configs, data_loader, sharding
import train


def final_norm_shards(params):
    value = params['PaliGemma']['llm']['final_norm_1']['Dense_0']['kernel'].value
    rows = []
    for shard in value.addressable_shards:
        array = np.asarray(shard.data)
        rows.append({'device': str(shard.device), 'shape': list(array.shape),
                     'finite': bool(np.isfinite(array).all()), 'max_abs': float(np.abs(array).max()),
                     'sha256': hashlib.sha256(array.tobytes()).hexdigest()})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--objective", choices=("flow", "paired"), default="paired")
    parser.add_argument("--methods", nargs="+", choices=("forward", "jax", "nnx"), default=["forward", "jax", "nnx"])
    parser.add_argument("--benchmark-rng", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, force=True)
    jax.config.update("jax_compilation_cache_dir", str(Path.home()/".cache/jax"))
    cfg = configs.get_config("vtla_pi05_base_bottle")
    mesh = sharding.make_mesh(cfg.fsdp_devices)
    ds = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    rep = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    batch = next(iter(data_loader.create_data_loader(dataclasses.replace(cfg, num_workers=0), sharding=ds, shuffle=True)))
    batch = jax.device_put(jax.tree.map(lambda x: np.asarray(x)[:args.batch], batch), ds)
    train_rng, init_rng = jax.random.split(jax.random.key(cfg.seed))
    loss_rng = jax.random.fold_in(train_rng, 0) if args.benchmark_rng else jax.random.key(23)
    state, ss = train.init_train_state(cfg, init_rng if args.benchmark_rng else jax.random.key(7), mesh, resume=False)
    jax.block_until_ready(state)
    report = {"batch": args.batch, "objective": args.objective, "benchmark_rng": args.benchmark_rng,
              "initial_audit": train._audit_device_parameters(state.params),
              "initial_final_norm_shards": final_norm_shards(state.params), "runs": []}
    out = Path(f"artifacts/gradient_probe_{args.objective}_b{args.batch}.json")
    out.write_text(json.dumps(report, indent=2)+"\n")

    def model_loss(model, batch):
        model.train()
        method = Pi0.compute_loss_and_metrics if args.objective == "flow" else type(model).compute_loss_and_metrics
        loss, metrics = method(model, loss_rng, *batch, train=True)
        return jnp.mean(loss), metrics

    def pure_loss(params, batch):
        return model_loss(nnx.merge(state.model_def, params), batch)

    def nnx_grad(params, batch):
        model = nnx.merge(state.model_def, params)
        return nnx.value_and_grad(model_loss, argnums=nnx.DiffState(0, cfg.trainable_filter), has_aux=True)(model, batch)

    grad_sharding = ss.params.filter(cfg.trainable_filter)
    # This disposable comparison is valid because this experiment trains all params.
    assert len(jax.tree.leaves(ss.params)) == len(jax.tree.leaves(grad_sharding))
    for method in args.methods:
        print("START", args.objective, method, args.batch, flush=True)
        raw = {"forward": pure_loss, "jax": jax.value_and_grad(pure_loss, has_aux=True), "nnx": nnx_grad}[method]
        fn = jax.jit(raw, in_shardings=(ss.params, ds),
                     out_shardings=rep if method == "forward" else (rep, grad_sharding))
        started = time.perf_counter()
        with sharding.set_mesh(mesh):
            result = fn(state.params, batch)
        jax.block_until_ready(result)
        if method == "forward":
            loss_info = result
            grad_info = {}
        else:
            loss_info, grads = result
            grad_info = {"norm": float(optax.global_norm(grads)), "audit": train._audit_device_parameters(grads)}
        loss, metrics = loss_info
        row = {"method": method, "seconds": time.perf_counter()-started, "loss": float(loss),
               "metrics": {k: float(v) for k, v in metrics.items()}, "gradients": grad_info,
               "parameter_audit": train._audit_device_parameters(state.params),
               "final_norm_shards": final_norm_shards(state.params)}
        report["runs"].append(row)
        out.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(row), flush=True)
        if row["parameter_audit"]["bad"]:
            raise FloatingPointError("Read-only gradient call changed a parameter to an unsafe value")
        del result, fn
        if method != "forward":
            del grads
        gc.collect()
    print("GRADIENT_PROBE_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
