"""Audit fresh GPU optimizer state and a zero-gradient update without training."""
import functools
import json
import logging
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

import train
from openpi.training import config, sharding


def audit(tree):
    leaves = jax.tree_util.tree_flatten_with_path(tree)[0]
    def summarize(*values):
        return tuple((jnp.max(jnp.abs(x)), jnp.count_nonzero(x), jnp.all(jnp.isfinite(x))) for x in values)
    summaries = jax.device_get(jax.jit(summarize)(*(value for _, value in leaves)))
    result = []
    for (path, value), (maximum, nonzero, finite) in zip(leaves, summaries, strict=True):
        record = {"name": jax.tree_util.keystr(path), "shape": list(value.shape),
                  "max_abs": float(maximum), "nonzero": int(nonzero), "finite": bool(finite)}
        result.append(record)
        if record["nonzero"]:
            logging.info("Nonzero optimizer leaf: %s", record)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train.init_logging()
    jax.config.update("jax_compilation_cache_dir", str(Path("~/.cache/jax").expanduser()))
    cfg = config.get_config("pi05_righthand_ee_fastslow_27k_b64")
    mesh = sharding.make_mesh(cfg.fsdp_devices)
    rng = jax.random.split(jax.random.key(cfg.seed))[1]
    state, _ = train.init_train_state(cfg, rng, mesh, resume=False)
    jax.block_until_ready(state)
    report = {"donate_train_buffers": cfg.donate_train_buffers,
              "initial_parameters": train._audit_device_parameters(state.params)}
    logging.info("Auditing all optimizer moment arrays before any update")
    report["initial_optimizer"] = audit(state.opt_state)
    out = Path("artifacts") / f"optimizer_probe_{os.getpid()}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    bad = [x for x in report["initial_optimizer"] if x["nonzero"] or not x["finite"]]
    if bad:
        raise RuntimeError(f"Fresh optimizer must be exactly zero: {bad}")

    @functools.partial(jax.jit, donate_argnums=(0, 1) if cfg.donate_train_buffers else ())
    def zero_update(params, opt_state):
        gradients = jax.tree.map(jnp.zeros_like, params)
        updates, next_opt = state.tx.update(gradients, opt_state, params)
        return optax.apply_updates(params, updates), next_opt

    logging.info("Fresh optimizer is zero; testing an isolated zero-gradient AdamW update")
    parameters, next_opt = zero_update(state.params, state.opt_state)
    jax.block_until_ready(parameters)
    report["after_zero_parameters"] = train._audit_device_parameters(parameters)
    report["after_zero_optimizer"] = audit(next_opt)
    out.write_text(json.dumps(report, indent=2) + "\n")
    logging.info("Probe saved to %s; bad parameters=%s", out, report["after_zero_parameters"]["bad"])
