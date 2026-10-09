"""CPU-only preflight for the requested 45999 policy + 26000 touch initialization."""
import dataclasses
import argparse
import hashlib
import json
from pathlib import Path

from flax import nnx
from flax.traverse_util import flatten_dict
import jax
import numpy as np

from openpi.training.config import get_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone-step", type=int, default=45999)
    args = parser.parse_args()
    cfg = get_config("pi05_righthand_ee_fastslow_46k_b64")
    if args.backbone_step != 45999:
        source = Path(cfg.weight_loader.backbone_path).parent.parent / str(args.backbone_step) / "params"
        cfg = dataclasses.replace(cfg, weight_loader=dataclasses.replace(cfg.weight_loader, backbone_path=str(source)))
    source_label = f"pi05_{args.backbone_step}"
    print(f"Auditing policy {args.backbone_step} and tactile 26000 on CPU", flush=True)
    shape = jax.eval_shape(lambda rng: nnx.state(cfg.model.create(rng)).to_pure_dict(), jax.random.key(0))
    loaded = cfg.weight_loader.load(shape)
    entries = []
    for key, value in flatten_dict(loaded, sep="/").items():
        if value is None:
            continue
        arr = np.asarray(value)
        finite = bool(np.isfinite(arr).all())
        maximum = float(np.max(np.abs(arr))) if arr.size else 0.0
        entries.append({"name": key, "shape": list(arr.shape), "max_abs": maximum, "finite": finite,
                        "source": "touch26000" if key.startswith(("tactile_encoder/", "tactile_adapter/")) else source_label})
    bad = [x for x in entries if not x["finite"] or x["max_abs"] > 1e6]
    backbone = Path(cfg.weight_loader.backbone_path).parent
    tactile = Path(cfg.weight_loader.tactile_path).parent
    norm_hashes = []
    for root in [backbone, tactile]:
        norm = json.loads((root / "assets/local/V1show_merged/norm_stats.json").read_text())
        norm_hashes.append(hashlib.sha256(json.dumps(norm, sort_keys=True).encode()).hexdigest())
    result = {"config": cfg.name, "weight_loader": dataclasses.asdict(cfg.weight_loader),
              "arrays": len(entries), "normalization_equal": norm_hashes[0] == norm_hashes[1],
              "normalization_hashes": norm_hashes, "bad": bad,
              "largest": sorted(entries, key=lambda x: x["max_abs"], reverse=True)[:10],
              "policy_arrays": sum(x["source"] == source_label for x in entries),
              "tactile_arrays": sum(x["source"] == "touch26000" for x in entries)}
    out = Path(f"artifacts/initialization_{args.backbone_step}_touch26000_audit.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if bad or not result["normalization_equal"]:
        raise ValueError("Initialization audit failed; do not launch training")


if __name__ == "__main__":
    main()
