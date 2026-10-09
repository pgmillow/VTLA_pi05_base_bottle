import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str
    # Keys present in the randomly-initialized model but missing from the checkpoint
    # are filled from init when they match this regex. Default keeps historical LoRA
    # behaviour; use ".*" when adding new modules such as the tactile encoder.
    missing_regex: str = ".*lora.*"

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        return _merge_params(loaded_params, params, missing_regex=self.missing_regex)


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


@dataclasses.dataclass(frozen=True)
class BackboneTactileWeightLoader(WeightLoader):
    """Initialize the policy from one run and only tactile modules from another.

    Every target parameter must exist in its designated source. This deliberately
    does not fall back to random initialization or import an optimizer state.
    """

    backbone_path: str
    tactile_path: str

    def load(self, params: at.Params) -> at.Params:
        backbone = _model.restore_params(download.maybe_download(self.backbone_path), restore_type=np.ndarray)
        tactile = _model.restore_params(download.maybe_download(self.tactile_path), restore_type=np.ndarray)
        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        sources = [flax.traverse_util.flatten_dict(p, sep="/") for p in (backbone, tactile)]
        selected = {}
        counts = [0, 0]
        for key, ref in flat_ref.items():
            source = int(key.startswith(("tactile_encoder/", "tactile_adapter/")))
            if key not in sources[source]:
                raise ValueError(f"Missing {'tactile' if source else 'backbone'} parameter: {key}")
            value = sources[source][key]
            if value is None or ref is None:
                if value is not None or ref is not None:
                    raise ValueError(f"Incompatible optional parameter: {key}")
            elif value.shape != ref.shape:
                raise ValueError(f"Shape mismatch for {key}: {value.shape} versus {ref.shape}")
            selected[key] = value
            counts[source] += 1
        if not all(counts):
            raise ValueError("A mixed initialization must load both policy and tactile parameters")
        logger.info("Mixed initialization: %d policy leaves from %s; %d tactile leaves from %s",
                    counts[0], self.backbone_path, counts[1], self.tactile_path)
        return _merge_params(flax.traverse_util.unflatten_dict(selected, sep="/"), params, missing_regex=r"(?!)")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            ref = flat_ref[k]
            # NNX modules without a bias store Param(None) in trained checkpoints.
            if v is None or ref is None:
                if v is not None or ref is not None:
                    raise ValueError(f"Checkpoint parameter {k!r} has incompatible None/array values")
                result[k] = None
            else:
                result[k] = v.astype(ref.dtype) if v.dtype != ref.dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
