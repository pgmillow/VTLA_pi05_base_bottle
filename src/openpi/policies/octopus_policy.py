import dataclasses

import einops
import numpy as np

from openpi import transforms


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class OctopusInputs(transforms.DataTransformFn):
    """Map three-camera Octopus demos to OpenPI inputs.

    Supports both schemas used locally:
    - 15-D action / 27-D state (L hand 2-DoF + R hand 1-DoF + EE)
    - 14-D action / 26-D state (L/R hand 1-DoF each + EE), e.g. 0727fulldata
    State/actions are padded later to the model action_dim (typically 32).
    """

    def __call__(self, data: dict) -> dict:
        images = data["images"]
        inputs = {
            "state": np.asarray(data["state"], dtype=np.float32),
            "image": {
                "base_0_rgb": _parse_image(images["top"]),
                "left_wrist_0_rgb": _parse_image(images["left_wrist"]),
                "right_wrist_0_rgb": _parse_image(images["right_wrist"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if "tactile" in data:
            inputs["tactile"] = np.asarray(data["tactile"], dtype=np.float32)
        if "tactile_valid" in data:
            inputs["tactile_valid"] = np.asarray(data["tactile_valid"], dtype=bool)
        if "tactile_frame_valid" in data:
            inputs["tactile_frame_valid"] = np.asarray(data["tactile_frame_valid"], dtype=bool)
        for key in ("tactile_sample_valid", "tactile_contact_valid"):
            if key in data:
                inputs[key] = np.asarray(data[key], dtype=bool)
        for key in ("tactile_baseline", "tactile_scale", "tactile_contact"):
            if key in data:
                inputs[key] = np.asarray(data[key], dtype=np.float32)
        # Keep sample ids for train-loop skip logging (ignored by Observation.from_dict).
        for key in ("sample_global_index", "sample_episode_index", "sample_frame_in_episode"):
            if key in data:
                inputs[key] = np.asarray(data[key])
        return inputs


@dataclasses.dataclass(frozen=True)
class OctopusOutputs(transforms.DataTransformFn):
    """Return the physical action prefix from the 32-D model output."""

    action_dim: int = 15

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}


@dataclasses.dataclass(frozen=True)
class ClipNormalizedInputs(transforms.DataTransformFn):
    """Clip quantile-normalized state and actions before model padding/tokenization."""

    limit: float = 1.0

    def __call__(self, data: dict) -> dict:
        data["state"] = np.clip(data["state"], -self.limit, self.limit).astype(np.float32, copy=False)
        if "actions" in data:
            data["actions"] = np.clip(data["actions"], -self.limit, self.limit).astype(np.float32, copy=False)
        return data


@dataclasses.dataclass(frozen=True)
class SelectStateDims(transforms.DataTransformFn):
    """Keep only selected state dims before pi05 discrete-state tokenization.

    Must run after Normalize/Clip and before TokenizePrompt. Training and inference
    must use the same `dims`. Actions are left unchanged; PadStatesAndActions still
    pads the sliced state to model action_dim (unused by pi05 after tokenization).
    """

    dims: tuple[int, ...]

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"])
        if state.shape[-1] <= max(self.dims):
            raise ValueError(
                f"state last dim is {state.shape[-1]}, but SelectStateDims requires index {max(self.dims)}"
            )
        data["state"] = state[..., list(self.dims)].astype(np.float32, copy=False)
        return data
