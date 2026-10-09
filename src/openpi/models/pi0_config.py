import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore
    # If true, adds the normalized state as a continuous token to the action expert input. This is mainly useful for
    # experiments that keep pi05's AdaRMS action expert while avoiding discrete state tokens in the prompt.
    continuous_state_input: bool = False
    # If set, only the first `loss_action_dim` action dimensions are used when computing training loss.
    loss_action_dim: int | None = None
    # If set, exactly these action dimensions are used when computing training loss. This takes precedence over
    # `loss_action_dim` and is useful when the active action dimensions are not a prefix.
    loss_action_dims: tuple[int, ...] | None = None

    # Image augmentation knobs (only applied when TrainConfig.train_image_augment=True). Defaults reproduce the
    # original hardcoded augmentation (crop+rotate on non-wrist cams + brightness/contrast/saturation/hue jitter
    # with p=0.3).
    image_aug_crop_enabled: bool = True
    image_aug_crop_ratio: float = 0.95
    image_aug_rotate_deg: float = 5.0
    image_aug_brightness: float = 0.3
    image_aug_contrast: float = 0.4
    image_aug_saturation: float = 0.5
    image_aug_hue: float = 0.1
    image_aug_prob: float = 0.3

    # Right-hand tactile token branch. Off by default so existing checkpoints keep
    # the original prefix (images + language only).
    tactile_enabled: bool = False
    tactile_num_sensors: int = 11
    tactile_num_frames: int = 16
    tactile_embed_dim: int = 256
    # prefix: legacy scalar-gated tokens; action_adapter: contact-supervised suffix residual.
    tactile_fusion: str = "prefix"
    contact_loss_weight: float = 0.01

    pytorch_compile_mode: str | None = "max-autotune"

    def __post_init__(self):
        if self.tactile_fusion not in ("prefix", "action_adapter"):
            raise ValueError("Unknown tactile_fusion")
        if self.tactile_fusion == "action_adapter" and not self.tactile_enabled:
            raise ValueError("action_adapter requires tactile_enabled")
        if self.contact_loss_weight < 0:
            raise ValueError("contact_loss_weight must be nonnegative")
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)
        if self.pytorch_compile_mode is not None:
            assert self.pytorch_compile_mode in [
                "default",
                "reduce-overhead",
                "max-autotune",
                "max-autotune-no-cudagraphs",
            ]
        if self.loss_action_dim is not None:
            assert 1 <= self.loss_action_dim <= self.action_dim
        if self.loss_action_dims is not None:
            assert len(self.loss_action_dims) > 0
            assert all(0 <= dim < self.action_dim for dim in self.loss_action_dims)
            assert len(set(self.loss_action_dims)) == len(self.loss_action_dims)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                tactile=(
                    jax.ShapeDtypeStruct(
                        [batch_size, self.tactile_num_sensors, self.tactile_num_frames, 14, 14],
                        jnp.float32,
                    )
                    if self.tactile_enabled
                    else None
                ),
                tactile_valid=(
                    jax.ShapeDtypeStruct([batch_size, self.tactile_num_sensors], jnp.bool_)
                    if self.tactile_enabled
                    else None
                ),
                tactile_frame_valid=(
                    jax.ShapeDtypeStruct([batch_size, self.tactile_num_frames], jnp.bool_)
                    if self.tactile_enabled
                    else None
                ),
                tactile_sample_valid=(jax.ShapeDtypeStruct([batch_size, self.tactile_num_sensors, self.tactile_num_frames], bool)
                                      if self.tactile_fusion == "action_adapter" else None),
                tactile_baseline=(jax.ShapeDtypeStruct([batch_size, self.tactile_num_sensors, 14, 14], jnp.float32)
                                  if self.tactile_fusion == "action_adapter" else None),
                tactile_scale=(jax.ShapeDtypeStruct([batch_size, self.tactile_num_sensors], jnp.float32)
                               if self.tactile_fusion == "action_adapter" else None),
                tactile_contact=(jax.ShapeDtypeStruct([batch_size], jnp.float32)
                                 if self.tactile_fusion == "action_adapter" else None),
                tactile_contact_valid=(jax.ShapeDtypeStruct([batch_size], bool)
                                       if self.tactile_fusion == "action_adapter" else None),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
