"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.octopus_policy as octopus_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.polaris_config as polaris_config
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Assemble a T-frame right-hand taxel window from observation.tactile.right.
    tactile_enabled: bool = False
    tactile_num_frames: int = 16
    tactile_key: str = "observation.tactile.right"
    tactile_valid_key: str = "observation.tactile.right_valid"
    tactile_contact_supervision: bool = False
    tactile_baseline_frames: int = 16
    tactile_contact_off: float = 1.0
    tactile_contact_on: float = 3.0

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = ()


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotOctopusDataConfig(DataConfigFactory):
    """Data adapter for the local three-camera dual-arm pouring dataset."""

    action_dim: int = 15
    clip_normalized_inputs: float | None = 1.0
    # If set, keep only these state dims after Normalize/Clip and before TokenizePrompt.
    # Example (0728 26-D schema, right arm + right hand + right EE):
    #   (6, 7, 8, 9, 10, 11, 13, 20, 21, 22, 23, 24, 25)
    state_input_dims: tuple[int, ...] | None = None
    include_tactile: bool = False
    contact_supervision: bool = False
    baseline_frames: int = 16
    contact_off: float = 1.0
    contact_on: float = 3.0
    tactile_num_frames: int = 16

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_map = {
            "images": {
                "top": "observation.images.top",
                "left_wrist": "observation.images.left_wrist",
                "right_wrist": "observation.images.right_wrist",
            },
            "state": "observation.state",
            "actions": "action",
            "prompt": "prompt",
            "sample_global_index": "sample_global_index",
            "sample_episode_index": "sample_episode_index",
            "sample_frame_in_episode": "sample_frame_in_episode",
        }
        if self.include_tactile:
            repack_map.update(
                {
                    "tactile": "tactile",
                    "tactile_valid": "tactile_valid",
                    "tactile_frame_valid": "tactile_frame_valid",
                }
            )
        if self.contact_supervision:
            repack_map.update({key: key for key in (
                "tactile_sample_valid", "tactile_baseline", "tactile_scale",
                "tactile_contact", "tactile_contact_valid",
            )})
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(repack_map)
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[octopus_policy.OctopusInputs()],
            outputs=[octopus_policy.OctopusOutputs(action_dim=self.action_dim)],
        )
        model_transforms = ModelTransformFactory()(model_config)
        prefix_transforms: list[_transforms.DataTransformFn] = []
        if self.clip_normalized_inputs is not None:
            prefix_transforms.append(octopus_policy.ClipNormalizedInputs(self.clip_normalized_inputs))
        if self.state_input_dims is not None:
            prefix_transforms.append(octopus_policy.SelectStateDims(self.state_input_dims))
        if prefix_transforms:
            model_transforms = dataclasses.replace(
                model_transforms,
                inputs=[*prefix_transforms, *model_transforms.inputs],
            )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
            tactile_enabled=self.include_tactile,
            tactile_num_frames=self.tactile_num_frames,
            tactile_contact_supervision=self.contact_supervision,
            tactile_baseline_frames=self.baseline_frames,
            tactile_contact_off=self.contact_off,
            tactile_contact_on=self.contact_on,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # LIBERO already represents actions as deltas, but we have some old Pi0 checkpoints that are trained with this
        # extra delta transform.
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.

    # List of datasets to sample from: name, version, weight, and optionally filter_dict_path
    datasets: Sequence[droid_rlds_dataset.RLDSDataset] = (
        droid_rlds_dataset.RLDSDataset(
            name="droid",
            version="1.0.1",
            weight=1.0,
            filter_dict_path="gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json",
        ),
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            datasets=self.datasets,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Split this global batch into equal microbatches and average gradients before
    # one optimizer update. Schedules/checkpoints still count optimizer steps.
    gradient_accumulation_steps: int = 1
    # Can be disabled when diagnosing GPU buffer aliasing in initialization or updates.
    donate_train_buffers: bool = True
    separate_gradient_update: bool = False
    fsdp_replicated_paths: tuple[str, ...] = ()
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # If true, will enable TensorBoard logging under checkpoint_dir/tensorboard.
    tensorboard_enabled: bool = False

    # Skip optimizer updates when loss/grad are non-finite or exceed these thresholds.
    skip_update_guard_enabled: bool = True
    skip_update_max_loss: float = 0.5
    skip_update_max_grad_norm: float = 5.0

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.gradient_accumulation_steps < 1 or self.batch_size % self.gradient_accumulation_steps:
            raise ValueError("Global batch size must be divisible by gradient_accumulation_steps >= 1")
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="pi0_libero",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0_config.Pi0Config(),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # Here is an example of loading a pi0-FAST model for full finetuning.
        # Modify action_dim and action_horizon to match your dataset (action horizon is equal to
        # the desired action chunk length).
        # The max_token_len is the maximum number of (non-image) tokens the model can handle.
        # This includes the tokenized prompt, proprioceptive state, and (FAST-tokenized) action tokens.
        # Choosing this value too small may chop off tokens at the end of your sequence (the code will throw
        # a warning), while choosing it too large will waste memory (since we pad each batch element to the
        # max_token_len). A good rule of thumb is to use approx 180 for single-arm robots, and approx 250 for
        # two-arm robots. Generally, err on the lower side here first, and potentially increase the value if
        # you see many warnings being thrown during training.
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # Note that we load the pi0-FAST base model checkpoint here.
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # Here is an example of loading a pi0-FAST model for LoRA finetuning.
        # For setting action_dim, action_horizon, and max_token_len, see the comments above.
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # Again, make sure to match the model config above when extracting the freeze filter
        # that specifies which parameters should be frozen during LoRA finetuning.
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instructions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    TrainConfig(
        name="pi05_aloha_pen_uncap",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=64,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/mnt/pi-data/kevin",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=10,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="your_hf_username/my_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    # 0727fulldata: merged left/right pour, action 14 / state 26 (1-DoF hands + EE).
    # Data: /data/0727fulldata -> ~/.cache/huggingface/lerobot/local/0727fulldata
    # frames=159273; steps/epoch=159273//64=2488; 10 epochs -> 24880 steps.
    # LR: warmup 200 -> 5e-5, cosine to 2e-6; FSDP=4; save every epoch.
    TrainConfig(
        name="pi05_0727fulldata",
        exp_name="pi05_0727fulldata",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            discrete_state_input=True,
            # Only supervise physical 14-D actions; pad dims 14..31 masked out of loss.
            loss_action_dims=tuple(range(14)),
        ),
        data=LeRobotOctopusDataConfig(
            repo_id="local/0727fulldata",
            assets=AssetsConfig(
                assets_dir="/home/xudi_ge/duibi/openpi/assets/0727fulldata_bimanual",
            ),
            action_dim=14,
            clip_normalized_inputs=1.0,
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/xudi_ge/openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=5e-5,
            decay_steps=24_880,
            decay_lr=2e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=nnx.Nothing(),
        ema_decay=None,
        batch_size=64,
        num_workers=8,
        num_train_steps=24_880,
        log_interval=50,
        save_interval=2_488,
        keep_period=2_488,
        checkpoint_base_dir="/data/final_double_hands_openpi",
        fsdp_devices=4,
        overwrite=True,
        resume=False,
        wandb_enabled=False,
    ),
    # 0728fulldata: merged left/right pour (0727 + 0728 sources), action 14 / state 26.
    # Data: /data/0728fulldata -> ~/.cache/huggingface/lerobot/local/0728fulldata
    # frames=356183; steps/epoch=356183//64=5565; 10 epochs -> 55650 steps.
    # LR: warmup 200 -> 2e-5, cosine to 2e-6; FSDP=4; save every epoch; TensorBoard on.
    TrainConfig(
        name="pi05_0728fulldata",
        exp_name="pi05_0728fulldata",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            discrete_state_input=True,
            # Only supervise physical 14-D actions; pad dims 14..31 masked out of loss.
            loss_action_dims=tuple(range(14)),
            # Mild color-only augmentation; no crop/rotate (wrist cams are sensitive).
            image_aug_crop_enabled=False,
            image_aug_brightness=0.08,
            image_aug_contrast=0.08,
            image_aug_saturation=0.08,
            image_aug_hue=0.03,
            image_aug_prob=0.1,
        ),
        data=LeRobotOctopusDataConfig(
            repo_id="local/0728fulldata",
            assets=AssetsConfig(
                assets_dir="/home/xudi_ge/duibi/openpi/assets/0728fulldata_bimanual",
            ),
            action_dim=14,
            clip_normalized_inputs=1.0,
            base_config=DataConfig(prompt_from_task=True),
        ),
        # Load from pi05_base checkpoint.
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/xudi_ge/openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=5e-5,
            decay_steps=55_650,
            decay_lr=2e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=nnx.Nothing(),
        ema_decay=None,
        batch_size=64,
        num_workers=8,
        num_train_steps=55_650,
        log_interval=50,
        save_interval=5_565,
        keep_period=5_565,
        checkpoint_base_dir="/data/final_double_hands_openpi",
        fsdp_devices=4,
        overwrite=True,
        resume=False,
        wandb_enabled=False,
        tensorboard_enabled=True,
    ),
    # Right-arm/hand only: same 14-D/26-D Octopus schema as 0728fulldata.
    # - State into pi05 prompt: right arm qpos + right hand + right EE (13 dims).
    # - Loss: right arm cmds + right hand cmd only (7 dims); left dims masked.
    # Data: /data/V1show_lerobot/merged -> ~/.cache/huggingface/lerobot/local/V1show_merged
    # frames=517944; steps/epoch=517944//64=8092; 46000 steps ≈ 5.68 epochs.
    # Timing vs 0728 (55600 step / 4.679d): ~3.87d from Mon 03:00 -> ~Thu 23:55.
    TrainConfig(
        name="pi05_righthand_ee",
        exp_name="pi05_righthand_ee",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            discrete_state_input=True,
            # Right arm (6-11) + right hand (13); left arm/hand and pads excluded from loss.
            loss_action_dims=(6, 7, 8, 9, 10, 11, 13),
            image_aug_crop_enabled=False,
            image_aug_brightness=0.08,
            image_aug_contrast=0.08,
            image_aug_saturation=0.08,
            image_aug_hue=0.03,
            image_aug_prob=0.1,
        ),
        data=LeRobotOctopusDataConfig(
            repo_id="local/V1show_merged",
            assets=AssetsConfig(
                assets_dir="/home/xudi_ge/duibi/openpi/assets/righthand_ee_bimanual",
            ),
            action_dim=14,
            clip_normalized_inputs=1.0,

            # 26-D state indices: right_arm_qpos0-5, right_hand_qpos0, right_ee_*.
            state_input_dims=(6, 7, 8, 9, 10, 11, 13, 20, 21, 22, 23, 24, 25),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/xudi_ge/openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=5e-5,
            decay_steps=46_000,
            decay_lr=2e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=nnx.Nothing(),
        ema_decay=None,
        batch_size=64,
        num_workers=8,
        num_train_steps=46_000,
        log_interval=50,
        save_interval=8_092,
        keep_period=8_092,
        checkpoint_base_dir="/data/final_double_hands_openpi",
        fsdp_devices=4,
        overwrite=True,
        resume=False,
        wandb_enabled=False,
        tensorboard_enabled=True,
    ),
    # 0804merge right-arm/hand training, following pi05_righthand_ee.
    # Data: /data/803haoran/lerobot_v21_0804merge/0804merge
    # frames=534261; steps/epoch=534261//64=8347; 50000 steps ≈ 5.99 epochs.
    # Save every 1000 steps and permanently retain every 3000-step checkpoint.
    TrainConfig(
        name="pi05_righthand_ee_0804merge",
        exp_name="pi05_righthand_ee_0804merge",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            discrete_state_input=True,
            # Right arm (6-11) + right hand (13); left arm/hand and pads excluded from loss.
            loss_action_dims=(6, 7, 8, 9, 10, 11, 13),
            image_aug_crop_enabled=False,
            image_aug_brightness=0.08,
            image_aug_contrast=0.08,
            image_aug_saturation=0.08,
            image_aug_hue=0.03,
            image_aug_prob=0.1,
        ),
        data=LeRobotOctopusDataConfig(
            repo_id="/data/803haoran/lerobot_v21_0804merge/0804merge",
            assets=AssetsConfig(
                assets_dir="/home/xudi_ge/duibi/openpi/assets/righthand_ee_0804merge",
                asset_id="local/0804merge",
            ),
            action_dim=14,
            clip_normalized_inputs=1.0,
            # 26-D state indices: right_arm_qpos0-5, right_hand_qpos0, right_ee_*.
            state_input_dims=(6, 7, 8, 9, 10, 11, 13, 20, 21, 22, 23, 24, 25),
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/xudi_ge/openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=50_000,
            decay_lr=3e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=nnx.Nothing(),
        ema_decay=None,
        # Use the original OpenPI behavior: always apply optimizer updates.
        skip_update_guard_enabled=False,
        skip_update_max_loss=float("inf"),
        skip_update_max_grad_norm=float("inf"),
        seed=43,
        batch_size=64,
        num_workers=8,
        num_train_steps=50_000,
        log_interval=20,
        save_interval=1_000,
        keep_period=3_000,
        checkpoint_base_dir="/data/final_double_hands_openpi",
        fsdp_devices=4,
        overwrite=True,
        resume=False,
        wandb_enabled=False,
        tensorboard_enabled=True,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    # Full resume of pi05_righthand_ee (params + AdamW + step) from latest ckpt (e.g. 32368).
    # Do NOT load params-only / reset optimizer — that caused NaN around step 100.
    # LR schedule must match the original run so restored opt_state lines up.
    # checkpoint_dir resolves to:
    #   /data/final_double_hands_openpi/pi05_righthand_ee/pi05_righthand_ee
    # (exp_name=".." cancels the unique config name in the path after resolve)
    TrainConfig(
        name="pi05_righthand_ee_continue",
        exp_name="..",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            discrete_state_input=True,
            loss_action_dims=(6, 7, 8, 9, 10, 11, 13),
            image_aug_crop_enabled=False,
            image_aug_brightness=0.08,
            image_aug_contrast=0.08,
            image_aug_saturation=0.08,
            image_aug_hue=0.03,
            image_aug_prob=0.1,
        ),
        data=LeRobotOctopusDataConfig(
            repo_id="local/V1show_merged",
            assets=AssetsConfig(
                assets_dir="/home/xudi_ge/duibi/openpi/assets/righthand_ee_bimanual",
            ),
            action_dim=14,
            clip_normalized_inputs=1.0,
            state_input_dims=(6, 7, 8, 9, 10, 11, 13, 20, 21, 22, 23, 24, 25),
            base_config=DataConfig(prompt_from_task=True),
        ),
        # Unused when resume=True; kept only for non-resume init shape.
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/xudi_ge/openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=5e-5,
            decay_steps=46_000,
            decay_lr=2e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=nnx.Nothing(),
        ema_decay=None,
        batch_size=64,
        num_workers=8,
        num_train_steps=46_000,
        log_interval=50,
        save_interval=8_092,
        keep_period=8_092,
        checkpoint_base_dir="/data/final_double_hands_openpi/pi05_righthand_ee/pi05_righthand_ee",
        fsdp_devices=4,
        overwrite=False,
        resume=True,
        wandb_enabled=False,
        tensorboard_enabled=True,
    ),
    # 2026-08-06: resume full train_state from righthand_ee step 24276 (not params-only).
    # Launch with scripts/train_righthand_ee_0806.sh (copies 24276 into this exp dir once).
    # Same LR schedule as the original run so AdamW opt_state matches; trains 24276 → 46000.
    TrainConfig(
        name="pi05_righthand_ee_0806",
        exp_name="pi05_righthand_ee_0806",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            discrete_state_input=True,
            loss_action_dims=(6, 7, 8, 9, 10, 11, 13),
            image_aug_crop_enabled=False,
            image_aug_brightness=0.08,
            image_aug_contrast=0.08,
            image_aug_saturation=0.08,
            image_aug_hue=0.03,
            image_aug_prob=0.1,
        ),
        data=LeRobotOctopusDataConfig(
            repo_id="local/V1show_merged",
            assets=AssetsConfig(
                assets_dir="/home/xudi_ge/duibi/openpi/assets/righthand_ee_bimanual",
            ),
            action_dim=14,
            clip_normalized_inputs=1.0,
            state_input_dims=(6, 7, 8, 9, 10, 11, 13, 20, 21, 22, 23, 24, 25),
            base_config=DataConfig(prompt_from_task=True),
        ),
        # Unused when resume=True.
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/xudi_ge/openpi-assets/checkpoints/pi05_base/params"
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=5e-5,
            decay_steps=46_000,
            decay_lr=2e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=nnx.Nothing(),
        ema_decay=None,
        batch_size=64,
        num_workers=8,
        num_train_steps=46_000,
        log_interval=10,
        save_interval=1_000,
        keep_period=3_000,
        checkpoint_base_dir="/data/final_double_hands_openpi",
        fsdp_devices=4,
        overwrite=False,
        resume=True,
        wandb_enabled=False,
        tensorboard_enabled=True,
    ),
    # Right-hand tactile tokens, initialized from pi05_base (not 0806).
    # Tactile encoder + Linear(256->2048) + scalar tanh gate are randomly
    # initialized; gate starts at 0. RGB/LLM come from pi05_base.
    TrainConfig(
        name="pi05_righthand_ee_tactile",
        exp_name="pi05_righthand_ee_tactile",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=30,
            discrete_state_input=True,
            loss_action_dims=(6, 7, 8, 9, 10, 11, 13),
            image_aug_crop_enabled=False,
            image_aug_brightness=0.08,
            image_aug_contrast=0.08,
            image_aug_saturation=0.08,
            image_aug_hue=0.03,
            image_aug_prob=0.1,
            tactile_enabled=True,
            tactile_num_sensors=11,
            tactile_num_frames=16,
        ),
        data=LeRobotOctopusDataConfig(
            repo_id="local/V1show_merged_tactile",
            assets=AssetsConfig(
                assets_dir="/home/xudi_ge/duibi/openpi/assets/righthand_ee_bimanual",
                asset_id="local/V1show_merged",
            ),
            action_dim=14,
            clip_normalized_inputs=1.0,
            state_input_dims=(6, 7, 8, 9, 10, 11, 13, 20, 21, 22, 23, 24, 25),
            include_tactile=True,
            tactile_num_frames=16,
            base_config=DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/xudi_ge/openpi-assets/checkpoints/pi05_base/params",
            missing_regex=".*",
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=5e-5,
            decay_steps=16_000,
            decay_lr=2e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=nnx.Nothing(),
        ema_decay=None,
        # From pi05_base, initial loss is ~0.6 and tactile_gate grads are large;
        # the 0.5/5.0 skip guard would drop every step.
        skip_update_guard_enabled=False,
        skip_update_max_loss=float("inf"),
        skip_update_max_grad_norm=float("inf"),
        batch_size=64,
        num_workers=8,
        num_train_steps=16_000,
        log_interval=10,
        save_interval=1_000,
        keep_period=2_000,
        checkpoint_base_dir="/data/final_double_hands_openpi",
        fsdp_devices=4,
        overwrite=True,
        resume=False,
        wandb_enabled=False,
        tensorboard_enabled=True,
    ),
    # RoboArena & PolaRiS configs.
    *roboarena_config.get_roboarena_configs(),
    *polaris_config.get_polaris_configs(),
]

# Isolated AT-VLA-inspired stage A+B. Keep all legacy configurations unchanged.
_atvla_base = next(c for c in _CONFIGS if c.name == "pi05_righthand_ee_tactile")
_CONFIGS.append(dataclasses.replace(
    _atvla_base,
    name="pi05_righthand_ee_atvla",
    exp_name="contact_adapter_v1",
    model=dataclasses.replace(_atvla_base.model, tactile_fusion="action_adapter", contact_loss_weight=0.01),
    data=dataclasses.replace(
        _atvla_base.data, repo_id="/data/V1show_lerobot/merged_tactile",
        contact_supervision=True, baseline_frames=16,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/home/xudi_ge/openpi-assets/checkpoints/pi05_base/params",
        missing_regex=r"(tactile_encoder|tactile_adapter)/.*",
    ),
    # Initial adaptation: train encoder/contact head/residual adapter; freeze pretrained policy.
    freeze_filter=nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex(r"(tactile_encoder|tactile_adapter)/.*"))),
    skip_update_guard_enabled=True,
    skip_update_max_loss=float("inf"),
    skip_update_max_grad_norm=float("inf"),
    checkpoint_base_dir="/data/final_double_hands_openpi_atvla",
    overwrite=False,
    resume=False,
))


# Stage 2: initialize all model weights from stage 1, with a fresh optimizer.
_atvla_stage1 = next(c for c in _CONFIGS if c.name == "pi05_righthand_ee_atvla")
_CONFIGS.append(dataclasses.replace(
    _atvla_stage1,
    name="pi05_righthand_ee_atvla_full",
    exp_name="full_from_10000_30k",
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/data/final_double_hands_openpi_atvla/pi05_righthand_ee_atvla/contact_adapter_v1/10000/params",
        missing_regex=r"(?!)",
    ),
    freeze_filter=nnx.Nothing(),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=200, peak_lr=5e-5, decay_steps=30_000, decay_lr=5e-6,
    ),
    num_train_steps=30_000,
    overwrite=False,
    resume=False,
))


from openpi.fastslow.model import Pi05FastSlowConfig

_fastslow_source = next(c for c in _CONFIGS if c.name == "pi05_righthand_ee_atvla_full")
_fastslow_checkpoint = "/data/final_double_hands_openpi_atvla/pi05_righthand_ee_atvla_full/full_from_10000_30k/29999"
_fastslow_config = dataclasses.replace(
    _fastslow_source,
    name="pi05_righthand_ee_fastslow",
    exp_name="from_29999_split6_4",
    model=Pi05FastSlowConfig(**dataclasses.asdict(_fastslow_source.model)),
    data=dataclasses.replace(_fastslow_source.data, assets=AssetsConfig(
        assets_dir=_fastslow_checkpoint + "/assets", asset_id="local/V1show_merged")),
    weight_loader=weight_loaders.CheckpointWeightLoader(_fastslow_checkpoint + "/params", missing_regex=r"(?!)"),
    lr_schedule=_optimizer.CosineDecaySchedule(
        warmup_steps=200, peak_lr=5e-5, decay_steps=30_000, decay_lr=2e-6, warmup_init_lr=2e-6,
    ),
    # Freeze vision/language, train the shared action expert and tactile modules.
    freeze_filter=nnx.All(nnx.Param, nnx_utils.PathRegex(r"PaliGemma/.*"),
                         nnx.Not(nnx_utils.PathRegex(r"PaliGemma/llm/.*_1/.*"))),
    batch_size=8,
    num_workers=2,
    checkpoint_base_dir="/data/final_double_hands_openpi_fastslow",
    overwrite=False, resume=False,
)
_CONFIGS.append(_fastslow_config)
# The old full-run checkpoints contain extreme action-expert K/V values. Start
# full training from the audited tactile-adapter checkpoint instead.
_fastslow_full_checkpoint = "/data/final_double_hands_openpi_atvla/pi05_righthand_ee_atvla/contact_adapter_v1/10000"
_CONFIGS.append(dataclasses.replace(
    _fastslow_config,
    name="pi05_righthand_ee_fastslow_full",
    freeze_filter=nnx.Nothing(),
    exp_name="from_10000_full_split6_4",
    weight_loader=weight_loaders.CheckpointWeightLoader(_fastslow_full_checkpoint + "/params", missing_regex=r"(?!)"),
    data=dataclasses.replace(_fastslow_config.data, assets=AssetsConfig(
        assets_dir=_fastslow_full_checkpoint + "/assets", asset_id="local/V1show_merged")),
))

# New optimizer/schedule: mature visual-action policy + the learned touch path.
_fastslow_46k = "/data/final_double_hands_openpi/pi05_righthand_ee_0806/pi05_righthand_ee_0806/45999"
_fastslow_touch26k = "/data/final_double_hands_openpi_fastslow/pi05_righthand_ee_fastslow_full/full_from_10000_30k_min2e6_20261002/26000"
_CONFIGS.append(dataclasses.replace(
    _fastslow_config,
    name="pi05_righthand_ee_fastslow_46k_b64",
    exp_name="from_45999_touch26000_b64_30k_20261004",
    freeze_filter=nnx.Nothing(),
    weight_loader=weight_loaders.BackboneTactileWeightLoader(
        backbone_path=_fastslow_46k + "/params", tactile_path=_fastslow_touch26k + "/params"),
    data=dataclasses.replace(_fastslow_config.data, assets=AssetsConfig(
        assets_dir=_fastslow_46k + "/assets", asset_id="local/V1show_merged")),
    batch_size=64,
    gradient_accumulation_steps=8,
    num_train_steps=30_000,
    save_interval=1_000,
    keep_period=2_000,
    overwrite=False,
    resume=False,
))

# Prepared fallback only; choose explicitly when the 45999 source fails audit.
_fastslow_27k = "/data/final_double_hands_openpi/pi05_righthand_ee_0806/pi05_righthand_ee_0806/27000"
_CONFIGS.append(dataclasses.replace(
    next(c for c in _CONFIGS if c.name == "pi05_righthand_ee_fastslow_46k_b64"),
    name="pi05_righthand_ee_fastslow_27k_b64",
    donate_train_buffers=False,
    exp_name="from_27000_touch26000_b64_30k_20261004",
    weight_loader=weight_loaders.BackboneTactileWeightLoader(
        backbone_path=_fastslow_27k + "/params", tactile_path=_fastslow_touch26k + "/params"),
    data=dataclasses.replace(_fastslow_config.data, assets=AssetsConfig(
        assets_dir=_fastslow_27k + "/assets", asset_id="local/V1show_merged")),
))


# Fresh pi05-base VTLA experiment. The 0806 run supplies hyperparameters only.
from openpi.vtla.model import Pi05VTLAConfig

_vtla_reference = next(c for c in _CONFIGS if c.name == "pi05_righthand_ee_0806")
_vtla_model_args = dataclasses.asdict(_vtla_reference.model)
_vtla_model_args.update(tactile_enabled=True, tactile_fusion="action_adapter", contact_loss_weight=0.01)
_vtla_model_args.update(tactile_layout='o10_268', tactile_num_sensors=1, loss_action_dims=tuple(range(17)))
_CONFIGS.append(dataclasses.replace(
    _vtla_reference,
    name="vtla_pi05_base_bottle",
    exp_name="base_fresh_gate_46k",
    model=Pi05VTLAConfig(**_vtla_model_args),
    data=dataclasses.replace(
        _atvla_base.data,
        repo_id="/data/open_bottle_cap_vtla_20261006/lerobot",
        assets=AssetsConfig(assets_dir="/home/xudi_ge/VTLA_pi05_base_bottle/assets/bottle", asset_id="open_bottle_cap"),
        contact_supervision=True, baseline_frames=16,
        state_input_dims=None, action_dim=17,
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/home/xudi_ge/openpi-assets/checkpoints/pi05_base/params",
        missing_regex=r"(tactile_encoder|tactile_adapter)/.*",
    ),
    batch_size=64, gradient_accumulation_steps=1, num_workers=8,
    num_train_steps=46_000,
    lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=200, peak_lr=5e-5, decay_steps=46_000, decay_lr=2e-6),
    optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
    freeze_filter=nnx.Nothing(), ema_decay=None, fsdp_devices=4,
    donate_train_buffers=False, separate_gradient_update=True,
    # Replicate this 12 MiB modulation matrix and its moments while diagnosing
    # observed four-GPU corruption in its column-sharded representation.
    fsdp_replicated_paths=("['final_norm_1']['Dense_0']['kernel']",),
    log_interval=10, save_interval=1_000, keep_period=3_000,
    skip_update_guard_enabled=True, skip_update_max_loss=float("inf"), skip_update_max_grad_norm=float("inf"),
    checkpoint_base_dir="/data/checkpoints_vtla_bottle", resume=False, overwrite=False,
    wandb_enabled=False, tensorboard_enabled=True,
))

# Three-stage alternative: visual task adaptation -> tactile adaptation -> joint fast/slow.
# Each stage loads parameters only and starts a fresh optimizer and schedule.
from openpi.vtla.stages import Pi05VisualAdaptConfig

_bottle_joint = next(c for c in _CONFIGS if c.name == "vtla_pi05_base_bottle")
_bottle_stage_root = "/data/checkpoints_vtla_bottle_staged"
_visual_args = {f.name: getattr(_bottle_joint.model, f.name)
                for f in dataclasses.fields(pi0_config.Pi0Config)}
_visual_args.update(tactile_enabled=False, tactile_fusion="prefix", contact_loss_weight=0.)
_bottle_visual = dataclasses.replace(
    _bottle_joint,
    name="vtla_bottle_stage1_visual", exp_name="visual_20k",
    seed=43,  # User-requested RNG control; episode split remains model.split_seed=42.
    model=Pi05VisualAdaptConfig(**_visual_args),
    data=dataclasses.replace(_bottle_joint.data, include_tactile=False, contact_supervision=False),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "/home/xudi_ge/openpi-assets/checkpoints/pi05_base/params", missing_regex=r"(?!)"),
    num_train_steps=20_000,
    lr_schedule=dataclasses.replace(_bottle_joint.lr_schedule, decay_steps=20_000),
    checkpoint_base_dir=_bottle_stage_root,
)
_bottle_touch = dataclasses.replace(
    _bottle_joint,
    name="vtla_bottle_stage2_tactile", exp_name="tactile_6k",
    model=dataclasses.replace(_bottle_joint.model, training_objective="flow"),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        _bottle_stage_root + "/vtla_bottle_stage1_visual/visual_20k/19999/params",
        missing_regex=r"(tactile_encoder|tactile_adapter)/.*"),
    freeze_filter=nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex(r"(tactile_encoder|tactile_adapter)/.*"))),
    num_train_steps=6_000,
    lr_schedule=dataclasses.replace(_bottle_joint.lr_schedule, decay_steps=6_000),
    checkpoint_base_dir=_bottle_stage_root,
)
_bottle_joint_stage = dataclasses.replace(
    _bottle_joint,
    name="vtla_bottle_stage3_joint", exp_name="joint_20k",
    weight_loader=weight_loaders.CheckpointWeightLoader(
        _bottle_stage_root + "/vtla_bottle_stage2_tactile/tactile_6k/5999/params",
        missing_regex=r"(?!)"),
    num_train_steps=20_000,
    lr_schedule=dataclasses.replace(_bottle_joint.lr_schedule, decay_steps=20_000),
    checkpoint_base_dir=_bottle_stage_root,
)
_CONFIGS.extend((_bottle_visual, _bottle_touch, _bottle_joint_stage))

# Direct full fine-tuning requested after the GPU peer-copy workaround passed.
# Keep the staged and original diagnostic runs available under their own names.
_CONFIGS.append(dataclasses.replace(
    _bottle_joint,
    name="vtla_bottle_direct_full",
    exp_name="base_touch_joint_46k_seed43",
    seed=43,
))

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
