from collections.abc import Iterator, Sequence
import bisect
import io
import json
import logging
import multiprocessing
import os
import pathlib
import typing
from typing import Any, Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import pandas as pd
import PIL.Image
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


class LocalLeRobotParquetDataset(Dataset):
    """Read the local OpenPI-compatible LeRobot parquet layout without HuggingFace access."""

    def __init__(
        self,
        repo_id: str,
        action_horizon: int,
        *,
        tactile_num_frames: int | None = None,
        tactile_key: str = "observation.tactile.right",
        tactile_valid_key: str = "observation.tactile.right_valid",
        contact_supervision: bool = False,
        baseline_frames: int = 16,
        contact_off: float = 1.0,
        contact_on: float = 3.0,
    ):
        home = pathlib.Path(os.environ.get("HF_LEROBOT_HOME", pathlib.Path.home() / ".cache/huggingface/lerobot"))
        root = pathlib.Path(repo_id)
        self._root = root if root.is_absolute() else home / repo_id
        self._action_horizon = action_horizon
        self._tactile_num_frames = tactile_num_frames
        self._tactile_key = tactile_key
        self._tactile_valid_key = tactile_valid_key
        self._contact_supervision = contact_supervision
        self._baseline_frames = baseline_frames
        self._contact_off = contact_off
        self._contact_on = contact_on
        if contact_supervision and (baseline_frames < 4 or not 0 <= contact_off < contact_on):
            raise ValueError("Contact calibration needs >=4 frames and 0 <= off < on")
        self._tactile_calibration = None
        self._episodes = [json.loads(line) for line in (self._root / "meta/episodes.jsonl").read_text().splitlines()]
        self._tasks = {
            int(item["task_index"]): item["task"]
            for item in (json.loads(line) for line in (self._root / "meta/tasks.jsonl").read_text().splitlines())
        }
        # Local v2.1 exports use `n_frames`; older converted datasets use `length`.
        self._episode_ends = np.cumsum(
            [int(ep["length"] if "length" in ep else ep["n_frames"]) for ep in self._episodes]
        ).tolist()
        self._cache_ep_idx = None
        self._cache_df = None

    @property
    def tasks(self) -> dict[int, str]:
        return self._tasks

    def __len__(self) -> int:
        return int(self._episode_ends[-1])

    def _load_episode(self, ep_idx: int) -> pd.DataFrame:
        if self._cache_ep_idx != ep_idx:
            chunk_dir = self._root / "data" / "chunk-000"
            candidates = [
                chunk_dir / f"file-{ep_idx:03d}.parquet",
                chunk_dir / f"episode_{ep_idx:06d}.parquet",
            ]
            path = next((candidate for candidate in candidates if candidate.exists()), None)
            if path is None:
                raise FileNotFoundError(f"No parquet for episode {ep_idx}; tried {[p.name for p in candidates]}")
            self._cache_df = pd.read_parquet(path)
            self._cache_ep_idx = ep_idx
            if self._contact_supervision:
                from openpi.models.tactile_contact import calibrate_baseline
                initial = self._cache_df.iloc[:self._baseline_frames]
                frames = np.stack([self._parse_tactile_patch(x) for x in initial[self._tactile_key]])
                valid = np.stack(initial[self._tactile_valid_key]).astype(bool)
                self._tactile_calibration = calibrate_baseline(frames, valid)
        return self._cache_df

    @staticmethod
    def _parse_image(value) -> np.ndarray:
        if isinstance(value, dict) and "bytes" in value:
            with PIL.Image.open(io.BytesIO(value["bytes"])) as image:
                return np.asarray(image.convert("RGB"))
        return np.asarray(value)

    @staticmethod
    def _parse_tactile_patch(value) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32)
        if arr.size == 11 * 14 * 14:
            return arr.reshape(11, 14, 14)
        raise ValueError(f"unexpected tactile patch size {arr.shape}")

    def _tactile_window(self, df, local_idx: int) -> dict:
        horizon = int(self._tactile_num_frames or 16)
        taxels = np.zeros((11, horizon, 14, 14), dtype=np.float32)
        sensor_valid = np.zeros((11,), dtype=bool)
        frame_valid = np.zeros((horizon,), dtype=bool)
        sample_valid = np.zeros((11, horizon), dtype=bool)
        for i in range(horizon):
            src = local_idx - (horizon - 1 - i)
            if src < 0:
                continue
            row = df.iloc[src]
            taxels[:, i] = self._parse_tactile_patch(row[self._tactile_key])
            valid = np.asarray(row[self._tactile_valid_key], dtype=bool).reshape(11)
            taxels[:, i] *= valid[:, None, None]
            sensor_valid = np.logical_or(sensor_valid, valid)
            frame_valid[i] = True
            sample_valid[:, i] = valid
        if self._contact_supervision:
            from openpi.models.tactile_contact import contact_fields
            return contact_fields(
                taxels, sample_valid, *self._tactile_calibration,
                calibrated=local_idx >= self._baseline_frames - 1,
                off_threshold=self._contact_off, on_threshold=self._contact_on,
            )
        return {
            "tactile": taxels,
            "tactile_valid": sensor_valid,
            "tactile_frame_valid": frame_valid,
        }

    def __getitem__(self, index: SupportsIndex) -> dict:
        idx = index.__index__()
        ep_pos = bisect.bisect_right(self._episode_ends, idx)
        ep_start = 0 if ep_pos == 0 else self._episode_ends[ep_pos - 1]
        local_idx = idx - ep_start
        ep_idx = int(self._episodes[ep_pos]["episode_index"])

        df = self._load_episode(ep_idx)
        row = df.iloc[local_idx]
        actions = np.stack(df["action"].to_numpy()).astype(np.float32)
        action_indices = np.minimum(np.arange(local_idx, local_idx + self._action_horizon), len(df) - 1)
        sample = {
            "observation.images.top": self._parse_image(row["observation.images.top"]),
            "observation.images.left_wrist": self._parse_image(row["observation.images.left_wrist"]),
            "observation.images.right_wrist": self._parse_image(row["observation.images.right_wrist"]),
            "observation.state": np.asarray(row["observation.state"], dtype=np.float32),
            "action": actions[action_indices],
            "task_index": np.asarray(row["task_index"], dtype=np.int64),
            # Sample ids for skip-update debugging (passed through transforms, peeled in DataLoaderImpl).
            "sample_global_index": np.asarray(idx, dtype=np.int64),
            "sample_episode_index": np.asarray(ep_idx, dtype=np.int64),
            "sample_frame_in_episode": np.asarray(local_idx, dtype=np.int64),
        }
        if self._tactile_num_frames is not None:
            if self._tactile_key not in df.columns:
                raise KeyError(f"{self._root} is missing {self._tactile_key}")
            sample.update(self._tactile_window(df, local_idx))
        return sample


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training."""
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    local_home = pathlib.Path(os.environ.get("HF_LEROBOT_HOME", pathlib.Path.home() / ".cache/huggingface/lerobot"))
    local_root = local_home / repo_id
    if (local_root / "meta/episodes.jsonl").exists() and (local_root / "data").exists():
        tactile_frames = data_config.tactile_num_frames if data_config.tactile_enabled else None
        dataset = LocalLeRobotParquetDataset(
            repo_id,
            action_horizon,
            tactile_num_frames=tactile_frames,
            tactile_key=data_config.tactile_key,
            tactile_valid_key=data_config.tactile_valid_key,
            contact_supervision=data_config.tactile_contact_supervision,
            baseline_frames=data_config.tactile_baseline_frames,
            contact_off=data_config.tactile_contact_off,
            contact_on=data_config.tactile_contact_on,
        )
        if data_config.prompt_from_task:
            dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset.tasks)])
        return dataset

    if data_config.tactile_contact_supervision:
        raise ValueError("Contact supervision currently requires the local LeRobot parquet loader")
    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    dataset = lerobot_dataset.LeRobotDataset(
        data_config.repo_id,
        delta_timestamps={
            key: [t / dataset_meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    from openpi.fastslow.model import Pi05FastSlowConfig
    from openpi.vtla.stages import Pi05VisualAdaptConfig
    if isinstance(model_config, (Pi05FastSlowConfig, Pi05VisualAdaptConfig)):
        if framework != "jax":
            raise ValueError("The pi05 fast/slow implementation uses the existing JAX model")
        from openpi.fastslow.data import FastSlowDataset
        dataset = FastSlowDataset(data_config, model_config, skip_norm_stats=skip_norm_stats)
    else:
        dataset = create_torch_dataset(data_config, action_horizon, model_config)
        dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    result = DataLoaderImpl(data_config, data_loader)
    if isinstance(model_config, Pi05FastSlowConfig):
        result.fastslow_metadata = model_config.runtime_metadata()
    return result


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


_SAMPLE_ID_KEYS = (
    "sample_global_index",
    "sample_episode_index",
    "sample_frame_in_episode",
)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader
        # Populated on each __iter__ yield for train-loop skip debugging.
        self.last_batch_meta: dict[str, Any] = {}

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            meta = {}
            for key in _SAMPLE_ID_KEYS:
                if key in batch:
                    meta[key] = batch[key]
            self.last_batch_meta = meta
            if "fast_tactile" in batch or "action_valid" in batch:
                from openpi.fastslow.model import FastSlowObservation
                yield FastSlowObservation.from_dict(batch), batch["actions"]
            else:
                yield _model.Observation.from_dict(batch), batch["actions"]
