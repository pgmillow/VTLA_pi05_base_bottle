"""Old visual/state context + causal fresh touch, aligned to the same action chunk."""
import bisect
import json

import numpy as np

from openpi import transforms
from openpi.training.data_loader import LocalLeRobotParquetDataset


class FastSlowDataset:
    def __init__(self, data_config, model_config, *, split="train", skip_norm_stats=False):
        if split not in ("train", "validation"):
            raise ValueError("Unknown dataset split")
        self.paired = getattr(model_config, 'training_objective', 'paired') == 'paired'
        self.use_touch = model_config.tactile_enabled
        if self.use_touch and not data_config.tactile_contact_supervision:
            raise ValueError("Use episode-calibrated tactile data")
        dataset_type = LocalLeRobotParquetDataset
        if getattr(model_config, 'use_frame_cache', False):
            from openpi.vtla.cache import CachedLeRobotDataset
            dataset_type = CachedLeRobotDataset
        self.base = dataset_type(
            data_config.repo_id, model_config.action_horizon,
            tactile_num_frames=data_config.tactile_num_frames if self.use_touch else None,
            tactile_key=data_config.tactile_key, tactile_valid_key=data_config.tactile_valid_key,
            contact_supervision=self.use_touch, baseline_frames=data_config.tactile_baseline_frames,
            contact_off=data_config.tactile_contact_off, contact_on=data_config.tactile_contact_on)
        self.model_config = model_config
        info_path = self.base._root / "meta/info.json"
        if info_path.exists() and abs(float(json.loads(info_path.read_text())["fps"])-model_config.fps) > .02:
            raise ValueError("Dataset fps differs from the model's pressure history and action timing")
        self.baseline_frames = data_config.tactile_baseline_frames
        if self.baseline_frames != 16:
            raise ValueError("Use the same 16-frame calibration as the online runtime")
        order = np.random.default_rng(model_config.split_seed).permutation(len(self.base._episodes))
        count = max(1, int(np.ceil(len(order)*model_config.validation_fraction)))
        if count >= len(order):
            raise ValueError("Need at least two episodes for train/validation split")
        self.episode_positions = sorted(order[count:] if split == "train" else order[:count])
        self.episode_ids = [int(self.base._episodes[p]["episode_index"]) for p in self.episode_positions]
        self.starts, self.lengths = [], []
        for p in self.episode_positions:
            start = 0 if p == 0 else self.base._episode_ends[p-1]
            length = self.base._episode_ends[p]-start
            self.starts.append(start+self.baseline_frames-1)
            self.lengths.append(max(0, length-self.baseline_frames+1))
        self.ends = np.cumsum(self.lengths).tolist()
        if not self.ends or self.ends[-1] == 0:
            raise ValueError("No samples after the contact-free calibration interval")
        if data_config.norm_stats is None and not skip_norm_stats:
            raise ValueError("Training requires normalization statistics from its declared assets")
        norm = {} if skip_norm_stats else data_config.norm_stats
        self.transform = transforms.compose([
            transforms.PromptFromLeRobotTask(self.base.tasks),
            *data_config.repack_transforms.inputs, *data_config.data_transforms.inputs,
            transforms.Normalize(norm, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs])
        self.split = split

    def __len__(self):
        return self.ends[-1]

    def pair_indices(self, index, delay=None):
        if not 0 <= index < len(self):
            raise IndexError(index)
        slot = bisect.bisect_right(self.ends, index)
        slow = self.starts[slot]+index-(0 if slot == 0 else self.ends[slot-1])
        end = self.base._episode_ends[self.episode_positions[slot]]
        max_delay = min(self.model_config.max_delay_frames, end-slow-1)
        if delay is None:
            delay = (int(np.random.randint(max_delay+1)) if self.split == "train"
                     else int(np.random.default_rng(index+self.model_config.split_seed).integers(max_delay+1)))
        if not 0 <= delay <= max_delay:
            raise ValueError("Delay crosses an episode boundary or exceeds the configured limit")
        return slow, slow+delay, end

    def __getitem__(self, index):
        slow_index, fast_index, end = self.pair_indices(int(index), delay=None if self.paired else 0)
        raw = self.base[slow_index]
        # Same episode is still resident: only read tactile at t1, never future
        # images/state/actions. Calibration is fixed from the first 16 frames.
        sample = self.transform(raw)
        if self.paired:
            episode_start = slow_index-int(raw["sample_frame_in_episode"])
            sample["fast_tactile"] = self.base._tactile_window(self.base._cache_df, fast_index-episode_start)
            sample["fast_start"] = np.asarray(fast_index-slow_index, np.int32)
        sample["action_valid"] = np.arange(self.model_config.action_horizon)+slow_index < end
        return sample
