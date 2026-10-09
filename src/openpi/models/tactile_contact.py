"""Causal, episode-fixed taxel calibration shared by training and online inference.

The user assumes the first baseline_frames frames of each episode are contact-free.
Labels below are noise-relative weak labels, not force measurements or human labels.
This module intentionally depends only on NumPy (safe in data-loader workers).
"""

from collections import deque

import numpy as np

PATCH_SHAPES = ((12, 8), (10, 8), (12, 8), (10, 8), (12, 8), (10, 8),
                (12, 8), (10, 8), (12, 8), (12, 8), (14, 8))
CONTACT_FIELDS = ("tactile_baseline", "tactile_scale", "tactile_sample_valid",
                  "tactile_contact", "tactile_contact_valid")


def spatial_mask():
    mask = np.zeros((11, 14, 14), dtype=bool)
    for i, (h, w) in enumerate(PATCH_SHAPES):
        mask[i, :h, :w] = True
    return mask


def calibrate_baseline(frames, valid, *, min_valid_frames=4):
    """frames [T,11,14,14]; validity [T,11]. Never uses post-calibration frames."""
    frames = np.asarray(frames, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    mask = spatial_mask()
    valid = valid & np.all(np.isfinite(np.where(mask[None], frames, 0)), axis=(-2, -1))
    baseline = np.zeros((11, 14, 14), dtype=np.float32)
    scale = np.ones(11, dtype=np.float32)
    usable = valid.sum(axis=0) >= min_valid_frames
    for sensor in range(11):
        if not usable[sensor]:
            continue
        samples = frames[valid[:, sensor], sensor]
        baseline[sensor] = np.median(samples, axis=0)
        noise = np.abs(samples - baseline[sensor])[:, mask[sensor]]
        scale[sensor] = max(float(np.quantile(noise, 0.995)), 1.0)
    baseline = np.where(mask, baseline, 0).astype(np.float32)
    return baseline, scale, usable


def contact_fields(taxels, sample_valid, baseline, scale, baseline_valid,
                   *, calibrated=True, off_threshold=1.0, on_threshold=3.0):
    """Return raw history plus masks and current-frame weak contact supervision.

    taxels [11,T,14,14]; sample_valid [11,T]. A positive requires any reliable
    sensor; a negative requires all 11 sensors reliable. Ambiguous scores are
    excluded from BCE. The warmup outputs an unavailable tactile branch.
    """
    if not 0 <= off_threshold < on_threshold:
        raise ValueError("Require 0 <= off_threshold < on_threshold")
    if not calibrated:
        # Offline loading already has the calibration segment in memory. Hide it
        # until all calibration frames have arrived, exactly as online inference.
        baseline = np.zeros_like(baseline)
        scale = np.ones_like(scale)
        baseline_valid = np.zeros_like(baseline_valid, dtype=bool)
    mask = spatial_mask()
    taxels = np.asarray(taxels, dtype=np.float32)
    finite = np.all(np.isfinite(np.where(mask[:, None], taxels, 0)), axis=(-2, -1))
    valid = np.asarray(sample_valid, dtype=bool) & finite & baseline_valid[:, None] & calibrated
    taxels = np.where(valid[:, :, None, None] & mask[:, None], taxels, 0).astype(np.float32)
    current_valid = valid[:, -1]
    normalized = (taxels[:, -1] - baseline) / scale[:, None, None]
    score = np.sum(np.abs(normalized) * mask, axis=(-2, -1)) / mask.sum(axis=(-2, -1))
    positive = np.any(current_valid & (score >= on_threshold))
    negative = np.all(current_valid) and np.all(score <= off_threshold)
    return {
        "tactile": taxels,
        "tactile_valid": current_valid,
        "tactile_frame_valid": valid.any(axis=0),
        "tactile_sample_valid": valid,
        "tactile_baseline": baseline.astype(np.float32),
        "tactile_scale": scale.astype(np.float32),
        "tactile_contact": np.asarray(positive, dtype=np.float32),
        "tactile_contact_valid": np.asarray(positive or negative, dtype=bool),
    }


class TactileContactPreprocessor:
    """Online equivalent of dataset calibration. Call reset at each episode.

    Feed all raw frames, including the initial contact-free calibration segment.
    Merge update()'s result into the normal Octopus policy input dictionary.
    """

    def __init__(self, num_frames=16, baseline_frames=16, off_threshold=1.0, on_threshold=3.0):
        if num_frames < 1 or baseline_frames < 4:
            raise ValueError("num_frames >= 1 and baseline_frames >= 4 required")
        self.num_frames = num_frames
        self.baseline_frames = baseline_frames
        self.off_threshold = off_threshold
        self.on_threshold = on_threshold
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.num_frames)
        self.valid_history = deque(maxlen=self.num_frames)
        self.calibration_frames = []
        self.calibration_valid = []
        self.calibration = None

    def update(self, frame, valid):
        frame = np.asarray(frame, dtype=np.float32).reshape(11, 14, 14)
        valid = np.asarray(valid, dtype=bool).reshape(11)
        self.history.append(frame.copy())
        self.valid_history.append(valid.copy())
        if self.calibration is None:
            self.calibration_frames.append(frame.copy())
            self.calibration_valid.append(valid.copy())
            if len(self.calibration_frames) == self.baseline_frames:
                self.calibration = calibrate_baseline(self.calibration_frames, self.calibration_valid)
        history = np.zeros((11, self.num_frames, 14, 14), dtype=np.float32)
        validity = np.zeros((11, self.num_frames), dtype=bool)
        n = len(self.history)
        history[:, -n:] = np.stack(self.history).transpose(1, 0, 2, 3)
        validity[:, -n:] = np.stack(self.valid_history).T
        calibration = self.calibration or (np.zeros((11, 14, 14), np.float32),
                                           np.ones(11, np.float32), np.zeros(11, bool))
        return contact_fields(history, validity, *calibration, calibrated=self.calibration is not None,
                              off_threshold=self.off_threshold, on_threshold=self.on_threshold)
