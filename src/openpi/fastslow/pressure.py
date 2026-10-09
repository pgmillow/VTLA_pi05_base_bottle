"""Causal rate-matched raw pressure histories for the existing pi05 encoder."""
from collections import deque
import numpy as np
from openpi.models.tactile_contact import calibrate_baseline, contact_fields


class PressureStream:
    """Accept *every sensor frame*, independently of slow/fast model requests."""
    def __init__(self, history=16, baseline_frames=16, fps=29.994):
        if history < 1 or baseline_frames < 4 or fps <= 0:
            raise ValueError("Invalid history/calibration length")
        self.history, self.baseline_frames = history, baseline_frames
        self.fps = float(fps)
        self.reset()

    def reset(self):
        self.frames = deque()
        self.valid = deque()
        self.timestamps = deque()
        self.initial, self.initial_valid, self.initial_times = [], [], []
        self.calibration = None
        self.calibration_time = float("inf")

    def update(self, raw, valid, timestamp):
        timestamp = float(timestamp)
        if not np.isfinite(timestamp) or (self.timestamps and timestamp <= self.timestamps[-1]):
            raise ValueError("Sensor timestamps must be finite and strictly increasing")
        raw = np.asarray(raw, np.float32).reshape(11, 14, 14).copy()
        valid = np.asarray(valid, bool).reshape(11).copy()
        self.frames.append(raw)
        self.valid.append(valid)
        self.timestamps.append(timestamp)
        # Keep enough raw samples to resample the same physical time span as
        # training, even if a sensor publishes faster than the dataset's fps.
        cutoff = timestamp - (max(self.history, self.baseline_frames)+2)/self.fps
        while len(self.timestamps) > 2 and self.timestamps[1] < cutoff:
            self.timestamps.popleft()
            self.frames.popleft()
            self.valid.popleft()
        if self.calibration is None:
            self.initial.append(raw)
            self.initial_valid.append(valid)
            self.initial_times.append(timestamp)
            target = self.initial_times[0] + np.arange(self.baseline_frames)/self.fps
            if timestamp >= target[-1]-1e-6:
                values, validity = self._sample(self.initial, self.initial_valid, self.initial_times, target)
                self.calibration = calibrate_baseline(values, validity)
                self.calibration_time = timestamp
                self.initial, self.initial_valid, self.initial_times = [], [], []

    def _sample(self, frames, valid, timestamps, target, max_gap_factor=1.5):
        timestamps = np.asarray(timestamps)
        # Causal zero-order hold: never interpolate using a future sensor frame.
        indices = np.searchsorted(timestamps, target+1e-6, side="right")-1
        chosen = indices.clip(min=0)
        fresh = (indices >= 0) & (target-timestamps[chosen] <= max_gap_factor/self.fps)
        values = np.asarray(frames, np.float32)[chosen]
        validity = np.asarray(valid, bool)[chosen] & fresh[:, None]
        return values, validity

    def snapshot(self, at_time):
        if self.calibration is None:
            raise ValueError("Feed 16 contact-free frames at the training rate before inference")
        if not np.isfinite(at_time):
            raise ValueError("Observation time must be finite")
        times = np.asarray(self.timestamps)
        eligible = np.flatnonzero(times <= at_time + 1e-6)
        if not len(eligible):
            raise ValueError("No causal pressure observation")
        end = int(eligible[-1])
        sensor_time = float(times[end])
        # Do not expose a calibration completed after the requested timestamp.
        if at_time + 1e-6 < self.calibration_time:
            raise ValueError("Calibration was not ready at the requested timestamp")
        target = sensor_time - np.arange(self.history-1, -1, -1)/self.fps
        raw, valid = self._sample(list(self.frames)[:end+1], list(self.valid)[:end+1], times[:end+1], target)
        fields = contact_fields(raw.transpose(1, 0, 2, 3), valid.T, *self.calibration)
        return fields, sensor_time
