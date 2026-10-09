"""Timestamped future-action replacement; independent of any model or robot SDK."""
from dataclasses import dataclass
import math
import threading
import numpy as np


@dataclass(frozen=True)
class ActionUpdate:
    session_id: str
    chunk_id: int
    revision: int
    observation_time: float
    sensor_time: float
    actions: np.ndarray
    valid_from: int
    fps: float

    def as_dict(self):
        return {**self.__dict__, "actions": self.actions.tolist()}


class ActionBuffer:
    """Executor-side buffer. Client timestamps must share one monotonic clock.

    It never returns past commands. It bounds per-command changes against the
    latest actual command, so a sequence of refinements cannot stack corrections
    inside one control tick. Limits are supplied in the robot's physical units.
    """
    def __init__(self, action_dim, max_step, initial_command, blend_steps=3):
        self.action_dim = action_dim
        self.max_step = np.asarray(max_step, np.float32)
        if self.max_step.shape != (action_dim,) or not np.isfinite(self.max_step).all() or np.any(self.max_step <= 0):
            raise ValueError("Provide one positive finite per-command bound per action dimension")
        if blend_steps < 1:
            raise ValueError("blend_steps must be positive")
        self.blend_steps, self.lock = blend_steps, threading.RLock()
        self.reset(initial_command=initial_command)

    def reset(self, session_id=None, initial_command=None):
        with self.lock:
            self.session_id = session_id
            self.chunk_id, self.revision = -1, -1
            self.actions = None
            self.next_index = 0
            if initial_command is None:
                self.last_command = None
            else:
                command = np.asarray(initial_command, np.float32)
                if command.shape != (self.action_dim,) or not np.isfinite(command).all():
                    raise ValueError("Initial command must be finite and match action dimensions")
                self.last_command = command.copy()
            self.last_query_time = -math.inf
            self.last_emit_time = -math.inf
            self.last_sensor_time = -math.inf

    def apply(self, update, now):
        with self.lock:
            if not np.isfinite(now) or now < self.last_query_time or update.observation_time > now or update.sensor_time > now:
                return False
            if self.session_id is not None and update.session_id != self.session_id:
                return False
            if update.chunk_id < self.chunk_id or (update.chunk_id == self.chunk_id and update.revision <= self.revision):
                return False
            a = np.asarray(update.actions, np.float32)
            if a.ndim != 2 or a.shape[1] != self.action_dim or not np.isfinite(a).all() or update.fps <= 0:
                return False
            if not all(np.isfinite(v) for v in (update.fps, update.observation_time, update.sensor_time)):
                return False
            # Ceil: the command timestamp must be >= arrival time.
            first = max(update.valid_from, math.ceil((now-update.observation_time)*update.fps - 1e-6))
            same = update.chunk_id == self.chunk_id
            if same:
                if update.observation_time != self.observation_time or update.fps != self.fps or a.shape != self.actions.shape:
                    return False
                if update.sensor_time < self.last_sensor_time:
                    return False
                first = max(first, self.next_index)
            elif self.actions is not None and update.observation_time <= self.observation_time:
                return False
            if not 0 <= first < len(a):
                return False
            a = a.copy()
            if same:
                a[:first] = self.actions[:first]
                count = min(self.blend_steps, len(a)-first)
                alpha = np.arange(1, count+1, dtype=np.float32)[:, None] / self.blend_steps
                a[first:first+count] = (1-alpha)*self.actions[first:first+count] + alpha*a[first:first+count]
            self.session_id, self.chunk_id, self.revision = update.session_id, update.chunk_id, update.revision
            self.actions, self.next_index = a, first
            self.observation_time, self.fps = update.observation_time, update.fps
            self.last_sensor_time = update.sensor_time
            return True

    def next_action(self, now):
        """Returns (chunk_id, index, action) once per trajectory step, else None."""
        with self.lock:
            if not np.isfinite(now) or now < self.last_query_time:
                raise ValueError("Executor clock must be finite and monotonic")
            self.last_query_time = now
            if self.actions is None or now < self.observation_time:
                return None
            if self.last_command is None:
                raise ValueError("Initialize the buffer with the robot's actual last command after reset")
            if now-self.last_emit_time < 1/self.fps - 1e-6:
                return None
            due = math.floor((now-self.observation_time)*self.fps + 1e-6)
            if due < self.next_index:
                return None
            index = max(due, self.next_index)
            if index >= len(self.actions):
                return None
            action = self.actions[index].copy()
            if self.last_command is not None:
                action = self.last_command + np.clip(action-self.last_command, -self.max_step, self.max_step)
            self.last_command = action.copy()
            self.last_emit_time = now
            self.next_index = index+1
            return self.chunk_id, index, action

