#!/usr/bin/env python3
"""Single-MCAP → LeRobot v2.1, aligned with mcap_to_lerobot_v3.py.

Output schema matches the v3 converter:
  observation.state (26): left/right arm qpos6, left/right hand qpos1,
                          left/right EE xyz+rpy6
  action (14): left/right arm cmd6, left/right hand cmd1

Numeric modes:
  safe:            arm/EE strict linear, RPY unwrap before interpolation,
                   hand state/cmd previous-value hold with a bounded age
  strict:         all numeric streams use raw linear interpolation with a gate
  continuous (default): keep every top-camera anchor; all numeric streams
                        interpolate across gaps, RPY is unwrapped, and image
                        streams use ungated nearest-neighbor sampling

Safe/strict image matching uses a gate; continuous mode has no image time gate.

One input .mcap → one output directory with:
  meta/{info.json,episodes.jsonl,tasks.jsonl}
  data/chunk-000/episode_000000.parquet
  viz/all_cmd.png
  convert_report.json

Usage:
  /home/xudi_ge/openpi/.venv/bin/python \\
    /data/mcap_to_lerobot_v21.py \\
    --mcap /data/show_data_V1/left_0722/2026-07-23_09:26:32.850801277.mcap \\
    --out-dir /tmp/lerobot_v21_demo \\
    --task "pour water from the right-hand bottle into the left-hand cup"
"""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from PIL import Image

# ---------------------------------------------------------------------------
# Schema / topics (aligned with mcap_to_lerobot_v3.py)
# ---------------------------------------------------------------------------
TOPIC_TOP = "/realsense/top/color/image_raw/compressed"
TOPIC_LEFT_IMG = "/realsense/left_hand/color/image_rect_raw/compressed"
TOPIC_RIGHT_IMG = "/realsense/right_hand/color/image_rect_raw/compressed"
TOPIC_VLA_ARM_STATE = "/vla_teleop/proprioception"
TOPIC_VLA_ARM_CMD = "/vla_teleop/action_label"
TOPIC_LEFT_EE_POS = "/left_arm/ee_position"
TOPIC_LEFT_EE_RPY = "/left_arm/ee_rpy"
TOPIC_RIGHT_EE_POS = "/right_arm/ee_position"
TOPIC_RIGHT_EE_RPY = "/right_arm/ee_rpy"
TOPIC_LEFT_HAND_JS = "/inspire/left_hand/joint_states"
TOPIC_RIGHT_HAND_JS = "/inspire/right_hand/joint_states"
TOPIC_LEFT_HAND_CMD = "/inspire/left_hand/joint_cmd"
TOPIC_RIGHT_HAND_CMD = "/inspire/right_hand/joint_cmd"

ARM_DOF = 6
# Match mcap_to_lerobot_v3.py: one mapped hand DoF on each side.
HAND_DOF = 1
EE_POSE_DOF = 6
ACTION_DIM = 2 * ARM_DOF + 2 * HAND_DOF  # 14
STATE_DIM = ACTION_DIM + 2 * EE_POSE_DOF  # 26
IMG_HWC = (224, 224, 3)

OBS_TOP = "observation.images.top"
OBS_LEFT = "observation.images.left_wrist"
OBS_RIGHT = "observation.images.right_wrist"


@dataclass
class NumBuf:
    t: list[float] = field(default_factory=list)
    v: list[np.ndarray] = field(default_factory=list)
    names: list[str] | None = None


@dataclass
class ImgBuf:
    t: list[float] = field(default_factory=list)
    raw: list[object] = field(default_factory=list)


def _stamp_s(msg: object, log_time_ns: int) -> Optional[float]:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None) if header is not None else None
    if stamp is not None:
        sec = int(getattr(stamp, "sec", 0))
        nanosec = int(getattr(stamp, "nanosec", getattr(stamp, "nsec", 0)))
        if sec != 0 or nanosec != 0:
            return float(sec) + float(nanosec) * 1e-9
    if log_time_ns > 0:
        return float(log_time_ns) * 1e-9
    return None


def _require_stamp(msg: object, topic: str, log_time_ns: int) -> float:
    t = _stamp_s(msg, log_time_ns)
    if t is None:
        raise RuntimeError(f"{topic}: missing header.stamp")
    return t


def _pad(vec: np.ndarray, dim: int) -> np.ndarray:
    out = np.zeros((dim,), dtype=np.float32)
    n = min(dim, int(vec.shape[0]))
    out[:n] = vec[:n]
    return out


def _hand_pos(msg: object, *, min_dim: int = 1) -> Optional[np.ndarray]:
    """Return published hand joints (variable length). Caller pads to left/right DoF."""
    position = getattr(msg, "position", None)
    if not position:
        return None
    vec = np.asarray(list(position), dtype=np.float32)
    if vec.shape[0] < min_dim:
        return _pad(vec, min_dim)
    return vec


def _hand_names(msg: object) -> list[str] | None:
    names = list(getattr(msg, "name", []) or [])
    return names or None


def _vec3(msg: object) -> np.ndarray:
    if all(hasattr(msg, a) for a in ("x", "y", "z")):
        return np.asarray([float(msg.x), float(msg.y), float(msg.z)], dtype=np.float32)
    data = getattr(msg, "data", None)
    if data is not None:
        return np.asarray(list(data), dtype=np.float32)[:3]
    raise RuntimeError("expected Vector3/Point")


def _decode_rgb(msg: object) -> np.ndarray:
    data = bytes(getattr(msg, "data", b"") or b"")
    if not data:
        raise ValueError("empty compressed image")
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _letterbox_224(img: np.ndarray) -> np.ndarray:
    th, tw, _ = IMG_HWC
    sh, sw = img.shape[:2]
    scale = min(tw / sw, th / sh)
    nw = max(1, int(round(sw * scale)))
    nh = max(1, int(round(sh * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    y0 = (th - nh) // 2
    x0 = (tw - nw) // 2
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas


def _jpeg_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return bytes(buf)


def _sanitize_num(buf: NumBuf) -> tuple[np.ndarray, np.ndarray, list[str] | None]:
    if not buf.t:
        return np.asarray([], dtype=np.float64), np.zeros((0, 1), dtype=np.float32), buf.names
    # pad jagged hand vectors to max dim in this stream
    max_dim = max(int(x.shape[0]) for x in buf.v)
    order = np.argsort(np.asarray(buf.t, dtype=np.float64))
    t = np.asarray(buf.t, dtype=np.float64)[order]
    v = np.stack([_pad(buf.v[i], max_dim) for i in order], axis=0).astype(np.float32)
    # drop non-monotonic duplicates by keeping increasing t
    keep = np.ones(len(t), dtype=bool)
    last = -np.inf
    for i, ti in enumerate(t):
        if ti < last:
            keep[i] = False
        else:
            last = ti
    return t[keep], v[keep], buf.names


def _sanitize_img(buf: ImgBuf) -> tuple[np.ndarray, list]:
    if not buf.t:
        return np.asarray([], dtype=np.float64), []
    order = np.argsort(np.asarray(buf.t, dtype=np.float64))
    t = np.asarray(buf.t, dtype=np.float64)[order]
    frames = [buf.raw[i] for i in order]
    keep = np.ones(len(t), dtype=bool)
    last = -np.inf
    for i, ti in enumerate(t):
        if ti < last:
            keep[i] = False
        else:
            last = ti
    t2 = t[keep]
    frames2 = [frames[i] for i, k in enumerate(keep) if k]
    return t2, frames2


def remove_glitches(
    t: np.ndarray, v: np.ndarray, *, jump_abs: float, glitch_ms: float
) -> tuple[np.ndarray, int]:
    """Hold previous level across short leave-and-return spikes. Works on 1D or ND (uses L2)."""
    if v.size == 0 or len(t) < 3:
        return v.copy(), 0
    cleaned = v.copy()
    glitch_s = glitch_ms / 1000.0
    n_fixed = 0
    i = 0
    n = len(cleaned)

    def _mag(a: np.ndarray, b: np.ndarray) -> float:
        d = a.astype(np.float64) - b.astype(np.float64)
        return float(np.linalg.norm(d))

    while i < n - 1:
        if _mag(cleaned[i + 1], cleaned[i]) < jump_abs:
            i += 1
            continue
        before = cleaned[i].copy()
        t_jump = float(t[i + 1])
        return_j = None
        for j in range(i + 2, n):
            if float(t[j]) > t_jump + glitch_s:
                break
            if _mag(cleaned[j], before) <= jump_abs * 0.5:
                return_j = j
                break
        if return_j is not None:
            cleaned[i + 1 : return_j] = before
            n_fixed += 1
            i = return_j
        else:
            i += 1
    return cleaned, n_fixed


def sample_nearest_idx(ts: np.ndarray, target: float, gate_s: float) -> Optional[int]:
    if ts.size == 0:
        return None
    right = int(np.searchsorted(ts, target, side="left"))
    cands = []
    if right < ts.size:
        cands.append(right)
    if right > 0:
        cands.append(right - 1)
    best = min(cands, key=lambda i: abs(float(ts[i]) - target))
    if abs(float(ts[best]) - target) > gate_s:
        return None
    return best


def sample_linear(
    ts: np.ndarray, values: np.ndarray, target: float, gate_s: float
) -> Optional[np.ndarray]:
    if ts.size == 0:
        return None
    # nearest must be within gate first
    if sample_nearest_idx(ts, target, gate_s) is None:
        return None
    right = int(np.searchsorted(ts, target, side="left"))
    if right < ts.size and abs(float(ts[right]) - target) < 1e-9:
        return np.asarray(values[right], dtype=np.float32)
    left = right - 1
    if left < 0 or right >= ts.size:
        return None
    prev_age = target - float(ts[left])
    next_age = float(ts[right]) - target
    if prev_age > gate_s or next_age > gate_s:
        return None
    denom = float(ts[right]) - float(ts[left])
    if denom <= 0:
        return None
    alpha = (target - float(ts[left])) / denom
    return ((1.0 - alpha) * values[left] + alpha * values[right]).astype(np.float32)


def sample_continuous_linear(
    ts: np.ndarray, values: np.ndarray, target: float
) -> Optional[np.ndarray]:
    """Linear interpolation without a gap gate; hold stream endpoints."""
    if ts.size == 0:
        return None
    right = int(np.searchsorted(ts, target, side="left"))
    if right <= 0:
        return np.asarray(values[0], dtype=np.float32)
    if right >= ts.size:
        return np.asarray(values[-1], dtype=np.float32)
    if abs(float(ts[right]) - target) < 1e-9:
        return np.asarray(values[right], dtype=np.float32)
    left = right - 1
    denom = float(ts[right]) - float(ts[left])
    if denom <= 0:
        return None
    alpha = (target - float(ts[left])) / denom
    return ((1.0 - alpha) * values[left] + alpha * values[right]).astype(np.float32)


def sample_hold(
    ts: np.ndarray,
    values: np.ndarray,
    target: float,
    gate_s: float,
) -> Optional[np.ndarray]:
    """Last sample at or before target, bounded by a maximum sample age."""
    if ts.size == 0:
        return None
    idx = int(np.searchsorted(ts, target, side="right") - 1)
    if idx < 0:
        return None
    if target - float(ts[idx]) > gate_s:
        return None
    return np.asarray(values[idx], dtype=np.float32)


def load_mcap(path: Path) -> dict:
    topics = [
        TOPIC_TOP,
        TOPIC_LEFT_IMG,
        TOPIC_RIGHT_IMG,
        TOPIC_VLA_ARM_STATE,
        TOPIC_VLA_ARM_CMD,
        TOPIC_LEFT_EE_POS,
        TOPIC_LEFT_EE_RPY,
        TOPIC_RIGHT_EE_POS,
        TOPIC_RIGHT_EE_RPY,
        TOPIC_LEFT_HAND_JS,
        TOPIC_RIGHT_HAND_JS,
        TOPIC_LEFT_HAND_CMD,
        TOPIC_RIGHT_HAND_CMD,
    ]
    top, left_img, right_img = ImgBuf(), ImgBuf(), ImgBuf()
    arm_q, arm_cmd = NumBuf(), NumBuf()
    lee_p, lee_r, ree_p, ree_r = NumBuf(), NumBuf(), NumBuf(), NumBuf()
    lhq, rhq, lhc, rhc = NumBuf(), NumBuf(), NumBuf(), NumBuf()

    with path.open("rb") as handle:
        reader = make_reader(handle, decoder_factories=[DecoderFactory()])
        for _schema, channel, message, ros_msg in reader.iter_decoded_messages(topics=topics):
            topic = channel.topic
            log_ns = int(message.log_time)

            if topic == TOPIC_TOP:
                t = _stamp_s(ros_msg, log_ns)
                if t is None:
                    continue
                top.t.append(t)
                top.raw.append(ros_msg)
            elif topic == TOPIC_LEFT_IMG:
                t = _stamp_s(ros_msg, log_ns)
                if t is None:
                    continue
                left_img.t.append(t)
                left_img.raw.append(ros_msg)
            elif topic == TOPIC_RIGHT_IMG:
                t = _stamp_s(ros_msg, log_ns)
                if t is None:
                    continue
                right_img.t.append(t)
                right_img.raw.append(ros_msg)
            elif topic == TOPIC_VLA_ARM_STATE:
                t = _require_stamp(ros_msg, topic, log_ns)
                pos = _pad(np.asarray(list(ros_msg.position), dtype=np.float32), 12)
                arm_q.t.append(t)
                arm_q.v.append(pos)
            elif topic == TOPIC_VLA_ARM_CMD:
                t = _require_stamp(ros_msg, topic, log_ns)
                cmd = _pad(np.asarray(list(ros_msg.position), dtype=np.float32), 12)
                arm_cmd.t.append(t)
                arm_cmd.v.append(cmd)
            elif topic == TOPIC_LEFT_EE_POS:
                t = _stamp_s(ros_msg, log_ns) or (log_ns * 1e-9)
                lee_p.t.append(t)
                lee_p.v.append(_vec3(ros_msg))
            elif topic == TOPIC_LEFT_EE_RPY:
                t = _stamp_s(ros_msg, log_ns) or (log_ns * 1e-9)
                lee_r.t.append(t)
                lee_r.v.append(_vec3(ros_msg))
            elif topic == TOPIC_RIGHT_EE_POS:
                t = _stamp_s(ros_msg, log_ns) or (log_ns * 1e-9)
                ree_p.t.append(t)
                ree_p.v.append(_vec3(ros_msg))
            elif topic == TOPIC_RIGHT_EE_RPY:
                t = _stamp_s(ros_msg, log_ns) or (log_ns * 1e-9)
                ree_r.t.append(t)
                ree_r.v.append(_vec3(ros_msg))
            elif topic == TOPIC_LEFT_HAND_JS:
                t = _require_stamp(ros_msg, topic, log_ns)
                hp = _hand_pos(ros_msg)
                if hp is not None:
                    lhq.t.append(t)
                    lhq.v.append(hp)
                    if lhq.names is None:
                        lhq.names = _hand_names(ros_msg)
            elif topic == TOPIC_RIGHT_HAND_JS:
                t = _require_stamp(ros_msg, topic, log_ns)
                hp = _hand_pos(ros_msg)
                if hp is not None:
                    rhq.t.append(t)
                    rhq.v.append(hp)
                    if rhq.names is None:
                        rhq.names = _hand_names(ros_msg)
            elif topic == TOPIC_LEFT_HAND_CMD:
                t = _require_stamp(ros_msg, topic, log_ns)
                hp = _hand_pos(ros_msg)
                if hp is not None:
                    lhc.t.append(t)
                    lhc.v.append(hp)
                    if lhc.names is None:
                        lhc.names = _hand_names(ros_msg)
            elif topic == TOPIC_RIGHT_HAND_CMD:
                t = _require_stamp(ros_msg, topic, log_ns)
                hp = _hand_pos(ros_msg)
                if hp is not None:
                    rhc.t.append(t)
                    rhc.v.append(hp)
                    if rhc.names is None:
                        rhc.names = _hand_names(ros_msg)

    return {
        "top": _sanitize_img(top),
        "left_img": _sanitize_img(left_img),
        "right_img": _sanitize_img(right_img),
        "arm_q": _sanitize_num(arm_q),
        "arm_cmd": _sanitize_num(arm_cmd),
        "lee_p": _sanitize_num(lee_p),
        "lee_r": _sanitize_num(lee_r),
        "ree_p": _sanitize_num(ree_p),
        "ree_r": _sanitize_num(ree_r),
        "lhq": _sanitize_num(lhq),
        "rhq": _sanitize_num(rhq),
        "lhc": _sanitize_num(lhc),
        "rhc": _sanitize_num(rhc),
    }


def features_dict() -> dict:
    h, w, c = IMG_HWC
    img = {"dtype": "image", "shape": [h, w, c], "names": ["height", "width", "channel"]}
    state_names = (
        [f"left_arm_qpos{i}" for i in range(ARM_DOF)]
        + [f"right_arm_qpos{i}" for i in range(ARM_DOF)]
        + [f"left_hand_qpos{i}" for i in range(HAND_DOF)]
        + [f"right_hand_qpos{i}" for i in range(HAND_DOF)]
        + [f"left_ee_{n}" for n in ("x", "y", "z", "roll", "pitch", "yaw")]
        + [f"right_ee_{n}" for n in ("x", "y", "z", "roll", "pitch", "yaw")]
    )
    action_names = (
        [f"left_arm_cmd_pos{i}" for i in range(ARM_DOF)]
        + [f"right_arm_cmd_pos{i}" for i in range(ARM_DOF)]
        + [f"left_hand_cmd_pos{i}" for i in range(HAND_DOF)]
        + [f"right_hand_cmd_pos{i}" for i in range(HAND_DOF)]
    )
    return {
        OBS_TOP: dict(img),
        OBS_LEFT: dict(img),
        OBS_RIGHT: dict(img),
        "observation.state": {"dtype": "float32", "shape": [STATE_DIM], "names": state_names},
        "action": {"dtype": "float32", "shape": [ACTION_DIM], "names": action_names},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }


def convert_one(
    mcap_path: Path,
    out_dir: Path,
    *,
    task: str,
    image_ms: float,
    arm_ms: float,
    jump_abs: float,
    glitch_ms: float,
    numeric_interp_mode: str,
    hand_ms: float,
    fps_override: float | None,
    episode_index: int = 0,
    index_offset: int = 0,
    chunks_size: int = 1000,
    write_meta: bool = True,
) -> dict:
    if numeric_interp_mode not in {"safe", "strict", "continuous"}:
        raise ValueError(f"unsupported numeric_interp_mode={numeric_interp_mode!r}")
    data = load_mcap(mcap_path)
    top_ts, top_frames = data["top"]
    left_ts, left_frames = data["left_img"]
    right_ts, right_frames = data["right_img"]
    if top_ts.size == 0:
        raise RuntimeError("no top camera frames")
    if left_ts.size == 0 or right_ts.size == 0:
        raise RuntimeError("missing wrist camera frames")

    image_s = image_ms / 1000.0
    arm_s = arm_ms / 1000.0
    hand_s = hand_ms / 1000.0

    # Remove known short hand leave-and-return glitches before resampling.
    hand_stats = {}
    hand_names: dict[str, list[str] | None] = {}
    for key in ("lhq", "rhq", "lhc", "rhc"):
        t, v, names = data[key]
        cleaned, n_fix = remove_glitches(t, v, jump_abs=jump_abs, glitch_ms=glitch_ms)
        data[key] = (t, cleaned)
        hand_names[key] = names
        hand_stats[key] = {
            "n": int(t.size),
            "glitch_fixed": int(n_fix),
            "dof": int(cleaned.shape[1]) if cleaned.ndim == 2 and cleaned.size else 0,
            "names": names,
        }

    # Safe mode interpolates continuous Euler trajectories rather than the raw
    # wrapped representation. Do this independently for each RPY component.
    if numeric_interp_mode in {"safe", "continuous"}:
        for key in ("lee_r", "ree_r"):
            ts, values, *_ = data[key]
            data[key] = (ts, np.unwrap(values.astype(np.float64), axis=0).astype(np.float32))

    # Decode caches
    left_cache: dict[int, np.ndarray] = {}
    right_cache: dict[int, np.ndarray] = {}

    def _wrist(frames, idx, cache):
        if idx not in cache:
            cache[idx] = _letterbox_224(_decode_rgb(frames[idx]))
        return cache[idx]

    rows = {
        OBS_TOP: [],
        OBS_LEFT: [],
        OBS_RIGHT: [],
        "observation.state": [],
        "action": [],
        "timestamp": [],
        "frame_index": [],
        "episode_index": [],
        "index": [],
        "task_index": [],
    }
    # for viz (kept frames) — full hand cmd vectors
    viz_t: list[float] = []
    viz_arm_cmd: list[np.ndarray] = []
    viz_l_hand: list[np.ndarray] = []
    viz_r_hand: list[np.ndarray] = []
    viz_hold_left: list[bool] = []
    viz_hold_right: list[bool] = []

    drop_reasons: dict[str, int] = {}
    n_hold_l = n_hold_r = 0
    t0 = float(top_ts[0])

    for i, target_t in enumerate(top_ts):
        reasons: list[str] = []

        if numeric_interp_mode == "continuous":
            li = sample_nearest_idx(left_ts, float(target_t), float("inf"))
            ri = sample_nearest_idx(right_ts, float(target_t), float("inf"))
        else:
            li = sample_nearest_idx(left_ts, float(target_t), image_s)
            ri = sample_nearest_idx(right_ts, float(target_t), image_s)
        if li is None:
            reasons.append("left_wrist")
        if ri is None:
            reasons.append("right_wrist")

        def _tv(key: str) -> tuple[np.ndarray, np.ndarray]:
            item = data[key]
            return item[0], item[1]

        numeric_sampler = (
            sample_continuous_linear
            if numeric_interp_mode == "continuous"
            else lambda ts, values, target: sample_linear(ts, values, target, arm_s)
        )
        arm_q = numeric_sampler(*_tv("arm_q"), float(target_t))
        arm_cmd = numeric_sampler(*_tv("arm_cmd"), float(target_t))
        lee_p = numeric_sampler(*_tv("lee_p"), float(target_t))
        lee_r = numeric_sampler(*_tv("lee_r"), float(target_t))
        ree_p = numeric_sampler(*_tv("ree_p"), float(target_t))
        ree_r = numeric_sampler(*_tv("ree_r"), float(target_t))
        if arm_q is None:
            reasons.append("arm_q")
        if arm_cmd is None:
            reasons.append("arm_cmd")
        if lee_p is None or lee_r is None:
            reasons.append("left_ee")
        if ree_p is None or ree_r is None:
            reasons.append("right_ee")

        if numeric_interp_mode == "safe":
            lhq = sample_hold(*_tv("lhq"), float(target_t), hand_s)
            rhq = sample_hold(*_tv("rhq"), float(target_t), hand_s)
            lhc = sample_hold(*_tv("lhc"), float(target_t), hand_s)
            rhc = sample_hold(*_tv("rhc"), float(target_t), hand_s)
        else:
            lhq = numeric_sampler(*_tv("lhq"), float(target_t))
            rhq = numeric_sampler(*_tv("rhq"), float(target_t))
            lhc = numeric_sampler(*_tv("lhc"), float(target_t))
            rhc = numeric_sampler(*_tv("rhc"), float(target_t))
        if lhq is None:
            reasons.append("left_hand_q")
        if rhq is None:
            reasons.append("right_hand_q")
        if lhc is None:
            reasons.append("left_hand_cmd")
        if rhc is None:
            reasons.append("right_hand_cmd")

        if reasons:
            for r in reasons:
                drop_reasons[r] = drop_reasons.get(r, 0) + 1
            continue

        assert li is not None and ri is not None
        assert arm_q is not None and arm_cmd is not None
        assert lee_p is not None and lee_r is not None and ree_p is not None and ree_r is not None
        assert lhq is not None and rhq is not None and lhc is not None and rhc is not None

        # Same position-only schema and left/right arm ordering as the v3 converter.
        state = np.concatenate(
            [
                arm_q[6:12],
                arm_q[:6],
                _pad(lhq, HAND_DOF),
                _pad(rhq, HAND_DOF),
                lee_p[:3],
                lee_r[:3],
                ree_p[:3],
                ree_r[:3],
            ],
            axis=0,
        ).astype(np.float32)
        action = np.concatenate(
            [
                arm_cmd[6:12],
                arm_cmd[:6],
                _pad(lhc, HAND_DOF),
                _pad(rhc, HAND_DOF),
            ],
            axis=0,
        ).astype(np.float32)
        if state.shape[0] != STATE_DIM or action.shape[0] != ACTION_DIM:
            raise RuntimeError(
                f"bad dims state={state.shape} action={action.shape} "
                f"expected {STATE_DIM}/{ACTION_DIM}"
            )

        top_img = _letterbox_224(_decode_rgb(top_frames[i]))
        left_img = _wrist(left_frames, li, left_cache)
        right_img = _wrist(right_frames, ri, right_cache)

        fi = len(rows["frame_index"])
        rows[OBS_TOP].append({"bytes": _jpeg_bytes(top_img), "path": f"frame_{fi:06d}_top.jpg"})
        rows[OBS_LEFT].append({"bytes": _jpeg_bytes(left_img), "path": f"frame_{fi:06d}_left.jpg"})
        rows[OBS_RIGHT].append({"bytes": _jpeg_bytes(right_img), "path": f"frame_{fi:06d}_right.jpg"})
        rows["observation.state"].append(state)
        rows["action"].append(action)
        rows["timestamp"].append(np.asarray([float(target_t - t0)], dtype=np.float32))
        rows["frame_index"].append(np.asarray([fi], dtype=np.int64))
        rows["episode_index"].append(np.asarray([int(episode_index)], dtype=np.int64))
        rows["index"].append(np.asarray([int(index_offset + fi)], dtype=np.int64))
        rows["task_index"].append(np.asarray([0], dtype=np.int64))

        # hold age markers for viz (vs nearest would differ; mark if prev age > arm_ms as "long hold")
        lhc_ts = data["lhc"][0]
        rhc_ts = data["rhc"][0]
        l_age = float(target_t) - float(lhc_ts[int(np.searchsorted(lhc_ts, target_t, side="right") - 1)])
        r_age = float(target_t) - float(rhc_ts[int(np.searchsorted(rhc_ts, target_t, side="right") - 1)])
        long_l = l_age > arm_s
        long_r = r_age > arm_s
        if long_l:
            n_hold_l += 1
        if long_r:
            n_hold_r += 1

        viz_t.append(float(target_t - t0))
        viz_arm_cmd.append(action[:12].copy())
        viz_l_hand.append(_pad(lhc, HAND_DOF))
        viz_r_hand.append(_pad(rhc, HAND_DOF))
        viz_hold_left.append(long_l)
        viz_hold_right.append(long_r)

    n_frames = len(rows["frame_index"])
    if n_frames == 0:
        raise RuntimeError("no frames produced after gates")

    # fps
    if fps_override is not None:
        fps = float(fps_override)
    else:
        if n_frames >= 2:
            dts = np.diff(np.asarray(viz_t, dtype=np.float64))
            fps = float(1.0 / np.median(dts)) if np.median(dts) > 0 else 30.0
        else:
            fps = 30.0

    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "meta").mkdir(exist_ok=True)
    (out_dir / "viz").mkdir(exist_ok=True)
    (out_dir / "convert_reports").mkdir(exist_ok=True)

    # Split after filtering whenever retained anchors are no longer contiguous.
    # This prevents one v2.1 episode from pretending to be uniformly sampled
    # across dropped-frame gaps.
    retained_t = np.asarray(viz_t, dtype=np.float64)
    split_gap_s = max(0.050, 1.5 / float(fps))
    split_points = (np.flatnonzero(np.diff(retained_t) > split_gap_s) + 1).tolist()
    boundaries = [0, *split_points, n_frames]
    segments = [
        (start, stop)
        for start, stop in zip(boundaries[:-1], boundaries[1:])
        if stop > start
    ]

    img_struct = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    parquet_paths: list[Path] = []
    viz_paths: list[Path] = []
    episode_rows: list[dict] = []
    written_frames = 0
    arm_cmd_all = np.stack(viz_arm_cmd, axis=0)
    l_hand_all = _stack_hand(viz_l_hand)
    r_hand_all = _stack_hand(viz_r_hand)
    hold_l_all = np.asarray(viz_hold_left)
    hold_r_all = np.asarray(viz_hold_right)

    for segment_index, (start, stop) in enumerate(segments):
        ep = int(episode_index) + segment_index
        chunk_id = ep // int(chunks_size)
        segment_len = stop - start
        data_dir = out_dir / "data" / f"chunk-{chunk_id:03d}"
        data_dir.mkdir(parents=True, exist_ok=True)
        segment_t = retained_t[start:stop] - retained_t[start]
        arrays = {
            OBS_TOP: pa.array(rows[OBS_TOP][start:stop], type=img_struct),
            OBS_LEFT: pa.array(rows[OBS_LEFT][start:stop], type=img_struct),
            OBS_RIGHT: pa.array(rows[OBS_RIGHT][start:stop], type=img_struct),
            "observation.state": pa.array(
                rows["observation.state"][start:stop], type=pa.list_(pa.float32())
            ),
            "action": pa.array(rows["action"][start:stop], type=pa.list_(pa.float32())),
            "timestamp": pa.array(segment_t, type=pa.float32()),
            "frame_index": pa.array(range(segment_len), type=pa.int64()),
            "episode_index": pa.array([ep] * segment_len, type=pa.int64()),
            "index": pa.array(
                range(
                    int(index_offset) + written_frames,
                    int(index_offset) + written_frames + segment_len,
                ),
                type=pa.int64(),
            ),
            "task_index": pa.array([0] * segment_len, type=pa.int64()),
        }
        pq_path = data_dir / f"episode_{ep:06d}.parquet"
        pq.write_table(pa.table(arrays), pq_path)
        parquet_paths.append(pq_path)
        episode_rows.append(
            {"episode_index": ep, "tasks": [task], "length": segment_len}
        )

        viz_path = out_dir / "viz" / f"episode_{ep:06d}_all_cmd.png"
        _plot_cmds(
            viz_path,
            t=segment_t,
            arm_cmd=arm_cmd_all[start:stop],
            l_hand=l_hand_all[start:stop],
            r_hand=r_hand_all[start:stop],
            hold_l=hold_l_all[start:stop],
            hold_r=hold_r_all[start:stop],
            left_names=hand_names.get("lhc"),
            right_names=hand_names.get("rhc"),
            title=f"{mcap_path.name} | segment {segment_index + 1}/{len(segments)}",
            arm_ms=hand_ms,
        )
        viz_paths.append(viz_path)
        written_frames += segment_len

    if write_meta:
        n_episodes = len(segments)
        last_ep = int(episode_index) + n_episodes - 1
        total_chunks = last_ep // int(chunks_size) + 1
        info = {
            "codebase_version": "v2.1",
            "robot_type": "dual_arm_inspire_hand",
            "total_episodes": n_episodes,
            "total_frames": n_frames,
            "total_tasks": 1,
            "total_videos": 0,
            "total_chunks": total_chunks,
            "chunks_size": int(chunks_size),
            "fps": round(fps, 3),
            "splits": {"train": f"0:{n_episodes}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": None,
            "features": features_dict(),
            "source_mcap": str(mcap_path),
            "processing": {
                "numeric_interp_mode": numeric_interp_mode,
                "hand": (
                    f"glitch_remove + hold_previous_gate_{hand_ms:g}ms"
                    if numeric_interp_mode == "safe"
                    else f"glitch_remove + {numeric_interp_mode}_linear"
                ),
                "arm_ee": (
                    f"unwrap_rpy + linear_bracket_gate_{arm_ms:g}ms"
                    if numeric_interp_mode == "safe"
                    else (
                        "unwrap_rpy + continuous_linear"
                        if numeric_interp_mode == "continuous"
                        else "strict_linear"
                    )
                ),
                "images": (
                    "nearest_no_gate"
                    if numeric_interp_mode == "continuous"
                    else f"nearest_gate_{image_ms:g}ms"
                ),
                "split_gap_ms": round(split_gap_s * 1000.0, 3),
            },
        }
        (out_dir / "meta" / "info.json").write_text(
            json.dumps(info, indent=2, ensure_ascii=False) + "\n"
        )
        with (out_dir / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as handle:
            for episode_row in episode_rows:
                handle.write(json.dumps(episode_row, ensure_ascii=False) + "\n")
        (out_dir / "meta" / "tasks.jsonl").write_text(
            json.dumps({"task_index": 0, "task": task}, ensure_ascii=False) + "\n"
        )

    report = {
        "mcap": str(mcap_path),
        "out_dir": str(out_dir),
        "task": task,
        "episode_index": int(episode_index),
        "episode_indices": [row["episode_index"] for row in episode_rows],
        "split_count": len(segments) - 1,
        "top_anchors": int(top_ts.size),
        "kept_frames": n_frames,
        "drop_rate": round(1.0 - n_frames / max(1, int(top_ts.size)), 6),
        "drop_reasons": drop_reasons,
        "hand_glitch": hand_stats,
        "long_hold_frames": {"left_cmd": n_hold_l, "right_cmd": n_hold_r},
        "fps": round(fps, 3),
        "gates_ms": {"image": image_ms, "arm_ee": arm_ms, "hand": hand_ms},
        "numeric_interp_mode": numeric_interp_mode,
        "schema": {
            "action_dim": ACTION_DIM,
            "state_dim": STATE_DIM,
            "hand_left_dof": HAND_DOF,
            "hand_right_dof": HAND_DOF,
        },
        "parquet": str(parquet_paths[0]),
        "parquets": [str(path) for path in parquet_paths],
        "viz": str(viz_paths[0]),
        "viz_files": [str(path) for path in viz_paths],
    }
    report_path = out_dir / "convert_reports" / f"episode_{int(episode_index):06d}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if write_meta:
        (out_dir / "convert_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def _stack_hand(rows: list[np.ndarray]) -> np.ndarray:
    if not rows:
        return np.zeros((0, 1), dtype=np.float32)
    max_d = max(int(r.shape[0]) for r in rows)
    out = np.zeros((len(rows), max_d), dtype=np.float32)
    for i, r in enumerate(rows):
        out[i, : int(r.shape[0])] = r
    return out


def _plot_cmds(
    path: Path,
    *,
    t: np.ndarray,
    arm_cmd: np.ndarray,
    l_hand: np.ndarray,
    r_hand: np.ndarray,
    hold_l: np.ndarray,
    hold_r: np.ndarray,
    left_names: list[str] | None,
    right_names: list[str] | None,
    title: str,
    arm_ms: float,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, constrained_layout=True)

    ax = axes[0]
    for j in range(6):
        ax.plot(t, arm_cmd[:, j], lw=0.9, label=f"L{j}")
    ax.set_ylabel("left arm cmd")
    ax.legend(loc="upper right", ncol=6, fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title("Left arm joint_cmd (6)")

    ax = axes[1]
    for j in range(6):
        ax.plot(t, arm_cmd[:, 6 + j], lw=0.9, label=f"R{j}")
    ax.set_ylabel("right arm cmd")
    ax.legend(loc="upper right", ncol=6, fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title("Right arm joint_cmd (6)")

    ax = axes[2]
    n_l = int(l_hand.shape[1]) if l_hand.ndim == 2 else 1
    # user request: yellow + blue for left hand joints
    colors_l = ["#ffd700", "#1f77b4"]  # yellow (index), blue (middle)
    default_l = ["index", "middle"]
    for j in range(n_l):
        label = (
            left_names[j]
            if left_names and j < len(left_names)
            else (default_l[j] if j < len(default_l) else f"joint{j}")
        )
        ax.step(
            t,
            l_hand[:, j],
            where="post",
            color=colors_l[j % len(colors_l)],
            lw=1.8,
            label=f"{label} [train]",
        )
    ax.set_ylabel("left hand")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Left hand cmd ({HAND_DOF} DoF train)")

    ax = axes[3]
    n_r = int(r_hand.shape[1]) if r_hand.ndim == 2 else 1
    colors_r = ["#ff7f0e", "#e377c2"]
    for j in range(min(n_r, HAND_DOF)):
        label = (right_names[j] if right_names and j < len(right_names) else f"joint{j}")
        ax.step(
            t,
            r_hand[:, j],
            where="post",
            color=colors_r[j % len(colors_r)],
            lw=1.8,
            label=f"{label} [train]",
        )
    ax.set_ylabel("right hand")
    ax.set_xlabel("time (s, relative to first top)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Right hand cmd ({HAND_DOF} DoF train)")

    fig.suptitle(f"{title}  |  kept={len(t)} frames", fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single MCAP → LeRobot v2.1 episode")
    parser.add_argument("--mcap", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--task",
        type=str,
        default="pour water from the right-hand bottle into the left-hand cup",
    )
    parser.add_argument("--image-ms", type=float, default=50.0)
    parser.add_argument("--arm-ms", type=float, default=50.0, help="Arm/ee linear bracket gate")
    parser.add_argument(
        "--hand-ms",
        type=float,
        default=200.0,
        help="Safe mode maximum age for previous hand sample",
    )
    parser.add_argument(
        "--numeric-interp-mode",
        choices=("safe", "strict", "continuous"),
        default="continuous",
        help=(
            "safe: unwrap RPY + bounded hand hold; strict: all numeric linear with gate; "
            "continuous: retain every top anchor, unwrap RPY, interpolate numeric streams "
            "across gaps, and use ungated nearest wrist images"
        ),
    )
    parser.add_argument("--jump-abs", type=float, default=50.0, help="Hand glitch jump threshold")
    parser.add_argument("--glitch-ms", type=float, default=80.0)
    parser.add_argument("--fps", type=float, default=None, help="Override fps in info.json")
    args = parser.parse_args()

    mcap_path = args.mcap.expanduser().resolve()
    if not mcap_path.is_file():
        raise FileNotFoundError(mcap_path)

    print(f"MCAP: {mcap_path}")
    print(f"Out:  {args.out_dir}")
    print(
        f"gates: image={args.image_ms:g}ms arm={args.arm_ms:g}ms  "
        f"hand={args.hand_ms:g}ms mode={args.numeric_interp_mode} "
        f"glitch={args.glitch_ms:g}ms"
    )
    report = convert_one(
        mcap_path,
        args.out_dir,
        task=args.task,
        image_ms=args.image_ms,
        arm_ms=args.arm_ms,
        jump_abs=args.jump_abs,
        glitch_ms=args.glitch_ms,
        numeric_interp_mode=args.numeric_interp_mode,
        hand_ms=args.hand_ms,
        fps_override=args.fps,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"OK → {report['parquet']}")
    print(f"VIZ → {report['viz']}")


if __name__ == "__main__":
    main()
