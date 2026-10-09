#!/usr/bin/env python3
"""MCAP → LeRobot v2.1 with native Inspire taxel patches.

RGB / state / action match `/data/mcap_to_lerobot_v21.py`. Tactile is stored as
raw mono16 counts padded to a 14×14 canvas — not fake RGB images.

Per frame (aligned to the top camera):
  observation.tactile.{left,right}        float32 [11, 14, 14]
  observation.tactile.{left,right}_valid  bool    [11]

Sensor order is the encoder's TACTILE_IDS. History T=16 is NOT baked into each
row; assemble it at train time from previous rows of the same episode.

Usage:
  /home/xudi_ge/openpi/.venv/bin/python /data/mcap_to_lerobot_v21_tactile.py \\
    --mcap /data/datasample/2026-08-03_09-28-28.907826269.mcap \\
    --out-dir /tmp/lerobot_v21_tactile_demo \\
    --task "sample episode"

  /home/xudi_ge/openpi/.venv/bin/python /data/mcap_to_lerobot_v21_tactile.py \\
    --mcap-dir /data/datasample \\
    --out-dir /data/lerobot_v21_tactile/datasample \\
    --task "sample episode"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

_V21_PATH = Path(__file__).resolve().parent / "mcap_to_lerobot_v21.py"
_SPEC = importlib.util.spec_from_file_location("mcap_to_lerobot_v21_base", _V21_PATH)
v21 = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = v21
_SPEC.loader.exec_module(v21)

ACTION_DIM = v21.ACTION_DIM
ARM_DOF = v21.ARM_DOF
EE_POSE_DOF = v21.EE_POSE_DOF
HAND_DOF = v21.HAND_DOF
IMG_HWC = v21.IMG_HWC
OBS_LEFT = v21.OBS_LEFT
OBS_RIGHT = v21.OBS_RIGHT
OBS_TOP = v21.OBS_TOP
STATE_DIM = v21.STATE_DIM
ImgBuf = v21.ImgBuf
NumBuf = v21.NumBuf

TACTILE_IDS: tuple[str, ...] = (
    "12",
    "13",
    "22",
    "23",
    "32",
    "33",
    "42",
    "43",
    "52",
    "54",
    "61",
)
TACTILE_LAYOUT_VERSION = "inspire_hand_v1"
HANDS: tuple[str, ...] = ("left", "right")
CANVAS_HW: tuple[int, int] = (14, 14)
NUM_TAXELS = len(TACTILE_IDS)
TACTILE_PATCH_SHAPE: dict[str, tuple[int, int]] = {
    "12": (12, 8),
    "13": (10, 8),
    "22": (12, 8),
    "23": (10, 8),
    "32": (12, 8),
    "33": (10, 8),
    "42": (12, 8),
    "43": (10, 8),
    "52": (12, 8),
    "54": (12, 8),
    "61": (14, 8),
}
OBS_TACTILE_LEFT = "observation.tactile.left"
OBS_TACTILE_RIGHT = "observation.tactile.right"
OBS_TACTILE_LEFT_VALID = "observation.tactile.left_valid"
OBS_TACTILE_RIGHT_VALID = "observation.tactile.right_valid"


def tactile_topic(hand: str, taxel_id: str) -> str:
    return f"/inspire/{hand}_hand/tactile_{taxel_id}"


def all_tactile_topics(hands: tuple[str, ...] = HANDS) -> list[str]:
    return [tactile_topic(hand, taxel_id) for hand in hands for taxel_id in TACTILE_IDS]


def parse_tactile_topic(topic: str) -> tuple[str, str] | None:
    prefix = "/inspire/"
    suffix_head = "_hand/tactile_"
    if not topic.startswith(prefix) or suffix_head not in topic:
        return None
    rest = topic[len(prefix) :]
    hand, _, taxel_id = rest.partition(suffix_head)
    if hand not in HANDS or taxel_id not in TACTILE_IDS:
        return None
    return hand, taxel_id


def tactile_layout_dict() -> dict:
    return {
        "layout_version": TACTILE_LAYOUT_VERSION,
        "hands": list(HANDS),
        "tactile_ids": list(TACTILE_IDS),
        "canonical_size": list(CANVAS_HW),
        "patch_shapes": {k: list(v) for k, v in TACTILE_PATCH_SHAPE.items()},
        "sensor_order": [[hand, taxel_id] for hand in HANDS for taxel_id in TACTILE_IDS],
        "topics": {
            hand: {taxel_id: tactile_topic(hand, taxel_id) for taxel_id in TACTILE_IDS}
            for hand in HANDS
        },
        "storage": {
            OBS_TACTILE_LEFT: "float32 [11,14,14] raw counts, zero-padded, row-major flatten in parquet",
            OBS_TACTILE_RIGHT: "float32 [11,14,14] raw counts, zero-padded, row-major flatten in parquet",
            OBS_TACTILE_LEFT_VALID: "bool [11], True if that taxel had a nearest sample",
            OBS_TACTILE_RIGHT_VALID: "bool [11], True if that taxel had a nearest sample",
        },
        "history_frames": (
            "not stored; assemble T=16 at train time from previous rows of the same episode"
        ),
        "normalization": "raw counts only; baseline / q99.5 / tanh belong in the encoder",
    }


def decode_tactile_msg(msg: object) -> np.ndarray:
    encoding = str(getattr(msg, "encoding", "") or "").lower()
    height = int(getattr(msg, "height", 0) or 0)
    width = int(getattr(msg, "width", 0) or 0)
    if height <= 0 or width <= 0:
        raise ValueError(f"invalid tactile image shape {(height, width)}")
    is_bigendian = bool(getattr(msg, "is_bigendian", 0))
    data = bytes(getattr(msg, "data", b"") or b"")
    if encoding in {"mono16", "16uc1"}:
        dtype = np.dtype(np.uint16).newbyteorder(">" if is_bigendian else "<")
        bytes_per_pixel = 2
    elif encoding in {"mono8", "8uc1"}:
        dtype = np.dtype(np.uint8)
        bytes_per_pixel = 1
    else:
        raise ValueError(f"unsupported tactile encoding {encoding!r}")
    expected_row_bytes = width * bytes_per_pixel
    step = int(getattr(msg, "step", 0) or 0) or expected_row_bytes
    if len(data) < step * height:
        raise ValueError(
            f"tactile payload too short: got {len(data)} bytes, "
            f"need at least {step * height}"
        )
    if step == expected_row_bytes:
        arr = np.frombuffer(data[: expected_row_bytes * height], dtype=dtype)
        arr = arr.reshape(height, width)
    else:
        rows = [
            np.frombuffer(data[row * step : row * step + expected_row_bytes], dtype=dtype)
            for row in range(height)
        ]
        arr = np.stack(rows, axis=0)
    return arr.astype(np.float32, copy=False)


def orient_patch(arr: np.ndarray, taxel_id: str) -> np.ndarray:
    expected = TACTILE_PATCH_SHAPE[taxel_id]
    if arr.shape == expected:
        return arr
    if arr.shape == expected[::-1]:
        return np.ascontiguousarray(arr.T)
    return arr


def pad_to_canvas(arr: np.ndarray, canvas_hw: tuple[int, int] = CANVAS_HW) -> np.ndarray:
    canvas_h, canvas_w = canvas_hw
    out = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    hh = min(int(arr.shape[0]), canvas_h)
    ww = min(int(arr.shape[1]), canvas_w)
    if hh > 0 and ww > 0:
        out[:hh, :ww] = arr[:hh, :ww]
    return out


def features_dict() -> dict:
    feats = v21.features_dict()
    canvas_h, canvas_w = CANVAS_HW
    patch = {
        "dtype": "float32",
        "shape": [NUM_TAXELS, canvas_h, canvas_w],
        "names": ["sensor", "height", "width"],
    }
    valid = {"dtype": "bool", "shape": [NUM_TAXELS], "names": list(TACTILE_IDS)}
    feats[OBS_TACTILE_LEFT] = dict(patch)
    feats[OBS_TACTILE_RIGHT] = dict(patch)
    feats[OBS_TACTILE_LEFT_VALID] = dict(valid)
    feats[OBS_TACTILE_RIGHT_VALID] = dict(valid)
    return feats


def load_mcap(path: Path, *, hands: tuple[str, ...] = HANDS) -> dict:
    topics = [
        v21.TOPIC_TOP,
        v21.TOPIC_LEFT_IMG,
        v21.TOPIC_RIGHT_IMG,
        v21.TOPIC_VLA_ARM_STATE,
        v21.TOPIC_VLA_ARM_CMD,
        v21.TOPIC_LEFT_EE_POS,
        v21.TOPIC_LEFT_EE_RPY,
        v21.TOPIC_RIGHT_EE_POS,
        v21.TOPIC_RIGHT_EE_RPY,
        v21.TOPIC_LEFT_HAND_JS,
        v21.TOPIC_RIGHT_HAND_JS,
        v21.TOPIC_LEFT_HAND_CMD,
        v21.TOPIC_RIGHT_HAND_CMD,
        *all_tactile_topics(hands),
    ]
    top, left_img, right_img = ImgBuf(), ImgBuf(), ImgBuf()
    arm_q, arm_cmd = NumBuf(), NumBuf()
    lee_p, lee_r, ree_p, ree_r = NumBuf(), NumBuf(), NumBuf(), NumBuf()
    lhq, rhq, lhc, rhc = NumBuf(), NumBuf(), NumBuf(), NumBuf()
    tactile_bufs: dict[tuple[str, str], ImgBuf] = {
        (hand, taxel_id): ImgBuf() for hand in hands for taxel_id in TACTILE_IDS
    }

    with path.open("rb") as handle:
        reader = make_reader(handle, decoder_factories=[DecoderFactory()])
        for _schema, channel, message, ros_msg in reader.iter_decoded_messages(topics=topics):
            topic = channel.topic
            log_ns = int(message.log_time)

            if topic == v21.TOPIC_TOP:
                t = v21._stamp_s(ros_msg, log_ns)
                if t is None:
                    continue
                top.t.append(t)
                top.raw.append(ros_msg)
            elif topic == v21.TOPIC_LEFT_IMG:
                t = v21._stamp_s(ros_msg, log_ns)
                if t is None:
                    continue
                left_img.t.append(t)
                left_img.raw.append(ros_msg)
            elif topic == v21.TOPIC_RIGHT_IMG:
                t = v21._stamp_s(ros_msg, log_ns)
                if t is None:
                    continue
                right_img.t.append(t)
                right_img.raw.append(ros_msg)
            elif topic == v21.TOPIC_VLA_ARM_STATE:
                t = v21._require_stamp(ros_msg, topic, log_ns)
                pos = v21._pad(np.asarray(list(ros_msg.position), dtype=np.float32), 12)
                arm_q.t.append(t)
                arm_q.v.append(pos)
            elif topic == v21.TOPIC_VLA_ARM_CMD:
                t = v21._require_stamp(ros_msg, topic, log_ns)
                cmd = v21._pad(np.asarray(list(ros_msg.position), dtype=np.float32), 12)
                arm_cmd.t.append(t)
                arm_cmd.v.append(cmd)
            elif topic == v21.TOPIC_LEFT_EE_POS:
                t = v21._stamp_s(ros_msg, log_ns) or (log_ns * 1e-9)
                lee_p.t.append(t)
                lee_p.v.append(v21._vec3(ros_msg))
            elif topic == v21.TOPIC_LEFT_EE_RPY:
                t = v21._stamp_s(ros_msg, log_ns) or (log_ns * 1e-9)
                lee_r.t.append(t)
                lee_r.v.append(v21._vec3(ros_msg))
            elif topic == v21.TOPIC_RIGHT_EE_POS:
                t = v21._stamp_s(ros_msg, log_ns) or (log_ns * 1e-9)
                ree_p.t.append(t)
                ree_p.v.append(v21._vec3(ros_msg))
            elif topic == v21.TOPIC_RIGHT_EE_RPY:
                t = v21._stamp_s(ros_msg, log_ns) or (log_ns * 1e-9)
                ree_r.t.append(t)
                ree_r.v.append(v21._vec3(ros_msg))
            elif topic == v21.TOPIC_LEFT_HAND_JS:
                t = v21._require_stamp(ros_msg, topic, log_ns)
                hp = v21._hand_pos(ros_msg)
                if hp is not None:
                    lhq.t.append(t)
                    lhq.v.append(hp)
                    if lhq.names is None:
                        lhq.names = v21._hand_names(ros_msg)
            elif topic == v21.TOPIC_RIGHT_HAND_JS:
                t = v21._require_stamp(ros_msg, topic, log_ns)
                hp = v21._hand_pos(ros_msg)
                if hp is not None:
                    rhq.t.append(t)
                    rhq.v.append(hp)
                    if rhq.names is None:
                        rhq.names = v21._hand_names(ros_msg)
            elif topic == v21.TOPIC_LEFT_HAND_CMD:
                t = v21._require_stamp(ros_msg, topic, log_ns)
                hp = v21._hand_pos(ros_msg)
                if hp is not None:
                    lhc.t.append(t)
                    lhc.v.append(hp)
                    if lhc.names is None:
                        lhc.names = v21._hand_names(ros_msg)
            elif topic == v21.TOPIC_RIGHT_HAND_CMD:
                t = v21._require_stamp(ros_msg, topic, log_ns)
                hp = v21._hand_pos(ros_msg)
                if hp is not None:
                    rhc.t.append(t)
                    rhc.v.append(hp)
                    if rhc.names is None:
                        rhc.names = v21._hand_names(ros_msg)
            else:
                parsed = parse_tactile_topic(topic)
                if parsed is None:
                    continue
                t = v21._stamp_s(ros_msg, log_ns)
                if t is None:
                    t = log_ns * 1e-9
                tactile_bufs[parsed].t.append(t)
                tactile_bufs[parsed].raw.append(ros_msg)

    tactile = {
        key: v21._sanitize_img(buf) for key, buf in tactile_bufs.items()
    }
    return {
        "top": v21._sanitize_img(top),
        "left_img": v21._sanitize_img(left_img),
        "right_img": v21._sanitize_img(right_img),
        "arm_q": v21._sanitize_num(arm_q),
        "arm_cmd": v21._sanitize_num(arm_cmd),
        "lee_p": v21._sanitize_num(lee_p),
        "lee_r": v21._sanitize_num(lee_r),
        "ree_p": v21._sanitize_num(ree_p),
        "ree_r": v21._sanitize_num(ree_r),
        "lhq": v21._sanitize_num(lhq),
        "rhq": v21._sanitize_num(rhq),
        "lhc": v21._sanitize_num(lhc),
        "rhc": v21._sanitize_num(rhc),
        "tactile": tactile,
    }


def _empty_hand_canvas() -> np.ndarray:
    return np.zeros((NUM_TAXELS, CANVAS_HW[0], CANVAS_HW[1]), dtype=np.float32)


def _empty_valid() -> np.ndarray:
    return np.zeros((NUM_TAXELS,), dtype=bool)


def sample_hand_tactile(
    tactile: dict[tuple[str, str], tuple[np.ndarray, list]],
    hand: str,
    target_t: float,
    gate_s: float,
    caches: dict[tuple[str, str], dict[int, tuple[np.ndarray, bool]]],
) -> tuple[np.ndarray, np.ndarray]:
    canvas = _empty_hand_canvas()
    valid = _empty_valid()
    for i, taxel_id in enumerate(TACTILE_IDS):
        key = (hand, taxel_id)
        ts, frames = tactile.get(key, (np.asarray([], dtype=np.float64), []))
        idx = v21.sample_nearest_idx(ts, target_t, gate_s)
        if idx is None:
            continue
        cache = caches.setdefault(key, {})
        if idx not in cache:
            try:
                raw = decode_tactile_msg(frames[idx])
                patch = orient_patch(raw, taxel_id)
                cache[idx] = (pad_to_canvas(patch), True)
            except Exception:
                cache[idx] = (np.zeros(CANVAS_HW, dtype=np.float32), False)
        patch, ok = cache[idx]
        canvas[i] = patch
        valid[i] = bool(ok)
    return canvas, valid


def write_tactile_layout(out_dir: Path) -> None:
    meta = out_dir / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "tactile_layout.json").write_text(
        json.dumps(tactile_layout_dict(), indent=2, ensure_ascii=False) + "\n"
    )


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
    tactile_ms: float,
    fps_override: float | None,
    episode_index: int = 0,
    index_offset: int = 0,
    chunks_size: int = 1000,
    write_meta: bool = True,
    hands: tuple[str, ...] = HANDS,
    drop_if_no_right_tactile: bool = False,
) -> dict:
    if numeric_interp_mode not in {"safe", "strict", "continuous"}:
        raise ValueError(f"unsupported numeric_interp_mode={numeric_interp_mode!r}")
    data = load_mcap(mcap_path, hands=hands)
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
    tactile_s = float("inf") if numeric_interp_mode == "continuous" else tactile_ms / 1000.0

    hand_stats = {}
    hand_names: dict[str, list[str] | None] = {}
    for key in ("lhq", "rhq", "lhc", "rhc"):
        t, v, names = data[key]
        cleaned, n_fix = v21.remove_glitches(t, v, jump_abs=jump_abs, glitch_ms=glitch_ms)
        data[key] = (t, cleaned)
        hand_names[key] = names
        hand_stats[key] = {
            "n": int(t.size),
            "glitch_fixed": int(n_fix),
            "dof": int(cleaned.shape[1]) if cleaned.ndim == 2 and cleaned.size else 0,
            "names": names,
        }

    if numeric_interp_mode in {"safe", "continuous"}:
        for key in ("lee_r", "ree_r"):
            ts, values, *_ = data[key]
            data[key] = (ts, np.unwrap(values.astype(np.float64), axis=0).astype(np.float32))

    left_cache: dict[int, np.ndarray] = {}
    right_cache: dict[int, np.ndarray] = {}
    tactile_caches: dict[tuple[str, str], dict[int, tuple[np.ndarray, bool]]] = {}

    def _wrist(frames, idx, cache):
        if idx not in cache:
            cache[idx] = v21._letterbox_224(v21._decode_rgb(frames[idx]))
        return cache[idx]

    rows = {
        OBS_TOP: [],
        OBS_LEFT: [],
        OBS_RIGHT: [],
        OBS_TACTILE_LEFT: [],
        OBS_TACTILE_RIGHT: [],
        OBS_TACTILE_LEFT_VALID: [],
        OBS_TACTILE_RIGHT_VALID: [],
        "observation.state": [],
        "action": [],
        "timestamp": [],
        "frame_index": [],
        "episode_index": [],
        "index": [],
        "task_index": [],
    }
    viz_t: list[float] = []
    viz_arm_cmd: list[np.ndarray] = []
    viz_l_hand: list[np.ndarray] = []
    viz_r_hand: list[np.ndarray] = []
    viz_hold_left: list[bool] = []
    viz_hold_right: list[bool] = []

    drop_reasons: dict[str, int] = {}
    n_hold_l = n_hold_r = 0
    t0 = float(top_ts[0])
    tactile_valid_counts = {
        "left": np.zeros((NUM_TAXELS,), dtype=np.int64),
        "right": np.zeros((NUM_TAXELS,), dtype=np.int64),
    }
    tactile_msg_counts = {
        hand: {taxel_id: int(data["tactile"].get((hand, taxel_id), (np.asarray([]), []))[0].size)
               for taxel_id in TACTILE_IDS}
        for hand in HANDS
    }

    for i, target_t in enumerate(top_ts):
        reasons: list[str] = []

        if numeric_interp_mode == "continuous":
            li = v21.sample_nearest_idx(left_ts, float(target_t), float("inf"))
            ri = v21.sample_nearest_idx(right_ts, float(target_t), float("inf"))
        else:
            li = v21.sample_nearest_idx(left_ts, float(target_t), image_s)
            ri = v21.sample_nearest_idx(right_ts, float(target_t), image_s)
        if li is None:
            reasons.append("left_wrist")
        if ri is None:
            reasons.append("right_wrist")

        def _tv(key: str) -> tuple[np.ndarray, np.ndarray]:
            item = data[key]
            return item[0], item[1]

        numeric_sampler = (
            v21.sample_continuous_linear
            if numeric_interp_mode == "continuous"
            else lambda ts, values, target: v21.sample_linear(ts, values, target, arm_s)
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
            lhq = v21.sample_hold(*_tv("lhq"), float(target_t), hand_s)
            rhq = v21.sample_hold(*_tv("rhq"), float(target_t), hand_s)
            lhc = v21.sample_hold(*_tv("lhc"), float(target_t), hand_s)
            rhc = v21.sample_hold(*_tv("rhc"), float(target_t), hand_s)
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

        left_taxel, left_valid = sample_hand_tactile(
            data["tactile"], "left", float(target_t), tactile_s, tactile_caches
        )
        right_taxel, right_valid = sample_hand_tactile(
            data["tactile"], "right", float(target_t), tactile_s, tactile_caches
        )
        if drop_if_no_right_tactile and not bool(right_valid.any()):
            reasons.append("right_tactile")

        if reasons:
            for r in reasons:
                drop_reasons[r] = drop_reasons.get(r, 0) + 1
            continue

        assert li is not None and ri is not None
        assert arm_q is not None and arm_cmd is not None
        assert lee_p is not None and lee_r is not None and ree_p is not None and ree_r is not None
        assert lhq is not None and rhq is not None and lhc is not None and rhc is not None

        state = np.concatenate(
            [
                arm_q[6:12],
                arm_q[:6],
                v21._pad(lhq, HAND_DOF),
                v21._pad(rhq, HAND_DOF),
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
                v21._pad(lhc, HAND_DOF),
                v21._pad(rhc, HAND_DOF),
            ],
            axis=0,
        ).astype(np.float32)
        if state.shape[0] != STATE_DIM or action.shape[0] != ACTION_DIM:
            raise RuntimeError(
                f"bad dims state={state.shape} action={action.shape} "
                f"expected {STATE_DIM}/{ACTION_DIM}"
            )

        top_img = v21._letterbox_224(v21._decode_rgb(top_frames[i]))
        left_img = _wrist(left_frames, li, left_cache)
        right_img = _wrist(right_frames, ri, right_cache)

        fi = len(rows["frame_index"])
        rows[OBS_TOP].append({"bytes": v21._jpeg_bytes(top_img), "path": f"frame_{fi:06d}_top.jpg"})
        rows[OBS_LEFT].append({"bytes": v21._jpeg_bytes(left_img), "path": f"frame_{fi:06d}_left.jpg"})
        rows[OBS_RIGHT].append({"bytes": v21._jpeg_bytes(right_img), "path": f"frame_{fi:06d}_right.jpg"})
        rows[OBS_TACTILE_LEFT].append(left_taxel.reshape(-1))
        rows[OBS_TACTILE_RIGHT].append(right_taxel.reshape(-1))
        rows[OBS_TACTILE_LEFT_VALID].append(left_valid.copy())
        rows[OBS_TACTILE_RIGHT_VALID].append(right_valid.copy())
        rows["observation.state"].append(state)
        rows["action"].append(action)
        rows["timestamp"].append(np.asarray([float(target_t - t0)], dtype=np.float32))
        rows["frame_index"].append(np.asarray([fi], dtype=np.int64))
        rows["episode_index"].append(np.asarray([int(episode_index)], dtype=np.int64))
        rows["index"].append(np.asarray([int(index_offset + fi)], dtype=np.int64))
        rows["task_index"].append(np.asarray([0], dtype=np.int64))

        tactile_valid_counts["left"] += left_valid.astype(np.int64)
        tactile_valid_counts["right"] += right_valid.astype(np.int64)

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
        viz_l_hand.append(v21._pad(lhc, HAND_DOF))
        viz_r_hand.append(v21._pad(rhc, HAND_DOF))
        viz_hold_left.append(long_l)
        viz_hold_right.append(long_r)

    n_frames = len(rows["frame_index"])
    if n_frames == 0:
        raise RuntimeError("no frames produced after gates")

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
    l_hand_all = v21._stack_hand(viz_l_hand)
    r_hand_all = v21._stack_hand(viz_r_hand)
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
            OBS_TACTILE_LEFT: pa.array(
                rows[OBS_TACTILE_LEFT][start:stop], type=pa.list_(pa.float32())
            ),
            OBS_TACTILE_RIGHT: pa.array(
                rows[OBS_TACTILE_RIGHT][start:stop], type=pa.list_(pa.float32())
            ),
            OBS_TACTILE_LEFT_VALID: pa.array(
                [np.asarray(v, dtype=bool).tolist() for v in rows[OBS_TACTILE_LEFT_VALID][start:stop]],
                type=pa.list_(pa.bool_()),
            ),
            OBS_TACTILE_RIGHT_VALID: pa.array(
                [np.asarray(v, dtype=bool).tolist() for v in rows[OBS_TACTILE_RIGHT_VALID][start:stop]],
                type=pa.list_(pa.bool_()),
            ),
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
        episode_rows.append({"episode_index": ep, "tasks": [task], "length": segment_len})

        viz_path = out_dir / "viz" / f"episode_{ep:06d}_all_cmd.png"
        v21._plot_cmds(
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

    tactile_coverage = {
        hand: {
            taxel_id: {
                "n_messages": tactile_msg_counts[hand][taxel_id],
                "n_valid_frames": int(tactile_valid_counts[hand][i]),
                "valid_rate": round(
                    float(tactile_valid_counts[hand][i]) / max(1, n_frames), 6
                ),
            }
            for i, taxel_id in enumerate(TACTILE_IDS)
        }
        for hand in HANDS
    }

    processing = {
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
        "tactile": (
            "nearest_no_gate_raw_counts"
            if numeric_interp_mode == "continuous"
            else f"nearest_gate_{tactile_ms:g}ms_raw_counts"
        ),
        "split_gap_ms": round(split_gap_s * 1000.0, 3),
        "schema": {"action_dim": ACTION_DIM, "state_dim": STATE_DIM},
    }

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
            "processing": processing,
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
        write_tactile_layout(out_dir)

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
        "gates_ms": {
            "image": image_ms,
            "arm_ee": arm_ms,
            "hand": hand_ms,
            "tactile": tactile_ms,
        },
        "numeric_interp_mode": numeric_interp_mode,
        "schema": {
            "action_dim": ACTION_DIM,
            "state_dim": STATE_DIM,
            "tactile": [NUM_TAXELS, CANVAS_HW[0], CANVAS_HW[1]],
        },
        "tactile_coverage": tactile_coverage,
        "parquet": str(parquet_paths[0]),
        "parquets": [str(path) for path in parquet_paths],
        "viz": str(viz_paths[0]),
        "viz_files": [str(path) for path in viz_paths],
        "processing": processing,
    }
    report_path = out_dir / "convert_reports" / f"episode_{int(episode_index):06d}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    if write_meta:
        (out_dir / "convert_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n"
        )
    return report


def _list_mcaps(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*.mcap") if p.is_file())


def _load_reports(out_dir: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    rep_dir = out_dir / "convert_reports"
    if not rep_dir.is_dir():
        return out
    for path in sorted(rep_dir.glob("episode_*.json")):
        try:
            report = json.loads(path.read_text())
        except Exception:
            continue
        mcap = report.get("mcap")
        parquet = Path(report.get("parquet", ""))
        viz = Path(report.get("viz", ""))
        if not mcap or int(report.get("kept_frames", 0)) <= 0:
            continue
        if not parquet.is_file() or not viz.is_file():
            continue
        out[str(Path(mcap).resolve())] = report
    return out


def _write_folder_meta(
    out_dir: Path,
    *,
    task: str,
    episodes: list[dict],
    sources: list[dict],
    fps_values: list[float],
    image_ms: float,
    arm_ms: float,
    hand_ms: float,
    tactile_ms: float,
    numeric_interp_mode: str,
    chunks_size: int,
) -> None:
    n_ep = len(episodes)
    n_frames = int(sum(int(e["length"]) for e in episodes))
    fps = float(sorted(fps_values)[len(fps_values) // 2]) if fps_values else 30.0
    n_chunks = (
        (max((e["episode_index"] for e in episodes), default=-1) // chunks_size) + 1
        if n_ep
        else 0
    )
    info = {
        "codebase_version": "v2.1",
        "robot_type": "dual_arm_inspire_hand",
        "total_episodes": n_ep,
        "total_frames": n_frames,
        "total_tasks": 1,
        "total_videos": 0,
        "total_chunks": int(n_chunks),
        "chunks_size": chunks_size,
        "fps": round(fps, 3),
        "splits": {"train": f"0:{n_ep}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "features": features_dict(),
        "processing": {
            "numeric_interp_mode": numeric_interp_mode,
            "hand": (
                "glitch_remove + continuous_linear"
                if numeric_interp_mode == "continuous"
                else f"glitch_remove + bounded_sampling_{hand_ms:g}ms"
            ),
            "arm_ee": (
                "unwrap_rpy + continuous_linear"
                if numeric_interp_mode == "continuous"
                else f"unwrap_rpy + linear_bracket_gate_{arm_ms:g}ms"
            ),
            "images": (
                "ungated_nearest"
                if numeric_interp_mode == "continuous"
                else f"nearest_gate_{image_ms:g}ms"
            ),
            "tactile": (
                "ungated_nearest_raw_counts"
                if numeric_interp_mode == "continuous"
                else f"nearest_gate_{tactile_ms:g}ms_raw_counts"
            ),
            "schema": {
                "action_dim": ACTION_DIM,
                "state_dim": STATE_DIM,
                "tactile": [NUM_TAXELS, CANVAS_HW[0], CANVAS_HW[1]],
            },
        },
    }
    meta = out_dir / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": task}, ensure_ascii=False) + "\n"
    )
    with (meta / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode, ensure_ascii=False) + "\n")
    with (meta / "episode_sources.jsonl").open("w", encoding="utf-8") as handle:
        for source in sources:
            handle.write(json.dumps(source, ensure_ascii=False) + "\n")
    write_tactile_layout(out_dir)


def load_mcap_list(path: Path) -> list[Path]:
    """Read episode_sources.jsonl (field `mcap`) or a plain text file of paths."""
    text = path.read_text(encoding="utf-8")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise FileNotFoundError(f"empty mcap list: {path}")
    out: list[Path] = []
    seen: set[str] = set()
    for line in lines:
        if line.startswith("{"):
            rec = json.loads(line)
            raw = rec.get("mcap")
            if not raw:
                raise ValueError(f"{path}: jsonl line missing 'mcap': {line[:120]}")
        else:
            raw = line
        mcap = Path(raw).expanduser().resolve()
        key = str(mcap)
        if key in seen:
            continue
        seen.add(key)
        out.append(mcap)
    return out


def convert_mcaps(
    mcaps: list[Path],
    out_dir: Path,
    *,
    src_label: str,
    task: str,
    image_ms: float,
    arm_ms: float,
    hand_ms: float,
    tactile_ms: float,
    numeric_interp_mode: str,
    jump_abs: float,
    glitch_ms: float,
    chunks_size: int,
    resume: bool,
    hands: tuple[str, ...],
    drop_if_no_right_tactile: bool,
    fps_override: float | None,
) -> dict:
    if not mcaps:
        raise FileNotFoundError("no mcaps to convert")

    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "convert_reports").mkdir(exist_ok=True)
    (out_dir / "viz").mkdir(exist_ok=True)

    existing = _load_reports(out_dir) if resume else {}
    next_ep = 0
    index_offset = 0
    if existing:
        next_ep = max(max(int(i) for i in r["episode_indices"]) for r in existing.values()) + 1
        index_offset = int(sum(int(r["kept_frames"]) for r in existing.values()))

    ok: list[dict] = []
    skipped: list[dict] = []
    failed: list[dict] = []

    for i, mcap in enumerate(mcaps):
        key = str(mcap.resolve())
        if key in existing:
            report = existing[key]
            skipped.append(
                {
                    "episode_index": int(report["episode_index"]),
                    "mcap": key,
                    "kept_frames": int(report["kept_frames"]),
                }
            )
            print(
                f"[{src_label}] skip resume ep={int(report['episode_index']):04d} {mcap.name}",
                flush=True,
            )
            continue

        ep = next_ep
        t0 = time.time()
        print(f"\n[{src_label}] ({i + 1}/{len(mcaps)}) → ep={ep:04d}  {mcap.name}", flush=True)
        try:
            report = convert_one(
                mcap,
                out_dir,
                task=task,
                image_ms=image_ms,
                arm_ms=arm_ms,
                jump_abs=jump_abs,
                glitch_ms=glitch_ms,
                numeric_interp_mode=numeric_interp_mode,
                hand_ms=hand_ms,
                tactile_ms=tactile_ms,
                fps_override=fps_override,
                episode_index=ep,
                index_offset=index_offset,
                chunks_size=chunks_size,
                write_meta=False,
                hands=hands,
                drop_if_no_right_tactile=drop_if_no_right_tactile,
            )
            n = int(report["kept_frames"])
            existing[key] = report
            ok.append(
                {
                    "episode_index": ep,
                    "mcap": key,
                    "kept_frames": n,
                    "drop_rate": report.get("drop_rate"),
                    "elapsed_s": round(time.time() - t0, 2),
                }
            )
            index_offset += n
            next_ep = max(int(idx) for idx in report["episode_indices"]) + 1
            print(
                f"  OK kept={n} drop={report.get('drop_rate')} elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
        except Exception as exc:
            failed.append({"mcap": key, "error": str(exc), "planned_episode_index": ep})
            err_path = out_dir / "convert_reports" / f"FAILED_{mcap.stem}.txt"
            err_path.write_text(traceback.format_exc())
            print(f"  FAIL: {exc}", flush=True)

    by_ep = sorted(existing.values(), key=lambda r: int(r["episode_index"]))
    episodes = [
        {"episode_index": int(r["episode_index"]), "tasks": [task], "length": int(r["kept_frames"])}
        for r in by_ep
    ]
    sources = [
        {
            "episode_index": int(r["episode_index"]),
            "mcap": r.get("mcap"),
            "kept_frames": int(r["kept_frames"]),
            "drop_rate": r.get("drop_rate"),
            "parquet": r.get("parquet"),
            "viz": r.get("viz"),
            "fps": r.get("fps"),
        }
        for r in by_ep
    ]
    fps_values = [float(r.get("fps", 30.0)) for r in by_ep]
    total_frames = int(sum(int(r["kept_frames"]) for r in by_ep))

    _write_folder_meta(
        out_dir,
        task=task,
        episodes=episodes,
        sources=sources,
        fps_values=fps_values,
        image_ms=image_ms,
        arm_ms=arm_ms,
        hand_ms=hand_ms,
        tactile_ms=tactile_ms,
        numeric_interp_mode=numeric_interp_mode,
        chunks_size=chunks_size,
    )

    summary = {
        "src_label": src_label,
        "out_dir": str(out_dir),
        "task": task,
        "numeric_interp_mode": numeric_interp_mode,
        "n_mcaps": len(mcaps),
        "n_ok_new": len(ok),
        "n_skipped_resume": len(skipped),
        "n_failed": len(failed),
        "n_episodes_in_meta": len(episodes),
        "total_frames": total_frames,
        "failed": failed,
        "viz_dir": str(out_dir / "viz"),
    }
    (out_dir / "batch_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    return summary


def convert_dir(
    src_dir: Path,
    out_dir: Path,
    *,
    task: str,
    image_ms: float,
    arm_ms: float,
    hand_ms: float,
    tactile_ms: float,
    numeric_interp_mode: str,
    jump_abs: float,
    glitch_ms: float,
    chunks_size: int,
    resume: bool,
    limit: int | None,
    hands: tuple[str, ...],
    drop_if_no_right_tactile: bool,
    fps_override: float | None,
) -> dict:
    mcaps = _list_mcaps(src_dir)
    if limit is not None:
        mcaps = mcaps[: int(limit)]
    if not mcaps:
        raise FileNotFoundError(f"no mcaps in {src_dir}")
    summary = convert_mcaps(
        mcaps,
        out_dir,
        src_label=src_dir.name,
        task=task,
        image_ms=image_ms,
        arm_ms=arm_ms,
        hand_ms=hand_ms,
        tactile_ms=tactile_ms,
        numeric_interp_mode=numeric_interp_mode,
        jump_abs=jump_abs,
        glitch_ms=glitch_ms,
        chunks_size=chunks_size,
        resume=resume,
        hands=hands,
        drop_if_no_right_tactile=drop_if_no_right_tactile,
        fps_override=fps_override,
    )
    summary["src_dir"] = str(src_dir)
    (out_dir / "batch_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    return summary


def _parse_hands(value: str) -> tuple[str, ...]:
    if value == "both":
        return HANDS
    if value in HANDS:
        return (value,)
    raise ValueError(f"unsupported --hands {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MCAP → LeRobot v2.1 with native tactile patches")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--mcap", type=Path, help="Single MCAP file")
    src.add_argument("--mcap-dir", type=Path, help="Directory of .mcap files (one episode each)")
    src.add_argument(
        "--mcap-list",
        type=Path,
        help="episode_sources.jsonl (field mcap) or a text file of MCAP paths",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--task",
        type=str,
        default="pour water from the right-hand bottle into the left-hand cup",
    )
    parser.add_argument("--image-ms", type=float, default=50.0)
    parser.add_argument("--arm-ms", type=float, default=50.0)
    parser.add_argument("--hand-ms", type=float, default=200.0)
    parser.add_argument(
        "--tactile-ms",
        type=float,
        default=50.0,
        help="Tactile nearest-neighbor gate; ignored in continuous mode (ungated)",
    )
    parser.add_argument(
        "--numeric-interp-mode",
        choices=("safe", "strict", "continuous"),
        default="continuous",
    )
    parser.add_argument("--jump-abs", type=float, default=50.0)
    parser.add_argument("--glitch-ms", type=float, default=80.0)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--chunks-size", type=int, default=1000)
    parser.add_argument("--hands", choices=("both", "left", "right"), default="both")
    parser.add_argument(
        "--drop-if-no-right-tactile",
        action="store_true",
        help="Drop a frame only when all 11 right taxels are missing",
    )
    parser.add_argument("--no-resume", action="store_true", help="Directory mode: redo all mcaps")
    parser.add_argument("--limit", type=int, default=None, help="Directory mode: first N mcaps")
    args = parser.parse_args()

    hands = _parse_hands(args.hands)
    common = dict(
        task=args.task,
        image_ms=args.image_ms,
        arm_ms=args.arm_ms,
        jump_abs=args.jump_abs,
        glitch_ms=args.glitch_ms,
        numeric_interp_mode=args.numeric_interp_mode,
        hand_ms=args.hand_ms,
        tactile_ms=args.tactile_ms,
        fps_override=args.fps,
        hands=hands,
        drop_if_no_right_tactile=bool(args.drop_if_no_right_tactile),
    )

    if args.mcap is not None:
        mcap_path = args.mcap.expanduser().resolve()
        if not mcap_path.is_file():
            raise FileNotFoundError(mcap_path)
        print(f"MCAP: {mcap_path}")
        print(f"Out:  {args.out_dir}")
        print(
            f"gates: image={args.image_ms:g}ms arm={args.arm_ms:g}ms "
            f"hand={args.hand_ms:g}ms tactile={args.tactile_ms:g}ms "
            f"mode={args.numeric_interp_mode} hands={'+'.join(hands)}"
        )
        report = convert_one(
            mcap_path,
            args.out_dir,
            chunks_size=args.chunks_size,
            write_meta=True,
            **common,
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print(f"OK → {report['parquet']}")
        print(f"VIZ → {report['viz']}")
        return

    if args.mcap_list is not None:
        list_path = args.mcap_list.expanduser().resolve()
        if not list_path.is_file():
            raise FileNotFoundError(list_path)
        mcaps = load_mcap_list(list_path)
        if args.limit is not None:
            mcaps = mcaps[: int(args.limit)]
        print(f"LIST: {list_path} ({len(mcaps)} mcaps)")
        print(f"Out:  {args.out_dir}")
        summary = convert_mcaps(
            mcaps,
            args.out_dir,
            src_label=list_path.name,
            chunks_size=args.chunks_size,
            resume=not bool(args.no_resume),
            **common,
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"OK → {summary['out_dir']}")
        return

    src_dir = args.mcap_dir.expanduser().resolve()
    if not src_dir.is_dir():
        raise FileNotFoundError(src_dir)
    print(f"DIR: {src_dir}")
    print(f"Out: {args.out_dir}")
    summary = convert_dir(
        src_dir,
        args.out_dir,
        chunks_size=args.chunks_size,
        resume=not bool(args.no_resume),
        limit=args.limit,
        **common,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"OK → {summary['out_dir']}")


if __name__ == "__main__":
    main()
