#!/usr/bin/env python3
"""Read-only quality classification for Octopus MCAP recordings.

No MCAP is modified. Results are checkpointed per file so interrupted scans can
resume without rereading completed files.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


REQUIRED_NUMERIC = (
    "arm_q",
    "arm_cmd",
    "lee_p",
    "lee_r",
    "ree_p",
    "ree_r",
    "lhq",
    "rhq",
    "lhc",
    "rhc",
)
IMAGE_KEYS = ("top", "left_img", "right_img")
HAND_KEYS = ("lhq", "rhq", "lhc", "rhc")
ARM_EE_KEYS = ("arm_q", "arm_cmd", "lee_p", "lee_r", "ree_p", "ree_r")


def load_converter(path: Path):
    spec = importlib.util.spec_from_file_location("quality_converter_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load converter: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def nearest_distance(source_t: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source_t, target_t, side="left")
    right = np.clip(right, 0, len(source_t) - 1)
    left = np.clip(right - 1, 0, len(source_t) - 1)
    return np.minimum(np.abs(source_t[right] - target_t), np.abs(source_t[left] - target_t))


def isolated_spikes(
    timestamps: np.ndarray,
    values: np.ndarray,
    *,
    jump_threshold: float,
    residual_threshold: float,
) -> list[dict]:
    if len(timestamps) < 3:
        return []
    incoming = values[1:-1] - values[:-2]
    outgoing = values[2:] - values[1:-1]
    residual = values[2:] - values[:-2]
    dt = np.diff(timestamps)
    mask = (
        (np.abs(incoming) > jump_threshold)
        & (np.abs(outgoing) > jump_threshold)
        & (incoming * outgoing < 0)
        & (np.abs(residual) < residual_threshold)
        & (dt[:-1, None] <= 0.05)
        & (dt[1:, None] <= 0.05)
    )
    events = []
    for row, channel in zip(*np.where(mask)):
        center = int(row + 1)
        events.append(
            {
                "sample_index": center,
                "channel": int(channel),
                "time_s": round(float(timestamps[center]), 6),
                "incoming": round(float(incoming[row, channel]), 6),
                "outgoing": round(float(outgoing[row, channel]), 6),
            }
        )
    return events


def analyze_file(path: Path, converter) -> dict:
    started = time.time()
    data = converter.load_mcap(path)
    top_t = np.asarray(data["top"][0], dtype=np.float64)
    if top_t.size == 0:
        raise RuntimeError("top camera stream is empty")
    t0 = float(top_t[0])
    top_end = float(top_t[-1])

    record: dict = {
        "path": str(path),
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "top_frames": int(top_t.size),
        "duration_s": round(top_end - t0, 6),
        "image": {},
        "numeric": {},
        "hand_gap_summary": {},
        "spikes": {},
        "rpy_wraps": [],
        "return_home": None,
        "large_nonhome_cmd_jumps": [],
        "nonfinite_streams": [],
        "classification": None,
        "reasons": [],
        "flags": [],
    }

    for key in IMAGE_KEYS:
        timestamps = np.asarray(data[key][0], dtype=np.float64)
        gaps = np.diff(timestamps)
        record["image"][key] = {
            "count": int(timestamps.size),
            "max_gap_ms": round(float(gaps.max(initial=0.0) * 1000.0), 3),
            "gaps_over_50ms": int(np.count_nonzero(gaps > 0.05)),
            "non_increasing": int(np.count_nonzero(gaps <= 0)),
        }

    for key in ("left_img", "right_img"):
        timestamps = np.asarray(data[key][0], dtype=np.float64)
        distances = nearest_distance(timestamps, top_t)
        record["image"][key]["sync_max_ms"] = round(float(distances.max() * 1000.0), 3)
        record["image"][key]["anchors_over_50ms"] = int(np.count_nonzero(distances > 0.05))

    for key in REQUIRED_NUMERIC:
        timestamps = np.asarray(data[key][0], dtype=np.float64)
        values = np.asarray(data[key][1], dtype=np.float64)
        gaps = np.diff(timestamps)
        if not np.isfinite(values).all() or not np.isfinite(timestamps).all():
            record["nonfinite_streams"].append(key)
        record["numeric"][key] = {
            "count": int(timestamps.size),
            "dim": int(values.shape[1]) if values.ndim == 2 else None,
            "max_gap_ms": round(float(gaps.max(initial=0.0) * 1000.0), 3),
            "gaps_over_50ms": int(np.count_nonzero(gaps > 0.05)),
            "gaps_over_100ms": int(np.count_nonzero(gaps > 0.1)),
            "gaps_over_200ms": int(np.count_nonzero(gaps > 0.2)),
            "gaps_over_500ms": int(np.count_nonzero(gaps > 0.5)),
            "gaps_over_1s": int(np.count_nonzero(gaps > 1.0)),
            "start_delay_ms": round(float((timestamps[0] - t0) * 1000.0), 3),
            "end_early_ms": round(float((top_end - timestamps[-1]) * 1000.0), 3),
            "non_increasing": int(np.count_nonzero(gaps <= 0)),
        }

    # Long hand gaps are safe to hold only when both endpoints are effectively
    # unchanged. Otherwise the transition time is unknowable.
    for key in HAND_KEYS:
        timestamps = np.asarray(data[key][0], dtype=np.float64)
        values = np.asarray(data[key][1], dtype=np.float64)
        gaps = np.diff(timestamps)
        indices = np.flatnonzero(gaps > 0.2)
        stable = []
        dynamic = []
        for index in indices:
            endpoint_delta = float(np.max(np.abs(values[index + 1] - values[index])))
            event = {
                "sample_before": int(index),
                "start_s": round(float(timestamps[index] - t0), 6),
                "end_s": round(float(timestamps[index + 1] - t0), 6),
                "gap_ms": round(float(gaps[index] * 1000.0), 3),
                "endpoint_delta": round(endpoint_delta, 3),
            }
            (stable if endpoint_delta <= 30.0 else dynamic).append(event)
        record["hand_gap_summary"][key] = {
            "stable_over_200ms": len(stable),
            "dynamic_over_200ms": len(dynamic),
            "dynamic_over_500ms": sum(event["gap_ms"] > 500.0 for event in dynamic),
            "dynamic_over_1s": sum(event["gap_ms"] > 1000.0 for event in dynamic),
            "largest_stable": max(stable, key=lambda event: event["gap_ms"], default=None),
            "largest_dynamic": max(dynamic, key=lambda event: event["gap_ms"], default=None),
            "dynamic_events": dynamic,
        }

    spike_configs = {
        "arm_q": (0.15, 0.03),
        "arm_cmd": (0.3, 0.05),
        "lee_p": (0.02, 0.005),
        "ree_p": (0.02, 0.005),
        "lhq": (100.0, 30.0),
        "rhq": (100.0, 30.0),
        "lhc": (100.0, 30.0),
        "rhc": (100.0, 30.0),
    }
    for key, (jump, residual) in spike_configs.items():
        record["spikes"][key] = isolated_spikes(
            np.asarray(data[key][0], dtype=np.float64),
            np.asarray(data[key][1], dtype=np.float64),
            jump_threshold=jump,
            residual_threshold=residual,
        )

    for key in ("lee_r", "ree_r"):
        timestamps = np.asarray(data[key][0], dtype=np.float64)
        values = np.asarray(data[key][1], dtype=np.float64)
        delta = np.diff(values, axis=0)
        circular = (delta + np.pi) % (2 * np.pi) - np.pi
        for row, channel in zip(*np.where((np.abs(delta) > 3.0) & (np.abs(circular) < 0.3))):
            record["rpy_wraps"].append(
                {
                    "stream": key,
                    "sample_before": int(row),
                    "channel": int(channel),
                    "time_s": round(float(timestamps[row] - t0), 6),
                    "raw_delta_rad": round(float(delta[row, channel]), 6),
                    "circular_delta_rad": round(float(circular[row, channel]), 6),
                }
            )

    cmd_t = np.asarray(data["arm_cmd"][0], dtype=np.float64)
    cmd = np.asarray(data["arm_cmd"][1], dtype=np.float64)
    cmd_delta = np.diff(cmd, axis=0)
    cmd_jump_l2 = np.linalg.norm(cmd_delta, axis=1)
    candidates = np.flatnonzero(cmd_jump_l2 > 0.5)
    for index in candidates:
        target = cmd[index + 1]
        initial_distance = float(np.linalg.norm(target - cmd[0]))
        persistence_stop = min(len(cmd), int(index + 22))
        persistence = float(
            np.max(np.linalg.norm(cmd[index + 1 : persistence_stop] - target, axis=1))
        )
        event = {
            "sample_after": int(index + 1),
            "time_s": round(float(cmd_t[index + 1] - t0), 6),
            "jump_l2_rad": round(float(cmd_jump_l2[index]), 6),
            "jump_linf_rad": round(float(np.max(np.abs(cmd_delta[index]))), 6),
            "target_to_initial_l2_rad": round(initial_distance, 6),
            "persistent_target_max_deviation_rad": round(persistence, 6),
        }
        if initial_distance <= 0.1 and persistence <= 0.05:
            if record["return_home"] is None or event["jump_l2_rad"] > record["return_home"]["jump_l2_rad"]:
                record["return_home"] = event
        else:
            record["large_nonhome_cmd_jumps"].append(event)

    reasons = record["reasons"]
    flags = record["flags"]

    # ---- collect all issues ----

    # Hard issues: data genuinely unfixable by the converter
    hard_reasons: list[str] = []
    if record["nonfinite_streams"]:
        hard_reasons.append("nonfinite_numeric")

    severe_spikes = sum(
        len(record["spikes"][key]) for key in ("arm_q", "arm_cmd", "lee_p", "ree_p")
    )
    if severe_spikes:
        hard_reasons.append("isolated_arm_or_ee_spike")

    dynamic_hand_1s = sum(
        record["hand_gap_summary"][key]["dynamic_over_1s"] for key in HAND_KEYS
    )
    if dynamic_hand_1s:
        hard_reasons.append("dynamic_hand_gap_over_1s")

    # Soft issues: the converter can handle these (sort, interpolate, etc.)
    soft_reasons: list[str] = []
    if any(record["image"][key]["non_increasing"] for key in IMAGE_KEYS):
        soft_reasons.append("non_monotonic_image_time")
    if any(record["numeric"][key]["non_increasing"] for key in REQUIRED_NUMERIC):
        soft_reasons.append("non_monotonic_numeric_time")
    if any(record["image"][key]["max_gap_ms"] > 100.0 for key in IMAGE_KEYS):
        soft_reasons.append("image_gap_over_100ms")
    if any(record["image"][key].get("sync_max_ms", 0.0) > 50.0 for key in ("left_img", "right_img")):
        soft_reasons.append("wrist_sync_over_50ms")

    dynamic_hand = sum(
        record["hand_gap_summary"][key]["dynamic_over_200ms"] for key in HAND_KEYS
    )
    stable_hand = sum(
        record["hand_gap_summary"][key]["stable_over_200ms"] for key in HAND_KEYS
    )
    max_arm_ee_gap = max(record["numeric"][key]["max_gap_ms"] for key in ARM_EE_KEYS)
    max_boundary_gap = max(
        max(
            max(0.0, record["numeric"][key]["start_delay_ms"]),
            max(0.0, record["numeric"][key]["end_early_ms"]),
        )
        for key in REQUIRED_NUMERIC
    )

    if record["rpy_wraps"]:
        flags.append("rpy_wrap_requires_unwrap")
    if record["return_home"] is not None:
        flags.append("return_home_detected")
    if record["large_nonhome_cmd_jumps"]:
        flags.append("large_nonhome_cmd_jump")

    # ---- decide classification ----

    has_soft = (
        len(soft_reasons) > 0
        or dynamic_hand > 0
        or max_arm_ee_gap > 200.0
        or max_boundary_gap > 500.0
    )
    has_mild = stable_hand > 0 or max_arm_ee_gap > 100.0 or max_boundary_gap > 100.0

    if hard_reasons:
        reasons.extend(hard_reasons)
        reasons.extend(soft_reasons)
        classification = "REJECT_REVIEW"
    elif has_soft:
        reasons.extend(soft_reasons)
        if dynamic_hand:
            reasons.append("dynamic_hand_gap_over_200ms")
        if max_arm_ee_gap > 200.0:
            reasons.append("arm_ee_gap_over_200ms")
        if max_boundary_gap > 500.0:
            reasons.append("required_stream_boundary_missing_over_500ms")
        classification = "SPLIT_REQUIRED"
    elif has_mild:
        reasons.extend(soft_reasons)
        if stable_hand:
            reasons.append("stable_hand_gap_over_200ms")
        if max_arm_ee_gap > 100.0:
            reasons.append("arm_ee_gap_100_to_200ms")
        if max_boundary_gap > 100.0:
            reasons.append("required_stream_boundary_missing_100_to_500ms")
        classification = "HOLD_REVIEW"
    else:
        reasons.extend(soft_reasons)
        classification = "PASS"

    record["classification"] = classification
    record["metrics"] = {
        "dynamic_hand_gaps_over_200ms": dynamic_hand,
        "dynamic_hand_gaps_over_1s": dynamic_hand_1s,
        "stable_hand_gaps_over_200ms": stable_hand,
        "max_arm_ee_gap_ms": round(float(max_arm_ee_gap), 3),
        "max_required_boundary_missing_ms": round(float(max_boundary_gap), 3),
        "isolated_arm_ee_spikes": severe_spikes,
    }
    record["elapsed_s"] = round(time.time() - started, 3)
    return record


def aggregate(records: list[dict], root: Path) -> dict:
    classification_counts = Counter(record["classification"] for record in records)
    flag_counts = Counter(flag for record in records for flag in record.get("flags", []))
    reason_counts = Counter(reason for record in records for reason in record.get("reasons", []))
    return {
        "root": str(root),
        "files_scanned": len(records),
        "classification_counts": dict(sorted(classification_counts.items())),
        "flag_counts": dict(sorted(flag_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "classes": {
            classification: [
                record["file"]
                for record in records
                if record["classification"] == classification
            ]
            for classification in ("PASS", "HOLD_REVIEW", "SPLIT_REQUIRED", "REJECT_REVIEW")
        },
        "return_home_files": [
            record["file"] for record in records if record.get("return_home") is not None
        ],
        "records": records,
    }


def write_lists(report: dict, out_dir: Path) -> None:
    for classification, files in report["classes"].items():
        (out_dir / f"{classification.lower()}.txt").write_text(
            "".join(f"{name}\n" for name in files),
            encoding="utf-8",
        )
    (out_dir / "return_home.txt").write_text(
        "".join(f"{name}\n" for name in report["return_home_files"]),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify MCAP quality without modifying data")
    parser.add_argument("mcap_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--converter",
        type=Path,
        default=Path("/data/mcap_to_lerobot_v21.py"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    root = args.mcap_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    records_dir = out_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(root.glob("*.mcap"))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"no MCAP files in {root}")

    converter = load_converter(args.converter.expanduser().resolve())
    records = []
    for number, path in enumerate(files, start=1):
        record_path = records_dir / f"{path.stem}.json"
        if record_path.is_file() and not args.overwrite:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            print(
                f"[{number}/{len(files)}] SKIP {record['classification']:14s} {path.name}",
                flush=True,
            )
        else:
            try:
                record = analyze_file(path, converter)
                print(
                    f"[{number}/{len(files)}] {record['classification']:14s} {path.name} "
                    f"hand_dynamic={record['metrics']['dynamic_hand_gaps_over_200ms']} "
                    f"hand_stable={record['metrics']['stable_hand_gaps_over_200ms']} "
                    f"arm_gap={record['metrics']['max_arm_ee_gap_ms']:.0f}ms "
                    f"home={record['return_home'] is not None}",
                    flush=True,
                )
            except Exception as exc:
                record = {
                    "path": str(path),
                    "file": path.name,
                    "classification": "REJECT_REVIEW",
                    "reasons": ["load_or_parse_error"],
                    "flags": [],
                    "error": f"{type(exc).__name__}: {exc}",
                    "return_home": None,
                }
                print(f"[{number}/{len(files)}] ERROR {path.name}: {exc}", flush=True)
            record_path.write_text(
                json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        records.append(record)
        current = aggregate(records, root)
        (out_dir / "classification_partial.json").write_text(
            json.dumps(current, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    report = aggregate(records, root)
    report_path = out_dir / "classification_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_lists(report, out_dir)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2, ensure_ascii=False))
    print(f"REPORT -> {report_path}")


if __name__ == "__main__":
    main()
