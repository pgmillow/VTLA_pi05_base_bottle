#!/usr/bin/env python3
"""Audit LeRobot v2.1 parquet episodes for temporal discontinuities.

The audit is read-only. It separates deterministic representation/data issues
from motion candidates that require review:

* angle_wrap: raw RPY jump across +/-pi while circular delta stays small
* severe_timestamp_gap: adjacent retained frames are more than 0.5 s apart
* fast_state_motion: large state delta at a normal frame interval
* large_action_step: large arm action delta at a normal frame interval
* structural: non-finite values, bad dimensions, or non-increasing timestamps
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


STATE_DIM = 27
ACTION_DIM = 15
NORMAL_DT_MAX_S = 0.05
SEVERE_GAP_S = 0.5
ANGLE_WRAP_RAW_RAD = 3.0
ANGLE_WRAP_CIRCULAR_MAX_RAD = 0.3
ARM_STATE_STEP_RAD = 0.1
EE_POSITION_STEP_M = 0.02
EE_ORIENTATION_STEP_RAD = 0.12
ARM_ACTION_STEP_RAD = 1.0

RPY_STATE_INDICES = [18, 19, 20, 24, 25, 26]


def circular_delta(delta: np.ndarray) -> np.ndarray:
    return (delta + np.pi) % (2 * np.pi) - np.pi


def load_jsonl_by_episode(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows[int(row["episode_index"])] = row
    return rows


def event_base(
    *,
    episode_index: int,
    frame_index: int,
    timestamps: np.ndarray,
    source: dict,
    dataset: Path,
) -> dict:
    return {
        "episode_index": episode_index,
        "frame_index_before": frame_index,
        "frame_index_after": frame_index + 1,
        "time_before_s": round(float(timestamps[frame_index]), 6),
        "time_after_s": round(float(timestamps[frame_index + 1]), 6),
        "dt_s": round(float(timestamps[frame_index + 1] - timestamps[frame_index]), 6),
        "source_mcap": source.get("mcap"),
        "source_folder": source.get("source_folder"),
        "source_episode_index": source.get("source_episode_index"),
        "task": source.get("task"),
        "state_viz": str(dataset / "viz" / f"episode_{episode_index:06d}_all_state.png"),
        "cmd_viz": str(dataset / "viz" / f"episode_{episode_index:06d}_all_cmd.png"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit LeRobot v2.1 temporal jumps")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    dataset = args.dataset.expanduser().resolve()
    out_dir = (args.out_dir or dataset / "anomaly_audit").expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
    features = info["features"]
    state_names = features["observation.state"].get("names") or [
        f"state_{i}" for i in range(STATE_DIM)
    ]
    action_names = features["action"].get("names") or [
        f"action_{i}" for i in range(ACTION_DIM)
    ]
    sources = load_jsonl_by_episode(dataset / "meta" / "episode_sources.jsonl")
    episodes_meta = load_jsonl_by_episode(dataset / "meta" / "episodes.jsonl")
    parquet_files = sorted((dataset / "data").glob("chunk-*/episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"no episode parquet files under {dataset / 'data'}")

    events: list[dict] = []
    structural_issues: list[dict] = []

    for number, parquet_path in enumerate(parquet_files, start=1):
        episode_index = int(parquet_path.stem.split("_")[1])
        print(f"[{number}/{len(parquet_files)}] episode_{episode_index:06d}", flush=True)
        table = pq.read_table(
            parquet_path,
            columns=["timestamp", "observation.state", "action"],
        )
        timestamps = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64)
        state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float64)
        action = np.asarray(table.column("action").to_pylist(), dtype=np.float64)
        source = sources.get(episode_index, {})

        if state.ndim != 2 or state.shape[1] != STATE_DIM:
            structural_issues.append(
                {
                    "episode_index": episode_index,
                    "issue": "bad_state_shape",
                    "value": list(state.shape),
                    "parquet": str(parquet_path),
                }
            )
            continue
        if action.ndim != 2 or action.shape[1] != ACTION_DIM:
            structural_issues.append(
                {
                    "episode_index": episode_index,
                    "issue": "bad_action_shape",
                    "value": list(action.shape),
                    "parquet": str(parquet_path),
                }
            )
            continue
        if not (len(timestamps) == len(state) == len(action)):
            structural_issues.append(
                {
                    "episode_index": episode_index,
                    "issue": "row_count_mismatch",
                    "value": [len(timestamps), len(state), len(action)],
                    "parquet": str(parquet_path),
                }
            )
            continue
        if not np.isfinite(timestamps).all() or not np.isfinite(state).all() or not np.isfinite(action).all():
            structural_issues.append(
                {
                    "episode_index": episode_index,
                    "issue": "non_finite_value",
                    "parquet": str(parquet_path),
                }
            )

        dt = np.diff(timestamps)
        state_delta = np.diff(state, axis=0)
        action_delta = np.diff(action, axis=0)
        normal_dt = (dt > 0) & (dt <= NORMAL_DT_MAX_S)

        for i in np.flatnonzero(dt <= 0):
            structural_issues.append(
                {
                    **event_base(
                        episode_index=episode_index,
                        frame_index=int(i),
                        timestamps=timestamps,
                        source=source,
                        dataset=dataset,
                    ),
                    "issue": "non_increasing_timestamp",
                }
            )

        for i in np.flatnonzero(dt > SEVERE_GAP_S):
            events.append(
                {
                    **event_base(
                        episode_index=episode_index,
                        frame_index=int(i),
                        timestamps=timestamps,
                        source=source,
                        dataset=dataset,
                    ),
                    "severity": "confirmed",
                    "category": "severe_timestamp_gap",
                    "channel": "timestamp",
                    "delta": round(float(dt[i]), 6),
                    "unit": "s",
                    "reason": "Adjacent retained frames are more than 0.5 s apart.",
                }
            )

        rpy = state[:, RPY_STATE_INDICES]
        raw_rpy_delta = np.diff(rpy, axis=0)
        circ_rpy_delta = circular_delta(raw_rpy_delta)
        for local_channel, state_index in enumerate(RPY_STATE_INDICES):
            wrap_mask = (
                (np.abs(raw_rpy_delta[:, local_channel]) > ANGLE_WRAP_RAW_RAD)
                & (np.abs(circ_rpy_delta[:, local_channel]) < ANGLE_WRAP_CIRCULAR_MAX_RAD)
            )
            for i in np.flatnonzero(wrap_mask):
                events.append(
                    {
                        **event_base(
                            episode_index=episode_index,
                            frame_index=int(i),
                            timestamps=timestamps,
                            source=source,
                            dataset=dataset,
                        ),
                        "severity": "confirmed",
                        "category": "angle_wrap",
                        "channel": state_names[state_index],
                        "value_before": round(float(state[i, state_index]), 6),
                        "value_after": round(float(state[i + 1, state_index]), 6),
                        "delta": round(float(raw_rpy_delta[i, local_channel]), 6),
                        "circular_delta": round(float(circ_rpy_delta[i, local_channel]), 6),
                        "unit": "rad",
                        "reason": "Raw Euler angle crosses +/-pi while circular motion stays small.",
                    }
                )

            fast_mask = (
                (np.abs(circ_rpy_delta[:, local_channel]) > EE_ORIENTATION_STEP_RAD)
                & normal_dt
                & ~wrap_mask
            )
            for i in np.flatnonzero(fast_mask):
                events.append(
                    {
                        **event_base(
                            episode_index=episode_index,
                            frame_index=int(i),
                            timestamps=timestamps,
                            source=source,
                            dataset=dataset,
                        ),
                        "severity": "review",
                        "category": "fast_ee_orientation",
                        "channel": state_names[state_index],
                        "delta": round(float(circ_rpy_delta[i, local_channel]), 6),
                        "unit": "rad",
                        "reason": "Circular EE orientation step exceeds 0.12 rad within 50 ms.",
                    }
                )

        for state_index in range(12):
            mask = (np.abs(state_delta[:, state_index]) > ARM_STATE_STEP_RAD) & normal_dt
            for i in np.flatnonzero(mask):
                events.append(
                    {
                        **event_base(
                            episode_index=episode_index,
                            frame_index=int(i),
                            timestamps=timestamps,
                            source=source,
                            dataset=dataset,
                        ),
                        "severity": "review",
                        "category": "fast_arm_state",
                        "channel": state_names[state_index],
                        "delta": round(float(state_delta[i, state_index]), 6),
                        "unit": "rad",
                        "reason": "Arm state step exceeds 0.1 rad within 50 ms.",
                    }
                )

        for state_index in [15, 16, 17, 21, 22, 23]:
            mask = (np.abs(state_delta[:, state_index]) > EE_POSITION_STEP_M) & normal_dt
            for i in np.flatnonzero(mask):
                events.append(
                    {
                        **event_base(
                            episode_index=episode_index,
                            frame_index=int(i),
                            timestamps=timestamps,
                            source=source,
                            dataset=dataset,
                        ),
                        "severity": "review",
                        "category": "fast_ee_position",
                        "channel": state_names[state_index],
                        "delta": round(float(state_delta[i, state_index]), 6),
                        "unit": "m",
                        "reason": "EE position step exceeds 0.02 m within 50 ms.",
                    }
                )

        for action_index in range(12):
            mask = (np.abs(action_delta[:, action_index]) > ARM_ACTION_STEP_RAD) & normal_dt
            for i in np.flatnonzero(mask):
                events.append(
                    {
                        **event_base(
                            episode_index=episode_index,
                            frame_index=int(i),
                            timestamps=timestamps,
                            source=source,
                            dataset=dataset,
                        ),
                        "severity": "review",
                        "category": "large_arm_action_step",
                        "channel": action_names[action_index],
                        "delta": round(float(action_delta[i, action_index]), 6),
                        "unit": "rad",
                        "reason": "Arm action step exceeds 1 rad within 50 ms; may be a task phase switch.",
                    }
                )

    severity_counts = Counter(event["severity"] for event in events)
    category_counts = Counter(event["category"] for event in events)
    episodes_by_category: dict[str, set[int]] = defaultdict(set)
    for event in events:
        episodes_by_category[event["category"]].add(event["episode_index"])

    episode_summary = []
    all_episode_indices = sorted(
        set(event["episode_index"] for event in events)
        | set(issue["episode_index"] for issue in structural_issues)
    )
    for episode_index in all_episode_indices:
        episode_events = [event for event in events if event["episode_index"] == episode_index]
        episode_issues = [
            issue for issue in structural_issues if issue["episode_index"] == episode_index
        ]
        source = sources.get(episode_index, {})
        episode_summary.append(
            {
                "episode_index": episode_index,
                "length": episodes_meta.get(episode_index, {}).get("length"),
                "source_mcap": source.get("mcap"),
                "task": source.get("task"),
                "confirmed_event_count": sum(
                    event["severity"] == "confirmed" for event in episode_events
                ),
                "review_event_count": sum(
                    event["severity"] == "review" for event in episode_events
                ),
                "categories": sorted(set(event["category"] for event in episode_events)),
                "structural_issue_count": len(episode_issues),
                "state_viz": str(dataset / "viz" / f"episode_{episode_index:06d}_all_state.png"),
                "cmd_viz": str(dataset / "viz" / f"episode_{episode_index:06d}_all_cmd.png"),
            }
        )

    report = {
        "dataset": str(dataset),
        "episodes_scanned": len(parquet_files),
        "frames_scanned": int(info.get("total_frames", 0)),
        "thresholds": {
            "normal_dt_max_s": NORMAL_DT_MAX_S,
            "severe_gap_s": SEVERE_GAP_S,
            "angle_wrap_raw_rad": ANGLE_WRAP_RAW_RAD,
            "angle_wrap_circular_max_rad": ANGLE_WRAP_CIRCULAR_MAX_RAD,
            "arm_state_step_rad": ARM_STATE_STEP_RAD,
            "ee_position_step_m": EE_POSITION_STEP_M,
            "ee_orientation_step_rad": EE_ORIENTATION_STEP_RAD,
            "arm_action_step_rad": ARM_ACTION_STEP_RAD,
        },
        "summary": {
            "event_count": len(events),
            "confirmed_event_count": severity_counts["confirmed"],
            "review_event_count": severity_counts["review"],
            "structural_issue_count": len(structural_issues),
            "category_event_counts": dict(sorted(category_counts.items())),
            "category_episode_counts": {
                category: len(episodes)
                for category, episodes in sorted(episodes_by_category.items())
            },
            "category_episode_indices": {
                category: sorted(episodes)
                for category, episodes in sorted(episodes_by_category.items())
            },
        },
        "events": events,
        "structural_issues": structural_issues,
        "episodes": episode_summary,
    }

    report_path = out_dir / "anomaly_audit.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    csv_path = out_dir / "anomaly_events.csv"
    csv_fields = [
        "severity",
        "category",
        "episode_index",
        "frame_index_before",
        "frame_index_after",
        "time_before_s",
        "time_after_s",
        "dt_s",
        "channel",
        "value_before",
        "value_after",
        "delta",
        "circular_delta",
        "unit",
        "source_mcap",
        "reason",
        "state_viz",
        "cmd_viz",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(events)

    episode_path = out_dir / "anomaly_episodes.csv"
    with episode_path.open("w", encoding="utf-8", newline="") as handle:
        fields = list(episode_summary[0].keys()) if episode_summary else ["episode_index"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in episode_summary:
            csv_row = dict(row)
            csv_row["categories"] = "|".join(row["categories"])
            writer.writerow(csv_row)

    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"JSON -> {report_path}")
    print(f"Events CSV -> {csv_path}")
    print(f"Episodes CSV -> {episode_path}")


if __name__ == "__main__":
    main()
