#!/usr/bin/env python3
"""Visualize timing gaps, stream alignment, and return-home jumps in one MCAP."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_converter(path: Path):
    spec = importlib.util.spec_from_file_location("mcap_to_lerobot_v21_runtime", path)
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


def interpolate_at(source_t: np.ndarray, values: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    return np.stack(
        [np.interp(target_t, source_t, values[:, j]) for j in range(values.shape[1])],
        axis=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize raw MCAP anomaly evidence")
    parser.add_argument("--mcap", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--converter",
        type=Path,
        default=Path("/data/mcap_to_lerobot_v21.py"),
    )
    args = parser.parse_args()

    mcap = args.mcap.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    converter = load_converter(args.converter.expanduser().resolve())
    data = converter.load_mcap(mcap)

    top_t = np.asarray(data["top"][0], dtype=np.float64)
    t0 = float(top_t[0])
    top_rel = top_t - t0
    stream_keys = [
        "arm_q",
        "arm_cmd",
        "lee_p",
        "ree_p",
        "lhq",
        "rhq",
        "lhc",
        "rhc",
    ]
    stream_labels = {
        "arm_q": "arm state",
        "arm_cmd": "arm cmd",
        "lee_p": "left EE",
        "ree_p": "right EE",
        "lhq": "left hand state",
        "rhq": "right hand state",
        "lhc": "left hand cmd",
        "rhc": "right hand cmd",
    }

    cmd_t = np.asarray(data["arm_cmd"][0], dtype=np.float64)
    cmd = np.asarray(data["arm_cmd"][1], dtype=np.float64)
    state_t = np.asarray(data["arm_q"][0], dtype=np.float64)
    state = np.asarray(data["arm_q"][1], dtype=np.float64)
    cmd_delta_l2 = np.linalg.norm(np.diff(cmd, axis=0), axis=1)
    jump_index = int(np.argmax(cmd_delta_l2))
    jump_t = float(cmd_t[jump_index + 1])
    jump_rel = jump_t - t0
    state_on_cmd = interpolate_at(state_t, state, cmd_t)
    state_cmd_l2 = np.linalg.norm(state_on_cmd - cmd, axis=1)

    fig, axes = plt.subplots(5, 1, figsize=(16, 17), constrained_layout=True)

    ax = axes[0]
    for key in ["arm_q", "lhq", "rhq", "lhc", "rhc"]:
        source_t = np.asarray(data[key][0], dtype=np.float64)
        distance_ms = nearest_distance(source_t, top_t) * 1000.0
        ax.plot(top_rel, distance_ms, lw=1.0, label=stream_labels[key])
    ax.axhline(50.0, color="red", linestyle="--", lw=1.2, label="50 ms gate")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylabel("nearest age (ms)")
    ax.set_title("Numeric stream distance to each top-camera anchor")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3, fontsize=8)

    ax = axes[1]
    row = 0
    for key in stream_keys:
        ts = np.asarray(data[key][0], dtype=np.float64)
        gap_ms = np.diff(ts) * 1000.0
        indices = np.flatnonzero(gap_ms > 50.0)
        if indices.size:
            ax.scatter(
                ts[indices] - t0,
                np.full(indices.size, row),
                s=np.clip(gap_ms[indices] / 4.0, 12, 180),
                c=gap_ms[indices],
                cmap="magma",
                vmin=50,
                vmax=1050,
                edgecolors="none",
            )
        row += 1
    ax.set_yticks(range(len(stream_keys)), [stream_labels[key] for key in stream_keys])
    ax.set_ylabel("stream")
    ax.set_title("Internal gaps >50 ms (marker size/color encode gap duration)")
    ax.grid(True, axis="x", alpha=0.3)

    ax = axes[2]
    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    # /vla first six values are verified against /right_arm/*.
    for j, color in enumerate(colors):
        ax.plot(
            cmd_t - t0,
            cmd[:, j],
            color=color,
            lw=1.3,
            label=f"right cmd j{j}",
        )
        ax.plot(
            state_t - t0,
            state[:, j],
            color=color,
            lw=0.9,
            linestyle="--",
            alpha=0.75,
        )
    ax.axvline(jump_rel, color="red", linestyle="--", lw=1.5, label="largest cmd jump")
    ax.set_ylabel("joint position (rad)")
    ax.set_title("Right arm: command (solid) vs measured state (dashed)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=7)

    ax = axes[3]
    ax.plot(cmd_t - t0, state_cmd_l2, color="tab:red", lw=1.2, label="||state-cmd||₂")
    ax.plot(
        cmd_t[1:] - t0,
        cmd_delta_l2,
        color="tab:blue",
        lw=1.0,
        label="||Δcmd||₂",
    )
    ax.axvline(jump_rel, color="red", linestyle="--", lw=1.5)
    ax.set_ylabel("L2 norm (rad)")
    ax.set_title(
        f"Return-home signature: Δcmd={cmd_delta_l2[jump_index]:.3f} rad at {jump_rel:.3f}s"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[4]
    for key in ["lhq", "rhq", "lhc", "rhc"]:
        ts = np.asarray(data[key][0], dtype=np.float64)
        values = np.asarray(data[key][1], dtype=np.float64)
        ax.plot(ts - t0, values[:, 0], lw=1.0, label=stream_labels[key])
    ax.axvline(jump_rel, color="red", linestyle="--", lw=1.5)
    ax.set_xlabel("time from first top frame (s)")
    ax.set_ylabel("hand value")
    ax.set_title("Hand state/cmd samples (line bridges indicate unobserved gaps)")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, fontsize=8)

    fig.suptitle(f"{mcap.name} — anomaly overview", fontsize=14)
    overview_path = out_dir / f"{mcap.stem}_anomaly_overview.png"
    fig.savefig(overview_path, dpi=140)
    plt.close(fig)

    zoom_start = max(0.0, jump_rel - 1.0)
    zoom_end = min(float(top_rel[-1]), jump_rel + 2.6)
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True, constrained_layout=True)
    ax = axes[0]
    for j, color in enumerate(colors):
        ax.plot(cmd_t - t0, cmd[:, j], color=color, lw=1.5, label=f"cmd j{j}")
        ax.plot(
            state_t - t0,
            state[:, j],
            color=color,
            lw=1.0,
            linestyle="--",
            label=f"state j{j}",
        )
    ax.axvline(jump_rel, color="red", linestyle="--", lw=1.5)
    ax.set_xlim(zoom_start, zoom_end)
    ax.set_ylabel("joint position (rad)")
    ax.set_title("Return-home zoom: command jumps while state remains continuous")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=6, fontsize=7)

    ax = axes[1]
    ax.plot(cmd_t - t0, state_cmd_l2, color="tab:red", lw=1.5, label="||state-cmd||₂")
    ax.plot(cmd_t[1:] - t0, cmd_delta_l2, color="tab:blue", lw=1.2, label="||Δcmd||₂")
    ax.axvline(jump_rel, color="red", linestyle="--", lw=1.5)
    ax.set_xlim(zoom_start, zoom_end)
    ax.set_xlabel("time from first top frame (s)")
    ax.set_ylabel("L2 norm (rad)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    zoom_path = out_dir / f"{mcap.stem}_return_home_zoom.png"
    fig.savefig(zoom_path, dpi=140)
    plt.close(fig)

    summary = {
        "mcap": str(mcap),
        "overview": str(overview_path),
        "return_home_zoom": str(zoom_path),
        "top_frames": int(len(top_t)),
        "largest_cmd_jump_time_s": round(jump_rel, 6),
        "largest_cmd_jump_l2_rad": round(float(cmd_delta_l2[jump_index]), 6),
        "state_cmd_l2_at_jump_rad": round(
            float(state_cmd_l2[jump_index + 1]), 6
        ),
        "stream_max_gap_ms": {
            key: round(float(np.diff(np.asarray(data[key][0])).max() * 1000.0), 3)
            for key in stream_keys
        },
    }
    summary_path = out_dir / f"{mcap.stem}_anomaly_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
