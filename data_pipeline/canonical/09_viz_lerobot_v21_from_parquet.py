#!/usr/bin/env python3
"""Visualize LeRobot v2.1 episodes from parquet (cmd + state).

Mirrors the style of mcap_to_lerobot_v21_single._plot_cmds, and adds a state
figure covering arm qpos, hand qpos, and EE pose.

Example:
  /home/xudi_ge/openpi/.venv/bin/python \\
    /home/xudi_ge/data/pipeline/09_viz_lerobot_v21_from_parquet.py \\
    --dataset /data/show_demo_v1_no_ok \\
    --out-dir /data/show_demo_v1_no_ok/viz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

SUPPORTED_SCHEMAS = {(14, 26), (15, 27)}


def _to_2d(col) -> np.ndarray:
    rows = col.to_pylist()
    if not rows:
        return np.zeros((0, 0), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _load_episode_meta(dataset: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    ep_path = dataset / "meta" / "episodes.jsonl"
    if ep_path.is_file():
        for line in ep_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            out[int(row["episode_index"])] = row
    src_path = dataset / "meta" / "episode_sources.jsonl"
    if src_path.is_file():
        for line in src_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            ep = int(row["episode_index"])
            out.setdefault(ep, {})
            out[ep].update(row)
    return out


def _list_parquets(dataset: Path) -> list[Path]:
    data_root = dataset / "data"
    files = sorted(data_root.glob("chunk-*/episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no episode parquet under {data_root}")
    return files


def _episode_index_from_path(path: Path) -> int:
    # episode_000012.parquet
    return int(path.stem.split("_")[1])


def plot_cmd(
    path: Path,
    *,
    t: np.ndarray,
    action: np.ndarray,
    title: str,
) -> None:
    action_dim = action.shape[1]
    if action_dim == 14:
        left_hand_indices = [12]
        right_hand_indices = [13]
    elif action_dim == 15:
        left_hand_indices = [12, 13]
        right_hand_indices = [14]
    else:
        raise ValueError(f"unsupported action dimension: {action_dim}")

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True, constrained_layout=True)

    ax = axes[0]
    for j in range(6):
        ax.plot(t, action[:, j], lw=0.9, label=f"L{j}")
    ax.set_ylabel("left arm cmd")
    ax.legend(loc="upper right", ncol=6, fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title("Left arm joint_cmd (6)")

    ax = axes[1]
    for j in range(6):
        ax.plot(t, action[:, 6 + j], lw=0.9, label=f"R{j}")
    ax.set_ylabel("right arm cmd")
    ax.legend(loc="upper right", ncol=6, fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title("Right arm joint_cmd (6)")

    ax = axes[2]
    colors_l = ["#ffd700", "#1f77b4"]
    labels_l = ["index", "middle"]
    for j, column in enumerate(left_hand_indices):
        ax.step(
            t,
            action[:, column],
            where="post",
            color=colors_l[j % len(colors_l)],
            lw=1.8,
            label=f"{labels_l[j]} [train]",
        )
    ax.set_ylabel("left hand")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Left hand cmd ({len(left_hand_indices)} DoF)")

    ax = axes[3]
    for j, column in enumerate(right_hand_indices):
        ax.step(
            t,
            action[:, column],
            where="post",
            color="#ff7f0e",
            lw=1.8,
            label=f"right_hand_{j} [train]",
        )
    ax.set_ylabel("right hand")
    ax.set_xlabel("time (s)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Right hand cmd ({len(right_hand_indices)} DoF)")

    fig.suptitle(f"{title}  |  kept={len(t)} frames", fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_state(
    path: Path,
    *,
    t: np.ndarray,
    state: np.ndarray,
    title: str,
) -> None:
    state_dim = state.shape[1]
    if state_dim == 26:
        left_hand_indices = [12]
        right_hand_indices = [13]
        left_ee_position = 14
        left_ee_rpy = 17
        right_ee_position = 20
        right_ee_rpy = 23
    elif state_dim == 27:
        left_hand_indices = [12, 13]
        right_hand_indices = [14]
        left_ee_position = 15
        left_ee_rpy = 18
        right_ee_position = 21
        right_ee_rpy = 24
    else:
        raise ValueError(f"unsupported state dimension: {state_dim}")

    fig, axes = plt.subplots(6, 1, figsize=(14, 14), sharex=True, constrained_layout=True)

    ax = axes[0]
    for j in range(6):
        ax.plot(t, state[:, j], lw=0.9, label=f"L{j}")
    ax.set_ylabel("left arm qpos")
    ax.legend(loc="upper right", ncol=6, fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title("Left arm joint_state (6)")

    ax = axes[1]
    for j in range(6):
        ax.plot(t, state[:, 6 + j], lw=0.9, label=f"R{j}")
    ax.set_ylabel("right arm qpos")
    ax.legend(loc="upper right", ncol=6, fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title("Right arm joint_state (6)")

    ax = axes[2]
    colors_l = ["#ffd700", "#1f77b4"]
    labels_l = ["index", "middle"]
    for j, column in enumerate(left_hand_indices):
        ax.step(
            t,
            state[:, column],
            where="post",
            color=colors_l[j % len(colors_l)],
            lw=1.8,
            label=f"{labels_l[j]} [train]",
        )
    for j, column in enumerate(right_hand_indices):
        ax.step(
            t,
            state[:, column],
            where="post",
            color="#ff7f0e",
            lw=1.8,
            label=f"right_hand_{j} [train]",
        )
    ax.set_ylabel("hand qpos")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(
        f"Hand joint_state (L{len(left_hand_indices)} + R{len(right_hand_indices)})"
    )

    ax = axes[3]
    for j, name in enumerate(("x", "y", "z")):
        ax.plot(t, state[:, left_ee_position + j], lw=1.0, label=f"left_ee_{name}")
    ax.set_ylabel("left ee xyz")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Left EE position (xyz)")

    ax = axes[4]
    for j, name in enumerate(("roll", "pitch", "yaw")):
        ax.plot(t, state[:, left_ee_rpy + j], lw=1.0, label=f"left_ee_{name}")
    for j, name in enumerate(("roll", "pitch", "yaw")):
        ax.plot(
            t,
            state[:, right_ee_rpy + j],
            lw=1.0,
            linestyle="--",
            label=f"right_ee_{name}",
        )
    ax.set_ylabel("ee rpy")
    ax.legend(loc="upper right", ncol=6, fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_title("EE orientation (left solid / right dashed)")

    ax = axes[5]
    for j, name in enumerate(("x", "y", "z")):
        ax.plot(t, state[:, right_ee_position + j], lw=1.0, label=f"right_ee_{name}")
    ax.set_ylabel("right ee xyz")
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Right EE position (xyz)")

    fig.suptitle(f"{title}  |  state  |  kept={len(t)} frames", fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def visualize_one(
    parquet_path: Path,
    out_dir: Path,
    *,
    meta: dict,
    overwrite: bool,
) -> dict:
    ep = _episode_index_from_path(parquet_path)
    cmd_path = out_dir / f"episode_{ep:06d}_all_cmd.png"
    state_path = out_dir / f"episode_{ep:06d}_all_state.png"
    if not overwrite and cmd_path.is_file() and state_path.is_file():
        return {
            "episode_index": ep,
            "skipped": True,
            "cmd": str(cmd_path),
            "state": str(state_path),
        }

    table = pq.read_table(
        parquet_path,
        columns=["timestamp", "action", "observation.state"],
    )
    t = np.asarray(table.column("timestamp").to_pylist(), dtype=np.float64)
    action = _to_2d(table.column("action"))
    state = _to_2d(table.column("observation.state"))
    if action.ndim != 2:
        raise RuntimeError(f"bad action shape {action.shape} in {parquet_path}")
    if state.ndim != 2:
        raise RuntimeError(f"bad state shape {state.shape} in {parquet_path}")
    if (action.shape[1], state.shape[1]) not in SUPPORTED_SCHEMAS:
        raise RuntimeError(
            f"unsupported action/state shapes {action.shape}/{state.shape} in {parquet_path}"
        )
    if t.size != action.shape[0]:
        raise RuntimeError(f"timestamp/action length mismatch in {parquet_path}")

    src = meta.get("mcap")
    tasks = meta.get("tasks") or []
    task = tasks[0] if tasks else meta.get("task")
    if src:
        title = Path(str(src)).name
    elif task:
        title = f"ep{ep:06d} | {task}"
    else:
        title = f"episode_{ep:06d}"

    plot_cmd(cmd_path, t=t, action=action, title=title)
    plot_state(state_path, t=t, state=state, title=title)
    return {
        "episode_index": ep,
        "skipped": False,
        "n_frames": int(t.size),
        "cmd": str(cmd_path),
        "state": str(state_path),
        "title": title,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Viz LeRobot v2.1 cmd+state from parquet")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    dataset = args.dataset.expanduser().resolve()
    out_dir = (args.out_dir or (dataset / "viz")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta_by_ep = _load_episode_meta(dataset)
    files = _list_parquets(dataset)
    if args.limit is not None:
        files = files[: int(args.limit)]

    ok = 0
    skipped = 0
    failed: list[dict] = []
    for i, pq_path in enumerate(files):
        ep = _episode_index_from_path(pq_path)
        print(f"[{i + 1}/{len(files)}] episode_{ep:06d}", flush=True)
        try:
            report = visualize_one(
                pq_path,
                out_dir,
                meta=meta_by_ep.get(ep, {}),
                overwrite=bool(args.overwrite),
            )
            if report.get("skipped"):
                skipped += 1
                print("  skip existing", flush=True)
            else:
                ok += 1
                print(f"  OK frames={report['n_frames']}", flush=True)
        except Exception as e:
            failed.append({"episode_index": ep, "parquet": str(pq_path), "error": str(e)})
            print(f"  FAIL: {e}", flush=True)

    summary = {
        "dataset": str(dataset),
        "out_dir": str(out_dir),
        "n_parquet": len(files),
        "n_ok": ok,
        "n_skipped": skipped,
        "n_failed": len(failed),
        "failed": failed,
    }
    (out_dir / "viz_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("DONE →", out_dir)


if __name__ == "__main__":
    main()
