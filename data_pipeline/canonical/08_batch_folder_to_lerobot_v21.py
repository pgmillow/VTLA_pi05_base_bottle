#!/usr/bin/env python3
"""Batch-convert mcap folders → LeRobot v2.1 datasets (one dataset per folder).

Layout:
  <out-root>/<folder_name>/
    meta/{info.json,episodes.jsonl,tasks.jsonl,episode_sources.jsonl}
    data/chunk-XXX/episode_XXXXXX.parquet
    viz/episode_XXXXXX_all_cmd.png
    convert_reports/episode_XXXXXX.json
    batch_summary.json

Example:
  /home/xudi_ge/openpi/.venv/bin/python \\
    /home/xudi_ge/data/pipeline/08_batch_folder_to_lerobot_v21.py \\
    --src-root /data/show_data_V1 \\
    --folders right_0722 right_0723 \\
    --out-root /data/lerobot_v21_show_V1 \\
    --task "Pour water into the glass"
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, "/data")
from mcap_to_lerobot_v21 import (  # noqa: E402
    ACTION_DIM,
    STATE_DIM,
    convert_one,
    features_dict,
)


def _list_mcaps(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.mcap"))


def _load_reports(out_dir: Path) -> dict[str, dict]:
    """Map resolved mcap path → report dict (successful only)."""
    out: dict[str, dict] = {}
    rep_dir = out_dir / "convert_reports"
    if not rep_dir.is_dir():
        return out
    for p in sorted(rep_dir.glob("episode_*.json")):
        try:
            r = json.loads(p.read_text())
        except Exception:
            continue
        mcap = r.get("mcap")
        pq = Path(r.get("parquet", ""))
        viz = Path(r.get("viz", ""))
        if not mcap or int(r.get("kept_frames", 0)) <= 0:
            continue
        if not pq.is_file() or not viz.is_file():
            continue
        out[str(Path(mcap).resolve())] = r
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
    numeric_interp_mode: str,
    chunks_size: int,
) -> None:
    n_ep = len(episodes)
    n_frames = int(sum(int(e["length"]) for e in episodes))
    fps = float(sorted(fps_values)[len(fps_values) // 2]) if fps_values else 30.0
    n_chunks = (max((e["episode_index"] for e in episodes), default=-1) // chunks_size) + 1 if n_ep else 0

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
            "schema": {"action_dim": ACTION_DIM, "state_dim": STATE_DIM},
        },
    }
    meta = out_dir / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
    (meta / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": task}, ensure_ascii=False) + "\n"
    )
    with (meta / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for e in episodes:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with (meta / "episode_sources.jsonl").open("w", encoding="utf-8") as f:
        for s in sources:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def convert_folder(
    src_dir: Path,
    out_dir: Path,
    *,
    task: str,
    image_ms: float,
    arm_ms: float,
    hand_ms: float,
    numeric_interp_mode: str,
    jump_abs: float,
    glitch_ms: float,
    chunks_size: int,
    resume: bool,
    limit: int | None,
) -> dict:
    mcaps = _list_mcaps(src_dir)
    if limit is not None:
        mcaps = mcaps[: int(limit)]
    if not mcaps:
        raise FileNotFoundError(f"no mcaps in {src_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "convert_reports").mkdir(exist_ok=True)
    (out_dir / "viz").mkdir(exist_ok=True)

    existing = _load_reports(out_dir) if resume else {}
    # Contiguous episode ids from existing reports
    used_eps = sorted(int(r["episode_index"]) for r in existing.values())
    next_ep = (max(used_eps) + 1) if used_eps else 0
    index_offset = int(sum(int(r["kept_frames"]) for r in existing.values()))

    ok: list[dict] = []
    skipped: list[dict] = []
    failed: list[dict] = []

    for i, mcap in enumerate(mcaps):
        key = str(mcap.resolve())
        if key in existing:
            r = existing[key]
            skipped.append(
                {
                    "episode_index": int(r["episode_index"]),
                    "mcap": key,
                    "kept_frames": int(r["kept_frames"]),
                }
            )
            print(
                f"[{src_dir.name}] skip resume ep={int(r['episode_index']):04d} {mcap.name}",
                flush=True,
            )
            continue

        ep = next_ep
        t0 = time.time()
        print(
            f"\n[{src_dir.name}] ({i + 1}/{len(mcaps)}) → ep={ep:04d}  {mcap.name}",
            flush=True,
        )
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
                fps_override=None,
                episode_index=ep,
                index_offset=index_offset,
                chunks_size=chunks_size,
                write_meta=False,
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
            next_ep += 1
            print(
                f"  OK kept={n} drop={report.get('drop_rate')} "
                f"elapsed={time.time() - t0:.1f}s",
                flush=True,
            )
        except Exception as e:
            failed.append({"mcap": key, "error": str(e), "planned_episode_index": ep})
            err_path = out_dir / "convert_reports" / f"FAILED_{mcap.stem}.txt"
            err_path.write_text(traceback.format_exc())
            print(f"  FAIL: {e}", flush=True)

    # Rebuild meta from all successful reports, ordered by episode_index
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
        numeric_interp_mode=numeric_interp_mode,
        chunks_size=chunks_size,
    )

    high_drop = sorted(
        [s for s in sources if float(s.get("drop_rate") or 0) >= 0.1],
        key=lambda x: float(x.get("drop_rate") or 0),
        reverse=True,
    )

    summary = {
        "src_dir": str(src_dir),
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
        "high_drop_ge_10pct": high_drop[:30],
        "viz_dir": str(out_dir / "viz"),
    }
    (out_dir / "batch_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Batch mcap folders → LeRobot v2.1")
    p.add_argument("--src-root", type=Path, required=True)
    p.add_argument("--folders", nargs="+", required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--image-ms", type=float, default=50.0)
    p.add_argument("--arm-ms", type=float, default=50.0)
    p.add_argument("--hand-ms", type=float, default=200.0)
    p.add_argument(
        "--numeric-interp-mode",
        choices=("safe", "strict", "continuous"),
        default="continuous",
    )
    p.add_argument("--jump-abs", type=float, default=50.0)
    p.add_argument("--glitch-ms", type=float, default=80.0)
    p.add_argument("--chunks-size", type=int, default=1000)
    p.add_argument(
        "--generate-cmd-state-viz",
        action="store_true",
        help="Generate all_cmd.png and all_state.png after each folder is converted",
    )
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="debug: only first N mcaps per folder")
    args = p.parse_args()

    resume = not bool(args.no_resume)
    all_summaries = []
    for name in args.folders:
        src = args.src_root / name
        out = args.out_root / name
        if not src.is_dir():
            raise FileNotFoundError(src)
        print("=" * 72, flush=True)
        print(f"FOLDER {name}: {src} → {out}", flush=True)
        print(f"task={args.task!r} resume={resume}", flush=True)
        summary = convert_folder(
            src,
            out,
            task=args.task,
            image_ms=args.image_ms,
            arm_ms=args.arm_ms,
            hand_ms=args.hand_ms,
            numeric_interp_mode=args.numeric_interp_mode,
            jump_abs=args.jump_abs,
            glitch_ms=args.glitch_ms,
            chunks_size=args.chunks_size,
            resume=resume,
            limit=args.limit,
        )
        all_summaries.append(summary)
        if args.generate_cmd_state_viz:
            viz_script = Path(__file__).resolve().parent / "09_viz_lerobot_v21_from_parquet.py"
            print(f"\nGenerating command + state visualization for {out}", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    str(viz_script),
                    "--dataset",
                    str(out),
                    "--out-dir",
                    str(out / "viz"),
                ],
                check=True,
            )
        brief = {k: summary[k] for k in summary if k not in ("failed", "high_drop_ge_10pct")}
        print(json.dumps(brief, indent=2), flush=True)

    root_summary = {
        "out_root": str(args.out_root),
        "task": args.task,
        "folders": all_summaries,
    }
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "batch_summary_all.json").write_text(
        json.dumps(root_summary, indent=2, ensure_ascii=False) + "\n"
    )
    print("\nALL DONE →", args.out_root / "batch_summary_all.json", flush=True)


if __name__ == "__main__":
    main()
