"""Read only tactile columns; audit weak labels without loading images or running training."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from openpi.models.tactile_contact import calibrate_baseline, spatial_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="/data/V1show_lerobot/merged_tactile")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--baseline-frames", type=int, default=16)
    parser.add_argument("--off", type=float, default=1.0)
    parser.add_argument("--on", type=float, default=3.0)
    args = parser.parse_args()
    if args.episodes < 1 or args.baseline_frames < 4 or not 0 <= args.off < args.on:
        parser.error("Need episodes >= 1, baseline-frames >= 4, 0 <= off < on")
    root = Path(args.dataset)
    episodes = [json.loads(line) for line in (root / "meta/episodes.jsonl").read_text().splitlines()]
    selected = np.linspace(0, len(episodes) - 1, min(args.episodes, len(episodes)), dtype=int)
    mask = spatial_mask()
    totals = {"frames": 0, "positive": 0, "negative": 0, "unknown": 0}
    for index in selected:
        ep = int(episodes[index]["episode_index"])
        paths = [root / "data/chunk-000" / f"episode_{ep:06d}.parquet",
                 root / "data/chunk-000" / f"file-{ep:03d}.parquet"]
        path = next(p for p in paths if p.is_file())
        table = pq.read_table(path, columns=["observation.tactile.right", "observation.tactile.right_valid"])
        raw = np.asarray(table.column(0).to_pylist(), np.float32).reshape(-1, 11, 14, 14)
        valid = np.asarray(table.column(1).to_pylist(), bool).reshape(-1, 11)
        baseline, scale, usable = calibrate_baseline(raw[:args.baseline_frames], valid[:args.baseline_frames])
        finite = np.all(np.isfinite(np.where(mask[None], raw, 0)), axis=(-2, -1))
        valid = valid & usable[None] & finite
        valid[:args.baseline_frames - 1] = False
        values = np.where(valid[..., None, None] & mask[None], raw, baseline[None])
        score = (np.abs((values - baseline) / scale[None, :, None, None]) * mask).sum(axis=(-2, -1)) / mask.sum(axis=(-2, -1))
        positive = np.any(valid & (score >= args.on), axis=1)
        negative = np.all(valid, axis=1) & np.all(score <= args.off, axis=1)
        row = {"episode": ep, "frames": len(raw), "positive": int(positive.sum()),
               "negative": int(negative.sum()), "unknown": int((~(positive | negative)).sum()),
               "calibrated_sensors": int(usable.sum()), "noise_scale_min": float(scale.min()),
               "noise_scale_max": float(scale.max())}
        print(json.dumps(row), flush=True)
        for key in totals:
            totals[key] += row[key]
    print(json.dumps({"sample_summary": totals, "label_type": "weak, noise-relative; not manual ground truth"}))


if __name__ == "__main__":
    main()
