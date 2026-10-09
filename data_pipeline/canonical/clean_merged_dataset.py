#!/usr/bin/env python3
"""Remove bad episodes from merged dataset and re-index for training."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

MERGED = Path("/data/V1show_lerobot/merged")
CHUNK_SIZE = 1000

# User-reported bad MCAPs (mapped via episode_sources.jsonl).
# Note: user wrote ...06-01-39.794627797.mcap; on disk it is ...794624797.mcap → ep 132.
REMOVE = {
    69,  # 2026-08-01_02-56-27.109467967.mcap
    118,  # 2026-08-01_05-49-30.022962650.mcap
    124,  # 2026-08-01_05-55-30.041894944.mcap
    130,  # 2026-08-01_06-00-02.561219368.mcap
    132,  # 2026-08-01_06-01-39.794624797.mcap
    133,  # 2026-08-01_06-03-23.211132375.mcap
    136,  # 2026-08-01_06-05-41.813534966.mcap
    138,  # 2026-08-01_06-08-38.717182214.mcap
    140,  # 2026-08-01_06-10-04.022743090.mcap
    148,  # 2026-08-01_06-16-25.802361486.mcap
    155,  # 2026-08-01_06-21-22.717915404.mcap
    158,  # 2026-08-01_06-23-54.014387692.mcap
    169,  # 2026-08-01_06-31-49.443611029.mcap
    193,  # 2026-08-01_06-57-49.628563018.mcap
    196,  # 2026-08-01_07-03-00.162501770.mcap
    201,  # 2026-08-01_07-10-01.759324427.mcap
    205,  # 2026-08-01_07-13-53.798649040.mcap
    209,  # 2026-08-01_07-16-53.590137462.mcap
    218,  # 2026-08-01_07-37-25.568646832.mcap
    224,  # 2026-08-01_07-47-04.243084665.mcap
    278,  # 2026-08-02_05-46-05.150100213.mcap
}

BAD_MCAP_NAMES = {
    "2026-08-01_02-56-27.109467967.mcap",
    "2026-08-01_05-49-30.022962650.mcap",
    "2026-08-01_05-55-30.041894944.mcap",
    "2026-08-01_06-00-02.561219368.mcap",
    "2026-08-01_06-01-39.794624797.mcap",
    "2026-08-01_06-01-39.794627797.mcap",  # user typo variant
    "2026-08-01_06-03-23.211132375.mcap",
    "2026-08-01_06-05-41.813534966.mcap",
    "2026-08-01_06-08-38.717182214.mcap",
    "2026-08-01_06-10-04.022743090.mcap",
    "2026-08-01_06-16-25.802361486.mcap",
    "2026-08-01_06-21-22.717915404.mcap",
    "2026-08-01_06-23-54.014387692.mcap",
    "2026-08-01_06-31-49.443611029.mcap",
    "2026-08-01_06-57-49.628563018.mcap",
    "2026-08-01_07-03-00.162501770.mcap",
    "2026-08-01_07-10-01.759324427.mcap",
    "2026-08-01_07-13-53.798649040.mcap",
    "2026-08-01_07-16-53.590137462.mcap",
    "2026-08-01_07-37-25.568646832.mcap",
    "2026-08-01_07-47-04.243084665.mcap",
    "2026-08-02_05-46-05.150100213.mcap",
}


def _parquet_path(root: Path, ep: int) -> Path:
    return root / "data" / f"chunk-{ep // CHUNK_SIZE:03d}" / f"episode_{ep:06d}.parquet"


def main() -> None:
    meta_dir = MERGED / "meta"
    ep_path = meta_dir / "episodes.jsonl"
    src_path = meta_dir / "episode_sources.jsonl"
    episodes = [json.loads(l) for l in ep_path.read_text().splitlines() if l.strip()]
    sources = [json.loads(l) for l in src_path.read_text().splitlines() if l.strip()]
    assert len(episodes) == len(sources)

    # Resolve REMOVE from mcap names if present (safety).
    remove = set(REMOVE)
    for s in sources:
        name = Path(s["mcap"]).name if s.get("mcap") else ""
        if name in BAD_MCAP_NAMES:
            remove.add(int(s["episode_index"]))

    missing_remove = sorted(ep for ep in remove if ep >= len(episodes))
    if missing_remove:
        raise RuntimeError(f"REMOVE indices out of range: {missing_remove}")

    print(f"Before: {len(episodes)} episodes")
    print(f"Removing {len(remove)}: {sorted(remove)}")
    for ep in sorted(remove):
        print(f"  - ep={ep:06d}  {Path(sources[ep].get('mcap','')).name}")

    kept_old = [i for i in range(len(episodes)) if i not in remove]
    print(f"After:  {len(kept_old)} episodes")

    staging = MERGED / "data" / "_reindex_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    global_idx = 0
    total_frames = 0
    new_episodes: list[dict] = []
    new_sources: list[dict] = []

    for new_idx, old_idx in enumerate(kept_old):
        old_pq = _parquet_path(MERGED, old_idx)
        if not old_pq.is_file():
            matches = list((MERGED / "data").glob(f"chunk-*/episode_{old_idx:06d}.parquet"))
            if not matches:
                raise FileNotFoundError(old_pq)
            old_pq = matches[0]

        new_chunk = new_idx // CHUNK_SIZE
        new_chunk_dir = staging / f"chunk-{new_chunk:03d}"
        new_chunk_dir.mkdir(parents=True, exist_ok=True)
        new_pq = new_chunk_dir / f"episode_{new_idx:06d}.parquet"

        df = pd.read_parquet(old_pq)
        n = len(df)
        df["episode_index"] = np.full(n, new_idx, dtype=np.int64)
        df["index"] = np.arange(global_idx, global_idx + n, dtype=np.int64)
        df.to_parquet(new_pq, index=False)

        tasks = episodes[old_idx]["tasks"]
        src = dict(sources[old_idx])
        src.update(
            {
                "episode_index": new_idx,
                "parquet": str(MERGED / "data" / f"chunk-{new_chunk:03d}" / f"episode_{new_idx:06d}.parquet"),
                "kept_frames": n,
            }
        )
        viz_cmd = MERGED / "viz" / f"episode_{new_idx:06d}_all_cmd.png"
        src["viz"] = str(viz_cmd)

        new_episodes.append({"episode_index": new_idx, "tasks": tasks, "length": n})
        new_sources.append(src)

        global_idx += n
        total_frames += n
        if (new_idx + 1) % 50 == 0 or new_idx + 1 == len(kept_old):
            print(f"  parquet {new_idx + 1}/{len(kept_old)}", flush=True)

    # Replace data/chunk-* with staging
    for d in sorted((MERGED / "data").glob("chunk-*")):
        shutil.rmtree(d)
    for d in sorted(staging.glob("chunk-*")):
        shutil.move(str(d), str(MERGED / "data" / d.name))
    shutil.rmtree(staging)

    # Reindex viz: stage then replace
    viz_dir = MERGED / "viz"
    if viz_dir.is_dir():
        viz_stage = MERGED / "_viz_staging"
        if viz_stage.exists():
            shutil.rmtree(viz_stage)
        viz_stage.mkdir()
        for new_idx, old_idx in enumerate(kept_old):
            for suffix in ("all_cmd.png", "all_state.png"):
                src = viz_dir / f"episode_{old_idx:06d}_{suffix}"
                dst = viz_stage / f"episode_{new_idx:06d}_{suffix}"
                if src.is_file():
                    shutil.copy2(src, dst)
        # keep summary if present; will rewrite below
        for p in viz_dir.glob("*"):
            p.unlink()
        for p in viz_stage.glob("*"):
            shutil.move(str(p), str(viz_dir / p.name))
        shutil.rmtree(viz_stage)
        print(f"  viz reindexed: {len(list(viz_dir.glob('*_all_cmd.png')))} cmd pngs")

    # Update meta
    info = json.loads((meta_dir / "info.json").read_text())
    info["total_episodes"] = len(new_episodes)
    info["total_frames"] = total_frames
    info["total_chunks"] = (len(new_episodes) - 1) // CHUNK_SIZE + 1
    info["splits"] = {"train": f"0:{len(new_episodes)}"}
    (meta_dir / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    (meta_dir / "episodes.jsonl").write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in new_episodes) + "\n"
    )
    (meta_dir / "episode_sources.jsonl").write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in new_sources) + "\n"
    )

    removed_log = {
        "removed_episode_indices_before": sorted(remove),
        "removed_mcap": [
            Path(sources[ep]["mcap"]).name for ep in sorted(remove) if sources[ep].get("mcap")
        ],
        "episodes_before": len(episodes),
        "episodes_after": len(new_episodes),
        "frames_after": total_frames,
    }
    (meta_dir / "last_clean_report.json").write_text(
        json.dumps(removed_log, indent=2, ensure_ascii=False) + "\n"
    )

    # Verify
    verify_info = json.loads((meta_dir / "info.json").read_text())
    verify_parquets = sorted((MERGED / "data").glob("chunk-*/episode_*.parquet"))
    verify_eps = [json.loads(l) for l in (meta_dir / "episodes.jsonl").read_text().splitlines() if l.strip()]
    verify_src = [json.loads(l) for l in (meta_dir / "episode_sources.jsonl").read_text().splitlines() if l.strip()]
    remaining_bad = [
        Path(s["mcap"]).name
        for s in verify_src
        if s.get("mcap") and Path(s["mcap"]).name in BAD_MCAP_NAMES
    ]

    print(f"\nDone. {len(new_episodes)} episodes, {total_frames} frames")
    print(f"  meta episodes: {verify_info['total_episodes']}")
    print(f"  parquet files: {len(verify_parquets)}")
    print(f"  remaining bad mcaps: {remaining_bad}")

    assert verify_info["total_episodes"] == len(verify_parquets) == len(verify_eps) == len(verify_src)
    assert [e["episode_index"] for e in verify_eps] == list(range(len(verify_eps)))
    assert not remaining_bad

    # Spot-check index continuity
    expected = 0
    for i, pq_path in enumerate(verify_parquets):
        df = pd.read_parquet(pq_path, columns=["episode_index", "index"])
        assert int(df["episode_index"].iloc[0]) == i
        assert int(df["index"].iloc[0]) == expected
        expected += len(df)
    assert expected == total_frames
    print("  VERIFY OK (contiguous episode_index + global index)")


if __name__ == "__main__":
    main()
