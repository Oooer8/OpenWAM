"""Prepare split, exact train-only min/max and immutable source-row offsets."""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from openwam.dataloader.cosmos3_droid import ACTION_POSE, STATE_POSE, HEAD, HEAD2, WRIST, prompt_variants
from openwam.dataloader.utils.lerobotv3 import compute_file_local_offsets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    root, out = Path(args.dataset_dir).resolve(), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    info_bytes = (root / "meta/info.json").read_bytes()
    info = json.loads(info_bytes)
    paths = sorted((root / "meta/episodes").glob("*/*.parquet"))
    columns = [name for name in pq.read_schema(paths[0]).names if not name.startswith("stats/")]
    for source in (ACTION_POSE, STATE_POSE):
        columns += [f"stats/{source}/{kind}" for kind in ("min", "max", "mean", "std", "count")]
    eps = pd.concat([pq.read_table(p, columns=columns).to_pandas() for p in paths], ignore_index=True)
    eps = eps.sort_values("episode_index").reset_index(drop=True)
    if eps["episode_index"].duplicated().any() or len(eps) != info["total_episodes"]:
        raise ValueError("Episode manifest does not match info.json")
    if int(eps["length"].sum()) != info["total_frames"]:
        raise ValueError("Frame count does not match info.json")
    # Compute from ALL source episodes before filtering to keep packed-file
    # offsets correct across held-out/invalid episodes.
    eps["_data_row_offset"] = compute_file_local_offsets(eps, "data/chunk_index", "data/file_index")
    for camera in (HEAD, HEAD2, WRIST):
        offset = np.rint(eps[f"videos/{camera}/from_timestamp"].to_numpy() * info["fps"]).astype(np.int64)
        expected = compute_file_local_offsets(eps, f"videos/{camera}/chunk_index", f"videos/{camera}/file_index")
        if not np.array_equal(offset, expected):
            raise ValueError(f"Video timestamps and physical row offsets disagree for {camera}")
        eps[f"_video_frame_offset/{camera}"] = offset

    # Deterministic session holdout: all episodes from one lab/date collection
    # session stay together. This is not a claim of scene-disjoint validation.
    sessions = eps["episode_id"].astype(str).str.rsplit("/", n=1).str[0]
    heldout = sessions.map(lambda s: int(hashlib.sha256(("42:" + s).encode()).hexdigest()[:8], 16) % 100 < 2)
    eps["training_split"] = np.where(heldout, "val", "train")
    tasks = pq.read_table(root / "meta/tasks.parquet").to_pylist()
    valid_prompts = {row["task"] for row in tasks if prompt_variants(row["task"])}
    valid = eps["tasks"].map(lambda x: any(str(t) in valid_prompts for t in x)) & (eps["length"] >= 33)
    excluded = eps.loc[~valid, "episode_index"].astype(int).tolist()
    eps = eps.loc[valid].reset_index(drop=True)
    train = eps[eps["training_split"] == "train"]
    val = eps[eps["training_split"] == "val"]
    if train.empty or val.empty:
        raise ValueError("Both training and holdout must contain episodes")
    normalization = {}
    for mode, source in (("eef", ACTION_POSE), ("eef_state", STATE_POSE)):
        values = {}
        for kind in ("min", "max", "mean", "std"):
            values[kind] = np.stack(train[f"stats/{source}/{kind}"]).astype(np.float64)[:, :3]
        counts = np.stack(train[f"stats/{source}/count"]).astype(np.float64).reshape(len(train), -1)[:, :1]
        mean = (values["mean"] * counts).sum(0) / counts.sum()
        variance = ((values["std"] ** 2 + values["mean"] ** 2) * counts).sum(0) / counts.sum() - mean ** 2
        # rot6d passes through; gripper is openness [0,1], mapped to [-1,1].
        minimum = np.r_[values["min"].min(0), np.full(6, -1.), 0.]
        maximum = np.r_[values["max"].max(0), np.full(6, 1.), 1.]
        s = {"min": minimum, "max": maximum, "mean": np.r_[mean, np.zeros(6), .5],
             "std": np.r_[np.sqrt(np.maximum(variance, 0)), np.ones(6), .5],
             "q01": minimum.copy(), "q99": maximum.copy()}
        normalization[mode] = {k: v.astype(np.float32).tolist() for k, v in s.items()}
    (out / "normalization.json").write_text(json.dumps(normalization, indent=2))
    np.save(out / "normalization_stats.npy", {k: {s: np.array(v, np.float32) for s, v in d.items()} for k, d in normalization.items()})

    # Verify each referenced physical Parquet range and all three video paths.
    jobs = []
    for (chunk, file), group in eps.groupby(["data/chunk_index", "data/file_index"]):
        p = root / info["data_path"].format(chunk_index=int(chunk), file_index=int(file))
        needed = int((group["_data_row_offset"] + group["length"]).max())
        jobs.append((p, needed))
    def check_parquet(job):
        p, needed = job
        if pq.ParquetFile(p).metadata.num_rows < needed:
            raise ValueError(f"Truncated Parquet shard: {p}")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(check_parquet, jobs))
    video_paths = set()
    for camera in (HEAD, HEAD2, WRIST):
        for c, f in eps[[f"videos/{camera}/chunk_index", f"videos/{camera}/file_index"]].drop_duplicates().itertuples(index=False, name=None):
            p = root / info["video_path"].format(video_key=camera, chunk_index=int(c), file_index=int(f))
            if not p.is_file() or p.stat().st_size == 0:
                raise FileNotFoundError(p)
            video_paths.add(str(p))
    eps.drop(columns=[c for c in eps if c.startswith("stats/")]).to_parquet(out / "episodes.parquet", index=False)
    manifest = {
        "dataset_dir": str(root), "info_sha256": hashlib.sha256(info_bytes).hexdigest(),
        "selection": "success only; session hash holdout at 2%; source metadata unchanged",
        "seed": 42, "excluded_episode_indices": excluded,
        "train_episodes": len(train), "val_episodes": len(val),
        "train_frames": int(train["length"].sum()), "val_frames": int(val["length"].sum()),
        "train_full_windows_stride1": int((train["length"] - 32).sum()),
        "val_full_windows_stride1": int((val["length"] - 32).sum()),
        "parquet_shards_checked": len(jobs), "video_paths_checked": len(video_paths),
        "action_source": ACTION_POSE, "state_source": STATE_POSE,
        "pose_semantics": "published Cartesian XYZ plus extrinsic XYZ Euler radians in robot-base frame; no additional TCP transform",
        "gripper": "raw closedness 0=open/1=closed; invert to openness, normalize to -1=closed/+1=open",
        "unified_slots": "0:3 xyz; 3:9 rotation matrix columns 0/1; 9 gripper; other 70 slots masked",
        "normalization": "exact train-only episode min/max for xyz; identity rot6d; fixed physical gripper range [0,1]",
        "normalization_auxiliary_stats": "rotation/gripper mean,std are neutral placeholders; q01,q99 alias min,max; only min-max is supported",
        "fps": info["fps"], "num_frames": 33, "video_stride": 4,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
