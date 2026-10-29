"""Offline, resumable RoboDojo LeRobot-v3 -> OpenWAM Alpha preparation.

Raw source is read-only. Deterministic transforms are cached losslessly.
Training supervision follows official RoboDojo: achieved state[t+1], not the
LeRobot command column. RGB reversal is verified against pinned source HDF5.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
SLOTS = ((0, 256, 0, 320), (256, 384, 0, 160), (256, 384, 160, 320))
REVISION = "cefcfbbf2497103fe46b99039dbd381976fe4a42"
FORMAT = 1


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(path)


def prepare_numeric(root, out):
    from openwam.dataloader.robodojo import read_calibrated_eef20
    from openwam.dataloader.robodojo_contract import arx_x5_calibration
    from openwam.dataloader.transforms.multiview import format_prompt_for_inference
    from openwam.dataloader.transforms.normalize import Normalizer
    from openwam.dataloader.utils.normalization import pin_rot6d_identity, ROT6D_DIMS_EEF20, STAT_KEYS
    from openwam.dataloader.utils.stats_computation.robocoin_stats_computation import Accumulator
    from openwam.dataloader.utils.stats_computation.robodojo_stats_computation import _metadata

    info = json.loads((root / "meta/info.json").read_text())
    if info["fps"] != 25 or info["features"]["observation.state"]["shape"] != [16]:
        raise ValueError("Only verified 25-FPS RoboDojo EEF16 is supported")
    verification = json.loads((root / "download_verified.json").read_text())
    if verification["revision"] != REVISION:
        raise ValueError("Source revision differs from the verified release")
    all_files = sorted(p for p in root.rglob("*") if p.is_file())
    source = [{"path": str(p.relative_to(root)), "size": p.stat().st_size,
               "mtime_ns": p.stat().st_mtime_ns} for p in all_files]
    fingerprint = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
    plan_path = out / "plan.json"
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        if old["source_fingerprint"] != fingerprint or old["format"] != FORMAT:
            raise ValueError("Source/config changed: select a fresh output directory")
        return old

    episodes = pq.read_table(root / "meta/episodes/chunk-000/file-000.parquet").to_pylist()
    tasks = pq.read_table(root / "meta/tasks.parquet").to_pylist()
    task_text = {int(t["task_index"]): t["__index_level_0__"] for t in tasks}
    columns = ["observation.state", "action", "episode_index", "frame_index", "task_index", "index", "timestamp"]
    tables = [pq.read_table(p, columns=columns) for p in sorted((root / "data").rglob("*.parquet"))]
    import pyarrow as pa
    table = pa.concat_tables(tables)
    n = len(table)
    if n != info["total_frames"] or len(episodes) != info["total_episodes"]:
        raise ValueError("Incomplete source")
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    command = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    if not np.isfinite(state).all() or not np.isfinite(command).all():
        raise ValueError("Non-finite source")
    np.testing.assert_array_equal(table["index"].to_numpy(), np.arange(n))
    ids = table["episode_index"].to_numpy()
    frame_ids = table["frame_index"].to_numpy()
    task_ids = table["task_index"].to_numpy()
    times = table["timestamp"].to_numpy()
    calibration = arx_x5_calibration()
    mapping = {
        "state/left_ee_poses": state[:, :7],
        "state/left_ee_joint_states": state[:, 7:8],
        "state/right_ee_poses": state[:, 8:15],
        "state/right_ee_joint_states": state[:, 15:16],
    }
    converted = read_calibrated_eef20(mapping, calibration)
    np.save(out / "states_eef20.npy", converted)
    accumulator = Accumulator(dim=20, reservoir_cap=1_000_000, seed=0)
    compact = []
    windows = []
    camera_jobs = defaultdict(list)
    repairs = []
    cursor = 0
    lag_errors = []
    for expected_ep, e in enumerate(episodes):
        ep, start, end, length = [int(e[k]) for k in ("episode_index", "dataset_from_index", "dataset_to_index", "length")]
        if ep != expected_ep or start != cursor or end - start != length or length < 2:
            raise ValueError(f"Invalid contiguous episode index: {ep}")
        np.testing.assert_array_equal(ids[start:end], ep)
        np.testing.assert_array_equal(frame_ids[start:end], np.arange(length))
        np.testing.assert_allclose(times[start:end], np.arange(length) / 25, atol=1e-4, rtol=0)
        task = int(task_ids[start])
        np.testing.assert_array_equal(task_ids[start:end], task)
        if e["tasks"] != [task_text[task]]:
            raise ValueError(f"Episode/task instruction mismatch {ep}")
        prompt = format_prompt_for_inference(task_text[task])
        compact.append(dict(episode_index=ep, start=start, end=end, length=length, task_index=task, prompt=prompt))
        windows.append(np.column_stack((np.full(length-1, ep), np.arange(length-1),
                                        np.minimum(33, length-np.arange(length-1)))).astype(np.int32))
        accumulator.update_batch(converted[start:end])
        accumulator.update_batch(converted[start+1:end])
        pos = [0, 1, 2, 8, 9, 10]
        lag_errors.extend(np.linalg.norm(command[start:end-1, pos] - state[start+1:end, pos], axis=1)[::50].tolist())
        for camera in CAMERAS:
            key = f"videos/observation.images.{camera}"
            a, b = [round(float(e[key + "/" + side + "_timestamp"]) * 25) for side in ("from", "to")]
            path = info["video_path"].format(video_key=f"observation.images.{camera}",
                                            chunk_index=e[key+"/chunk_index"], file_index=e[key+"/file_index"])
            record = dict(episode_index=ep, start=start, length=length, video_start=a, video_end=b)
            if b - a != length:
                if b - a != length - 1:
                    raise ValueError(f"Unexpected video deficit ep={ep}, {camera}")
                repairs.append(dict(**record, camera=camera, task_index=task, path=path))
                record["repair"] = True
            camera_jobs[(camera, path)].append(record)
        cursor = end
    if cursor != n:
        raise ValueError("Trailing unindexed rows")
    stats = {k: np.asarray(v, np.float32) for k, v in accumulator.finalize().items() if k in STAT_KEYS}
    pin_rot6d_identity(stats, ROT6D_DIMS_EEF20)
    metadata = _metadata(tasks=[str(k) for k in sorted(task_text)], calibration=calibration, variant="sim",
                         embodiment="arx_x5", state_rows=n, action_rows=n-len(episodes),
                         reservoir_cap=1_000_000, reservoir_rows=1_000_000)
    metadata["source_revision"] = REVISION
    np.save(out / "normalization_stats.npy", {"eef": stats, "metadata": metadata,
                                             "num_timesteps": metadata["num_timesteps"]})
    normal = Normalizer("min_max", stats).normalize(converted)
    unified = np.lib.format.open_memmap(out / "states_unified.npy", mode="w+", dtype=np.float32, shape=(n, 80))
    unified[:] = 0
    unified[:, list(range(10)) + list(range(34, 44))] = normal
    unified.flush()
    del unified
    np.save(out / "windows.npy", np.concatenate(windows))
    atomic_json(out / "episodes.json", compact)
    atomic_json(out / "source_inventory.json", source)
    jobs = [dict(camera=c, path=p, episodes=rs) for (c, p), rs in sorted(camera_jobs.items())]
    plan = dict(format=FORMAT, dataset_dir=str(root), source_revision=REVISION,
                source_fingerprint=fingerprint, info_sha256=sha(root / "meta/info.json"),
                frames=n, episodes=len(episodes), tasks=len(tasks), windows=n-len(episodes),
                canvas=[384, 320, 3], dtype="uint8", rgb_transform="reverse_rgb_verified_against_official_hdf5",
                action_target="achieved_state[t+1]", fps=25, num_frames=33, video_stride=4,
                window_stride=1, calibration=calibration, jobs=jobs, repairs=repairs,
                action_vs_next_state_position_l2_m_quantiles=np.quantile(lag_errors, [.5, .99, 1]).tolist())
    atomic_json(plan_path, plan)
    return plan


def video_job(args):
    root_s, out_s, job, total = args
    root, out = Path(root_s), Path(out_s)
    key = hashlib.sha256(job["path"].encode()).hexdigest()[:20]
    receipt = out / "video_receipts" / (key + ".json")
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        cache = out / previous["cache"]
        if cache.exists() and cache.stat().st_size == previous["bytes"]:
            return previous
    begin = time.monotonic()
    camera = job["camera"]
    y0, y1, x0, x1 = SLOTS[CAMERAS.index(camera)]
    cache_path = out / "camera_cache" / (key + ".bin")
    temporary = cache_path.with_suffix(".tmp")
    episodes = job["episodes"]
    expected = max(e["video_end"] for e in episodes)
    for a, b in zip(episodes, episodes[1:]):
        if a["video_end"] != b["video_start"]:
            raise ValueError("Packed-video episode gaps/overlaps")
    if episodes[0]["video_start"] != 0:
        raise ValueError("Packed video does not start at zero")
    path = root / job["path"]
    digest = sha(path)
    with av.open(str(path)) as container, open(temporary, "wb", buffering=8 << 20) as output:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = 2
        segment = 0
        count = 0
        for number, frame in enumerate(container.decode(stream)):
            if number >= expected:
                raise ValueError(f"Extra source video frame: {path} at {number}")
            if abs(float(frame.pts * frame.time_base) - number / 25) > 1e-5:
                raise ValueError(f"Nonuniform PTS {path} at {number}")
            while number >= episodes[segment]["video_end"]:
                segment += 1
            e = episodes[segment]
            # Official JPEG decoder returns RGB opposite to this AV1's
            # rgb24 interpretation; independently verified on raw HDF5.
            rgb = frame.to_ndarray(format="rgb24")[:, :, ::-1]
            image = Image.fromarray(rgb).resize((x1-x0, y1-y0), Image.Resampling.BILINEAR)
            output.write(np.asarray(image).tobytes())
            count += 1
    if count != expected:
        raise ValueError(f"Short video {path}: {count} != {expected}")
    temporary.replace(cache_path)
    result = dict(path=job["path"], sha256=digest, frames=count, cache=str(cache_path.relative_to(out)),
                  bytes=cache_path.stat().st_size, seconds=time.monotonic()-begin)
    atomic_json(receipt, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--numeric-only", action="store_true")
    args = parser.parse_args()
    root, out = Path(args.source).resolve(), Path(args.output).resolve()
    if out == root or root in out.parents:
        raise ValueError("Use a separate output directory")
    out.mkdir(parents=True, exist_ok=True)
    # Never allow two preparers to race on shared frames or plan files.
    import fcntl
    lock = open(out / ".prepare.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = prepare_numeric(root, out)
    print(json.dumps({k: plan[k] for k in ("frames", "episodes", "tasks", "windows")}), flush=True)
    print("REQUIRED_RAW_REPAIRS", len(plan["repairs"]), flush=True)
    if args.numeric_only:
        return
    (out / "camera_cache").mkdir(exist_ok=True)
    (out / "video_receipts").mkdir(exist_ok=True)
    started = time.monotonic()
    jobs = [(str(root), str(out), job, plan["frames"]) for job in plan["jobs"]]
    with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
        for i, result in enumerate(pool.map(video_job, jobs), 1):
            progress = dict(completed=i, total=len(jobs), seconds=time.monotonic()-started, last=result["path"])
            atomic_json(out / "video_progress.json", progress)
            print(json.dumps(progress), flush=True)
    atomic_json(out / "videos_complete.json", dict(status="decoded", total=len(jobs), seconds=time.monotonic()-started))
    print("VIDEOS_COMPLETE; raw repairs and validation required before ready manifest", flush=True)


if __name__ == "__main__":
    main()
