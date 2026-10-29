"""Fail-closed corpus checks, source parity, reader parity and IO benchmark."""
import argparse
import concurrent.futures
import multiprocessing
import hashlib
import json
import random
import statistics
import time
import io
from pathlib import Path

import av
import numpy as np
import torch
import yaml
import cv2
from PIL import Image

from prepare_robodojo_openwam import CAMERAS, SLOTS, atomic_json, sha
from openwam.dataloader.robodojo_prepared import PreparedRoboDojoDataset
from openwam.dataloader.transforms.normalize import Normalizer


def jpeg_roundtrip(frame):
    buffer = io.BytesIO()
    Image.fromarray(frame).save(buffer, format="JPEG", quality=95, subsampling=0)
    bgr = cv2.imdecode(np.frombuffer(buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class JPEGShard:
    def __init__(self, root, shard):
        self.blob = np.memmap(root / shard["path"], mode="r", dtype=np.uint8)
        self.offsets = np.load(root / shard["offsets"], mmap_mode="r")
        assert self.offsets.shape == (shard["end"]-shard["start"]+1,)
        assert self.offsets[0] == 0 and self.offsets[-1] == len(self.blob)
        assert np.all(np.diff(self.offsets) > 0)

    def __getitem__(self, index):
        a, b = self.offsets[index:index+2]
        bgr = cv2.imdecode(self.blob[a:b], cv2.IMREAD_COLOR)
        assert bgr is not None and bgr.shape == (384, 320, 3)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def decode_indices(path, indices):
    """Reference hot path: random-access AV1 decode then official PIL resize."""
    wanted = set(map(int, indices))
    images = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.codec_context.thread_count = 2
        container.seek(int(min(indices) / 25 / stream.time_base), stream=stream, backward=True)
        for f in container.decode(stream):
            index = round(float(f.pts * f.time_base) * 25)
            if index in wanted:
                images[index] = Image.fromarray(f.to_ndarray(format="rgb24")[:, :, ::-1])
            if index >= max(indices):
                break
    if set(images) != wanted:
        raise ValueError(f"Reference decoder missing frames: {path}")
    return [images[int(i)] for i in indices]


_IMAGE_CHECK_CONTEXT = None


def check_episode_images(ep):
    out, shard_map, images, records, cache_maps = _IMAGE_CHECK_CONTEXT
    s = shard_map[ep["task_index"]]
    final = images[ep["task_index"]]
    repaired = any(records[(ep["episode_index"], c)][1].get("repair") for c in CAMERAS)
    indices = np.arange(ep["length"]) if repaired else np.array([0, ep["length"]//2, ep["length"]-1])
    reference_canvas = np.empty((len(indices), 384, 320, 3), np.uint8)
    for camera, (y0, y1, x0, x1) in zip(CAMERAS, SLOTS):
        path, record = records[(ep["episode_index"], camera)]
        if record.get("repair"):
            reference = np.load(out / "repairs" / f"episode_{ep['episode_index']:04d}_{camera}.npy")
            reference_canvas[:, y0:y1, x0:x1] = reference[indices]
        else:
            reference = cache_maps[path][record["video_start"]+indices]
            reference_canvas[:, y0:y1, x0:x1] = reference
    for index, reference in zip(indices, reference_canvas):
        actual = final[ep["start"]-s["start"]+index]
        np.testing.assert_array_equal(actual, jpeg_roundtrip(reference))
    return ep["episode_index"]


def main():
    cv2.setNumThreads(1)
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", required=True)
    p.add_argument("--image-workers", type=int, default=32)
    args = p.parse_args()
    out = Path(args.prepared)
    plan = json.loads((out / "plan.json").read_text())
    assembly = json.loads((out / "assembly_complete.json").read_text())
    repair = json.loads((out / "repairs_complete.json").read_text())
    references = json.loads((out / "raw_reference_checks.json").read_text())
    text = json.loads((out / "text_cache/validation.json").read_text())
    assert references["status"] == text["status"] == "passed"
    assert repair["cameras"] == len(plan["repairs"])
    source = Path(plan["dataset_dir"])
    expected = json.loads((source / "download_verified.json").read_text())["files"]
    receipts = [json.loads(p.read_text()) for p in (out / "video_receipts").glob("*.json")]
    digests = {r["path"]: r["sha256"] for r in receipts}
    for entry in expected:
        name = entry["path"]
        if name not in digests:
            digests[name] = sha(source / name)
        if digests[name] != entry["sha256"]:
            raise ValueError(f"Source SHA-256 mismatch: {name}")
    assert len(receipts) == len(plan["jobs"]) == 319
    source_inventory = json.loads((out / "source_inventory.json").read_text())
    for item in source_inventory:
        stat = (source / item["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (item["size"], item["mtime_ns"]):
            raise ValueError("Source changed during preparation")
    print("PASS full source hashes", len(expected), flush=True)
    episodes = json.loads((out / "episodes.json").read_text())
    raw = np.load(out / "states_eef20.npy", mmap_mode="r")
    state = np.load(out / "states_unified.npy", mmap_mode="r")
    windows = np.load(out / "windows.npy", mmap_mode="r")
    stats = np.load(out / "normalization_stats.npy", allow_pickle=True).item()["eef"]
    norm = Normalizer("min_max", stats)
    slots = [*range(10), *range(34, 44)]
    empty_slots = sorted(set(range(80))-set(slots))
    for start in range(0, len(raw), 65536):
        x = np.asarray(raw[start:start+65536])
        assert np.isfinite(x).all()
        y = state[start:start+65536]
        np.testing.assert_array_equal(y[:, slots], norm.normalize(x))
        assert not y[:, empty_slots].any()
    cursor = 0
    for ep in episodes:
        count = ep["length"]-1
        w = windows[cursor:cursor+count]
        np.testing.assert_array_equal(w[:, 0], ep["episode_index"])
        np.testing.assert_array_equal(w[:, 1], np.arange(count))
        np.testing.assert_array_equal(w[:, 2], np.minimum(33, ep["length"]-np.arange(count)))
        cursor += count
    assert cursor == len(windows) == plan["windows"]
    print("PASS all states and all window boundaries", flush=True)

    shard_map = {s["task_index"]: s for s in assembly["shards"]}
    images = {task: JPEGShard(out, s) for task, s in shard_map.items()}
    for task, array in images.items():
        s = shard_map[task]
        assert (out / s["path"]).stat().st_size == s["bytes"]
        assert s["encoded_frames"] == s["end"]-s["start"]
    # Bind the receipts to bytes actually persisted, before discarding intermediates.
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        digests_written = pool.map(sha, [out / s["path"] for s in assembly["shards"]])
        for shard, digest in zip(assembly["shards"], digests_written):
            if digest != shard["sha256"]:
                raise ValueError(f"Written JPEG shard checksum mismatch: {shard['path']}")
    print("PASS all written JPEG shard hashes", len(assembly["shards"]), flush=True)

    # Check assembly and episode offset on both boundaries of EVERY episode,
    # all three cameras. Repairs are verified against their entire raw segment.
    records = {}
    cache_maps = {}
    for job in plan["jobs"]:
        key = hashlib.sha256(job["path"].encode()).hexdigest()[:20]
        record = json.loads((out / "video_receipts" / (key+".json")).read_text())
        camera = job["camera"]
        y0, y1, x0, x1 = SLOTS[CAMERAS.index(camera)]
        cache_maps[job["path"]] = np.memmap(out / record["cache"], dtype=np.uint8, mode="r",
                                        shape=(record["frames"], y1-y0, x1-x0, 3))
        for ep in job["episodes"]:
            records[(ep["episode_index"], camera)] = (job["path"], ep)
    # Linux fork shares read-only mmaps; every child owns its decoder buffers.
    # Use processes so Python validation also runs concurrently with JPEG work.
    global _IMAGE_CHECK_CONTEXT
    _IMAGE_CHECK_CONTEXT = out, shard_map, images, records, cache_maps
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=args.image_workers, mp_context=multiprocessing.get_context("fork")
    ) as pool:
        for completed, _ in enumerate(pool.map(check_episode_images, episodes, chunksize=10), 1):
            if completed % 500 == 0:
                print(f"Image verification {completed}/{len(episodes)}", flush=True)
    print("PASS image assembly across every episode and all repaired frames", flush=True)
    # AV1 is lossy; compare to original JPEG, with a conservative bounded MAE.
    # Corrected RGB must be closer than reversing it again unless near grayscale.
    color_checks = []
    for check in references["results"]:
        ep = episodes[check["episode"]]
        reference = np.load(out / "raw_references" / f"episode_{ep['episode_index']:04d}.npz")
        final = images[ep["task_index"]]
        base = ep["start"] - shard_map[ep["task_index"]]["start"]
        for camera, (y0, y1, x0, x1) in zip(CAMERAS, SLOTS):
            for index in check["video_sample_frames"]:
                a = final[base+index][y0:y1, x0:x1].astype(float)
                b = reference[f"{camera}_{index}"].astype(float)
                error = float(np.abs(a-b).mean())
                wrong = float(np.abs(a[..., ::-1]-b).mean())
                if error > 6 or error > wrong + .25:
                    raise ValueError(f"Source image/color mismatch task={ep['task_index']} cam={camera} frame={index}: {error}, {wrong}")
                color_checks.append(dict(task=ep["task_index"], camera=camera, frame=index, mae=error, reversed_mae=wrong))
    print("PASS original-source color and alignment", len(color_checks), flush=True)

    artifact_names = ["episodes.json", "windows.npy", "states_unified.npy", "states_eef20.npy",
                      "normalization_stats.npy", "plan.json"] + [name for s in assembly["shards"]
                                                                 for name in (s["path"], s["offsets"])]
    artifacts = {}
    for name in artifact_names:
        path = out / name
        artifacts[name] = dict(size=path.stat().st_size)
        if not name.endswith(".jpgs"):
            artifacts[name]["sha256"] = sha(path)
    manifest = {k: plan[k] for k in ("format", "dataset_dir", "source_revision", "frames", "episodes",
                                     "tasks", "windows", "fps", "num_frames", "video_stride", "window_stride",
                                     "rgb_transform", "action_target", "calibration")}
    manifest.update(status="ready", image_codec="jpeg", jpeg_quality=95, jpeg_subsampling=0,
                    artifacts=artifacts, frame_shards=assembly["shards"])
    # Keep the production manifest absent until the reader itself passes.
    candidate = out / "validation_candidate"
    candidate.mkdir(exist_ok=True)
    for name in ("frames", "episodes.json", "windows.npy", "states_unified.npy", "states_eef20.npy",
                 "normalization_stats.npy", "plan.json"):
        link = candidate / name
        if not link.exists():
            link.symlink_to(out / name)
    atomic_json(candidate / "manifest.json", manifest)
    cfg = yaml.safe_load(Path("configs/dataloader/robodojo_prepared.yaml").read_text())
    cfg["prepared_dir"] = str(candidate)
    ds = PreparedRoboDojoDataset(cfg, split="val")
    selected = np.random.default_rng(42).integers(0, len(ds), 500)
    for i in selected:
        item = ds[int(i)]
        ep, start, length = map(int, windows[i])
        source_index = episodes[ep]["start"] + start + np.minimum(np.arange(33), length-1)
        np.testing.assert_array_equal(item["action"], state[source_index[1:]])
        np.testing.assert_array_equal(item["proprio"], state[source_index[:1]])
        assert item["action_mask"].sum() == (length-1)*20
        assert item["proprio_mask"].sum() == 20
        assert item["video_mask"].sum() == (length-1)//4+1
        assert len(item["video"]) == 9 and item["video"][0].size == (320, 384)
    # Same clip, warm source files: direct mmap vs AV1 seek/decode/resize/layout.
    cached_times, hot_times = [], []
    for ep in episodes[::100]:
        if any(records[(ep["episode_index"], c)][1].get("repair") for c in CAMERAS):
            continue
        start = min(50, ep["length"]-33)
        window = ep["start"] - ep["episode_index"] + start
        t = time.perf_counter()
        sample = ds[window]
        cached_times.append(time.perf_counter()-t)
        t = time.perf_counter()
        canvas = np.empty((9, 384, 320, 3), np.uint8)
        for camera, (y0, y1, x0, x1) in zip(CAMERAS, SLOTS):
            path, record = records[(ep["episode_index"], camera)]
            indices = record["video_start"] + start + np.arange(0, 33, 4)
            frames = decode_indices(source/path, indices)
            canvas[:, y0:y1, x0:x1] = np.stack([np.asarray(f.resize((x1-x0, y1-y0), Image.Resampling.BILINEAR))
                                              for f in frames])
        hot_times.append(time.perf_counter()-t)
        np.testing.assert_array_equal(np.stack([jpeg_roundtrip(f) for f in canvas]), np.stack(sample["video"]))
    training_ds = PreparedRoboDojoDataset(cfg, split="train")
    # Test real stochastic augmentation and 8-worker IPC at official batch 4.
    generator = torch.Generator().manual_seed(42)
    subset = torch.utils.data.Subset(training_ds, torch.randperm(len(ds), generator=generator)[:512].tolist())
    loader = torch.utils.data.DataLoader(subset, batch_size=4, num_workers=8, collate_fn=list)
    times = []
    last = time.perf_counter()
    for batch in loader:
        now = time.perf_counter()
        times.append(now-last)
        assert len(batch) == 4
        last = time.perf_counter()
    steady = times[8:]
    result = dict(status="passed", source_files_sha256_checked=len(expected), decoded_video_files=len(receipts),
                  frames=plan["frames"], episodes=plan["episodes"], tasks=plan["tasks"], windows=plan["windows"],
                  raw_source_tasks=references["tasks"], raw_state_max_abs_error=max(x["state_max_abs_error"] for x in references["results"]),
                  raw_video_checks=len(color_checks), raw_video_mae_max=max(x["mae"] for x in color_checks),
                  assembly_episode_checks=len(episodes), repaired_camera_episodes=repair["cameras"],
                  image_validation_workers=args.image_workers,
                  jpeg_shards_sha256_checked=len(assembly["shards"]),
                  reader_random_windows=len(selected), color_checks=color_checks,
                  io=dict(comparison_samples=len(cached_times),
                          cached_clip_median_s=statistics.median(cached_times),
                          av1_hot_clip_median_s=statistics.median(hot_times),
                          median_speedup=statistics.median(hot_times)/statistics.median(cached_times),
                          batch4_workers8_with_augmentation_samples_s=4*len(steady)/sum(steady),
                          batch4_workers8_wait_p95_s=float(np.quantile(steady, .95))),
                  jpeg=dict(quality=95, subsampling=0, all_frames_decoded_and_checked=True,
                            bytes=sum(s["bytes"] for s in assembly["shards"]),
                            mae=sum(s["absolute_error"] for s in assembly["shards"])/sum(s["pixel_values"] for s in assembly["shards"]),
                            max_frame_mae=max(s["max_frame_mae"] for s in assembly["shards"]),
                            psnr_db=float(10*np.log10(255**2/(sum(s["squared_error"] for s in assembly["shards"])/sum(s["pixel_values"] for s in assembly["shards"]))))),
                  note="Source AV1 and chosen JPEG cache are lossy; errors quantified. Metadata-short cameras restored from original JPEG. No held-out validation split.")
    atomic_json(out / "validation.json", result)
    atomic_json(out / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in result.items() if k != "color_checks"}), flush=True)


if __name__ == "__main__":
    main()
