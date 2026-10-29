"""Parallel assembly into indexed JPEG shards; quality 95 and 4:4:4."""
import argparse
import concurrent.futures
import hashlib
import json
import time
import io
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

from prepare_robodojo_openwam import CAMERAS, SLOTS, atomic_json


def assemble_task(arguments):
    cv2.setNumThreads(1)
    root_s, task, episodes, jobs = arguments
    root = Path(root_s)
    destination = root / "frames" / f"task_{task:02d}.jpgs"
    offsets_path = root / "frames" / f"task_{task:02d}.offsets.npy"
    receipt = root / "assembly_receipts" / f"task_{task:02d}.json"
    if receipt.exists():
        result = json.loads(receipt.read_text())
        if destination.stat().st_size != result["bytes"]:
            raise ValueError("Truncated completed shard")
        return result
    source_records, mapped = {}, {}
    selected_eps = {e["episode_index"] for e in episodes}
    for job in jobs:
        if not any(e["episode_index"] in selected_eps for e in job["episodes"]):
            continue
        key = hashlib.sha256(job["path"].encode()).hexdigest()[:20]
        record = json.loads((root / "video_receipts" / (key+".json")).read_text())
        camera = job["camera"]
        y0, y1, x0, x1 = SLOTS[CAMERAS.index(camera)]
        file = root / record["cache"]
        expected = record["frames"] * (y1-y0) * (x1-x0) * 3
        if file.stat().st_size != expected:
            raise ValueError(f"Truncated camera cache: {file}")
        mapped[job["path"]] = np.memmap(file, mode="r", dtype=np.uint8,
                                       shape=(record["frames"], y1-y0, x1-x0, 3))
        for e in job["episodes"]:
            source_records[(e["episode_index"], camera)] = (job["path"], e)
    temp = destination.with_suffix(".tmp")
    started = time.monotonic()
    content_hash = hashlib.sha256()
    offsets = [0]
    abs_error = squared_error = pixels = 0
    max_frame_mae = 0.
    start, end = episodes[0]["start"], episodes[-1]["end"]
    with open(temp, "wb", buffering=16 << 20) as output:
        for ep in episodes:
            frames = np.empty((ep["length"], 384, 320, 3), np.uint8)
            for camera, (y0, y1, x0, x1) in zip(CAMERAS, SLOTS):
                path, e = source_records[(ep["episode_index"], camera)]
                if e.get("repair"):
                    source = np.load(root / "repairs" / f"episode_{ep['episode_index']:04d}_{camera}.npy")
                else:
                    source = mapped[path][e["video_start"]:e["video_end"]]
                if len(source) != ep["length"]:
                    raise ValueError("Unresolved deficient camera")
                frames[:, y0:y1, x0:x1] = source
            for frame in frames:
                buffer = io.BytesIO()
                Image.fromarray(frame).save(buffer, format="JPEG", quality=95, subsampling=0)
                block = buffer.getvalue()
                # Verify every encoded image offline, not on the first epoch.
                decoded = cv2.cvtColor(cv2.imdecode(np.frombuffer(block, np.uint8), cv2.IMREAD_COLOR),
                                       cv2.COLOR_BGR2RGB)
                if decoded.shape != frame.shape:
                    raise ValueError("Encoded JPEG shape mismatch")
                delta = decoded.astype(np.int16)-frame.astype(np.int16)
                total = int(np.abs(delta).sum())
                mae = total/frame.size
                if mae > 4:
                    raise ValueError(f"Unexpected JPEG quality degradation: MAE={mae}")
                abs_error += total
                squared_error += float(np.square(delta.astype(np.float32)).sum())
                pixels += frame.size
                max_frame_mae = max(max_frame_mae, mae)
                output.write(block)
                content_hash.update(block)
                offsets.append(offsets[-1]+len(block))
    temp.replace(destination)
    with open(offsets_path.with_suffix(".tmp"), "wb") as f:
        np.save(f, np.asarray(offsets, np.int64))
    offsets_path.with_suffix(".tmp").replace(offsets_path)
    result = dict(task_index=task, path=str(destination.relative_to(root)), start=start, end=end,
                  offsets=str(offsets_path.relative_to(root)), codec="jpeg", quality=95, subsampling=0,
                  sha256=content_hash.hexdigest(), bytes=destination.stat().st_size,
                  encoded_frames=len(offsets)-1, pixel_values=pixels, absolute_error=abs_error,
                  squared_error=squared_error, mae=abs_error/pixels, max_frame_mae=max_frame_mae,
                  psnr_db=float(10*np.log10(255**2/max(squared_error/pixels, 1e-12))),
                  seconds=time.monotonic()-started)
    atomic_json(receipt, result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", required=True)
    p.add_argument("--workers", type=int, default=24)
    args = p.parse_args()
    root = Path(args.prepared)
    plan = json.loads((root / "plan.json").read_text())
    json.loads((root / "videos_complete.json").read_text())
    json.loads((root / "repairs_complete.json").read_text())
    episodes = json.loads((root / "episodes.json").read_text())
    for folder in ("frames", "assembly_receipts"):
        (root / folder).mkdir(exist_ok=True)
    started = time.monotonic()
    jobs = [(str(root), task, [e for e in episodes if e["task_index"] == task], plan["jobs"])
            for task in range(plan["tasks"])]
    results = []
    with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(assemble_task, j) for j in jobs]
        for f in concurrent.futures.as_completed(futures):
            result = f.result()
            results.append(result)
            progress = dict(completed=len(results), total=len(jobs), seconds=time.monotonic()-started,
                            last=result["task_index"])
            atomic_json(root / "assembly_progress.json", progress)
            print(json.dumps(progress), flush=True)
    atomic_json(root / "assembly_complete.json", dict(status="assembled", frames=plan["frames"],
                shards=sorted(results, key=lambda r: r["task_index"]), seconds=time.monotonic()-started))


if __name__ == "__main__":
    main()
