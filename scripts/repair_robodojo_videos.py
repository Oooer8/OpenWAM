"""Restore entire deficient camera episodes from pinned official HDF5."""
import argparse
import concurrent.futures
import hashlib
import io
import json
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import cv2
import h5py
import numpy as np
import requests
from PIL import Image

from prepare_robodojo_openwam import CAMERAS, SLOTS, REVISION, atomic_json


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url, size, block_size=8 << 20):
        self.session = requests.Session()
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        self.session.mount("https://", HTTPAdapter(max_retries=Retry(total=4, backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504])))
        self.session.proxies.update({"https": "http://192.168.10.3:7890"})
        self.url, self.size, self.pos, self.block_size = url, size, 0, block_size
        self.bytes_read = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset + (0 if whence == 0 else self.pos if whence == 1 else self.size)
        return self.pos

    @lru_cache(maxsize=16)
    def block(self, number):
        start = number * self.block_size
        end = min(start + self.block_size, self.size) - 1
        response = self.session.get(self.url + f"?range_start={start}&range_end={end}",
                                    headers={"Range": f"bytes={start}-{end}"}, timeout=120)
        response.raise_for_status()
        if response.status_code != 206 or len(response.content) != end - start + 1:
            raise RuntimeError(f"Server did not honor range {start}-{end}")
        self.bytes_read += len(response.content)
        return response.content

    def readinto(self, target):
        view = memoryview(target).cast("B")
        size = min(len(view), self.size - self.pos)
        done = 0
        while done < size:
            number, offset = divmod(self.pos, self.block_size)
            block = self.block(number)
            take = min(size - done, len(block) - offset)
            view[done:done+take] = block[offset:offset+take]
            done += take
            self.pos += take
        return done


def api(path):
    s = requests.Session()
    s.proxies.update({"https": "http://192.168.10.3:7890"})
    r = s.get(f"https://huggingface.co/api/datasets/RoboDojo-Benchmark/RoboDojo/tree/{REVISION}/{path}",
              params={"recursive": "false", "limit": 1000}, timeout=60)
    r.raise_for_status()
    return r.json()


def source_records(out, episodes):
    path = out / "raw_source_map.json"
    if path.exists():
        return json.loads(path.read_text())
    dirs = sorted(x["path"] for x in api("data/RoboDojo") if x["type"] == "directory")
    if len(dirs) != 35:
        raise ValueError("Unexpected official task directory count")
    result = {}
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        listings = list(pool.map(api, [d + "/arx_x5/data" for d in dirs]))
    for task, listing in enumerate(listings):
        files = sorted((x for x in listing if x["path"].endswith(".hdf5")), key=lambda x: x["path"])
        rows = [e for e in episodes if e["task_index"] == task]
        if len(files) != len(rows):
            raise ValueError(f"Task {task} count mismatch: {len(files)} vs {len(rows)}")
        for e, item in zip(rows, files):
            result[str(e["episode_index"])] = item
    atomic_json(path, result)
    return result


def repair_job(arguments):
    out_s, source_s, records, episode, item = arguments
    import pyarrow.parquet as pq
    out, source = Path(out_s), Path(source_s)
    ep = episode["episode_index"]
    receipt = out / "repair_receipts" / f"episode_{ep:04d}.json"
    if receipt.exists():
        return json.loads(receipt.read_text())
    url = f"https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/resolve/{REVISION}/{item['path']}"
    remote = HTTPRangeFile(url, item["size"])
    # Only three source numeric shards; memory-map reading is small beside video.
    tables = sorted((source / "data").rglob("*.parquet"))
    source_states = None
    offset = 0
    for p in tables:
        t = pq.read_table(p, columns=["observation.state"])
        if offset <= episode["start"] < offset + len(t):
            source_states = np.asarray(t.slice(episode["start"]-offset, episode["length"])["observation.state"].to_pylist())
            break
        offset += len(t)
    if source_states is None or len(source_states) != episode["length"]:
        raise ValueError("Reference episode crosses a numeric shard")
    checksums = {}
    with h5py.File(remote, "r") as handle:
        instruction = handle["instruction"][()]
        if isinstance(instruction, bytes):
            instruction = instruction.decode()
        from openwam.dataloader.transforms.multiview import format_prompt_for_inference
        if format_prompt_for_inference(instruction.strip()) != episode["prompt"]:
            raise ValueError(f"Raw source instruction mismatch: episode {ep}")
        raw = np.concatenate([handle[f"state/{name}"][:] for name in
            ("left_ee_poses", "left_ee_joint_states", "right_ee_poses", "right_ee_joint_states")], axis=1)
        np.testing.assert_allclose(raw, source_states, atol=1e-6, rtol=0)
        for record in records:
            camera = record["camera"]
            raw_camera = "cam_head" if camera == "cam_high" else camera
            data = handle[f"vision/{raw_camera}/colors"]
            if len(data) != episode["length"]:
                raise ValueError(f"Raw source camera length mismatch {ep} {camera}")
            y0, y1, x0, x1 = SLOTS[CAMERAS.index(camera)]
            target = np.empty((episode["length"], y1-y0, x1-x0, 3), dtype=np.uint8)
            h = hashlib.sha256()
            # Sequential slabs avoid hundreds of tiny network round trips.
            for start in range(0, len(data), 32):
                for j, encoded in enumerate(data[start:start+32]):
                    buf = np.frombuffer(bytes(encoded), np.uint8)
                    h.update(buf)
                    rgb = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if rgb is None:
                        raise ValueError(f"Bad official JPEG {ep} {camera} {start+j}")
                    resized = Image.fromarray(rgb).resize((x1-x0, y1-y0), Image.Resampling.BILINEAR)
                    target[start + j] = np.asarray(resized)
            checksums[camera] = h.hexdigest()
            destination = out / "repairs" / f"episode_{ep:04d}_{camera}.npy"
            temporary = destination.with_suffix(".tmp")
            with open(temporary, "wb") as f:
                np.save(f, target)
            temporary.replace(destination)
    result = dict(episode=ep, source=item, revision=REVISION, frames=episode["length"],
                  cameras=checksums, bytes_transferred=remote.bytes_read,
                  semantics="entire camera episode restored from original RGB JPEG; state and instruction checked")
    atomic_json(receipt, result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", required=True)
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args()
    out = Path(args.prepared)
    plan = json.loads((out / "plan.json").read_text())
    episodes = json.loads((out / "episodes.json").read_text())
    mapping = source_records(out, episodes)
    grouped = defaultdict(list)
    for r in plan["repairs"]:
        grouped[r["episode_index"]].append(r)
    (out / "repair_receipts").mkdir(exist_ok=True)
    (out / "repairs").mkdir(exist_ok=True)
    jobs = [(str(out), plan["dataset_dir"], rs, episodes[ep], mapping[str(ep)]) for ep, rs in grouped.items()]
    with concurrent.futures.ProcessPoolExecutor(args.workers) as pool:
        for i, result in enumerate(pool.map(repair_job, jobs), 1):
            print(json.dumps(dict(done=i, total=len(jobs), result=result)), flush=True)
    atomic_json(out / "repairs_complete.json", dict(status="repaired", episodes=len(jobs), cameras=len(plan["repairs"])))


if __name__ == "__main__":
    main()
