"""Independent raw-HDF5 references covering every task, pinned to source revision."""
import argparse
import concurrent.futures
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image

from repair_robodojo_videos import HTTPRangeFile, source_records
from prepare_robodojo_openwam import CAMERAS, SLOTS, REVISION, atomic_json


def audit(arguments):
    output, episode, item = arguments
    out = Path(output)
    ep = episode["episode_index"]
    path = out / "raw_references" / f"episode_{ep:04d}.npz"
    receipt = path.with_suffix(".json")
    if path.exists() and receipt.exists():
        return json.loads(receipt.read_text())
    url = f"https://huggingface.co/datasets/RoboDojo-Benchmark/RoboDojo/resolve/{REVISION}/{item['path']}"
    remote = HTTPRangeFile(url, item["size"], block_size=1 << 20)
    arrays = {}
    from openwam.dataloader.robodojo import read_calibrated_eef20
    from openwam.dataloader.robodojo_contract import arx_x5_calibration
    from openwam.dataloader.transforms.multiview import format_prompt_for_inference
    with h5py.File(remote) as h:
        instruction = h["instruction"][()]
        if isinstance(instruction, bytes):
            instruction = instruction.decode()
        if format_prompt_for_inference(instruction.strip()) != episode["prompt"]:
            raise ValueError(f"Wrong reference task: {ep}")
        arrays["eef20"] = read_calibrated_eef20(h, arx_x5_calibration())
        if len(arrays["eef20"]) != episode["length"]:
            raise ValueError("Reference/source episode length mismatch")
        transformed = np.load(out / "states_eef20.npy", mmap_mode="r")[episode["start"]:episode["end"]]
        error = float(np.max(np.abs(transformed - arrays["eef20"])))
        # LeRobot stores float32 while HDF5 sometimes stores float64.
        np.testing.assert_allclose(transformed, arrays["eef20"], rtol=0, atol=3e-6)
        frames = [0, min(10, episode["length"]-1), episode["length"]-1]
        for camera, (y0, y1, x0, x1) in zip(CAMERAS, SLOTS):
            raw_camera = "cam_head" if camera == "cam_high" else camera
            for index in frames:
                buf = np.frombuffer(bytes(h[f"vision/{raw_camera}/colors"][index]), np.uint8)
                rgb = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if rgb is None:
                    raise ValueError("Bad source JPEG")
                arrays[f"{camera}_{index}"] = np.asarray(Image.fromarray(rgb).resize(
                    (x1-x0, y1-y0), Image.Resampling.BILINEAR))
    np.savez(path, **arrays)
    result = dict(episode=ep, task=episode["task_index"], source=item["path"], revision=REVISION,
                  state_max_abs_error=error, video_sample_frames=frames, bytes_transferred=remote.bytes_read)
    atomic_json(receipt, result)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", required=True)
    args = p.parse_args()
    out = Path(args.prepared)
    episodes = json.loads((out / "episodes.json").read_text())
    mapping = source_records(out, episodes)
    selected = {}
    for e in episodes:
        selected.setdefault(e["task_index"], e)
    (out / "raw_references").mkdir(exist_ok=True)
    jobs = [(str(out), e, mapping[str(e["episode_index"])]) for e in selected.values()]
    results = []
    with concurrent.futures.ProcessPoolExecutor(8) as pool:
        for result in pool.map(audit, jobs):
            results.append(result)
            print(json.dumps(result), flush=True)
    atomic_json(out / "raw_reference_checks.json", dict(status="passed", tasks=len(results), results=results))


if __name__ == "__main__":
    main()
