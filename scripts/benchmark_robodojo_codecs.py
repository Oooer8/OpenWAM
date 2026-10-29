"""Measure identical RGB clips with raw, LZ4-frame and Zstd lossless storage."""
import argparse
import json
import statistics
import time
import io
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
import cv2
from PIL import Image

from openwam.dataloader.transforms.video import VideoColorJitter
from prepare_robodojo_openwam import CAMERAS, SLOTS, atomic_json


class Clips(torch.utils.data.Dataset):
    def __init__(self, root, name, jitter=False, count=512):
        self.path = Path(root) / name
        self.name, self.jitter, self.count = name, jitter, count
        self.blob = np.memmap(self.path.with_suffix(".bin"), mode="r", dtype=np.uint8)
        self.offsets = np.load(self.path.with_suffix(".npy"))
        self.codec = None if name in ("raw", "jpeg") else pa.Codec(name, compression_level=1 if name == "zstd" else None)
        self.augment = VideoColorJitter(.2, .2, .2, 0)
        self.selection = np.random.default_rng(42).integers(0, len(self.offsets)-1, (count, 9))

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        result = []
        for i in self.selection[index]:
            buffer = memoryview(self.blob)[int(self.offsets[i]):int(self.offsets[i+1])]
            if self.codec is not None:
                buffer = self.codec.decompress(buffer, 384*320*3)
            if self.name == "jpeg":
                bgr = cv2.imdecode(np.frombuffer(buffer, np.uint8), cv2.IMREAD_COLOR)
                array = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            else:
                array = np.frombuffer(buffer, np.uint8).reshape(384, 320, 3)
            result.append(Image.fromarray(array))
        if self.jitter:
            result = self.augment.apply({"video": result})["video"]
        return result


def main():
    cv2.setNumThreads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--codecs", nargs="+", default=["raw", "lz4_frame", "zstd"])
    args = parser.parse_args()
    out = Path(args.prepared)
    folder = out / "codec_benchmark"
    folder.mkdir(exist_ok=True)
    checks = json.loads((out / "raw_reference_checks.json").read_text())["results"]
    images = []
    for check in checks:
        data = np.load(out / "raw_references" / f"episode_{check['episode']:04d}.npz")
        for index in check["video_sample_frames"]:
            canvas = np.empty((384, 320, 3), np.uint8)
            for camera, (y0, y1, x0, x1) in zip(CAMERAS, SLOTS):
                canvas[y0:y1, x0:x1] = data[f"{camera}_{index}"]
            images.append(canvas)
    raw_bytes = len(images)*384*320*3
    results = {}
    for name in args.codecs:
        codec = None if name in ("raw", "jpeg") else pa.Codec(name, compression_level=1 if name == "zstd" else None)
        offsets = [0]
        t = time.perf_counter()
        with open(folder / f"{name}.bin", "wb") as f:
            for image in images:
                blob = image.tobytes()
                if codec is not None:
                    blob = codec.compress(blob)
                    np.testing.assert_array_equal(np.frombuffer(codec.decompress(blob, image.nbytes), np.uint8),
                                                  image.reshape(-1))
                elif name == "jpeg":
                    buffer = io.BytesIO()
                    Image.fromarray(image).save(buffer, format="JPEG", quality=95, subsampling=0)
                    blob = buffer.getvalue()
                f.write(blob)
                offsets.append(offsets[-1]+len(blob))
        compression_s = time.perf_counter()-t
        np.save(folder / f"{name}.npy", np.array(offsets, np.int64))
        ds = Clips(folder, name)
        times = []
        for i in range(len(ds)):
            t = time.perf_counter()
            result = ds[i]
            times.append(time.perf_counter()-t)
            if i == 0 and name != "jpeg":
                np.testing.assert_array_equal(np.stack(result), np.stack([images[j] for j in ds.selection[i]]))
        augmented = Clips(folder, name, jitter=True)
        loader = torch.utils.data.DataLoader(augmented, batch_size=4, num_workers=8, collate_fn=list)
        last = time.perf_counter()
        waits = []
        for batch in loader:
            now = time.perf_counter()
            waits.append(now-last)
            last = time.perf_counter()
        results[name] = dict(bytes=offsets[-1], ratio=offsets[-1]/raw_bytes,
                             compression_verification_s=compression_s,
                             clip9_median_ms=statistics.median(times)*1000,
                             clips_s_workers8_augmented=4*len(waits[8:])/sum(waits[8:]),
                             batch4_p95_ms=float(np.quantile(waits[8:], .95))*1000)
        print(name, json.dumps(results[name]), flush=True)
    report = dict(status="passed", images=len(images), tasks=len(checks), jpeg_quality=95, jpeg_subsampling=0,
                  semantics="warm files, identical random clips, PIL construction and official augmentation; no cold-disk claim",
                  results=results)
    atomic_json(out / ("jpeg_benchmark.json" if "jpeg" in args.codecs else "codec_benchmark.json"), report)


if __name__ == "__main__":
    main()
