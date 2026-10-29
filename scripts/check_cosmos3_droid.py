"""Check packed-file alignment, alpha masks, poses and actual AV1 decoding."""
from pathlib import Path
import json
import argparse
import random
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from openwam.dataloader.registry import build_dataset
from openwam.dataloader.cosmos3_droid import arm10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--window-stride', type=int, default=None)
    parser.add_argument('--output-dir', default='work/cosmos3_droid_prepared')
    args = parser.parse_args()
    random.seed(42)
    cfg = OmegaConf.load("configs/dataloader/cosmos3_droid.yaml")
    if args.window_stride is not None:
        if args.window_stride < 1:
            raise ValueError('window stride must be positive')
        cfg.window_stride = args.window_stride
    train = build_dataset(cfg, split="train")
    val = build_dataset(cfg, split="val")
    train_ids = set(train._eps_df.episode_index)
    val_ids = set(val._eps_df.episode_index)
    assert not train_ids & val_ids
    sessions = lambda ds: set(ds._eps_df.episode_id.str.rsplit("/", n=1).str[0])
    assert not sessions(train) & sessions(val)
    closed = arm10(np.zeros((2, 6)), [0, 1])
    np.testing.assert_array_equal(closed[:, -1], [1, 0])
    np.testing.assert_array_equal(closed[0, 3:9], [1, 0, 0, 0, 1, 0])
    summary = {"window_stride": int(cfg.window_stride), "train_episodes": len(train_ids), "val_episodes": len(val_ids),
               "train_windows": len(train), "val_windows": len(val), "samples": []}
    panels = []
    for split, ds in (("train", train), ("val", val)):
        starts = ds._cum_n_starts
        indices = [0, len(ds) - 1] + [int(starts[i]) for i in [1, len(ds._eps_df)//2]]
        indices += random.sample(range(len(ds)), 4)
        for i in indices:
            begin = time.monotonic()
            sample = ds[i]
            assert sample["action"].shape == (32, 80)
            assert sample["proprio"].shape == (1, 80)
            assert sample["action_mask"][:, :10].all() and not sample["action_mask"][:, 10:].any()
            assert sample["proprio_mask"][:, :10].all() and not sample["proprio_mask"][:, 10:].any()
            assert sample["video_mask"].all()
            assert len(sample["video"]) == 9
            assert all(im.size == (320, 384) for im in sample["video"])
            assert sample["prompt"].strip()
            for key in ("action", "proprio"):
                x = sample[key]
                assert torch.isfinite(x).all()
                assert not x[:, 10:].any()
                r1, r2 = x[:, 3:6], x[:, 6:9]
                torch.testing.assert_close(r1.norm(dim=-1), torch.ones(len(x)), atol=1e-5, rtol=0)
                torch.testing.assert_close(r2.norm(dim=-1), torch.ones(len(x)), atol=1e-5, rtol=0)
                assert (torch.sum(r1*r2, dim=-1).abs() < 1e-5).all()
            image = np.asarray(sample["video"][0])
            if image[:256].std() <= 5 or image[256:, :160].std() <= 5:
                dst = Path(args.output_dir)
                dst.mkdir(parents=True, exist_ok=True)
                sample['video'][0].save(dst / f'low_contrast_{split}_{i}.png')
                print(json.dumps({'low_contrast': i, 'split': split,
                      'head_std': float(image[:256].std()),
                      'wrist_std': float(image[256:, :160].std()),
                      'wrist_min': int(image[256:, :160].min()),
                      'wrist_max': int(image[256:, :160].max())}), flush=True)
            assert image[:256].std() > 5 and image[256:, :160].std() > 5
            assert image[256:, 160:].max() == 0
            item = {"split": split, "index": i, "seconds": round(time.monotonic()-begin, 3),
                    "prompt": sample["prompt"], "xyz_abs_max": float(sample["action"][:, :3].abs().max())}
            summary["samples"].append(item)
            print(json.dumps(item), flush=True)
            if len(panels) < 8:
                canvas = Image.new("RGB", (320, 410), "white")
                canvas.paste(sample["video"][0], (0, 26))
                ImageDraw.Draw(canvas).text((4, 5), f"{split} window {i}", fill="black")
                panels.append(canvas)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sheet = Image.new("RGB", (320*4, 410*2), "white")
    for n, im in enumerate(panels):
        sheet.paste(im, (n%4*320, n//4*410))
    sheet.save(out / "sample_check.png")
    summary["status"] = "passed"
    (out / "sample_check.json").write_text(json.dumps(summary, indent=2))
    print("DATA_CHECK_PASSED", flush=True)


if __name__ == "__main__":
    main()
