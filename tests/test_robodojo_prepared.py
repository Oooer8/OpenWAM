"""Parity tests against the official HDF5 reader, including all tail windows."""
import hashlib
import json
import random
import io
from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from openwam.dataloader.robodojo import RoboDojoDataset, read_calibrated_eef20
from openwam.dataloader.robodojo_contract import arx_x5_calibration
from openwam.dataloader.robodojo_prepared import PreparedRoboDojoDataset
from openwam.dataloader.transforms.normalize import Normalizer
from openwam.dataloader.utils.stats_computation.robodojo_stats_computation import build_and_save_robodojo_stats


@pytest.fixture
def pair(tmp_path):
    source = tmp_path / "raw"
    folder = source / "synthetic" / "arx_x5" / "data"
    folder.mkdir(parents=True)
    length = 41
    data = np.load("work/robodojo_adapter/reference_episode_0.npz")
    path = folder / "episode_0000000.hdf5"
    with h5py.File(path, "w") as h:
        for name in ("left_ee_poses", "right_ee_poses", "left_ee_joint_states", "right_ee_joint_states"):
            h[f"state/{name}"] = data[name][:length]
        h["instruction"] = "Test official temporal and image semantics."
        rng = np.random.default_rng(17)
        for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            d = h.create_dataset(f"vision/{camera}/colors", (length,), dtype=h5py.vlen_dtype(np.dtype("uint8")))
            for i in range(length):
                image = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
                ok, buf = cv2.imencode(".jpg", image)
                assert ok
                d[i] = buf
    out = tmp_path / "prepared"
    out.mkdir()
    stats = out / "normalization_stats.npy"
    build_and_save_robodojo_stats(source, stats)
    jitter = dict(enabled=True, brightness=.2, contrast=.2, saturation=.2, hue=0)
    official = RoboDojoDataset(source, task_name="synthetic", normalization_stats_path=stats,
                              unify_action_map=["0-9", "34-43"], color_jitter=jitter)
    raw = read_calibrated_eef20(path, arx_x5_calibration())
    normal = Normalizer("min_max", np.load(stats, allow_pickle=True).item()["eef"]).normalize(raw)
    state = np.zeros((length, 80), np.float32)
    state[:, [*range(10), *range(34, 44)]] = normal
    np.save(out / "states_unified.npy", state)
    with h5py.File(path) as h:
        frames = np.stack([np.asarray(official._video_at(h, i)) for i in range(length)])
    offsets = [0]
    with open(out / "frames.jpgs", "wb") as f:
        for frame in frames:
            buffer = io.BytesIO()
            Image.fromarray(frame).save(buffer, format="JPEG", quality=95, subsampling=0)
            blob = buffer.getvalue()
            f.write(blob)
            offsets.append(offsets[-1]+len(blob))
    np.save(out / "frames.offsets.npy", np.array(offsets, np.int64))
    # JPEG storage is intentionally lossy. Compare official sample assembly
    # using the exact same JPEG round-trip before its random augmentation.
    original_video_at = official._video_at
    def reference_video_at(handle, index):
        buffer = io.BytesIO()
        original_video_at(handle, index).save(buffer, format="JPEG", quality=95, subsampling=0)
        bgr = cv2.imdecode(np.frombuffer(buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    official._video_at = reference_video_at
    windows = np.array([[0, i, min(33, length-i)] for i in range(length-1)], np.int32)
    np.save(out / "windows.npy", windows)
    episodes = [dict(episode_index=0, start=0, end=length, length=length, task_index=0,
                     prompt="A video recorded from a robot's point of view executing the following instruction: Test official temporal and image semantics.")]
    (out / "episodes.json").write_text(json.dumps(episodes))
    artifacts = {p.name: dict(size=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                 for p in out.iterdir()}
    (out / "manifest.json").write_text(json.dumps(dict(status="ready", format=1, image_codec="jpeg", dataset_dir=str(source),
                 windows=len(windows), frames=length, artifacts=artifacts,
                 frame_shards=[dict(task_index=0, path="frames.jpgs", offsets="frames.offsets.npy", start=0, end=length)])))
    cfg = yaml.safe_load(Path("configs/dataloader/robodojo_prepared.yaml").read_text())
    cfg.update(dataset_dir=str(source), prepared_dir=str(out), color_jitter=jitter)
    return official, PreparedRoboDojoDataset(cfg), cfg


def test_every_window_matches_official(pair):
    official, prepared, _ = pair
    assert len(official) == len(prepared)
    for index in range(len(official)):
        random.seed(123 + index)
        a = official[index]
        random.seed(123 + index)
        b = prepared[index]
        for key in ("action", "proprio", "action_mask", "video_mask", "proprio_mask"):
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        assert a["prompt"] == b["prompt"]
        for x, y in zip(a["video"], b["video"]):
            np.testing.assert_array_equal(x, y)
    assert prepared[-1]["action_mask"].sum() == 20
    assert prepared[-1]["video_mask"].sum() == 1
    with pytest.raises(IndexError):
        prepared[len(prepared)]


def test_reject_stale_config_and_incomplete_cache(pair):
    _, _, cfg = pair
    for key, bad in (("video_stride", 3), ("window_stride", 2), ("height", 256),
                     ("unify_action_map", ["0-19"]), ("action_mode", "joint")):
        with pytest.raises(ValueError):
            PreparedRoboDojoDataset(dict(cfg, **{key: bad}))
    manifest = Path(cfg["prepared_dir"]) / "manifest.json"
    m = json.loads(manifest.read_text())
    m["status"] = "preparing"
    manifest.write_text(json.dumps(m))
    with pytest.raises(ValueError):
        PreparedRoboDojoDataset(cfg)


def test_val_disables_jitter_and_denormalization_matches(pair):
    official, _, cfg = pair
    prepared = PreparedRoboDojoDataset(cfg, split="val")
    official._color_jitter = None
    a, b = official[3], prepared[3]
    for x, y in zip(a["video"], b["video"]):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_allclose(official.denormalize_action(a["action"]),
                               prepared.denormalize_action(b["action"]), rtol=0, atol=0)
