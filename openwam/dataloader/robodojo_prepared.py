"""Strict reader for the offline-validated RoboDojo Alpha cache.

Only indexed JPEG decoding and stochastic augmentation remain online. No AV1
decoding, parquet reads, pose conversion, normalization or resizing occurs.
"""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path

import numpy as np
import cv2
import torch
from PIL import Image

from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.transforms.normalize import Normalizer
from openwam.dataloader.transforms.video import VideoColorJitter, color_jitter_enabled

SLOTS = np.array([*range(10), *range(34, 44)])


class PreparedRoboDojoDataset(BaseDataset):
    DEPLOY_ACTION_MODE = "eef"

    @classmethod
    def from_config(cls, config, split="train"):
        return cls(dict(config), split=split)

    def __init__(self, config, *, split="train"):
        self.root = Path(config["prepared_dir"])
        manifest_bytes = (self.root / "manifest.json").read_bytes()
        self.manifest = json.loads(manifest_bytes)
        m = self.manifest
        if (m.get("status"), m.get("format"), m.get("image_codec")) != ("ready", 1, "jpeg"):
            raise ValueError("Complete offline preparation and validation before training")
        expected = dict(action_mode="eef", embodiment="arx_x5", variant="sim", num_frames=33,
                        video_stride=4, window_stride=1, height=384, width=320, multiview=True,
                        normalize_mode="min-max", unify_action=True,
                        unify_action_map=["0-9", "34-43"],
                        camera_layout=["cam_head", "cam_left_wrist", "cam_right_wrist"])
        for key, value in expected.items():
            actual = config.get(key)
            if isinstance(value, list):
                actual = list(actual) if actual is not None else actual
            if actual != value:
                raise ValueError(f"Prepared cache requires {key}={value!r}, got {actual!r}")
        if config.get("unify_state_map") is not None:
            raise ValueError("Prepared state uses the canonical action mapping")
        if Path(config["dataset_dir"]).resolve() != Path(m["dataset_dir"]).resolve():
            raise ValueError("Source root differs from preparation provenance")
        if split not in ("train", "val"):
            raise ValueError(split)
        # Official RoboDojo exposes the same corpus for train/val; val only
        # disables augmentation. It is not a held-out validation score.
        self.split = split
        for name, entry in m["artifacts"].items():
            p = self.root / name
            if p.stat().st_size != entry["size"]:
                raise ValueError(f"Truncated cache: {p}")
            if entry.get("sha256") and p.stat().st_size < (100 << 20):
                if hashlib.sha256(p.read_bytes()).hexdigest() != entry["sha256"]:
                    raise ValueError(f"Cache metadata checksum mismatch: {p}")
        self.episodes = json.loads((self.root / "episodes.json").read_text())
        self._shards = {int(s["task_index"]): s for s in m["frame_shards"]}
        self._decoded = OrderedDict()
        self._decode_cache_size = int(config.get("decode_cache_size", 0))
        self._frames = {}
        self._states = self._windows = None
        self._length = m["windows"]
        self.normalization_stats_path = str(self.root / "normalization_stats.npy")
        self._stats = np.load(self.normalization_stats_path, allow_pickle=True).item()["eef"]
        self._normalizer = Normalizer("min_max", self._stats)
        self.action_mode, self.embodiment, self.variant = "eef", "arx_x5", "sim"
        self.num_frames, self.num_action_steps, self.num_video_frames = 33, 32, 9
        self._action_masks = np.zeros((34, 32, 80), dtype=bool)
        self._video_masks = np.arange(0, 33, 4)[None, :] < np.arange(34)[:, None]
        for length in range(2, 34):
            self._action_masks[length, :length-1, SLOTS] = True
        self._proprio_mask = np.zeros((1, 80), bool)
        self._proprio_mask[:, SLOTS] = True
        jitter = config.get("color_jitter")
        self._latent_cache = None
        self._latent_cache_dir = config.get("latent_cache_dir")
        if self._latent_cache_dir:
            if color_jitter_enabled(jitter):
                raise ValueError("VAE cache requires color_jitter.enabled=false")
            from openwam.dataloader.robodojo_latent_cache import LatentCache
            self._latent_cache = LatentCache(self._latent_cache_dir, self.root, len(self))
        self._jitter = None
        if split == "train" and color_jitter_enabled(jitter):
            self._jitter = VideoColorJitter(**{k: float(jitter.get(k, d)) for k, d in
                                            (("brightness", .2), ("contrast", .2), ("saturation", .2), ("hue", 0))})

    @property
    def action_dim(self):
        return 80

    @property
    def normalization_stats(self):
        return {k: np.asarray(v).copy() for k, v in self._stats.items()}

    def denormalize_action(self, action):
        return self._normalizer.unnormalize(np.asarray(action)[..., SLOTS])

    def _open(self):
        if self._states is None:
            cv2.setNumThreads(1)
            self._states = np.load(self.root / "states_unified.npy", mmap_mode="r")
            self._windows = np.load(self.root / "windows.npy", mmap_mode="r")
            if self._states.dtype != np.float32 or self._states.shape != (self.manifest["frames"], 80):
                raise ValueError("Invalid cached state array")
            if self._windows.dtype != np.int32 or self._windows.shape != (len(self), 3):
                raise ValueError("Invalid cached window array")

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_frames"] = {}
        state["_decoded"] = OrderedDict()
        state["_states"] = state["_windows"] = None
        return state

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        self._open()
        ep, offset, actual = map(int, self._windows[index])
        episode = self.episodes[ep]
        task = episode["task_index"]
        start = episode["start"] + offset
        state_indices = start + np.minimum(np.arange(33), actual - 1)
        frames = []
        if self._latent_cache is None:
            shard = self._shards[task]
            if task not in self._frames:
                blob = np.memmap(self.root / shard["path"], mode="r", dtype=np.uint8)
                offsets = np.load(self.root / shard["offsets"], mmap_mode="r")
                if offsets.dtype != np.int64 or offsets.shape != (shard["end"]-shard["start"]+1,):
                    raise ValueError("Invalid cached JPEG index")
                if offsets[0] != 0 or offsets[-1] != len(blob):
                    raise ValueError("Truncated cached JPEG shard")
                self._frames[task] = (blob, offsets)
            start = episode["start"] + offset
            # All expensive deterministic work is already done; only tiny index
            # and boundary-mask lookups remain, plus stochastic clip augmentation.
            state_indices = start + np.minimum(np.arange(33), actual - 1)
            video_indices = state_indices[::4]
            blob, offsets = self._frames[task]
            frames = []
            decoded = {}
            for i in video_indices:
                local = i-shard["start"]
                if local not in decoded:
                    key = (task, local)
                    if key in self._decoded:
                        decoded[local] = self._decoded[key]
                        self._decoded.move_to_end(key)
                    else:
                        bgr = cv2.imdecode(blob[offsets[local]:offsets[local+1]], cv2.IMREAD_COLOR)
                        if bgr is None or bgr.shape != (384, 320, 3):
                            raise ValueError(f"Invalid cached JPEG: task={task}, frame={local}")
                        decoded[local] = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                        if self._decode_cache_size:
                            self._decoded[key] = decoded[local]
                            if len(self._decoded) > self._decode_cache_size:
                                self._decoded.popitem(last=False)
                frames.append(decoded[local])
            if self._jitter is not None:
                frames = self._jitter.apply({"video": frames})["video"]
        states = np.array(self._states[state_indices], copy=True)
        result = {
            "video": frames, "vace_video": None, "first_frame_image": [frames[0]] if frames else None,
            "action": torch.from_numpy(states[1:]), "proprio": torch.from_numpy(states[:1]),
            "action_mask": torch.from_numpy(self._action_masks[actual].copy()),
            "video_mask": torch.from_numpy(self._video_masks[actual].copy()),
            "proprio_mask": torch.from_numpy(self._proprio_mask.copy()),
            "prompt": episode["prompt"], "episode_index": ep, "start_frame": offset,
            "end_frame": offset + actual, "episode_length": episode["length"],
            "task_name": str(episode["task_index"]), "variant": "sim", "embodiment": "arx_x5",
            "source_frame": "env_origin_relative_position_world_orientation_wxyz", "active_arm": "both",
        }

        if self._latent_cache is not None:
            result["video_latents"] = self._latent_cache[index]
        return result
