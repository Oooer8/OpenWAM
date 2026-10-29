"""Read NVIDIA Cosmos3-DROID without modifying the shared source dataset.

This conversion's Cartesian columns are kept in their published robot-base
frame. No undocumented wrist/TCP transform is inferred. Deployment must use
the same endpoint convention. Actions are commanded poses, not future states.
"""
from __future__ import annotations

import hashlib
import functools
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from openwam.dataloader.bases import LeRobotV3Reader
from openwam.dataloader.bases.lerobot_v3_reader import _read_data_table_cached
from openwam.dataloader.oxe_droid import _clean_text
from openwam.dataloader.utils.eef import euler_xyz_to_rot6d
from openwam.dataloader.utils.normalization import apply_normalization

logger = logging.getLogger(__name__)

ACTION_POSE = "action.cartesian_position"
ACTION_GRIP = "action.gripper_position"
STATE_POSE = "observation.state.cartesian_position"
STATE_GRIP = "observation.state.gripper_position"
HEAD = "observation.image.exterior_image_1_left"
HEAD2 = "observation.image.exterior_image_2_left"
WRIST = "observation.image.wrist_image_left"


@functools.lru_cache(maxsize=64)
def _read_cosmos_table_cached(path, columns):
    # This one-bucket corpus has 62 numeric shards. The training node has
    # 1.8 TiB RAM; a bounded full-corpus cache per worker avoids re-reading
    # random shards every batch. Keep the generic multi-bucket cache small.
    return _read_data_table_cached.__wrapped__(path, columns)


def arm10(pose, closedness):
    pose = np.asarray(pose, dtype=np.float32).reshape(-1, 6)
    grip = np.asarray(closedness, dtype=np.float32).reshape(-1, 1)
    if not np.isfinite(pose).all() or not np.isfinite(grip).all():
        raise ValueError("Non-finite Cartesian pose/gripper target")
    if np.any(grip < -1e-4) or np.any(grip > 1.0001):
        raise ValueError("DROID gripper closedness is outside [0,1]")
    return np.concatenate((pose[:, :3], euler_xyz_to_rot6d(pose[:, 3:]), 1 - grip), axis=1)


def prompt_variants(value):
    if not isinstance(value, str):
        return []
    # Match the official DROID placeholder/fallback policy for each annotation.
    return list(dict.fromkeys(text for x in value.split("|") if (text := _clean_text(x))))


class Cosmos3DroidDataset(LeRobotV3Reader):
    DATASET_NAME = "Cosmos3-DROID"
    ACTION_DIM = 10
    NEEDED_COLS = (ACTION_POSE, ACTION_GRIP, STATE_POSE, STATE_GRIP, "task_index", "episode_index")
    HEAD_CAMERA = HEAD
    LEFT_WRIST_CAMERA = WRIST
    RIGHT_WRIST_CAMERA = None
    DEPLOY_ACTION_MODE = "eef"
    WRIST_DECODE_TOLERATED = ()  # Every observed camera must decode successfully.
    CONFIG_KEYS = LeRobotV3Reader.CONFIG_KEYS + ("prepared_dir", "action_mode", "view_layout")

    def __init__(self, dataset_dir, *, prepared_dir, action_mode="eef", view_layout="openwam_two_view", **kwargs):
        if view_layout not in ("openwam_two_view", "cosmos3_three_view"):
            raise ValueError(f"Unknown Cosmos3-DROID view_layout: {view_layout}")
        self.view_layout = view_layout
        if view_layout == "cosmos3_three_view":
            if not kwargs.get("multiview", False):
                raise ValueError("cosmos3_three_view requires multiview=true")
            if kwargs.get("head_camera_choices"):
                raise ValueError("Three-view packing uses all cameras; head sampling must be disabled")
            layout = list(kwargs.get("camera_layout") or (WRIST, HEAD, HEAD2))
            if layout != [WRIST, HEAD, HEAD2]:
                raise ValueError("Cosmos3 order must be [wrist, exterior_1, exterior_2]")
            kwargs["camera_layout"] = layout
        self._prepared = Path(prepared_dir)
        self._manifest = json.loads((self._prepared / "manifest.json").read_text())
        if Path(dataset_dir).resolve() != Path(self._manifest["dataset_dir"]).resolve():
            raise ValueError("Prepared index belongs to a different dataset root")
        info_bytes = (Path(dataset_dir) / "meta/info.json").read_bytes()
        if hashlib.sha256(info_bytes).hexdigest() != self._manifest["info_sha256"]:
            raise ValueError("Dataset info changed since preparation")
        if action_mode != "eef" or kwargs.get("normalize_mode") != "min-max":
            raise ValueError("This alpha adapter requires action_mode=eef and normalize_mode=min-max")
        if (kwargs.get("num_frames", 33), kwargs.get("video_stride", 4)) != (33, 4):
            raise ValueError("Alpha expects a 33-row window and video_stride=4")
        if not kwargs.get("unify_action", False) or kwargs.get("unify_action_map") != ["0-9"]:
            raise ValueError('Alpha requires unify_action=true, unify_action_map=["0-9"]')
        self.action_mode = action_mode
        super().__init__(dataset_dir, **kwargs)

    def _resolve_cameras(self, info):
        if self.view_layout == "cosmos3_three_view":
            # Reader roles name spatial slots, not the camera's physical mount:
            # top (full width), bottom-left, bottom-right. This also gives each
            # source its own packed-file offsets and the correct decode size.
            return WRIST, HEAD, HEAD2
        return super()._resolve_cameras(info)

    def _build_episode_index(self, info):
        if float(info["fps"]) != 15:
            raise ValueError("Cosmos3-DROID must retain its native 15 FPS")
        eps = pd.read_parquet(self._prepared / "episodes.parquet")
        return eps[eps["training_split"] == self._split].reset_index(drop=True)

    def _load_excluded_episode_indices(self):
        return set()  # Preparation already records and applies exclusions.

    def _load_data_table(self, chunk_idx, file_idx):
        path = self._dataset_dir / self._data_path_template.format(chunk_index=chunk_idx, file_index=file_idx)
        return _read_cosmos_table_cached(str(path), tuple(self.NEEDED_COLS))

    def _train_min_window_len(self):
        return self._num_frames  # Avoid windows mostly containing boundary padding.

    def _load_prompts(self):
        table = pq.read_table(self._dataset_dir / "meta/tasks.parquet")
        self._task_variants = {
            int(row["task_index"]): prompt_variants(row["task"])
            for row in table.to_pylist()
        }

    def _resolve_prompt(self, row, win):
        ep = int(row["episode_index"])
        if not (win["episode_index"].to_numpy() == ep).all():
            raise ValueError(f"Parquet index crosses episode boundary for episode {ep}")
        choices = self._task_variants.get(int(win["task_index"].iloc[0]), [])
        if not choices:
            raise ValueError(f"Episode {ep} has no usable language instruction")
        return random.choice(choices) if self._split == "train" else choices[0]

    def _load_stats(self, info):
        raw = json.loads((self._prepared / "normalization.json").read_text())
        self._state_stats = {k: np.array(v, np.float32) for k, v in raw["eef_state"].items()}
        self.normalization_stats_path = str(self._prepared / "normalization_stats.npy")
        return {k: np.array(v, np.float32) for k, v in raw["eef"].items()}

    def _decode_one_camera(self, camera, row, ep_local, offset, real_local_indices, *, kind):
        frames = super()._decode_one_camera(camera, row, ep_local, offset, real_local_indices, kind=kind)
        if self.view_layout == "cosmos3_three_view":
            if not frames:
                raise ValueError(f"Required three-view camera is missing: {camera}")
            # Preserve source pixels, including an actually black exterior
            # stream. Do not replace one exterior with the other or relabel it.
            return frames
        if kind != "head" or camera not in (HEAD, HEAD2):
            return frames
        def all_black(images):
            return all(max(hi for lo, hi in image.getextrema()) <= 1 for image in images)
        if not all_black(frames):
            return frames
        # Some source recordings contain an entirely black external stream.
        # Use the synchronized second external view, with its own packed-file
        # offset. Never substitute another episode or fabricate image pixels.
        alternate = HEAD2 if camera == HEAD else HEAD
        other = super()._decode_one_camera(alternate, row, ep_local, offset, real_local_indices, kind=kind)
        if all_black(other):
            raise ValueError(f"Both exterior views are black for episode {int(row['episode_index'])}, offset {offset}")
        count = getattr(self, "_black_head_fallback_count", 0) + 1
        self._black_head_fallback_count = count
        if count == 1 or count % 1000 == 0:
            logger.warning("Black exterior-view fallback #%d: episode=%d offset=%d %s -> %s",
                           count, int(row["episode_index"]), offset, camera, alternate)
        return other

    def _action_20d(self, win):
        a = arm10(np.stack(win[ACTION_POSE]), win[ACTION_GRIP].to_numpy())
        return apply_normalization(a, self._normalization_stats, "min-max")

    def _proprio_20d(self, win):
        p = arm10(np.stack(win[STATE_POSE].iloc[:1]), win[STATE_GRIP].iloc[:1].to_numpy())
        return apply_normalization(p, self._state_stats, "min-max")

    def __getitem__(self, index):
        # Fail with the actual corrupt sample instead of silently advancing to
        # another episode. Upfront checks run before expensive training starts.
        return self._getitem_impl(index)

