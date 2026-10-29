"""Exact-index scene/robot RGB pairing; never decode or substitute another index."""
import numpy as np

from .libero_latent_cache import CachedLiberoDataset

ROBOT_CAMERAS = ('observation.images.robot_rgb', 'observation.images.wrist_robot_rgb', None)
SCENE_CAMERAS = ('observation.images.image', 'observation.images.wrist_image', None)


class PairedRobotRGBCachedLiberoDataset(CachedLiberoDataset):
    CONFIG_KEYS = CachedLiberoDataset.CONFIG_KEYS + ('robot_rgb_latent_cache_dir',)

    def __init__(self, *args, robot_rgb_latent_cache_dir=None, **kwargs):
        if not robot_rgb_latent_cache_dir:
            raise ValueError('Missing separately encoded robot_rgb_latent_cache_dir')
        if str(robot_rgb_latent_cache_dir) == str(kwargs.get('libero_latent_cache_dir')):
            raise ValueError('V2 may not reuse scene RGB cache')
        super().__init__(*args, **kwargs)
        if tuple(self._camera_layout) != SCENE_CAMERAS:
            raise ValueError('V1 must retain the original external/wrist scene RGB layout')
        robot_kwargs = dict(kwargs, libero_latent_cache_dir=robot_rgb_latent_cache_dir,
            camera_layout=list(ROBOT_CAMERAS), target_camera=ROBOT_CAMERAS[0],
            head_camera_priority=[ROBOT_CAMERAS[0]], wrist_camera_priority=[ROBOT_CAMERAS[1]])
        self._robot_dataset = CachedLiberoDataset(*args, **robot_kwargs)
        other = self._robot_dataset
        if tuple(other._camera_layout) != ROBOT_CAMERAS:
            raise ValueError('Robot RGB streams missing; scene/wrist fallback is forbidden')
        if self._dataset_dir.resolve() != other._dataset_dir.resolve() or len(self) != len(other):
            raise ValueError('Paired RGB streams must use the same dataset and window count')
        for name in ('_cum_n_starts', '_ep_valid_start', '_ep_valid_end', '_video_sample_indices'):
            if not np.array_equal(getattr(self,name), getattr(other,name)):
                raise ValueError('Scene/robot temporal window mismatch: '+name)
        a,b = self._latent_cache.manifest,other._latent_cache.manifest
        if a['vae_weights_sha256'] != b['vae_weights_sha256']:
            raise ValueError('Scene/robot VAE weights differ')
        if a['dataset_binding']['episode_window_index_sha256'] != b['dataset_binding']['episode_window_index_sha256']:
            raise ValueError('Scene/robot cache window indices differ')

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        sample['robot_video_latents'] = self._robot_dataset._latent_cache[index]
        return sample
