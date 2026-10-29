"""Validated per-window LIBERO cache; actions and masks stay canonical."""
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from openwam.dataloader.libero import LiberoDataset
from openwam.dataloader.robodojo_latent_cache import SHAPE, sha256_file, state_digest
from openwam.dataloader.transforms.video import color_jitter_enabled


def dataset_binding(ds, *, hashes=True):
    root = ds._dataset_dir.resolve()
    paths = sorted(p for folder in ('meta', 'data', 'videos') for p in (root / folder).rglob('*') if p.is_file())
    files = []
    for p in paths:
        item = dict(path=str(p.relative_to(root)), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
        if hashes:
            item['sha256'] = sha256_file(p)
        files.append(item)
    return dict(root=str(root), files=files, windows=len(ds),
                settings=dict(num_frames=ds._num_frames, video_stride=ds._video_stride,
                    window_stride=ds._window_stride, height=ds._height, width=ds._width,
                    multiview=ds._multiview, camera_layout=list(ds._camera_layout), split=ds._split),
                episode_window_index_sha256=__import__('hashlib').sha256(
                    b''.join(np.asarray(x, dtype=np.int64).tobytes() for x in
                        (ds._cum_n_starts, ds._ep_valid_start, ds._ep_valid_end))).hexdigest())


def validate_binding(ds, binding):
    current = dataset_binding(ds, hashes=False)
    expected = dict(binding, files=[{k: v for k, v in p.items() if k != 'sha256'} for p in binding['files']])
    if current != expected:
        raise ValueError('LIBERO cache dataset files or sampling settings changed')
    # All videos are hashed at cache construction. Rehash small metadata on load;
    # file size/mtime also bind every immutable copied video and data shard.
    for item in binding['files']:
        if item['path'].startswith('meta/') and sha256_file(ds._dataset_dir / item['path']) != item['sha256']:
            raise ValueError('LIBERO metadata checksum changed: ' + item['path'])


class LiberoLatentCache:
    def __init__(self, root, ds):
        self.root = Path(root)
        self.manifest = m = json.loads((self.root / 'manifest.json').read_text())
        if (m.get('status'), m.get('format'), m.get('dtype'), m.get('shape'), m.get('windows')) != (
                'ready', 'libero_window_v1', 'bfloat16_uint16_bits', list(SHAPE), len(ds)):
            raise ValueError('Incomplete or incompatible LIBERO cache')
        if m.get('augmentation') != 'none' or m.get('encoding') != 'independent_9_frame_windows':
            raise ValueError('LIBERO cache temporal/augmentation mismatch')
        validate_binding(ds, m['dataset_binding'])
        receipt = json.loads((self.root / 'validation.json').read_text())
        if receipt.get('status') != 'passed' or receipt.get('manifest_sha256') != sha256_file(self.root / 'manifest.json'):
            raise ValueError('LIBERO cache validation receipt mismatch')
        cursor = 0
        for shard in m['shards']:
            expected = (shard['end'] - shard['start']) * int(np.prod(SHAPE)) * 2
            if shard['start'] != cursor or shard['end'] <= cursor or shard['size'] != expected:
                raise ValueError('LIBERO cache coverage gap/overlap')
            if (self.root / shard['path']).stat().st_size != expected:
                raise ValueError('Truncated LIBERO latent shard')
            cursor = shard['end']
        if cursor != len(ds):
            raise ValueError('Incomplete LIBERO cache coverage')
        self._ends = np.array([s['end'] for s in m['shards']])
        self._maps = OrderedDict()

    def __getstate__(self):
        return dict(self.__dict__, _maps=OrderedDict())

    def __getitem__(self, index):
        if not 0 <= index < self.manifest['windows']:
            raise IndexError(index)
        sid = int(np.searchsorted(self._ends, index, side='right'))
        shard = self.manifest['shards'][sid]
        if sid not in self._maps:
            self._maps[sid] = np.memmap(self.root / shard['path'], mode='r', dtype=np.uint16,
                shape=(shard['end'] - shard['start'], *SHAPE))
            if len(self._maps) > 16:
                self._maps.popitem(last=False)
        self._maps.move_to_end(sid)
        return torch.from_numpy(np.array(self._maps[sid][index-shard['start']], copy=True)).view(torch.bfloat16)


class CachedLiberoDataset(LiberoDataset):
    CONFIG_KEYS = LiberoDataset.CONFIG_KEYS + ('libero_latent_cache_dir',)

    def __init__(self, *args, libero_latent_cache_dir=None, **kwargs):
        if color_jitter_enabled(kwargs.get('color_jitter')) or kwargs.get('head_camera_choices'):
            raise ValueError('LIBERO cache requires deterministic cameras and disabled augmentation')
        super().__init__(*args, **kwargs)
        if not libero_latent_cache_dir:
            raise ValueError('Missing libero_latent_cache_dir')
        self._latent_cache = LiberoLatentCache(libero_latent_cache_dir, self)

    def _decode_window_video(self, *args, **kwargs):
        return []

    def __getitem__(self, index):
        # Bypass retry/skip: a cached index must never be paired with another
        # window's action, mask or language after a recoverable reader error.
        if not 0 <= index < len(self):
            raise IndexError(index)
        sample = self._getitem_impl(index)
        sample['video_latents'] = self._latent_cache[index]
        sample['first_frame_image'] = None
        return sample


def validate_training_config(cfg):
    from omegaconf import OmegaConf, open_dict
    import os
    root = OmegaConf.select(cfg, 'dataloader.libero_latent_cache_dir')
    os.environ.pop('OPENWAM_VALIDATED_LIBERO_LATENT_CACHE', None)
    if not root:
        return
    if cfg.training.mixed_precision != 'bf16' or cfg.dataloader.type != 'libero_cached':
        raise ValueError('LIBERO cache requires BF16 and type=libero_cached')
    ds = CachedLiberoDataset.from_config(cfg.dataloader, split='train')
    if int(cfg.training.batch_size) != ds._latent_cache.manifest['batch_size']:
        raise ValueError('Training microbatch must match the native VAE cache microbatch')
    if ds._latent_cache.manifest.get('execution_mode') != 'native_training_vae':
        raise ValueError('LIBERO requires the native training VAE cache requested by the user')
    os.environ['OPENWAM_VALIDATED_LIBERO_LATENT_CACHE'] = str(Path(root).resolve())
    with open_dict(cfg.project):
        cfg.project.vae_cache_directory = str(root)
        cfg.project.vae_cache_manifest_sha256 = sha256_file(Path(root) / 'manifest.json')
        cfg.project.vae_cache_weights_sha256 = ds._latent_cache.manifest['vae_weights_sha256']


def apply_training_cache(architecture):
    import os
    root = os.environ.get('OPENWAM_VALIDATED_LIBERO_LATENT_CACHE')
    if not root:
        return
    m = json.loads((Path(root) / 'manifest.json').read_text())
    repo = Path(__file__).resolve().parents[2]
    for name, digest in m['encoding_source_sha256'].items():
        if sha256_file(repo / name) != digest:
            raise ValueError('LIBERO cache encoder source changed: ' + name)
    vae = architecture.video_backbone.vae
    if vae.training or any(p.requires_grad for p in vae.parameters()):
        raise ValueError('LIBERO cached VAE must be frozen/eval')
    if state_digest(vae.state_dict()) != m['vae_weights_sha256']:
        raise ValueError('LIBERO cached and loaded VAE weights differ')
    # The existing native TI2V cached preprocessing contract is shared: exact
    # same 9-frame canvas, latent shape and clean first-frame conditioning.
    architecture.video_backbone._robodojo_vae_cache_validated = True
    vae.to(device='cpu')
    torch.cuda.empty_cache()
