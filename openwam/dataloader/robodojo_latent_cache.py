"""Lossless BF16 window cache, published only after coverage/integrity validation."""
import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

SHAPE = (48, 3, 24, 20)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for data in iter(lambda: f.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def state_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous().to(torch.bfloat16)
        h.update(name.encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(value.view(torch.uint16).numpy().tobytes())
    return h.hexdigest()


class LatentCache:
    def __init__(self, root, prepared_root, length):
        self.root = Path(root)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        m = self.manifest
        if (m.get('status'), m.get('format'), m.get('dtype'), m.get('shape'), m.get('windows')) != (
                'ready', 1, 'bfloat16_uint16_bits', list(SHAPE), length):
            raise ValueError('Incomplete or incompatible VAE cache')
        if m.get('augmentation') != 'none' or m.get('encoding') != 'independent_9_frame_windows':
            raise ValueError('Cache augmentation/temporal semantics mismatch')
        if m['prepared_manifest_sha256'] != sha256_file(Path(prepared_root) / 'manifest.json'):
            raise ValueError('VAE cache belongs to a different prepared dataset')
        receipt = json.loads((self.root / 'validation.json').read_text())
        if receipt.get('status') != 'passed' or receipt.get('manifest_sha256') != sha256_file(self.root / 'manifest.json'):
            raise ValueError('VAE cache is not validated')
        cursor = 0
        for shard in m['shards']:
            if shard['start'] != cursor or shard['end'] <= cursor:
                raise ValueError('Gap/overlap in latent window coverage')
            path = self.root / shard['path']
            expected = (shard['end']-cursor) * int(np.prod(SHAPE)) * 2
            if path.stat().st_size != expected or shard['size'] != expected:
                raise ValueError(f'Truncated latent shard: {path}')
            cursor = shard['end']
        if cursor != length:
            raise ValueError('Incomplete latent window coverage')
        self._ends = np.array([x['end'] for x in m['shards']])
        self._maps = OrderedDict()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_maps'] = OrderedDict()
        return state

    def __getitem__(self, index):
        if not 0 <= index < self.manifest['windows']:
            raise IndexError(index)
        sid = int(np.searchsorted(self._ends, index, side='right'))
        shard = self.manifest['shards'][sid]
        if sid not in self._maps:
            self._maps[sid] = np.memmap(self.root / shard['path'], dtype=np.uint16, mode='r',
                                      shape=(shard['end']-shard['start'], *SHAPE))
            if len(self._maps) > 32:
                self._maps.popitem(last=False)
        self._maps.move_to_end(sid)
        bits = np.array(self._maps[sid][index-shard['start']], copy=True)
        return torch.from_numpy(bits).view(torch.bfloat16)
