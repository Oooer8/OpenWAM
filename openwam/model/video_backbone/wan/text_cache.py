"""Read compact BF16 embeddings from a completed, checkpoint-bound T5 cache."""
from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch


class FrozenTextCache:
    def __init__(self, directory):
        self.root = Path(directory)
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        m = self.manifest
        if (m.get('status'), m.get('format'), m.get('dtype'), m.get('max_length')) != ('ready', 1, 'bfloat16', 512):
            raise ValueError('Incomplete or incompatible frozen-text cache')
        index_bytes = (self.root / 'index.json').read_bytes()
        if hashlib.sha256(index_bytes).hexdigest() != m['index_sha256']:
            raise ValueError('Text-cache index checksum mismatch')
        self.index = json.loads(index_bytes)
        self.hidden = int(m['hidden_dim'])
        self.maps = []
        for shard in m['shards']:
            path = self.root / shard['filename']
            if path.stat().st_size != shard['bytes']:
                raise ValueError(f'Truncated text cache: {path}')
            self.maps.append(np.memmap(path, dtype=np.uint16, mode='r', shape=(shard['tokens'], self.hidden)))
        if len(self.index) != m['num_prompts']:
            raise ValueError('Text-cache prompt count mismatch')

    def validate_checkpoint(self, checkpoint):
        p = Path(checkpoint).resolve()
        expected = Path(self.manifest['checkpoint']).resolve()
        if p.is_dir():
            p = p / expected.name
        if p != expected:
            raise ValueError(f'Text cache was generated from {expected}, not {p}')
        stat = p.stat()
        if (stat.st_size, stat.st_mtime_ns) != (self.manifest['checkpoint_size'], self.manifest['checkpoint_mtime_ns']):
            raise ValueError('Foundation checkpoint changed after text-cache generation')

    def encode(self, prompts, *, text_encoder, device, compact=False):
        if text_encoder.training or any(p.requires_grad for p in text_encoder.parameters()):
            raise ValueError('Text caching requires a frozen eval-mode encoder')
        if not prompts:
            raise ValueError('Cannot encode an empty prompt batch')
        # Missing prompts fail explicitly: silently computing a different path
        # could conceal incomplete coverage or stale prompt-cleaning rules.
        records = [self.index[p] for p in prompts]
        lengths = [r[2] for r in records]
        width = min(512, max(32, ((max(lengths) + 31) // 32) * 32)) if compact else 512
        context = torch.zeros((len(prompts), width, self.hidden), dtype=torch.bfloat16)
        for i, (shard, offset, length) in enumerate(records):
            context[i, :length].view(torch.uint16).numpy()[:] = self.maps[shard][offset:offset + length]
        return context.to(device), torch.tensor(lengths, dtype=torch.long, device=device)


@functools.lru_cache(maxsize=1)
def get_text_cache(directory):
    return FrozenTextCache(directory)


def configured_text_cache():
    path = os.environ.get('OPENWAM_TEXT_CACHE_DIR')
    return None if not path else get_text_cache(str(Path(path).resolve()))
