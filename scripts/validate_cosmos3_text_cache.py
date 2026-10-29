"""Validate complete coverage, file integrity, and full-context numerical equality."""
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch

from openwam.dataloader.cosmos3_droid import prompt_variants
from openwam.model.video_backbone.wan import encode
from openwam.model.video_backbone.wan.text_cache import FrozenTextCache
from precompute_cosmos3_text import load_encoder, atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True)
    parser.add_argument('--prepared', required=True)
    args = parser.parse_args()
    os.environ.pop('OPENWAM_TEXT_CACHE_DIR', None)
    cache = FrozenTextCache(args.cache)
    cache.validate_checkpoint(cache.manifest['checkpoint'])
    rows = pq.read_table(Path(args.prepared) / 'episodes.parquet', columns=['tasks', 'training_split']).to_pylist()
    prompts = sorted({p for r in rows if r['training_split'] in ('train', 'val')
                      for text in r['tasks'] for p in prompt_variants(text)})
    assert set(prompts) == set(cache.index)
    checksums = {}
    for shard in cache.manifest['shards']:
        h = hashlib.sha256()
        with (cache.root / shard['filename']).open('rb') as f:
            while chunk := f.read(16 << 20):
                h.update(chunk)
        checksums[shard['filename']] = h.hexdigest()
    encoder, tokenizer, _ = load_encoder(Path(cache.manifest['checkpoint']).parent, 'cuda:0')
    rng = random.Random(178)
    selected = rng.sample(prompts, 128)
    # The runtime loader must reconstruct exact 512-wide contexts, including
    # untouched BF16 bits and all-zero padding, for reordered instruction batches.
    max_abs = 0.0
    fresh_seconds, cache_seconds = [], []
    with torch.no_grad():
        for start in range(0, 128, 16):
            batch = selected[start:start+16]
            torch.cuda.synchronize(); begin = time.monotonic()
            fresh, seq = encode.encode_text(batch, tokenizer=tokenizer, text_encoder=encoder, device='cuda:0')
            torch.cuda.synchronize(); fresh_seconds.append(time.monotonic()-begin)
            begin = time.monotonic()
            cached, cached_seq = cache.encode(batch, text_encoder=encoder, device='cuda:0')
            torch.cuda.synchronize(); cache_seconds.append(time.monotonic()-begin)
            assert torch.equal(seq, cached_seq)
            diff = float((fresh.float()-cached.float()).abs().max())
            max_abs = max(max_abs, diff)
            torch.testing.assert_close(fresh, cached, rtol=0, atol=0)
    result = dict(status='passed', index_sha256=cache.manifest['index_sha256'], coverage=len(prompts),
                  checked_prompts=128, max_absolute_difference=max_abs, equality='bitwise BF16, full 512-token context',
                  fresh_seconds=fresh_seconds, cached_seconds=cache_seconds, file_sha256=checksums)
    atomic_json(cache.root / 'validation.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
