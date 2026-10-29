"""Encode all prepared train/val instruction variants with the frozen original T5.

Run with torchrun on eight GPUs. Full 512-token input and batch 16 match
training. Store only valid output tokens; padding is restored as exact zeros.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import yaml
from safetensors import safe_open

from openwam.dataloader.cosmos3_droid import prompt_variants
from openwam.model.video_backbone.wan import encode
from openwam.model.video_backbone.wan.pipeline_builder import _build_tokenizer, _import_class


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False))
    tmp.replace(path)


def load_encoder(root, device):
    cfg = yaml.safe_load((root / 'config.yaml').read_text())['model']['video_backbone']
    spec = next(c for c in cfg['components'] if c['attr'] == 'text_encoder')
    with torch.device('meta'):
        encoder = _import_class(spec['model_class'])(**spec.get('extra_kwargs', {}))
    weights = root / 'checkpoint_step_154000.safetensors'
    prefix = 'video_backbone.text_encoder.'
    with safe_open(weights, framework='pt', device='cpu') as f:
        state = {k[len(prefix):]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}
    encoder.load_state_dict(state, strict=True, assign=True)
    encoder.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    return encoder, _build_tokenizer(cfg['tokenizer'], str(root)), weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--prepared', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('gloo')
    root, output = Path(args.checkpoint).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.pop('OPENWAM_TEXT_CACHE_DIR', None)
    if (output / 'manifest.json').exists():
        raise RuntimeError('Output already has a completed cache; use it or select another directory')
    rows = pq.read_table(Path(args.prepared) / 'episodes.parquet', columns=['tasks', 'training_split']).to_pylist()
    prompts = sorted({p for r in rows if r['training_split'] in ('train', 'val')
                      for text in r['tasks'] for p in prompt_variants(text)})
    prompt_digest = hashlib.sha256(json.dumps(prompts, ensure_ascii=False).encode()).hexdigest()
    assigned = prompts[rank::world]
    encoder, tokenizer, weights = load_encoder(root, device)
    lengths = []
    for start in range(0, len(assigned), 1024):
        _, mask = tokenizer(assigned[start:start+1024], return_mask=True, add_special_tokens=True,
                            max_length=512, padding='max_length', truncation=True)
        lengths.extend(mask.gt(0).sum(1).tolist())
    hidden = 4096
    tokens = sum(lengths)
    filename = f'embeddings_rank{rank:02d}.bin'
    tmp = output / (filename + '.tmp')
    blob = np.memmap(tmp, dtype=np.uint16, mode='w+', shape=(tokens, hidden))
    index = {}
    offset = 0
    started = time.monotonic()
    print(json.dumps(dict(rank=rank, total_prompts=len(prompts), assigned=len(assigned), tokens=tokens,
                          bytes=tokens*hidden*2)), flush=True)
    with torch.no_grad():
        for start in range(0, len(assigned), 16):
            batch = assigned[start:start+16]
            n = len(batch)
            padded = batch + [batch[-1]]*(16-n)
            context, seq = encode.encode_text(padded, tokenizer=tokenizer, text_encoder=encoder, device=device)
            cpu = context[:n].cpu().view(torch.uint16).numpy()
            for j, prompt in enumerate(batch):
                length = lengths[start+j]
                if length != int(seq[j]):
                    raise ValueError('Token length changed between planning and encoding')
                blob[offset:offset+length] = cpu[j, :length]
                index[prompt] = [rank, offset, length]
                offset += length
            if start % 1600 == 0 or start+n == len(assigned):
                elapsed = time.monotonic()-started
                done = start+n
                progress = dict(rank=rank, done=done, total=len(assigned), elapsed_seconds=elapsed,
                                eta_seconds=elapsed/max(done,1)*(len(assigned)-done))
                atomic_json(output / f'progress_rank{rank}.json', progress)
                print(json.dumps(progress), flush=True)
    if offset != tokens:
        raise ValueError('Cache token count mismatch')
    blob.flush()
    del blob
    tmp.replace(output / filename)
    atomic_json(output / f'index_rank{rank}.json', index)
    atomic_json(output / f'shard_rank{rank}.json', dict(filename=filename, tokens=tokens, bytes=tokens*hidden*2))
    del encoder
    torch.cuda.empty_cache()
    if world > 1:
        dist.barrier()
    if rank == 0:
        merged = {}
        shards = []
        for i in range(world):
            local = json.loads((output / f'index_rank{i}.json').read_text())
            if merged.keys() & local.keys():
                raise ValueError('Duplicate prompt assignments')
            merged.update(local)
            shards.append(json.loads((output / f'shard_rank{i}.json').read_text()))
        if set(merged) != set(prompts):
            raise ValueError('Missing prompt coverage')
        atomic_json(output / 'index.json', merged)
        stat = weights.stat()
        manifest = dict(status='ready', format=1, dtype='bfloat16', max_length=512, hidden_dim=hidden,
                        batch_size=16, num_prompts=len(prompts), prompts_sha256=prompt_digest,
                        checkpoint=str(weights.resolve()), checkpoint_size=stat.st_size,
                        checkpoint_mtime_ns=stat.st_mtime_ns,
                        foundation_revision='52df4e66c82c5c8b480adcc8d01f4db7415dfb56',
                        index_sha256=hashlib.sha256((output/'index.json').read_bytes()).hexdigest(),
                        shards=shards, total_bytes=sum(s['bytes'] for s in shards),
                        semantics='Full 512-token frozen eval T5, unchanged BF16 values; valid tokens only stored')
        atomic_json(output / 'manifest.json', manifest)
        print('TEXT_CACHE_COMPLETE', json.dumps(manifest), flush=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
