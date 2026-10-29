"""Cache all frozen T5 instructions and verify cached vs live batch-4 output."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from precompute_cosmos3_text import load_encoder
from prepare_robodojo_openwam import atomic_json
from openwam.model.video_backbone.wan import encode
from openwam.model.video_backbone.wan.text_cache import FrozenTextCache


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", required=True)
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args()
    out = Path(args.prepared) / "text_cache"
    out.mkdir(exist_ok=True)
    os.environ.pop("OPENWAM_TEXT_CACHE_DIR", None)
    episodes = json.loads((Path(args.prepared) / "episodes.json").read_text())
    prompts = sorted({e["prompt"] for e in episodes})
    encoder, tokenizer, weights = load_encoder(Path(args.checkpoint), "cuda:0")
    values, index = [], {}
    offset = 0
    with torch.no_grad():
        for start in range(0, len(prompts), 4):
            batch = prompts[start:start+4]
            batch += [batch[-1]] * (4-len(batch))
            context, lengths = encode.encode_text(batch, tokenizer=tokenizer, text_encoder=encoder, device="cuda:0")
            for j, prompt in enumerate(prompts[start:start+4]):
                length = int(lengths[j])
                values.append(context[j, :length].cpu().view(torch.uint16).numpy().copy())
                index[prompt] = [0, offset, length]
                offset += length
    matrix = np.concatenate(values)
    tmp = out / "embeddings_rank00.bin.tmp"
    matrix.tofile(tmp)
    tmp.replace(out / "embeddings_rank00.bin")
    atomic_json(out / "index.json", index)
    stat = weights.stat()
    manifest = dict(status="ready", format=1, dtype="bfloat16", max_length=512, hidden_dim=4096,
                    num_prompts=len(prompts), checkpoint=str(weights.resolve()),
                    checkpoint_size=stat.st_size, checkpoint_mtime_ns=stat.st_mtime_ns,
                    index_sha256=hashlib.sha256((out / "index.json").read_bytes()).hexdigest(),
                    shards=[dict(filename="embeddings_rank00.bin", tokens=offset, bytes=matrix.nbytes)],
                    semantics="Frozen eval T5 full 512-token context; batch 4; valid tokens stored as original BF16 bits")
    atomic_json(out / "manifest.json", manifest)
    cache = FrozenTextCache(out)
    cache.validate_checkpoint(weights)
    max_error = 0.
    with torch.no_grad():
        for start in range(0, len(prompts), 4):
            batch = prompts[start:start+4]
            batch += [batch[-1]] * (4-len(batch))
            online, lengths = encode.encode_text(batch, tokenizer=tokenizer, text_encoder=encoder, device="cuda:0")
            cached, cached_lengths = cache.encode(batch, text_encoder=encoder, device="cuda:0")
            torch.testing.assert_close(cached_lengths, lengths, rtol=0, atol=0)
            error = (online.float()-cached.float()).abs().max().item()
            max_error = max(max_error, error)
            torch.testing.assert_close(cached, online, rtol=0, atol=0)
    result = dict(status="passed", prompts=len(prompts), bytes=matrix.nbytes, max_abs_error=max_error,
                  index_sha256=manifest["index_sha256"],
                  comparison="all prompts, full 512-token context, batch 4; bitwise equality")
    atomic_json(out / "validation.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
