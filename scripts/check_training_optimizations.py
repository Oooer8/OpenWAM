"""Compare pixel values and compiled FFN forward/backward with eager execution."""
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from openwam.train.utils.runtime_optimizations import compile_training_ffns
from types import SimpleNamespace
from openwam.model.video_backbone.wan.encode import preprocess_video, preprocess_video_batch


def error(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return dict(max_abs=float((a-b).abs().max()), relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-8)),
                cosine=float(torch.nn.functional.cosine_similarity(a, b, dim=0)))


def main():
    torch.set_num_threads(2)
    rng = np.random.default_rng(34)
    frames = []
    for _ in range(16):
        clip = []
        for t in range(9):
            arr = rng.integers(0, 256, (384, 320, 3), dtype=np.uint8)
            arr[192:, 160:] = 0
            if t == 0:
                arr[:256, :256] = np.arange(256, dtype=np.uint8)[None, :, None]
            clip.append(Image.fromarray(arr))
        frames.append(clip)
    result = {'pixel': {}, 'ffn': []}
    for device in ('cpu', 'cuda:0'):
        old = torch.cat([preprocess_video(v, dtype=torch.bfloat16, device=device) for v in frames])
        new = preprocess_video_batch(frames, dtype=torch.bfloat16, device=device)
        torch.testing.assert_close(old, new, rtol=0, atol=0)
        result['pixel'][device] = {'shape': list(new.shape), 'equality': 'bitwise BF16'}
        del old, new
    for dim, hidden, tokens in ((3072, 14336, 360), (1024, 4096, 33)):
        torch.manual_seed(59)
        eager = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate='tanh'), nn.Linear(hidden, dim)).cuda().bfloat16()
        compiled = copy.deepcopy(eager)
        keys = list(compiled.state_dict())
        compile_training_ffns((SimpleNamespace(blocks=[SimpleNamespace(ffn=compiled)]),))
        assert keys == list(compiled.state_dict())
        x = torch.randn(16, tokens, dim, device='cuda', dtype=torch.bfloat16, requires_grad=True)
        z = x.detach().clone().requires_grad_(True)
        weight = torch.randn_like(x) / 1000
        begin = time.monotonic()
        a = checkpoint(eager, x, use_reentrant=False)
        b = checkpoint(compiled, z, use_reentrant=False)
        (a.float() * weight).sum().backward()
        (b.float() * weight).sum().backward()
        torch.cuda.synchronize()
        metrics = {'shape': [16, tokens, dim], 'forward': error(a, b), 'input_gradient': error(x.grad, z.grad),
                   'parameter_gradients': [error(p.grad, q.grad) for p,q in zip(eager.parameters(), compiled.parameters())],
                   'first_compile_and_backward_seconds': time.monotonic()-begin, 'checkpoint_keys_unchanged': True}
        for e in [metrics['forward'], metrics['input_gradient'], *metrics['parameter_gradients']]:
            assert e['relative_l2'] < .015 and e['cosine'] > .9998, e
        result['ffn'].append(metrics)
        print(json.dumps(metrics), flush=True)
        del eager, compiled, x, z, a, b, weight
        torch.cuda.empty_cache()
    # More than the default 8-layer graph limit, with distinct module identities.
    # Reusing this tensor-only function must compile only the two shape variants.
    blocks = [SimpleNamespace(ffn=nn.Sequential(nn.Linear(d,h), nn.GELU(approximate='tanh'), nn.Linear(h,d)).cuda().bfloat16())
              for d,h in [(64,128),(32,64)] * 30]
    compile_training_ffns((SimpleNamespace(blocks=blocks),))
    for i, block in enumerate(blocks):
        d = block.ffn[0].in_features
        x = torch.randn(2,8,d,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        y = checkpoint(block.ffn, x, use_reentrant=False) if i%2 == 0 else block.ffn(x)
        y.float().square().mean().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.ffn.parameters())
    result['multi_layer_compile_check'] = {'distinct_layers':60,'status':'passed'}
    result['status'] = 'passed'
    Path('work/setup_logs/training_optimization_numerics.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
