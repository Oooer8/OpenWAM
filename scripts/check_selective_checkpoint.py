"""Check selective checkpoint numerical parity and compiled-FFN compatibility."""
import copy
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint, create_selective_checkpoint_contexts

from openwam.model.architectures.utils.selective_checkpoint import matmul_policy
from openwam.model.architectures.utils.checkpoint_scope import checkpoint_ffn_scope
from openwam.train.utils.runtime_optimizations import compile_training_ffns
from check_training_optimizations import error


def main():
    torch.set_num_threads(2)
    torch.manual_seed(732)
    results = []
    for dim, hidden, tokens in ((3072, 14336, 360), (1024, 4096, 32)):
        ffn = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate='tanh'),
                            nn.Linear(hidden, dim)).cuda().bfloat16()
        compile_training_ffns((SimpleNamespace(blocks=[SimpleNamespace(ffn=ffn)]),))
        proj = nn.Linear(dim, dim).cuda().bfloat16()
        modules = nn.ModuleList([proj, ffn])
        x = torch.randn(16, tokens, dim, device='cuda', dtype=torch.bfloat16)
        weight = torch.randn_like(x) / 1000
        counts = Counter()

        def policy(ctx, op, *args, **kwargs):
            counts[(str(op), ctx.is_recompute)] += 1
            return matmul_policy(ctx, op, *args, **kwargs)

        def contexts():
            return create_selective_checkpoint_contexts(policy)

        def fn(z):
            z = z + proj(z).tanh()
            return z + ffn(z)

        runs = []
        for mode in ('block', 'selective', 'ffn_only'):
            modules.zero_grad(set_to_none=True)
            z = x.detach().clone().requires_grad_(True)
            options = {'context_fn': contexts} if mode == 'selective' else {}
            if mode == 'ffn_only':
                with checkpoint_ffn_scope():
                    y = fn(z)
            else:
                y = checkpoint(fn, z, use_reentrant=False, **options)
            (y.float() * weight).sum().backward()
            runs.append((y.detach(), z.grad.detach(),
                         *[p.grad.detach().clone() for p in modules.parameters()]))
        metrics = [error(a, b) for b_run in runs[1:] for a, b in zip(runs[0], b_run)]
        for e in metrics:
            assert e['relative_l2'] < .001 and e['cosine'] > .99999, e
        selected = {str(k): v for k, v in counts.items() if 'mm.' in k[0]}
        assert selected, counts
        results.append({'shape': list(x.shape), 'errors': metrics,
                        'matrix_operation_policy_calls': selected})
        print(json.dumps(results[-1]), flush=True)
        del modules, ffn, proj, x, z, y, runs, weight
        torch.cuda.empty_cache()
    result = {'status': 'passed', 'checks': results}
    Path('work/setup_logs/selective_checkpoint_check.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
