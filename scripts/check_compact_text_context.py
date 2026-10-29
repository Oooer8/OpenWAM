"""Check masked context compaction using actual checkpoint cross-attention weights."""
import copy
import json
import random
from pathlib import Path

import torch
from torch import nn
from safetensors import safe_open
from openwam.model.video_backbone.wan.models.dit import CrossAttention
from openwam.model.action_backbone.separate_action_dit import BridgeCrossAttention
from openwam.model.video_backbone.wan.encode import compact_masked_text_context
from openwam.model.video_backbone.wan.text_cache import FrozenTextCache
from check_training_optimizations import error


def load_part(handle, module, prefix):
    state = {k[len(prefix):]: handle.get_tensor(k) for k in handle.keys() if k.startswith(prefix)}
    module.load_state_dict(state, strict=True, assign=True)
    return module.cuda().bfloat16()


def main():
    torch.set_num_threads(2)
    torch.manual_seed(32)
    cache = FrozenTextCache('work/cosmos3_text_cache')
    records = sorted(cache.index, key=lambda p: cache.index[p][2])
    rng = random.Random(91)
    prompt_sets = [rng.sample([p for p in records if cache.index[p][2] <= 32], 16), records[-16:]]
    dummy = nn.Linear(1,1).eval().requires_grad_(False)
    result = []
    with safe_open(cache.manifest['checkpoint'], framework='pt', device='cpu') as handle:
        for backbone, dim, tokens in [('video_backbone.dit', 3072, 360), ('action_backbone', 1024, 32)]:
            for layer in (0, 14, 29):
                embedding = nn.Sequential(nn.Linear(4096,dim), nn.GELU(approximate='tanh'), nn.Linear(dim,dim))
                embedding = load_part(handle, embedding, backbone+'.text_embedding.')
                attention = CrossAttention(dim,24) if dim == 3072 else BridgeCrossAttention(dim,24,128)
                attention = load_part(handle, attention, f'{backbone}.blocks.{layer}.cross_attn.')
                proprio_encoder = load_part(handle, nn.Linear(80,4096), 'proprio_encoder.')
                modules = nn.ModuleList([embedding, attention, proprio_encoder])
                for prompts in prompt_sets:
                    context, lengths = cache.encode(prompts, text_encoder=dummy, device='cuda')
                    direct, direct_lengths = cache.encode(prompts, text_encoder=dummy, device='cuda', compact=True)
                    torch.testing.assert_close(direct, compact_masked_text_context(context, lengths), rtol=0, atol=0)
                    assert torch.equal(lengths, direct_lengths)
                    proprio = torch.randn(16,80,device='cuda',dtype=torch.bfloat16)
                    query = torch.randn(16,tokens,dim,device='cuda',dtype=torch.bfloat16)
                    weight = torch.randn_like(query)/1000
                    runs = []
                    for compact in (False, True):
                        modules.zero_grad(set_to_none=True)
                        raw = context.detach().clone().requires_grad_(True)
                        q = query.detach().clone().requires_grad_(True)
                        ctx = compact_masked_text_context(raw,lengths) if compact else raw
                        mask = torch.arange(ctx.shape[1],device='cuda')[None,:] < lengths[:,None]
                        ctx = torch.cat([ctx,proprio_encoder(proprio)[:,None]],dim=1)
                        mask = torch.cat([mask,torch.ones((16,1),device='cuda',dtype=torch.bool)],dim=1)
                        output = attention(q,embedding(ctx),ctx_mask=mask[:,None,None,:])
                        (output.float()*weight).sum().backward()
                        runs.append((output.detach(), q.grad.detach(), raw.grad.detach(),
                                     [p.grad.detach().clone() for p in modules.parameters()]))
                    a,b = runs
                    metrics = {'backbone':backbone, 'layer':layer,'full_context_width':513,
                               'compact_context_width':ctx.shape[1], 'output':error(a[0],b[0]),
                               'query_gradient':error(a[1],b[1]), 'context_gradient':error(a[2],b[2]),
                               'parameter_gradients':[error(p,q) for p,q in zip(a[3],b[3])]}
                    for e in [metrics['output'],metrics['query_gradient'],metrics['context_gradient'],*metrics['parameter_gradients']]:
                        assert e['relative_l2'] < .02 and e['cosine'] > .9995, (metrics,e)
                    result.append(metrics)
                    print(json.dumps(metrics),flush=True)
                del modules,embedding,attention,proprio_encoder,runs,a,b
                torch.cuda.empty_cache()
    Path('work/setup_logs/compact_text_context_validation.json').write_text(json.dumps({'status':'passed','checks':result},indent=2))
    print('COMPACT_CONTEXT_CHECK_PASSED',flush=True)


if __name__ == '__main__':
    main()
