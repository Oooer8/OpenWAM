"""Measure deterministic cuDNN algorithm selection on the frozen native VAE."""
import json
import statistics
import time
from pathlib import Path

import torch
from safetensors import safe_open
from openwam.model.video_backbone.wan.models.vae import WanVideoVAE38


def main():
    torch.set_num_threads(2)
    torch.manual_seed(234)
    vae=WanVideoVAE38()
    weights='assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model/checkpoint_step_154000.safetensors'
    prefix='video_backbone.vae.'
    with safe_open(weights,framework='pt',device='cpu') as h:
        state={k[len(prefix):]:h.get_tensor(k) for k in h.keys() if k.startswith(prefix)}
    vae.load_state_dict(state,strict=True,assign=True)
    del state
    vae.cuda().bfloat16().eval().requires_grad_(False)
    pixels=torch.rand(16,3,9,384,320,device='cuda',dtype=torch.bfloat16)*2-1
    torch.backends.cudnn.deterministic=True
    result={'cudnn_version':torch.backends.cudnn.version(),'deterministic':True,'modes':{}}
    outputs={}
    for name,benchmark in [('original',False),('autotuned',True),('original_recheck',False)]:
        torch.backends.cudnn.benchmark=benchmark
        if benchmark:
            torch.backends.cudnn.benchmark_limit=4
        timings=[]
        with torch.no_grad():
            for _ in range(8):
                torch.cuda.synchronize();begin=time.monotonic()
                out=vae.batch_encode(pixels,device='cuda')
                torch.cuda.synchronize();timings.append(time.monotonic()-begin)
        outputs[name]=out.detach().float().cpu()
        result['modes'][name]={'seconds':timings,'steady_median_seconds':statistics.median(timings[2:]),
                               'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
        print(name,json.dumps(result['modes'][name]),flush=True)
    a,b=outputs['original'],outputs['autotuned']
    result['numerics']={'max_abs':float((a-b).abs().max()),'relative_l2':float((a-b).norm()/a.norm()),
                       'cosine':float(torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0))}
    assert result['numerics']['relative_l2'] < .01 and result['numerics']['cosine'] > .9999,result['numerics']
    result['status']='passed'
    Path('work/setup_logs/vae_autotune_check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
