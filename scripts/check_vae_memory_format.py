"""Measure native frozen-VAE channels-last execution without changing inputs."""
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    vae = WanVideoVAE38()
    weights = 'assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model/checkpoint_step_154000.safetensors'
    prefix = 'video_backbone.vae.'
    with safe_open(weights, framework='pt', device='cpu') as h:
        state = {k[len(prefix):]: h.get_tensor(k) for k in h.keys() if k.startswith(prefix)}
    vae.load_state_dict(state, strict=True, assign=True)
    del state
    vae.cuda().bfloat16().eval().requires_grad_(False)
    pixels = torch.rand(16, 3, 9, 384, 320, device='cuda', dtype=torch.bfloat16) * 2 - 1
    outputs = {}
    result = {'modes': {}, 'deterministic': True}
    for name in ('baseline', 'channels_last_3d', 'channels_last_2d_3d', 'baseline_recheck'):
        fmt = torch.channels_last_3d if name.startswith('channels_last') else torch.contiguous_format
        torch.nn.utils.memory_format.convert_conv3d_weight_memory_format(vae, fmt)
        fmt2 = torch.channels_last if name == 'channels_last_2d_3d' else torch.contiguous_format
        torch.nn.utils.memory_format.convert_conv2d_weight_memory_format(vae, fmt2)
        inputs = pixels.contiguous(memory_format=fmt)
        timings = []
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for _ in range(10):
                torch.cuda.synchronize()
                start = time.monotonic()
                out = vae.batch_encode(inputs, device='cuda')
                torch.cuda.synchronize()
                timings.append(time.monotonic() - start)
        outputs[name] = out.detach().float().cpu()
        result['modes'][name] = {'seconds': timings, 'steady_median_seconds': statistics.median(timings[2:]),
                                  'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30}
        print(name, json.dumps(result['modes'][name]), flush=True)
    a = outputs['baseline']
    result['numerics'] = {}
    for name in ('channels_last_3d', 'channels_last_2d_3d', 'baseline_recheck'):
        b = outputs[name]
        result['numerics'][name] = {'max_abs': float((a-b).abs().max()), 'relative_l2': float((a-b).norm()/a.norm()),
                                   'cosine': float(torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0))}
    result['status'] = 'passed' if all(v['relative_l2'] < .01 and v['cosine'] > .9999 for v in result['numerics'].values()) else 'rejected'
    Path('work/setup_logs/vae_memory_format_check.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result['numerics']), result['status'], flush=True)


if __name__ == '__main__':
    main()
