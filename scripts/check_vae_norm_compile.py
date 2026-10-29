"""Compare frozen VAE fusion/layout with its actual foundation weights."""
import json
import statistics
import time
from pathlib import Path

import torch
from safetensors import safe_open
from openwam.model.video_backbone.wan.models.vae import WanVideoVAE38
from openwam.train.utils.vae_optimizations import compile_vae_encoder_norms, use_vae_channels_last


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
    originals = [(m, m.forward) for m in vae.model.encoder.modules()]
    outputs = {}
    result = {'modes': {}, 'deterministic': True}
    for name in ('baseline', 'fused_norm', 'fused_norm_channels_last', 'baseline_recheck'):
        if name == 'fused_norm':
            result['compiled_norms'] = compile_vae_encoder_norms(vae)
        if name == 'fused_norm_channels_last':
            use_vae_channels_last(vae)
        if name == 'baseline_recheck':
            for module, forward in originals:
                module.forward = forward
            torch.nn.utils.memory_format.convert_conv3d_weight_memory_format(vae, torch.contiguous_format)
        inputs = pixels.contiguous(memory_format=torch.channels_last_3d) if name == 'fused_norm_channels_last' else pixels
        timings = []
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for _ in range(10):
                torch.cuda.synchronize()
                start = time.monotonic()
                out = vae.batch_encode(inputs, device='cuda')
                torch.cuda.synchronize()
                timings.append(time.monotonic() - start)
        outputs[name] = out.detach().cpu().double()
        result['modes'][name] = {'seconds': timings, 'steady_median_seconds': statistics.median(timings[2:]),
                                  'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30}
        print(name, json.dumps(result['modes'][name]), flush=True)
    a = outputs['baseline']
    result['numerics'] = {}
    for name in ('fused_norm', 'fused_norm_channels_last', 'baseline_recheck'):
        b = outputs[name]
        result['numerics'][name] = {'max_abs': float((a-b).abs().max()), 'relative_l2': float((a-b).norm()/a.norm()),
                                   'cosine': float((a.flatten() @ b.flatten())/(a.norm()*b.norm()))}
    result['status'] = 'passed' if all(v['relative_l2'] < .01 and v['cosine'] > .9999 for v in result['numerics'].values()) else 'rejected'
    Path('work/setup_logs/vae_norm_compile_check.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result['numerics']), result['status'], flush=True)


if __name__ == '__main__':
    main()
