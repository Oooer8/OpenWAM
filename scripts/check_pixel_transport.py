"""Bitwise and latency comparison for original, float-batch and uint8-batch pixels."""
import json
import statistics
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from openwam.model.video_backbone.wan.encode import preprocess_video, preprocess_video_batch


def float_batch(frames,device):
    array=np.stack([np.stack([np.asarray(image,dtype=np.float32) for image in clip]) for clip in frames])
    p=torch.from_numpy(array).to(device=device,dtype=torch.bfloat16)
    return (p*(2/255)-1).permute(0,4,1,2,3).contiguous()


def main():
    torch.set_num_threads(2)
    rng=np.random.default_rng(39)
    frames=[[Image.fromarray(rng.integers(0,256,(384,320,3),dtype=np.uint8)) for _ in range(9)] for _ in range(16)]
    result={'status':'passed','equality':'bitwise BF16 for CPU and CUDA','timings':{}}
    for device in ('cpu','cuda'):
        a=torch.cat([preprocess_video(v,dtype=torch.bfloat16,device=device) for v in frames])
        b=preprocess_video_batch(frames,dtype=torch.bfloat16,device=device)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        del a,b
    for name,fn in [('original',lambda:torch.cat([preprocess_video(v,dtype=torch.bfloat16,device='cuda') for v in frames])),
                    ('float32_batch',lambda:float_batch(frames,'cuda')),
                    ('uint8_batch',lambda:preprocess_video_batch(frames,dtype=torch.bfloat16,device='cuda'))]:
        timings=[]
        for _ in range(12):
            torch.cuda.synchronize();begin=time.monotonic()
            out=fn()
            torch.cuda.synchronize();timings.append(time.monotonic()-begin)
            del out
        result['timings'][name]={'seconds':timings,'steady_median':statistics.median(timings[2:])}
    Path('work/setup_logs/pixel_transport_check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
