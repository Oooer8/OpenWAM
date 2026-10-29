"""Check shared RMSNorm/gate fusion, gradients, changing lengths, and latency."""
import json
import statistics
import time
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint
from openwam.train.utils.runtime_optimizations import _functional_rms, _functional_gate
from check_training_optimizations import error


def main():
    torch.set_num_threads(2)
    torch.manual_seed(719)
    compiled_rms=torch.compile(_functional_rms,fullgraph=True,dynamic=True)
    compiled_gate=torch.compile(_functional_gate,fullgraph=True,dynamic=True)
    results=[]
    for length in (360,32,33,65,97,129):
        x=torch.randn(16,length,3072,device='cuda',dtype=torch.bfloat16)
        scale=torch.randn(3072,device='cuda',dtype=torch.bfloat16)
        for eps in (1e-6,1e-5):
            weight=torch.randn_like(x)/1000
            runs=[]
            for fn in (_functional_rms,compiled_rms):
                a=x.detach().clone().requires_grad_(True)
                b=scale.detach().clone().requires_grad_(True)
                y=checkpoint(fn,a,b,eps,use_reentrant=False)
                (y.float()*weight).sum().backward()
                runs.append((y.detach(),a.grad,b.grad))
            metrics=[error(p,q) for p,q in zip(*runs)]
            for e in metrics:
                assert e['relative_l2'] < .01 and e['cosine'] > .9999,e
            results.append({'kind':'rms','length':length,'eps':eps,'errors':metrics})
            print(json.dumps(results[-1]),flush=True)
        for scale_length in (1,length):
            g=torch.randn(16,scale_length,3072,device='cuda',dtype=torch.bfloat16)
            z=torch.randn_like(x)
            weight=torch.randn_like(x)/1000
            runs=[]
            for fn in (_functional_gate,compiled_gate):
                args=[v.detach().clone().requires_grad_(True) for v in (x,g,z)]
                y=checkpoint(fn,*args,use_reentrant=False)
                (y.float()*weight).sum().backward()
                runs.append((y.detach(),*[v.grad for v in args]))
            metrics=[error(p,q) for p,q in zip(*runs)]
            for e in metrics:
                assert e['relative_l2'] < .01 and e['cosine'] > .9999,e
            results.append({'kind':'gate','length':length,'scale_length':scale_length,'errors':metrics})
        del x,scale,weight,runs,args,y
    timings={}
    for label,fn in [('eager_rms',_functional_rms),('compiled_rms',compiled_rms)]:
        x=torch.randn(16,360,3072,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        w=torch.randn(3072,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        times=[]
        for i in range(22):
            x.grad=None;w.grad=None
            torch.cuda.synchronize();begin=time.monotonic()
            y=fn(x,w,1e-6)
            y.float().square().mean().backward()
            torch.cuda.synchronize();times.append(time.monotonic()-begin)
        timings[label]={'steady_median_seconds':statistics.median(times[2:])}
    result={'status':'passed','checks':results,'timings':timings}
    Path('work/setup_logs/norm_gate_compile_check.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(timings),flush=True)


if __name__=='__main__':
    main()
