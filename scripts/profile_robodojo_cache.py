"""Bounded phase profiling and exact CUDA Graph comparison for offline VAE."""
import argparse, json, time, statistics
from pathlib import Path
import numpy as np
import torch
import yaml
from cache_robodojo_vae import load_state, make_dataset, DEFAULT_CHECKPOINT, atomic_json
from openwam.model.video_backbone.wan.models.vae import WanVideoVAE38
from openwam.model.video_backbone.wan.encode import preprocess_video_batch
from openwam.train.utils.robodojo_exact_optimizations import install_exact_vae_norms

p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
a=p.parse_args()
out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(16*2**30/torch.cuda.get_device_properties(0).total_memory)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
ds=make_dataset(yaml.safe_load(Path('configs/dataloader/robodojo_prepared.yaml').read_text()))
vae=WanVideoVAE38()
vae.load_state_dict(load_state(DEFAULT_CHECKPOINT),strict=True,assign=True)
vae.cuda().bfloat16().eval().requires_grad_(False)
install_exact_vae_norms(vae,divide=True)
report=dict(status='running',gpu=torch.cuda.get_device_name(),batch=16,phase_rows=[],comparisons=[])

def save(): atomic_json(out/'profile.json',report)
def sync(): torch.cuda.synchronize()

with torch.inference_mode():
    for batch in range(10):
        start=time.perf_counter()
        frames=[ds[500000+16*batch+i]['video'] for i in range(16)]
        loaded=time.perf_counter()
        pixels=preprocess_video_batch(frames,dtype=torch.bfloat16,device='cuda'); sync()
        prepared=time.perf_counter()
        latents=vae.batch_encode(pixels,device='cuda'); sync()
        encoded=time.perf_counter()
        assert torch.isfinite(latents).all(); sync()
        checked=time.perf_counter()
        bits=latents.cpu().contiguous().view(torch.uint16).numpy()
        copied=time.perf_counter()
        if batch==0:
            scratch=np.memmap(out/'profile_scratch.bf16',mode='w+',dtype=np.uint16,shape=(10*16,*bits.shape[1:]))
        scratch[batch*16:(batch+1)*16]=bits
        written=time.perf_counter()
        row=dict(read_decode=loaded-start,pixel_prepare_h2d=prepared-loaded,vae=encoded-prepared,
                 finite_check=checked-encoded,d2h=copied-checked,mmap_write=written-copied,total=written-start)
        report['phase_rows'].append(row); save(); print('PHASE',json.dumps(row),flush=True)
        del pixels,latents,bits,frames
    start=time.perf_counter();scratch.flush();del scratch
    report['flush_seconds']=time.perf_counter()-start
    report['phases_mean']={k:statistics.mean(row[k] for row in report['phase_rows'][2:]) for k in report['phase_rows'][0]}
    # Reference outputs are moved to CPU BEFORE capture. Running eager alongside
    # the graph would require two activation pools and exceed the 16GiB budget.
    ds._open()
    tail=np.flatnonzero(ds._windows[:,2]<33)
    rng=np.random.default_rng(751)
    test_indices=[rng.integers(0,len(ds),16).tolist() for _ in range(4)]
    test_indices+=[rng.choice(tail,16).tolist() for _ in range(4)]
    pixels_cpu=[];references=[];before=[]
    for indices in test_indices:
        pixels=preprocess_video_batch([ds[i]['video'] for i in indices],dtype=torch.bfloat16,device='cuda')
        pixels_cpu.append(pixels.cpu())
        times=[]
        for _ in range(2):
            sync();begin=time.perf_counter()
            result=vae.batch_encode(pixels,device='cuda');sync()
            times.append(time.perf_counter()-begin)
        before.append(statistics.mean(times));references.append(result.cpu())
        del result,pixels
    vae.scale=[value.to(device='cuda',dtype=torch.bfloat16) for value in vae.scale]
    report['eager_before_seconds']=before
    torch.cuda.empty_cache()
    static_input=pixels_cpu[0].cuda()
    warm_stream=torch.cuda.Stream()
    warm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm_stream):
        for _ in range(3): tmp=vae.batch_encode(static_input,device='cuda')
    torch.cuda.current_stream().wait_stream(warm_stream);sync()
    assert torch.equal(references[0],tmp.cpu())
    report['prestaged_scale_bitwise_equal']=True
    del tmp
    torch.cuda.empty_cache()
    graph=torch.cuda.CUDAGraph()
    begin=time.perf_counter()
    try:
        with torch.cuda.graph(graph):
            static_output=vae.batch_encode(static_input,device='cuda')
    except Exception as error:
        report.update(status='graph_rejected',error=repr(error));save();raise
    report['capture_seconds']=time.perf_counter()-begin
    for iteration,(cpu,indices) in enumerate(zip(pixels_cpu,test_indices)):
        static_input.copy_(cpu);sync()
        runs=[]
        for _ in range(3):
            sync();begin=time.perf_counter();graph.replay();sync()
            runs.append(time.perf_counter()-begin)
        equal=torch.equal(references[iteration],static_output.cpu())
        row=dict(iteration=iteration,indices=indices,
                 timings=dict(eager_before=before[iteration],graph=statistics.median(runs)),bitwise_equal=equal)
        report['comparisons'].append(row);save();print('COMPARE',json.dumps(row),flush=True)
        assert equal,'Graph changed BF16 values'
    report['graph_peak_allocated_gib']=torch.cuda.max_memory_allocated()/2**30
    report['graph_peak_reserved_gib']=torch.cuda.max_memory_reserved()/2**30
    del graph,static_output,static_input
    import gc
    gc.collect();torch.cuda.empty_cache()
    after=[]
    for iteration,cpu in enumerate(pixels_cpu):
        pixels=cpu.cuda();sync();begin=time.perf_counter()
        result=vae.batch_encode(pixels,device='cuda');sync()
        after.append(time.perf_counter()-begin)
        assert torch.equal(references[iteration],result.cpu())
        report['comparisons'][iteration]['timings']['eager_after']=after[-1]
        del result,pixels
    report['eager_after_seconds']=after
    report['eager_mean_seconds']=(statistics.mean(before)+statistics.mean(after))/2
    report['graph_mean_seconds']=statistics.mean(r['timings']['graph'] for r in report['comparisons'])
    report['graph_speedup']=report['eager_mean_seconds']/report['graph_mean_seconds']
    report['status']='passed';save()
print(json.dumps(report['phases_mean']),flush=True)
print('GRAPH_SPEEDUP',report['graph_speedup'],flush=True)
