"""Native robot RGB cache across all four workers; verified shards are reused."""
import os
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from runtime_common import ROOT,SOURCE,child,write,idle


def worker_ranks(world,node):
    assert world in [8,32] and node in range(4)
    return list(range(node*8,min((node+1)*8,world)))


def main():
    from omegaconf import OmegaConf
    from openwam.dataloader.libero_latent_cache import CachedLiberoDataset
    from openwam.dataloader.libero_robot_rgb_pair import PairedRobotRGBCachedLiberoDataset
    cfg=OmegaConf.load(ROOT/'config.yaml')
    CachedLiberoDataset.from_config(cfg.dataloader,split='train')
    cache=Path(cfg.dataloader.robot_rgb_latent_cache_dir)
    phase=os.environ['LIBERO_CACHE_PHASE'];assert phase in ['init','encode','finalize']
    job=Path(os.environ['LIBERO_JOB_DIR']);node=int(os.environ['NODE_RANK'])
    if phase=='init' and (cache/'manifest.json').exists():
        PairedRobotRGBCachedLiberoDataset.from_config(cfg.dataloader,split='train')
        write(job/'cache_plan.json',dict(ready=True,world_size=0))
        print('Robot RGB cache ready; no encoding repeated.',flush=True);return
    if phase=='init':
        world=json.loads((cache/'build.json').read_text())['world_size'] if (cache/'build.json').exists() else 32
        assert world in [8,32], 'Unsupported previous cache topology'
    else:
        plan=json.loads((job/'cache_plan.json').read_text())
        if plan['ready']:return
        world=plan['world_size']
    out=ROOT/'cache_attempts'/f'node{node}'/str(time.time_ns());out.mkdir(parents=True)
    cmd=[sys.executable,'-u',str(SOURCE/'scripts/cache_libero_vae.py'),
        '--root',str(cache),'--config',str(ROOT/'robot_cache_dataset.yaml'),
        '--training-config',str(ROOT/'cache_reference_config.yaml'),
        '--checkpoint',str(cfg.training.finetune_ckpt_path),'--batch-size','4','--world-size',str(world)]
    if phase=='init':
        assert node==0
        print(f'CACHE_INIT: {world} native VAE workers; existing verified shards are reused.',flush=True)
        child('init',cmd+['--init'],os.environ.copy(),out)
        write(job/'cache_plan.json',dict(ready=False,world_size=world))
        return
    if phase=='finalize':
        assert node==0
        child('finalize',cmd+['--finalize'],os.environ.copy(),out)
        return
    ranks=worker_ranks(world,node)
    if not ranks:
        print('Resuming an existing 8-worker cache on worker0; this node waits.',flush=True);return
    idle(os.environ)
    processes=[]
    def interrupted(sig,_):raise InterruptedError(str(sig))
    previous={s:signal.signal(s,interrupted) for s in [signal.SIGINT,signal.SIGTERM]}
    try:
        for rank,device in zip(ranks,os.environ['CUDA_VISIBLE_DEVICES'].split(',')):
            log=(out/f'worker{rank}.log').open('xb')
            p=subprocess.Popen(cmd+['--worker',str(rank)],cwd=SOURCE,
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=device),stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            processes.append((rank,p,log))
            write(out/f'worker{rank}_launch.json',dict(pid=p.pid,device=device))
        pending=list(processes)
        next_report=0
        while pending:
            for item in list(pending):
                rank,p,log=item
                if p.poll() is not None:
                    code=p.wait();pending.remove(item)
                    write(out/f'worker{rank}_exit.json',dict(pid=p.pid,returncode=code,actual_worker_wait=True))
                    assert code==0,(rank,code)
            if time.monotonic()>=next_report:
                completed=0
                for receipt in cache.glob('*.json'):
                    try:
                        item=json.loads(receipt.read_text())
                        if all(k in item for k in ['start','end','path','build_sha256']):completed+=item['end']-item['start']
                    except (OSError,ValueError):pass
                tails={}
                for rank,_,_ in processes:
                    with (out/f'worker{rank}.log').open('rb') as f:
                        f.seek(max(0,(out/f'worker{rank}.log').stat().st_size-4096))
                        rows=f.read().decode('utf-8',errors='replace').splitlines()
                    tails[str(rank)]=next((line for line in reversed(rows) if line.startswith('{')), 'initializing')
                report=dict(status='encoding_or_native_validation',completed_shard_windows=completed,
                    total_windows=273465,active_workers=len(pending),worker_progress=tails,checked_unix=time.time())
                write(ROOT/f'cache_status.node{node}.json',report)
                print('CACHE_PROGRESS '+json.dumps(report),flush=True)
                next_report=time.monotonic()+30
            if pending:time.sleep(1)
    finally:
        for rank,p,log in processes:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
            code=p.wait();log.close()
            write(out/f'worker{rank}_exit.json',dict(pid=p.pid,returncode=code,actual_worker_wait=True))
        for s,handler in previous.items():signal.signal(s,handler)


if __name__=='__main__':main()
