"""Supervise exactly eight cache workers; publish only on full success."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/mnt/vepfs01/output/tiecheng.guo/artifacts/waw/libero_4suite_vae_cache_native_train_v2')
CHECKPOINT = '/mnt/vepfs01/output/tiecheng.guo/openwam/assets/openwam_ckpt/openwam_alpha/OpenWAM-Alpha-Pretrain-Foundation-Model/checkpoint_step_154000.safetensors'
BASE = [sys.executable, '-u', '-m', 'scripts.cache_libero_vae', '--root', str(ROOT),
        '--config', 'configs/libero_cache_data.yaml', '--checkpoint', CHECKPOINT]


def status(value):
    path = ROOT/'run_status.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(value, updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),indent=2)+'\n')
    os.replace(temp,path)


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    import fcntl
    with open(ROOT/'supervisor.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status(dict(status='initializing',pid=os.getpid(),training_started=False))
        subprocess.run(BASE+['--init'],check=True)
        workers=[]
        for gpu in range(8):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
                     OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
            log=open(ROOT/f'worker_{gpu}.log','a',buffering=1)
            process=subprocess.Popen(BASE+['--worker',str(gpu),'--max-gpu-memory-gib','24'],
                env=env,stdout=log,stderr=subprocess.STDOUT)
            workers.append((process,log))
        while True:
            states=[dict(gpu=i,pid=p.pid,returncode=p.poll()) for i,(p,_) in enumerate(workers)]
            status(dict(status='encoding',pid=os.getpid(),workers=states,training_started=False))
            if all(x['returncode'] is not None for x in states):
                break
            time.sleep(20)
        for _,log in workers:
            log.close()
        failed=[x for x in states if x['returncode']!=0]
        if failed:
            status(dict(status='failed',workers=states,training_started=False))
            raise RuntimeError(f'Cache workers failed: {failed}')
        status(dict(status='validating_all_shards',workers=states,training_started=False))
        subprocess.run(BASE+['--finalize'],check=True)
        status(dict(status='ready',workers=states,training_started=False))


if __name__=='__main__':
    main()
