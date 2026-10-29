"""Launch the authorized 8-GPU LIBERO SFT after validated cache readiness."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

BASE=Path('/mnt/vepfs01/output/tiecheng.guo/artifacts/waw')
CACHE=BASE/'libero_4suite_vae_cache_native_train_v2'
RUN=BASE/'libero_training_20261022'
REPO=Path('/mnt/vepfs01/output/tiecheng.guo/openwam')


def save(value):
    path=RUN/'run_status.json'
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(value,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),indent=2)+'\n')
    os.replace(temp,path)


def main():
    RUN.mkdir(parents=True,exist_ok=True)
    with open(RUN/'supervisor.lock','w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert json.loads((CACHE/'run_status.json').read_text())['status']=='ready'
        assert json.loads((CACHE/'integration_validation.json').read_text())['status']=='passed'
        validation=json.loads((CACHE/'validation.json').read_text())
        assert validation['status']=='passed'
        assert validation['manifest_sha256']==hashlib.sha256((CACHE/'manifest.json').read_bytes()).hexdigest()
        # Start from the established environment without opt-in settings from
        # other experiments. No unrelated process or file is modified.
        env={k:v for k,v in os.environ.items() if not k.startswith('OPENWAM_')}
        env.update(CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',OMP_NUM_THREADS='4',
            MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
            WANDB_MODE='offline',WANDB_DIR=str(RUN),PYTHONUNBUFFERED='1',
            OPENWAM_LOCAL_METRICS='1',OPENWAM_PROFILE_DIR=str(RUN/'baseline_profile'))
        command=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1',
            '--nproc_per_node=8','--max_restarts=0','scripts/profile_libero_training.py',
            '--config-name','train_libero_alpha_cached',
            'training.output_path='+str(BASE/'openwam_libero_alpha_official_20261022'),
            'training.save_full_states_for_resume=true',
            'project.wandb.run_name=openwam_libero_alpha_official_20261022']
        receipt=dict(command=command,repo=str(REPO),cache=str(CACHE),
            cache_manifest_sha256=validation['manifest_sha256'],world_size=8,
            global_batch=256,batch_per_gpu=4,gradient_accumulation=8,num_epochs=10,
            learning_rate=1e-4,mixed_precision='bf16',zero_stage=2,
            augmentation=False,vae_encoding='native_training_vae_cached',
            performance_phase='baseline',wandb_mode='offline')
        (RUN/'launch_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        with open(RUN/'training.log','a',buffering=1) as log:
            process=subprocess.Popen(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
            while process.poll() is None:
                save(dict(status='training',supervisor_pid=os.getpid(),torchrun_pid=process.pid))
                time.sleep(20)
        save(dict(status='completed' if process.returncode==0 else 'failed',returncode=process.returncode))
        if process.returncode:
            raise SystemExit(process.returncode)


if __name__=='__main__':
    main()
