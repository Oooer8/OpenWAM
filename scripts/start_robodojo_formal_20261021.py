"""Launch one authorized formal training job; save a credential-free receipt."""
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/mnt/vepfs01/output/tiecheng.guo/openwam')
RUN = Path('/mnt/vepfs01/output/tiecheng.guo/robodojo_runs/20261021_8x16_gc6')
CACHE = Path('/mnt/vepfs01/output/tiecheng.guo/RoboDojo_openwam_vae_cache_v1')
KEY = Path('/mnt/vepfs01/output/tiecheng.guo/.secrets/swanlab_api_key')
assert json.loads((CACHE / 'validation.json').read_text())['status'] == 'passed'
assert KEY.is_file() and KEY.stat().st_mode & 0o077 == 0
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                               '--format=csv,noheader,nounits'], text=True)
assert len(gpu.splitlines()) == 8 and all(int(row.split(',')[1]) < 1024 for row in gpu.splitlines()), gpu
RUN.mkdir(parents=True, exist_ok=False)
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', NPROC_PER_NODE='8',
           NNODES='1', NODE_RANK='0', MASTER_ADDR='127.0.0.1', MASTER_PORT='29635',
           PYTHONPATH=str(ROOT), OPENWAM_TRACKER='swanlab', WANDB_MODE='disabled',
           OPENWAM_SWANLAB_KEY_FILE=str(KEY), OPENWAM_SWANLAB_PROJECT='openwam-robodojo',
           OPENWAM_SWANLAB_RUN_NAME='robodojo_alpha_cached_8x16_gc6_20261021',
           OPENWAM_SWANLAB_LOGDIR=str(RUN/'swanlog'), OPENWAM_CHECKPOINT_PATTERN='100000',
           PYTHONUNBUFFERED='1')
for name in ('OPENWAM_TRAIN_ENTRY', 'OPENWAM_PROFILE_DIR', 'OPENWAM_BENCH_ACCUMULATION',
             'SWANLAB_API_KEY', 'SWANLAB_MODE'):
    env.pop(name, None)
command = ['bash', 'scripts/train_robodojo_cached.sh', '--config-name', 'train_robodojo_cached_swanlab']
with (RUN/'train.log').open('w') as log:
    child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
receipt = dict(pid=child.pid, pgid=child.pid, command=command, cwd=str(ROOT),
               unix_start=time.time(), gpu_ids=list(range(8)), batch_per_gpu=16,
               gradient_accumulation=1, global_batch=128, checkpoint_pattern='100000',
               max_steps=60000, learning_rate=1e-4, warmup_ratio=0.05, scheduler='cosine',
               save_steps=2000, save_full_states_for_resume=True, keep_last_k_ckpts=10,
               run_root=str(RUN), status='launched', swanlab_project='openwam-robodojo',
               swanlab_experiment=env['OPENWAM_SWANLAB_RUN_NAME'])
(RUN/'launch.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt, indent=2), flush=True)
