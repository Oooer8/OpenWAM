"""Start the early-save phase or resume its actual full-state checkpoint."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('phase', choices=['save', 'resume'])
a = p.parse_args()
ROOT = Path('/mnt/vepfs01/output/tiecheng.guo/openwam')
RUN = Path('/mnt/vepfs01/output/tiecheng.guo/robodojo_runs/20261021_8x16_gc6_verified')
KEY = '/mnt/vepfs01/output/tiecheng.guo/.secrets/swanlab_api_key'
gpu = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
assert len(gpu.splitlines())==8 and all(int(row.split(',')[1])<1024 for row in gpu.splitlines()),gpu
if a.phase == 'save':
    RUN.mkdir(parents=True,exist_ok=False)
else:
    assert not (RUN/'launch_resume.json').exists(), 'Resume already launched; inspect it first'
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', NPROC_PER_NODE='8',
           NNODES='1', NODE_RANK='0', MASTER_ADDR='127.0.0.1', MASTER_PORT='29635',
           PYTHONPATH=str(ROOT), OPENWAM_TRACKER='swanlab', WANDB_MODE='disabled',
           OPENWAM_SWANLAB_KEY_FILE=KEY, OPENWAM_SWANLAB_PROJECT='openwam-robodojo',
           OPENWAM_SWANLAB_RUN_NAME='robodojo_alpha_cached_8x16_gc6_verified_20261021',
           OPENWAM_SWANLAB_LOGDIR=str(RUN/f'swanlog_{a.phase}'),
           OPENWAM_CHECKPOINT_PATTERN='100000', PYTHONUNBUFFERED='1')
for name in ('OPENWAM_TRAIN_ENTRY','OPENWAM_PROFILE_DIR','OPENWAM_BENCH_ACCUMULATION',
             'SWANLAB_API_KEY','SWANLAB_MODE','OPENWAM_SWANLAB_RESUME_ID'):
    env.pop(name,None)
command=['bash','scripts/train_robodojo_cached.sh','--config-name','train_robodojo_cached_swanlab',
         f'training.output_path={RUN}/training',
         'project.wandb.run_name=robodojo_alpha_cached_8x16_gc6_verified_20261021']
if a.phase=='save':
    env['OPENWAM_TRAIN_ENTRY']='scripts/train_robodojo_verify_save.py'
else:
    folders=list((RUN/'training').glob('*/early_save_passed.json'))
    assert len(folders)==1
    folder=folders[0].parent
    state=json.loads((folder/'accel_state_step_20/trainer_state.json').read_text())
    assert state['global_step']==state['opt_step']==20
    env['OPENWAM_SWANLAB_RESUME_ID']=json.loads((RUN/'swanlog_save/openwam_run.json').read_text())['id']
    command+=['training.finetune_ckpt_path=null',f'training.resume_ckpt_path={folder}']
with (RUN/f'{a.phase}.log').open('w') as log:
    child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,
                           stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(phase=a.phase,pid=child.pid,pgid=child.pid,command=command,cwd=str(ROOT),
             unix_start=time.time(),run_root=str(RUN),global_batch=128,batch_per_gpu=16,
             gpus=8,accumulation=1,max_steps=60000,checkpoint_pattern='100000')
(RUN/f'launch_{a.phase}.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2),flush=True)
