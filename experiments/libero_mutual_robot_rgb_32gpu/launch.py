"""Volcano 4x8 launch: cache -> full-state gate -> fresh foundation training."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'source'
sys.path[:0]=[str(SOURCE),str(SOURCE/'scripts')]
from runtime_common import sha,write


def topology(env):
    def number(key,default):
        value=env.get(key,str(default))
        if not re.fullmatch('[0-9]+',value):raise ValueError('Invalid '+key)
        return int(value)
    nodes=number('NNODES',4);gpus=number('NPROC_PER_NODE',8);rank=number('NODE_RANK',0)
    port=number('MASTER_PORT',29659);host=env.get('MASTER_ADDR','')
    if (nodes,gpus)!=(4,8) or rank not in range(4) or not 0<port<65536:
        raise ValueError('Requires four nodes, eight GPUs per node, node ranks 0..3')
    if not host or host.lower() in ['localhost','::1','0.0.0.0'] or host.startswith('127.'):
        raise ValueError('Use Volcano MLP_WORKER_0_HOST or a reachable non-loopback MASTER_ADDR')
    return dict(nodes=nodes,gpus_per_node=gpus,node_rank=rank,master_addr=host,master_port=port)


def source_digest():
    h=hashlib.sha256()
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and p.suffix in ['.py','.yaml','.sh'] and not any(
                x in p.relative_to(ROOT).parts for x in ['runs','cache_attempts','__pycache__']):
            h.update(str(p.relative_to(ROOT)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def checked_gpu_idle():
    devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    assert len(devices)==len(set(devices))==8,'Each worker must see exactly eight GPUs'
    rows=subprocess.check_output(['nvidia-smi','--id='+os.environ['CUDA_VISIBLE_DEVICES'],
        '--query-gpu=memory.total,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
    parsed=[list(map(int,row.split(','))) for row in rows]
    assert len(parsed)==8 and all(total>=75*1024 for total,_ in parsed),'80GB-class GPUs required'
    assert all(used<1024 for _,used in parsed),'Existing GPU job detected; nothing stopped'


def run_child(stage,cmd,env,job,node):
    path=job/'status'/f'{stage}.node{node}.json'
    logpath=job/'logs'/f'{stage}.node{node}.log';logpath.parent.mkdir(parents=True,exist_ok=True)
    print('LOG '+str(logpath),flush=True)
    with logpath.open('xb') as log:
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,start_new_session=True)
        write(path,dict(status='running',pid=p.pid,started_unix=time.time(),log=str(logpath)))
        def stop(sig,_):
            if p.poll() is None:
                try:os.killpg(p.pid,sig)
                except ProcessLookupError:pass
        previous={s:signal.signal(s,stop) for s in [signal.SIGINT,signal.SIGTERM]}
        selector=selectors.DefaultSelector();selector.register(p.stdout,selectors.EVENT_READ)
        try:
            while selector.get_map():
                failures=list((job/'status').glob('failure.node*.json'))
                peer_failed=any(json.loads(f.read_text()).get('node_rank')!=node for f in failures)
                if peer_failed and p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
                for key,_ in selector.select(timeout=1):
                    data=os.read(key.fileobj.fileno(),65536)
                    if not data:selector.unregister(key.fileobj);continue
                    log.write(data);log.flush();sys.stdout.buffer.write(data);sys.stdout.buffer.flush()
            code=p.wait()
        finally:
            selector.close()
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)
            code=p.wait()
            for s,handler in previous.items():signal.signal(s,handler)
            write(path,dict(status='passed' if code==0 else 'failed',pid=p.pid,returncode=code,
                actual_subprocess_wait=True,finished_unix=time.time(),log=str(logpath)))
        assert code==0,f'{stage} node{node} failed ({code}); inspect {logpath}'


def wait_stage(job,stage,nodes=range(4)):
    deadline=time.monotonic()+float(os.environ.get('STAGE_TIMEOUT_SECONDS','172800'))
    next_print=0
    while time.monotonic()<deadline:
        failures=list((job/'status').glob('failure.node*.json'))
        assert not failures,'Peer failed: '+','.join(str(f) for f in failures)
        states=[]
        for node in nodes:
            p=job/'status'/f'{stage}.node{node}.json'
            states.append(json.loads(p.read_text()) if p.exists() else {})
        assert not any(v.get('status')=='failed' for v in states),'Stage failed: '+stage
        if all(v.get('status')=='passed' for v in states):return
        if time.monotonic()>=next_print:
            print('WAIT '+stage+' '+json.dumps([v.get('status','not_started') for v in states]),flush=True)
            next_print=time.monotonic()+30
        time.sleep(2)
    raise TimeoutError(stage+' did not finish within STAGE_TIMEOUT_SECONDS')


def main():
    from omegaconf import OmegaConf
    t=topology(os.environ);node=t['node_rank']
    action=os.environ.get('ACTION','run');dry=os.environ.get('DRY_RUN','1')
    assert action in ['run','validate'] and dry in ['0','1']
    assert not os.environ.get('RESUME_CKPT_DIR') and not os.environ.get('RESUME_RUN'),'This entry starts fresh'
    name=os.environ.get('RUN_ID','libero-mutual-robot-rgb-32gpu-20261025')
    assert re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,99}',name),'Invalid RUN_ID'
    cfg=OmegaConf.load(ROOT/'config.yaml')
    assert cfg.training.batch_size==8 and cfg.training.gradient_accumulation_steps==1
    assert cfg.training.keep_last_k_ckpts==3 and cfg.training.num_epochs==10
    assert cfg.training.resume_ckpt_path is None and cfg.training.save_full_states_for_resume
    job=Path(os.environ.get('OUTPUT_ROOT',str(ROOT/'runs')))/name
    summary=dict(**t,world_size=32,batch_per_gpu=8,gradient_accumulation_steps=1,global_batch=256,
        epochs=10,prepared_updates_per_epoch=1069,total_updates=10690,seed=42,learning_rate=1e-4,
        graph=[[0,1],[0,1],[1,2]],v1='scene RGB external+wrist',v2='robot RGB external+wrist',
        tail_global_batch=64,tail_per_rank=2,padding_windows_per_epoch=7,
        gradient_checkpointing=True,compile_ffn=False,compile_norm=False,
        keep_last_models=3,keep_last_complete_states=3,save_every_updates=2000,
        start_from='OpenWAM foundation154000',action=action,job=str(job),dry_run=dry,
        formal_training_started=False)
    print(json.dumps(summary,indent=2),flush=True)
    if dry=='1':return
    checked_gpu_idle()
    job.mkdir(parents=True,exist_ok=True);(job/'status').mkdir(exist_ok=True)
    lock=(job/f'.node{node}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not list((job/'artifacts').glob('*/config.yaml')),'Existing formal run; choose a new RUN_ID'
    assert not (job/'status'/f'cache.node{node}.json').exists(),'Attempt already exists; use a new RUN_ID'
    contract=dict(summary);contract.pop('node_rank');contract.pop('dry_run')
    contract['source_sha256']=source_digest()
    with (job/'.contract.lock').open('a') as cf:
        fcntl.flock(cf,fcntl.LOCK_EX)
        p=job/'contract.json'
        if p.exists():assert json.loads(p.read_text())==contract,'Nodes disagree about config/source/job'
        else:write(p,contract)
    env={k:v for k,v in os.environ.items() if not k.startswith('OPENWAM_')}
    env.update(PYTHONPATH=str(SOURCE),OPENWAM_TEXT_CACHE_DIR=str(cfg.project.text_cache_directory),
        OPENWAM_CACHED_TEXT_ENCODER_CPU='1',OPENWAM_LOCAL_METRICS='1',OPENWAM_COMPILE_TRAIN_FFN='0',
        OPENWAM_COMPILE_NORM_GATE='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',WANDB_MODE='disabled',
        LIBERO_PAIRED_CACHE_PROOF=str(job/'paired_cache_grouping.json'),LIBERO_JOB_DIR=str(job))
    try:
        if node==0:
            # Initialize once, then shard native cache generation over all 32 GPUs.
            run_child('raw_rgb_audit',[sys.executable,'-u',str(ROOT/'audit_rgb_inputs.py')],dict(env,CUDA_VISIBLE_DEVICES=''),job,node)
            run_child('cache_init',[sys.executable,'-u',str(ROOT/'cache_stage.py')],dict(env,LIBERO_CACHE_PHASE='init'),job,node)
        wait_stage(job,'cache_init',[0])
        run_child('cache_encode',[sys.executable,'-u',str(ROOT/'cache_stage.py')],dict(env,LIBERO_CACHE_PHASE='encode'),job,node)
        wait_stage(job,'cache_encode')
        if node==0:
            run_child('cache_finalize',[sys.executable,'-u',str(ROOT/'cache_stage.py')],dict(env,LIBERO_CACHE_PHASE='finalize'),job,node)
            run_child('cache_grouping',[sys.executable,'-u',str(ROOT/'check_paired_cache_grouping.py')],dict(env,CUDA_VISIBLE_DEVICES=''),job,node)
            write(job/'paired_cache_grouping.json',json.loads((ROOT/'paired_cache_grouping.json').read_text()))
            write(job/'status'/'cache.node0.json',dict(status='passed',finished_unix=time.time()))
        else:
            wait_stage(job,'cache',[0])
            write(job/'status'/f'cache.node{node}.json',dict(status='passed',finished_unix=time.time()))
        wait_stage(job,'cache')
        command=[sys.executable,'-u','-m','torch.distributed.run','--nnodes','4','--nproc_per_node','8',
            '--node_rank',str(node),'--master_addr',t['master_addr'],'--master_port',str(t['master_port']),
            '--max_restarts','0',str(ROOT/'train_worker.py')]
        for phase in ['gate_save','gate_restore']:
            c=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True))
            c.training.output_path=c.project.output_dir=str(job/'gate_artifacts')
            c.project.vav.formal_training=False
            if phase=='gate_restore':
                proof=json.loads((job/'gate_source.json').read_text())
                c.training.finetune_ckpt_path=None;c.training.resume_ckpt_path=proof['run']
            cp=job/'configs'/f'{phase}.node{node}.yaml';cp.parent.mkdir(exist_ok=True)
            OmegaConf.save(c,cp)
            run_child(phase,command,dict(env,LIBERO_CONFIG=str(cp),LIBERO_PHASE=phase,
                LIBERO_GATE_REPORT_DIR=str(job/phase)),job,node)
            wait_stage(job,phase)
            if node==0:
                reports=[json.loads((job/phase/f'rank{i}.json').read_text()) for i in range(32)]
                assert all(v['status']=='passed' and v['captures']==2 for v in reports)
                if phase=='gate_save':
                    runs=[p.parent for p in (job/'gate_artifacts').glob('*/accel_state_step_2/trainer_state.json')]
                    assert len(runs)==1
                    run=runs[0].parent;state=run/'accel_state_step_2'
                    assert len(list(state.glob('random_states_*.pkl')))==32
                    assert len(list(state.rglob('*optim_states.pt')))==32
                    assert (run/'checkpoint_step_2.safetensors').is_file()
                    write(job/'gate_source.json',dict(run=str(run),state=str(state),world_size=32,optimizer_step=2))
                else:
                    run=Path(json.loads((job/'gate_source.json').read_text())['run'])
                    receipts=[json.loads((run/'checkpoint_validation'/f'after_restore_rank_{i}.json').read_text()) for i in range(32)]
                    assert all(v['all_match'] for v in receipts)
                    write(job/'gpu_acceptance.json',dict(status='passed',world_size=32,
                        actual_node_exit_codes=[json.loads((job/'status'/f'{phase}.node{i}.json').read_text())['returncode'] for i in range(4)],
                        actual_node_save_exit_codes=[json.loads((job/'status'/f'gate_save.node{i}.json').read_text())['returncode'] for i in range(4)],
                        checkpoint_model_optimizer_scheduler_lr_steps_rng_exactly_restored=True,
                        full_and_tail_loss_gradients_finite=True,full_state_save_restore_verified=True,
                        graph=summary['graph'],source_sha256=contract['source_sha256'],
                        report_hashes={str(p.relative_to(job)):sha(p) for folder in ['gate_save','gate_restore'] for p in sorted((job/folder).glob('rank*.json'))}))
                    if action=='run':
                        # Remove only this bounded drill's payloads after all 32
                        # ranks restored them and all four torchrun parents exited.
                        assert run.resolve().is_relative_to((job/'gate_artifacts').resolve())
                        (run/'checkpoint_step_2.safetensors').unlink()
                        shutil.rmtree(run/'accel_state_step_2')
                        write(job/'gate_payload_cleanup.json',dict(status='passed',
                            verified_restore_before_cleanup=True,drill_run=str(run),
                            formal_checkpoints_untouched=True))
                write(job/'status'/f'{phase}_accepted.node0.json',dict(status='passed'))
            wait_stage(job,phase+'_accepted',[0])
        if action=='validate':
            print('32-GPU validation passed; formal training was not started.',flush=True);return
        acceptance=json.loads((job/'gpu_acceptance.json').read_text())
        assert acceptance['status']=='passed' and acceptance['full_state_save_restore_verified']
        assert acceptance['source_sha256']==source_digest()
        c=OmegaConf.create(OmegaConf.to_container(cfg,resolve=True))
        c.training.output_path=c.project.output_dir=str(job/'artifacts')
        cp=job/'configs'/f'formal.node{node}.yaml';OmegaConf.save(c,cp)
        print('Gates passed. Starting FRESH formal training from foundation154000.',flush=True)
        run_child('formal',command,dict(env,LIBERO_CONFIG=str(cp),LIBERO_PHASE='formal'),job,node)
        wait_stage(job,'formal')
        if node==0:
            runs=list((job/'artifacts').glob('*/config.yaml'));assert len(runs)==1
            run=runs[0].parent
            weights=sorted(int(p.stem.split('_')[-1]) for p in run.glob('checkpoint_step_*.safetensors'))
            states=sorted(int(p.name.split('_')[-1]) for p in run.glob('accel_state_step_*'))
            assert weights==states==[8000,10000,10690],(weights,states)
            for step in states:
                state=run/f'accel_state_step_{step}'
                assert json.loads((state/'trainer_state.json').read_text())['opt_step']==step
                assert len(list(state.glob('random_states_*.pkl')))==32
                assert len(list(state.rglob('*optim_states.pt')))==32
            write(job/'completed.json',dict(status='passed',optimizer_updates=10690,
                retained_steps=weights,keep_last_models=3,keep_last_states=3,
                actual_node_exit_codes=[json.loads((job/'status'/f'formal.node{i}.json').read_text())['returncode'] for i in range(4)]))
    except BaseException as error:
        write(job/'status'/f'failure.node{node}.json',dict(node_rank=node,error=repr(error),finished_unix=time.time()))
        raise


if __name__=='__main__':
    def interrupted(sig,_):raise InterruptedError('Launcher signal '+str(sig))
    signal.signal(signal.SIGTERM,interrupted)
    try:main()
    except BaseException as error:
        # Notify peers even when an early topology/device/dependency check fails.
        if os.environ.get('DRY_RUN','1')=='0':
            name=os.environ.get('RUN_ID','libero-mutual-robot-rgb-32gpu-20261025')
            node=os.environ.get('NODE_RANK','0')
            if re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,99}',name) and node.isdigit():
                job=Path(os.environ.get('OUTPUT_ROOT',str(ROOT/'runs')))/name
                write(job/'status'/f'failure.node{node}.json',dict(node_rank=int(node),error=repr(error),finished_unix=time.time()))
        raise
