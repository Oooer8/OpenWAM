"""CPU preparation -> robot RGB VAE cache -> eight-rank gate -> manual fresh train."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'source'
sys.path[:0]=[str(SOURCE),str(SOURCE/'scripts')]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(16<<20),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def stop_requested():
    assert not (ROOT/'STOP_REQUESTED.json').exists(),'Paused by user'


def idle(env):
    stop_requested()
    devices=env['CUDA_VISIBLE_DEVICES'].split(',')
    assert len(devices)==8 and len(set(devices))==8
    rows=subprocess.check_output(['nvidia-smi','--id='+env['CUDA_VISIBLE_DEVICES'],
        '--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
    assert len(rows)==8 and all(int(float(x))<1024 for x in rows),'GPUs busy; no existing job changed'


def child(name,cmd,env,out,cwd=SOURCE):
    stop_requested();out.mkdir(parents=True,exist_ok=True)
    with (out/(name+'.log')).open('xb') as log:
        p=subprocess.Popen(cmd,cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        receipt=dict(command=cmd,pid=p.pid,started_unix=time.time(),status='running')
        write(out/(name+'_exit.json'),receipt)
        def stop(sig,_):
            if p.poll() is None:os.killpg(p.pid,sig)
        previous={s:signal.signal(s,stop) for s in [signal.SIGINT,signal.SIGTERM]}
        try:
            while True:
                try:
                    code=p.wait(timeout=30);break
                except subprocess.TimeoutExpired:
                    print(f'{name}: still running; log={out/(name+".log")}',flush=True)
        finally:
            for sig,handler in previous.items():signal.signal(sig,handler)
        receipt.update(returncode=code,finished_unix=time.time(),status='passed' if code==0 else 'failed',
            evidence='actual subprocess.Popen.wait return value')
        write(out/(name+'_exit.json'),receipt)
        assert code==0,f'{name} failed; inspect {out/(name+".log")}'

