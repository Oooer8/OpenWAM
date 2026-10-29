"""One heartbeat check: start this LIBERO run once, only when all eight GPUs are idle."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import time
import uuid

BASE = Path('/mnt/vepfs01/output/tiecheng.guo/artifacts/waw')
RUN = BASE / 'libero_training_autostart_20261022'
OUTPUT = BASE / 'openwam_libero_alpha_autostart_20261022'
REPO = Path('/mnt/vepfs01/output/tiecheng.guo/openwam')
PYTHON = '/mnt/vepfs01/output/tiecheng.guo/envs/openwam-alpha/bin/python'


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)


def gpu_snapshot():
    def query(fields, flag):
        return subprocess.check_output(
            ['nvidia-smi', flag + '=' + fields, '--format=csv,noheader,nounits'],
            text=True, timeout=15).strip().splitlines()
    gpus = []
    for line in query('index,uuid,memory.used,utilization.gpu', '--query-gpu'):
        index, identity, memory, utilization = [s.strip() for s in line.split(',')]
        gpus.append(dict(index=int(index), uuid=identity,
                         memory_mib=int(memory), utilization=int(utilization)))
    if sorted(g['index'] for g in gpus) != list(range(8)):
        raise RuntimeError('Expected exactly GPUs 0 through 7; refusing to launch')
    identities = {g['uuid'] for g in gpus}
    processes = []
    for line in query('gpu_uuid,pid,used_gpu_memory', '--query-compute-apps'):
        identity, pid, memory = [s.strip() for s in line.split(',')]
        if identity in identities:
            processes.append(dict(gpu_uuid=identity, pid=int(pid), memory_mib=memory))
    idle = not processes and all(g['memory_mib'] <= 1024 and g['utilization'] <= 5 for g in gpus)
    return dict(utc=utc(), idle=idle, gpus=gpus, compute_processes=processes)


def active_libero_processes():
    found = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = (p / 'cmdline').read_bytes().replace(b'\0', b' ')
        except (FileNotFoundError, PermissionError):
            continue
        if (b'scripts.start_libero_' in cmd or b'scripts/profile_libero_training.py' in cmd
                or (b'train_libero_alpha_cached' in cmd and b'scripts/train.py' in cmd)):
            found.append(dict(pid=int(p.name), command=cmd.decode(errors='replace')))
    return found


def record(value):
    value = dict(value, checked_utc=utc())
    atomic_json(RUN / 'gate_state.json', value)
    return value


def inspect_launched(state):
    status_file = RUN / 'run_status.json'
    if not status_file.exists():
        if time.time() - state['launch_unix'] > 900:
            return record(dict(state, status='needs_attention', reason='Reserved launch has no supervisor status; do not relaunch'))
        return dict(state, observation='Supervisor initializing; do not launch again')
    status = json.loads(status_file.read_text())
    if status['status'] == 'launch_deferred_gpu_busy':
        return record(dict(status='armed', reason='Supervisor recheck found a new GPU job; no training started'))
    if status['status'] in ('failed', 'completed', 'interrupted_by_user'):
        return record(dict(state, status=status['status'], supervisor_status=status))
    metrics = sorted(OUTPUT.glob('*/metrics.jsonl'))
    if metrics:
        lines = metrics[-1].read_text().splitlines()
        if lines:
            last = json.loads(lines[-1])
            if not all(math.isfinite(last[k]) for k in ('loss_total', 'loss_video', 'loss_action', 'grad_norm')):
                return record(dict(state, status='needs_attention', reason='Non-finite training metrics', last=last))
            if last['micro_step'] >= 8 and time.time() - last['unix_time'] < 120:
                import yaml
                config = yaml.safe_load((metrics[-1].parent / 'config.yaml').read_text())
                if config['training']['keep_last_k_ckpts'] != 3:
                    return record(dict(state, status='needs_attention', reason='Actual checkpoint retention is not three'))
                return record(dict(state, status='training_confirmed', last=last,
                                   output_path=str(metrics[-1].parent), keep_last_k_ckpts=3))
    if time.time() - state['launch_unix'] > 900:
        return record(dict(state, status='needs_attention', reason='Training not confirmed within 15 minutes; do not relaunch', supervisor_status=status))
    return dict(state, observation='Model initializing; do not launch again', supervisor_status=status)


def check_and_launch(dry_run=False):
    path = RUN / 'gate_state.json'
    state = json.loads(path.read_text()) if path.exists() else dict(status='armed')
    if state['status'] in ('training_confirmed', 'completed', 'failed', 'needs_attention', 'interrupted_by_user', 'cancelled'):
        return state
    if state['status'] in ('launch_reserved', 'launched'):
        return inspect_launched(state)
    if state['status'] != 'armed':
        raise ValueError('Unknown gate state; refusing to launch: ' + state['status'])
    active = active_libero_processes()
    if active:
        result = dict(status='armed', waiting_reason='Existing LIBERO launcher/training process', existing=active)
        return result if dry_run else record(result)
    observations = []
    for sample in range(3):
        observations.append(gpu_snapshot())
        if not observations[-1]['idle']:
            result = dict(status='armed', waiting_reason='GPUs not fully idle', observations=observations)
            return result if dry_run else record(result)
        if sample < 2:
            time.sleep(5)
    if active_libero_processes():
        result = dict(status='armed', waiting_reason='A LIBERO launcher appeared during idle checks')
        return result if dry_run else record(result)
    if dry_run:
        return dict(status='would_launch', observations=observations, keep_last_k_ckpts=3)
    # Reserve before Popen: an unknown outcome must never produce a second run.
    state = record(dict(status='launch_reserved', reservation_id=str(uuid.uuid4()),
                        launch_unix=time.time(), observations=observations, keep_last_k_ckpts=3))
    command = [PYTHON, '-u', '-m', 'scripts.start_libero_autostart_20261022']
    with (RUN / 'supervisor.log').open('a') as log:
        process = subprocess.Popen(command, cwd=REPO, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    state = record(dict(state, status='launched', supervisor_pid=process.pid, command=command))
    with (RUN / 'launch_events.jsonl').open('a') as f:
        f.write(json.dumps(state) + '\n')
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    RUN.mkdir(exist_ok=True)
    with (RUN / 'gate.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps(dict(status='gate_locked', action='No duplicate check or launch')))
            return
        result = check_and_launch(args.dry_run)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
