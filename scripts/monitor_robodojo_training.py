"""Inspect the exact authorized run and prune only older verified checkpoints."""
import datetime
import fcntl
import json
import math
from pathlib import Path
import re
import shutil
import socket
import struct
import time
import zipfile

RUN = Path('/mnt/vepfs01/output/tiecheng.guo/robodojo_runs/20261021_8x16_gc6_verified')
FOLDER = RUN / 'training/2026-10-21_03-07-54'
KEEP = 3


def weights_valid(path):
    if path.is_symlink() or not path.is_file():
        return False
    size = path.stat().st_size
    with path.open('rb') as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            return False
        length, = struct.unpack('<Q', raw)
        if not 0 < length < min(size, 64 * 1024 * 1024):
            return False
        header = json.loads(stream.read(length))
    tensors = [value for key, value in header.items() if key != '__metadata__']
    return bool(tensors) and max(value['data_offsets'][1] for value in tensors) + 8 + length == size


def state_valid(path, step):
    if path.is_symlink() or not path.is_dir():
        return False
    marker = path / 'trainer_state.json'
    if not marker.is_file():
        return False
    meta = json.loads(marker.read_text())
    if meta.get('global_step') != step or meta.get('opt_step') != step:
        return False
    optimizer = list(path.rglob('*optim_states.pt'))
    if len(optimizer) != 8:
        return False
    for file in path.rglob('*'):
        if file.is_symlink():
            return False
        if file.is_file() and file.suffix in ('.pt', '.bin', '.pkl'):
            with zipfile.ZipFile(file) as archive:
                if not archive.namelist():
                    return False
    return True


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def main():
    assert FOLDER.resolve() == FOLDER and RUN.resolve() == RUN
    with (RUN / 'monitor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        now = time.time()
        previous_path = RUN / 'monitor_latest.json'
        previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
        launch = json.loads((RUN / 'launch_resume.json').read_text())
        proc = Path(f'/proc/{launch["pid"]}')
        try:
            alive = b'train_robodojo_cached_swanlab' in (proc / 'cmdline').read_bytes()
        except FileNotFoundError:
            alive = False
        rows = [json.loads(line) for line in (FOLDER / 'metrics.jsonl').read_text().splitlines()[-200:]]
        latest = rows[-1]
        stop_path = RUN / 'planned_stop_20000.json'
        planned_stop = json.loads(stop_path.read_text()) if stop_path.exists() else None
        intentionally_stopped = bool(planned_stop and planned_stop.get('status') == 'stopped'
                                     and latest['micro_step'] <= max(20001, planned_stop.get('last_observed_step', 20000)))
        worker_path = RUN / 'volcano_resume_launch.json'
        worker = json.loads(worker_path.read_text()) if worker_path.exists() else None
        job_path = RUN / 'volcano_job_status_node_0.json'
        job_status = json.loads(job_path.read_text()) if job_path.exists() else None
        worker_alive = False
        if worker and worker['hostname'] == socket.gethostname():
            try:
                worker_proc = Path(f'/proc/{worker["pid"]}')
                stat = (worker_proc / 'stat').read_text()
                fields = stat[stat.rfind(')') + 2:].split()
                worker_alive = fields[19] == worker['start_ticks'] and fields[0] != 'Z'
            except FileNotFoundError:
                pass
        age = now - latest['unix_time']
        external_progress = bool(worker and not intentionally_stopped and age < 900)
        running = alive or worker_alive
        finite = all(math.isfinite(row[key]) for row in rows for key in ('loss_total', 'loss_video', 'loss_action', 'grad_norm', 'lr'))
        # Native finalization removes resume states after the final weight save.
        # A last metrics row alone must never turn an active final save into completion.
        finished = (not running and latest['micro_step'] >= 60000
                    and not any(FOLDER.glob('accel_state_step_*'))
                    and weights_valid(FOLDER / 'checkpoint_step_60000.safetensors'))
        verified, incomplete, invalid = [], [], []
        for weight in sorted(FOLDER.glob('checkpoint_step_*.safetensors')):
            match = re.fullmatch(r'checkpoint_step_(\d+)\.safetensors', weight.name)
            if not match:
                continue
            step = int(match[1])
            state = FOLDER / f'accel_state_step_{step}'
            if not finished and not (state / 'trainer_state.json').is_file():
                incomplete.append(step)
                continue
            try:
                valid = weights_valid(weight) and (finished or state_valid(state, step))
            except (OSError, ValueError, KeyError, zipfile.BadZipFile):
                valid = False
            (verified if valid else invalid).append(step)
        verified.sort()
        deleted = []
        # A new full checkpoint must validate before an older one becomes eligible.
        # In-progress or uncertain files never enter this list.
        for step in verified[:-KEEP]:
            weight = FOLDER / f'checkpoint_step_{step}.safetensors'
            state = FOLDER / f'accel_state_step_{step}'
            assert weight.parent == FOLDER and state.parent == FOLDER
            assert not weight.is_symlink() and not state.is_symlink()
            if state.exists():
                shutil.rmtree(state)
            weight.unlink()
            deleted.append(step)
        active_writes = []
        for state in FOLDER.glob('accel_state_step_*'):
            if state.is_dir() and not (state / 'trainer_state.json').exists():
                active_writes.append(state.name)
        alerts = []
        if job_status and job_status.get('status') in ('failed_holding', 'failed_hold_expired'):
            alerts.append('platform_entry_failed_' + job_status['status'])
        if not finite:
            alerts.append('nonfinite_metrics')
        if not running and not finished and not intentionally_stopped and not external_progress:
            alerts.append('training_process_not_running_or_remote_worker_needs_check')
        if age > 900 and (running or worker) and not intentionally_stopped:
            alerts.append('metrics_older_than_15_minutes_check_save_io')
        if invalid:
            alerts.append('invalid_completed_checkpoints')
        free = shutil.disk_usage(FOLDER).free
        if free < 250_000_000_000:
            alerts.append('disk_free_below_250GB')
        report = dict(checked_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                      pid=launch['pid'], process_running=alive, finished=finished, latest=latest,
                      planned_stop=planned_stop, intentionally_stopped=intentionally_stopped,
                      volcano_worker=worker, volcano_worker_running_locally=worker_alive,
                      volcano_job_status=job_status,
                      recent_external_training_progress=external_progress,
                      metrics_age_seconds=age, recent_metrics_finite=finite,
                      advanced_since_previous=latest['micro_step'] - previous.get('latest', latest)['micro_step'],
                      keep_last_k=KEEP, kept_verified_steps=verified[-KEEP:], deleted_steps=deleted,
                      new_verified_steps=sorted(set(verified[-KEEP:]) - set(previous.get('kept_verified_steps', []))),
                      incomplete_weight_steps=incomplete, invalid_completed_steps=invalid,
                      unfinished_state_directories=active_writes, disk_free_bytes=free, alerts=alerts,
                      validation_scope='Safetensors header/extents, full-state completion marker and archive directories; not full payload checksums',
                      run_directory=str(FOLDER), swanlab_url='https://swanlab.cn/@oooer8/openwam-robodojo/runs/e47552zz')
        with (RUN / 'monitor_history.jsonl').open('a') as stream:
            stream.write(json.dumps(report, ensure_ascii=False) + '\n')
        atomic_json(previous_path, report)
        print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
