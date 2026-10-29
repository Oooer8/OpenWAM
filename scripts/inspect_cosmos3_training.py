"""Read-only training inspection; writes only a supervision snapshot."""
import datetime
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess

import yaml

base = Path(__file__).resolve().parents[1]
active_path = base / 'work/setup_logs/active_formal_training.json'
active = json.loads(active_path.read_text()) if active_path.is_file() else {}
root = base / active.get('output_root', 'outputs/cosmos3_droid_alpha_stride3')
# Never attribute metrics from a failed prior launch to a newly initialized job.
started = datetime.datetime.fromisoformat(active['started_at_shanghai']).timestamp() if active.get('started_at_shanghai') else 0
if active.get('run_directory'):
    candidates = [Path(active['run_directory']) / 'metrics.jsonl']
else:
    candidates = list(root.glob('*/metrics.jsonl'))
candidates = [p for p in candidates if p.is_file() and p.stat().st_size and p.stat().st_mtime >= started]
if not candidates:
    pid = int(active['torchrun_pid'])
    proc = Path(f'/proc/{pid}')
    cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode() if proc.exists() else ''
    identity = 'torchrun' in cmd and root.name in cmd
    snapshot = dict(status='preparing_threeview' if identity else 'requires_inspection',
        checked_at_shanghai=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        torchrun_pid=pid, run_directory=active.get('run_directory'), process_identity_verified=identity,
        completed_updates=0, total_updates=66790, progress_percent=0,
        intervention='Waiting for the active launch to initialize; no duplicate launch', log_file=active['log_file'])
    (base / 'work/setup_logs/formal_training_status.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2))
    print(json.dumps(snapshot,ensure_ascii=False))
    raise SystemExit(0)
metrics = max(candidates, key=lambda p: p.stat().st_mtime)
rows = []
partial_line = False
for line in metrics.read_text().splitlines():
    if not line.strip():
        continue
    try:
        rows.append(json.loads(line))
    except json.JSONDecodeError:
        partial_line = True  # The writer may be appending its latest line.
last = rows[-1]
cfg = yaml.safe_load((metrics.parent / 'config.yaml').read_text())
t = cfg['training']
runtime = cfg['project']['runtime_optimizations']
checks = {
    'window_stride': cfg['dataloader']['window_stride'] == 3,
    'clip_shape': cfg['dataloader']['num_frames'] == 33 and cfg['dataloader']['video_stride'] == 4,
    'training_target': t['num_epochs'] == 2 and t['max_steps'] is None,
    'batch': t['batch_size'] == 16 and t['gradient_accumulation_steps'] == 1,
    'checkpoint_policy': t['save_steps'] == 33395 and t['keep_last_k_ckpts'] == 2 and t['save_full_states_for_resume'],
    'precision_and_zero': t['mixed_precision'] == 'bf16' and t['zero_stage'] == 2,
    'no_recomputation': not t['use_gradient_checkpointing'] and not cfg['model']['architecture']['mot_checkpoint_mixed_attn'],
    'execution_flags': all(runtime[k] for k in ('zero_overlap_comm', 'cached_text_encoder_on_cpu', 'vae_channels_last', 'compiled_train_ffn', 'compiled_norm_and_gate', 'compact_masked_text_context', 'batch_video_preprocess')),
}
if 'learning_rate' in active:
    checks['learning_rate_schedule'] = (
        abs(float(t['learning_rate']) - float(active['learning_rate'])) < 1e-12
        and t.get('action_lr') is None and t.get('video_lr') is None
        and t['warmup_ratio'] == .05 and t['lr_scheduler'] == 'cosine'
        and t['lr_min_ratio'] == .01)
if active.get('view_layout') == 'cosmos3_three_view':
    checks['three_view_layout'] = (cfg['dataloader'].get('view_layout') == 'cosmos3_three_view'
        and cfg['dataloader'].get('camera_layout') == ['observation.image.wrist_image_left',
            'observation.image.exterior_image_1_left', 'observation.image.exterior_image_2_left']
        and not cfg['dataloader'].get('head_camera_choices')
        and cfg['dataloader']['height'] == 384 and cfg['dataloader']['width'] == 320)
nonfinite = [r['micro_step'] for r in rows if not all(math.isfinite(r[k]) for k in ('loss_total', 'loss_video', 'loss_action', 'grad_norm'))]
pid = int((base / active.get('pid_file', 'work/setup_logs/train_cosmos3_droid_stride3.pid')).read_text())
proc = Path(f'/proc/{pid}')
cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode() if proc.exists() else ''
identity = 'torchrun' in cmd and root.name in cmd
ranks = []
children_path = proc / f'task/{pid}/children'
for child in children_path.read_text().split() if children_path.exists() else []:
    child_cmd = Path(f'/proc/{child}/cmdline')
    if child_cmd.exists() and b'scripts/train.py' in child_cmd.read_bytes():
        ranks.append(int(child))
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
seconds = statistics.median(1 / r['micro_steps_per_sec'] for r in rows[-100:])
age = now.timestamp() - last['unix_time']
log_path = base / active.get('log_file', 'work/setup_logs/train_cosmos3_droid_stride3.log')
log = log_path.read_text(errors='replace')
alerts = [word for word in ('Traceback (most recent call last)', 'CUDA out of memory', 'ChildFailedError', 'No space left on device') if word in log]
gpu_result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=20)
healthy = identity and len(ranks) == 8 and age < 180 and not nonfinite and not alerts and all(checks.values())
status = 'training' if healthy else 'requires_inspection'
if last['optimizer_step'] == 66790 and not identity:
    status = 'completion_pending_validation'
snapshot = dict(status=status, automation_id='openwam', automation_status='ACTIVE', check_interval_minutes=15,
    checked_at_shanghai=now.isoformat(), server='root@192.168.52.168:2222', repository=str(base),
    torchrun_pid=pid, process_identity_verified=identity, rank_pids=ranks, run_directory=str(metrics.parent),
    log_file=str(log_path), completed_updates=last['optimizer_step'], total_updates=66790,
    epoch_index_zero_based=last['epoch'], progress_percent=last['optimizer_step']/66790*100,
    seconds_per_update_recent_median=seconds, estimated_remaining_training_hours=max(0,66790-last['optimizer_step'])*seconds/3600,
    metric_age_seconds=age, last_metrics=last, nonfinite_steps=nonfinite, configuration_checks=checks,
    configuration=dict(gpus=8,batch_per_gpu=16,gradient_accumulation_steps=1,window_stride=3,epochs=2,save_steps=33395,keep_last_k_ckpts=2,zero_stage=2,overlap_comm=True,gradient_checkpointing=False),
    expected_weights=['checkpoint_step_33395.safetensors','checkpoint_step_66790.safetensors'],
    checkpoints_present=[p.name for p in metrics.parent.glob('checkpoint_step_*.safetensors')],
    errors_seen=bool(alerts or nonfinite), error_markers=alerts, partial_metrics_line=partial_line,
    gpu_snapshot_csv_index_used_mib_total_mib_util_percent=gpu_result.stdout.strip(), gpu_query_exit_code=gpu_result.returncode,
    disk_free_gib=shutil.disk_usage(base).free/2**30, intervention='none; healthy training' if healthy else 'inspection required; no automatic mutation')
snapshot['view_layout'] = cfg['dataloader'].get('view_layout', 'openwam_two_view')
snapshot['camera_layout'] = cfg['dataloader'].get('camera_layout')
snapshot['canvas_width_height'] = [cfg['dataloader']['width'], cfg['dataloader']['height']]
snapshot['configuration']['view_layout'] = snapshot['view_layout']
snapshot['configuration']['learning_rate'] = t['learning_rate']
snapshot['configuration']['warmup_ratio'] = t['warmup_ratio']
snapshot['configuration']['lr_scheduler'] = t['lr_scheduler']
(base / 'work/setup_logs/formal_training_status.json').write_text(json.dumps(snapshot,ensure_ascii=False,indent=2))
print(json.dumps({k:snapshot[k] for k in ('status','completed_updates','progress_percent','seconds_per_update_recent_median','metric_age_seconds','last_metrics','configuration_checks','error_markers','rank_pids','gpu_snapshot_csv_index_used_mib_total_mib_util_percent','disk_free_gib','checkpoints_present')},indent=2))
