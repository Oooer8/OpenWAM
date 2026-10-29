#!/usr/bin/env bash
# Execute once on each allocated worker; do not run on the developer machine.
set -euo pipefail
export PATH=/mnt/vepfs01/output/tiecheng.guo/envs/openwam-alpha/bin:$PATH
export OPENWAM_JOB_ENTRY="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
source "$(dirname "$0")/robodojo_job_guard.sh"
cd "$(dirname "$0")/.."
fail() { echo "FATAL: $*" >&2; exit 1; }
[[ $# == 0 ]] || fail 'This resume entry takes no CLI arguments; use the documented environment variables.'
export PATH=/mnt/vepfs01/output/tiecheng.guo/envs/openwam-alpha/bin:$PATH
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
# Bound import-time compiler workers; this changes compilation concurrency only.
export TORCHINDUCTOR_COMPILE_THREADS=1
export NNODES="${MLP_WORKER_NUM:-${NNODES:-1}}"
export NODE_RANK="${MLP_ROLE_INDEX:-${NODE_RANK:-0}}"
export NPROC_PER_NODE="${MLP_WORKER_GPU:-${NPROC_PER_NODE:-8}}"
export MASTER_ADDR="${MLP_WORKER_0_HOST:-${MASTER_ADDR:-}}"
export MASTER_PORT="${MLP_WORKER_0_PORT:-${MASTER_PORT:-29635}}"
for name in NNODES NODE_RANK NPROC_PER_NODE MASTER_PORT; do
    [[ ${!name} =~ ^[0-9]+$ ]] || fail "$name must be an integer"
done
((NNODES > 0 && NPROC_PER_NODE > 0 && NODE_RANK < NNODES && MASTER_PORT > 0 && MASTER_PORT < 65536)) || fail 'Invalid topology'
((NNODES * NPROC_PER_NODE == 8)) || fail 'Full-state resume requires total8 GPUs to retain ZeRO optimizer partitioning and global batch128.'
if ((NNODES > 1)); then
    [[ -n $MASTER_ADDR && $MASTER_ADDR != localhost && $MASTER_ADDR != 127.* && $MASTER_ADDR != ::1 ]] || fail 'Worker0 host is required for multiple nodes'
else
    export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi
# Respect scheduler GPU assignment; only use all8 when no selection exists.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NPROC_PER_NODE-1)))}"
export RESUME_CKPT_DIR="${RESUME_CKPT_DIR:-/mnt/vepfs01/output/tiecheng.guo/robodojo_runs/20261021_8x16_gc6_verified/training/2026-10-21_03-07-54}"
export OPENWAM_TRACKER=swanlab WANDB_MODE=disabled
export OPENWAM_SWANLAB_PROJECT=openwam-robodojo
export OPENWAM_SWANLAB_RUN_NAME=robodojo_alpha_cached_8x16_gc6_verified_20261021
export OPENWAM_SWANLAB_RESUME_ID=e47552zz
export OPENWAM_SWANLAB_KEY_FILE="${OPENWAM_SWANLAB_KEY_FILE:-/mnt/vepfs01/output/tiecheng.guo/.secrets/swanlab_api_key}"
export OPENWAM_SWANLAB_LOGDIR="$RESUME_CKPT_DIR/../../swanlog_volcano"
export OPENWAM_TRAIN_ENTRY=scripts/train.py
export OPENWAM_CHECKPOINT_PATTERN=100000 OPENWAM_ZERO_OVERLAP=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
unset OPENWAM_VALIDATE_CHECKPOINT_STEP OPENWAM_VERIFY_CHECKPOINT_RESTORE OPENWAM_PROFILE_DIR OPENWAM_BENCH_ACCUMULATION
# Proxies must not intercept scheduler rendezvous/NCCL. SwanLab uses direct HTTPS.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
if [[ ${DRY_RUN:-0} != 1 ]]; then
source scripts/robodojo_runtime_env.sh
export OPENWAM_JOB_STAGE=runtime_preflight
openwam_job_record preflight 0
export OPENWAM_PREFLIGHT_TIMEOUT_SECONDS="${OPENWAM_PREFLIGHT_TIMEOUT_SECONDS:-900}"
[[ $OPENWAM_PREFLIGHT_TIMEOUT_SECONDS =~ ^[0-9]+$ ]] && ((OPENWAM_PREFLIGHT_TIMEOUT_SECONDS >= 60 && OPENWAM_PREFLIGHT_TIMEOUT_SECONDS <= 3600)) || fail 'Preflight timeout must be60..3600 seconds'
export OPENWAM_PREFLIGHT_LOG="$RESUME_CKPT_DIR/../../runtime_preflight_node_${NODE_RANK}.log"
echo "PREFLIGHT: timeout=${OPENWAM_PREFLIGHT_TIMEOUT_SECONDS}s; stack dumps every60s; log=$OPENWAM_PREFLIGHT_LOG"
timeout --kill-after=10s "${OPENWAM_PREFLIGHT_TIMEOUT_SECONDS}s" python -u -X faulthandler scripts/check_robodojo_runtime.py --checkpoint-dir "$RESUME_CKPT_DIR" --gpus "$NPROC_PER_NODE" > >(tee -a "$OPENWAM_PREFLIGHT_LOG") 2>&1 &
OPENWAM_TRAIN_PID=$!
openwam_wait_child
python - <<'PY'
import datetime, json, os, socket, sys
from pathlib import Path
import torch
sys.path.insert(0, 'scripts')
from monitor_robodojo_training import weights_valid, state_valid
p = Path(os.environ['RESUME_CKPT_DIR']).resolve()
assert torch.cuda.device_count() == int(os.environ['NPROC_PER_NODE']), 'Scheduler-visible GPU count differs'
completed = [int(x.name.split('_')[-1]) for x in p.glob('accel_state_step_*')
             if (x/'trainer_state.json').is_file() and x.name.split('_')[-1].isdigit()]
assert completed and max(completed) >= 20000, 'No completed full state at or beyond20k'
step = max(completed)
assert weights_valid(p/f'checkpoint_step_{step}.safetensors') and state_valid(p/f'accel_state_step_{step}',step), 'Latest completed checkpoint is not complete/readable'
stop=p.parent.parent/'planned_stop_20000.json'
assert stop.exists() and json.loads(stop.read_text())['status']=='stopped', 'Developer-machine job has not confirmed stopped at checkpoint20k'
assert os.environ.get('SWANLAB_API_KEY') or Path(os.environ['OPENWAM_SWANLAB_KEY_FILE']).is_file(), 'Provide SWANLAB_API_KEY or a mounted key file'
if int(os.environ['NODE_RANK']) == 0:
    pid = os.getppid()  # Shell exec chain preserves this PID through torchrun.
    stat = Path(f'/proc/{pid}/stat').read_text()
    receipt = dict(hostname=socket.gethostname(), pid=pid,
                   start_ticks=stat[stat.rfind(')')+2:].split()[19],
                   launched_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   resume_step=step, total_steps=60000, world_size=8,
                   batch_size_per_gpu=16, keep_last_k=3, checkpoint_directory=str(p))
    destination=p.parent.parent/'volcano_resume_launch.json'
    temporary=destination.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(receipt, indent=2)+'\n')
    temporary.replace(destination)
print(f'Preflight passed: checkpoint{step}, original job stopped, total8 GPUs, global batch128, scheduler total60000.')
PY
else
    echo "DRY_RUN: preflight skipped; no training or GPU work started."
fi
args=(--config-name train_robodojo_cached_swanlab
    training.finetune_ckpt_path=null "training.resume_ckpt_path=$RESUME_CKPT_DIR"
    "training.output_path=$(dirname "$RESUME_CKPT_DIR")"
    "project.wandb.run_name=$OPENWAM_SWANLAB_RUN_NAME"
    training.num_epochs=null training.max_steps=60000
    training.batch_size=16 training.gradient_accumulation_steps=1
    training.save_steps=2000 training.save_full_states_for_resume=true
    training.keep_last_k_ckpts=3)
echo "nodes=$NNODES rank=$NODE_RANK GPUs/node=$NPROC_PER_NODE total_GPUs=8 global_batch=128 resume=latest_complete_at_or_after_20000 total_steps=60000"
if [[ ${DRY_RUN:-0} == 1 ]]; then
    printf 'command: bash scripts/train_robodojo_cached.sh'; printf ' %q' "${args[@]}"; printf '\n'
    exit 0
fi
export OPENWAM_JOB_STAGE=training
openwam_job_record training 0
bash scripts/train_robodojo_cached.sh "${args[@]}" &
OPENWAM_TRAIN_PID=$!
openwam_wait_child
