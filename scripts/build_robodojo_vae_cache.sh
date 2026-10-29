#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/mnt/vepfs01/output/tiecheng.guo/envs/openwam-alpha/bin:$PATH
export PYTHONPATH=. OMP_NUM_THREADS=2 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cache_root=/mnt/vepfs01/output/tiecheng.guo/RoboDojo_openwam_vae_cache_v1
# Avoid sharing devices with unrelated evaluation or training jobs.
python - <<'PY'
import subprocess
rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()
assert len(rows) == 8, rows
assert all(int(row.split(',')[1]) < 1024 for row in rows), 'GPU busy; cache generation has not started: ' + repr(rows)
PY
python scripts/cache_robodojo_vae.py --root "$cache_root" --init
mkdir -p "$cache_root/logs"
worker_pids=()
cleanup() {
    for worker_pid in "${worker_pids[@]}"; do
        kill -TERM "$worker_pid" 2>/dev/null || true
    done
}
trap cleanup INT TERM EXIT
for gpu_id in {0..7}; do
    CUDA_VISIBLE_DEVICES="$gpu_id" python scripts/cache_robodojo_vae.py --root "$cache_root" --worker "$gpu_id" --world-size 8 "$@" > "$cache_root/logs/worker_$gpu_id.log" 2>&1 &
    worker_pids+=("$!")
done
for worker_pid in "${worker_pids[@]}"; do
    wait "$worker_pid"
done
worker_pids=()
# A pilot intentionally does not publish a complete training cache.
if [[ " $* " != *" --limit-chunks "* ]]; then
    python scripts/cache_robodojo_vae.py --root "$cache_root" --finalize
fi
trap - INT TERM EXIT
