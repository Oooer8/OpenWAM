#!/usr/bin/env bash
# Numerically validated native-layout VAE fusion + 2/3 block checkpointing.
# Defaults to the requested eight GPUs, per-GPU batch 32. Keeps the base
# dataset, augmentation, optimizer, precision, full text context, and LR schedule.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/mnt/vepfs01/output/tiecheng.guo/envs/openwam-alpha/bin:$PATH
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export OPENWAM_CHECKPOINT_PATTERN="${OPENWAM_CHECKPOINT_PATTERN:-110}"
export OPENWAM_EXACT_VAE_NORM="${OPENWAM_EXACT_VAE_NORM:-divide_affine}"
python - <<'PY'
import json
import os
import subprocess
from pathlib import Path

root = Path("/mnt/vepfs01/output/tiecheng.guo/robodojo_performance_20261020/exact_optimization")
assert json.loads((root/"checkpoint_deterministic_check.json").read_text())["status"] == "passed"
assert "divide_affine" in json.loads((root/"vae_check.json").read_text())["accepted_modes"]
comparison = json.loads((root/"comparison.json").read_text())
assert comparison["status"] == "completed" and comparison["same_model_data_training_config"]
devices = os.environ["CUDA_VISIBLE_DEVICES"]
rows = subprocess.check_output(
    ["nvidia-smi", "--id=" + devices, "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
    text=True,
).splitlines()
if any(int(row.split(",")[1]) >= 1024 for row in rows):
    raise SystemExit("Selected GPUs are busy; choose free GPUs with CUDA_VISIBLE_DEVICES.")
PY
exec bash scripts/train_robodojo_prepared.sh \
  training.batch_size=32 training.use_gradient_checkpointing=true "$@"
