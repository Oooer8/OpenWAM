#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Preflight verified 4,274,500 windows / global batch 128 -> 33,395
# updates per epoch with Accelerate's padded 8-rank BatchSamplerShard.
# The two retained checkpoint_step files are exactly the two epoch endpoints.
/mnt/vepfs01/output/tiecheng.guo/envs/openwam-alpha/bin/python - <<'CHECK'
import json
from pathlib import Path
p = json.loads(Path('work/setup_logs/stride3_training_preflight.json').read_text())
assert p['status'] == 'passed' and p['window_stride'] == 3
assert p['micro_steps_per_epoch'] == p['save_steps'] == 33395
assert p['batch_per_gpu'] == 16 and p['num_gpus'] == 8
CHECK
exec bash scripts/train_cosmos3_droid_fast.sh \
    dataloader.window_stride=3 \
    training.num_epochs=2 training.max_steps=null \
    training.batch_size=16 training.gradient_accumulation_steps=1 \
    training.save_steps=33395 training.keep_last_k_ckpts=2 \
    training.save_full_states_for_resume=true \
    training.output_path=/mnt/vepfs01/output/tiecheng.guo/openwam/outputs/cosmos3_droid_alpha_stride3
