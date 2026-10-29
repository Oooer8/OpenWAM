#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Same execution settings as the completed batch-16, 48-step benchmark.
export OPENWAM_TRAIN_CONFIG=train_cosmos3_droid_fast
export OPENWAM_TRAIN_ENTRY=scripts/train.py
unset OPENWAM_MEMORY_STRESS_AFTER_STEP
export OPENWAM_CHECKPOINT_REGION=block OPENWAM_SELECTIVE_CHECKPOINT=none
export OPENWAM_CACHED_TEXT_ENCODER_CPU=1 OPENWAM_ZERO_OVERLAP=1
export OPENWAM_VAE_CHANNELS_LAST=1 OPENWAM_COMPILE_VAE_NORM=0
export OPENWAM_COMPILE_NORM_GATE=1 OPENWAM_COMPILE_TRAIN_FFN=1
export OPENWAM_BATCH_VIDEO_PREPROCESS=1 OPENWAM_COMPACT_TEXT_CONTEXT=1
/mnt/vepfs01/output/tiecheng.guo/envs/openwam-alpha/bin/python - <<'CHECK'
import json
from pathlib import Path
for name in ('no_checkpoint_validation.json', 'vae_memory_format_check.json'):
    result = json.loads((Path('work/setup_logs') / name).read_text())
    if result.get('status') != 'passed':
        raise RuntimeError(f'Optimization validation missing or failed: {name}')
CHECK
exec bash scripts/train_cosmos3_droid.sh "$@"
