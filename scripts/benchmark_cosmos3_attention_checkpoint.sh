#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENWAM_CHECKPOINT_EVERY_N_LAYER=2 OPENWAM_CHECKPOINT_REGION=attention
export OPENWAM_SELECTIVE_CHECKPOINT=none OPENWAM_CACHED_TEXT_ENCODER_CPU=1
export OPENWAM_ZERO_OVERLAP=1 OPENWAM_VAE_CHANNELS_LAST=1
export OPENWAM_COMPILE_VAE_NORM="${OPENWAM_COMPILE_VAE_NORM:-0}"
export OPENWAM_COMPILE_NORM_GATE=1 OPENWAM_COMPILE_TRAIN_FFN=1
export OPENWAM_BATCH_VIDEO_PREPROCESS=1 OPENWAM_COMPACT_TEXT_CONTEXT=1
/mnt/vepfs01/output/tiecheng.guo/envs/openwam-alpha/bin/python - <<'CHECK'
import json, os
from pathlib import Path
name = 'vae_norm_compile_check.json' if os.environ['OPENWAM_COMPILE_VAE_NORM']=='1' else 'vae_memory_format_check.json'
assert json.loads((Path('work/setup_logs')/name).read_text())['status']=='passed'
CHECK
exec bash scripts/train_cosmos3_droid.sh \
 training.num_epochs=null training.max_steps=36 training.save_steps=0 \
 training.save_full_states_for_resume=false training.lr_scheduler=null \
 training.output_path=/mnt/vepfs01/output/tiecheng.guo/openwam/outputs/cosmos3_droid_b16_attention_checkpoint_benchmark
