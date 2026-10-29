#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENWAM_CHECKPOINT_REGION=block OPENWAM_CHECKPOINT_EVERY_N_LAYER=2
export OPENWAM_SELECTIVE_CHECKPOINT=none OPENWAM_CACHED_TEXT_ENCODER_CPU=1
export OPENWAM_ZERO_OVERLAP=1 OPENWAM_VAE_CHANNELS_LAST=1 OPENWAM_COMPILE_VAE_NORM=0
export OPENWAM_COMPILE_NORM_GATE=1 OPENWAM_COMPILE_TRAIN_FFN=1
export OPENWAM_BATCH_VIDEO_PREPROCESS=1 OPENWAM_COMPACT_TEXT_CONTEXT=1
export OPENWAM_TRAIN_ENTRY=scripts/benchmark_no_checkpoint.py
export OPENWAM_MEMORY_STRESS_AFTER_STEP=36
bench_args=()
bench_name=cosmos3_droid_b16_no_checkpoint_benchmark
if [[ "${OPENWAM_BENCH_BATCH:-16}" == "8" ]]; then
    export OPENWAM_MEMORY_STRESS_AFTER_STEP=72
    bench_args=(training.batch_size=8 training.gradient_accumulation_steps=2)
    bench_name=cosmos3_droid_b8_ga2_no_checkpoint_benchmark
fi
/mnt/vepfs01/output/tiecheng.guo/envs/openwam-alpha/bin/python - <<'CHECK'
import json
from pathlib import Path
assert json.loads(Path('work/setup_logs/vae_memory_format_check.json').read_text())['status']=='passed'
CHECK
exec bash scripts/train_cosmos3_droid.sh \
 training.use_gradient_checkpointing=false model.architecture.mot_checkpoint_mixed_attn=false \
 training.num_epochs=null training.max_steps=$((OPENWAM_MEMORY_STRESS_AFTER_STEP * 4 / 3)) training.save_steps=0 \
 training.save_full_states_for_resume=false training.lr_scheduler=null \
 training.output_path=/mnt/vepfs01/output/tiecheng.guo/openwam/outputs/$bench_name "${bench_args[@]}"
