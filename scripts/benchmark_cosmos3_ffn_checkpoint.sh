#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENWAM_CHECKPOINT_EVERY_N_LAYER=2 OPENWAM_CHECKPOINT_REGION=ffn
export OPENWAM_SELECTIVE_CHECKPOINT=none
export OPENWAM_COMPILE_NORM_GATE=1 OPENWAM_COMPILE_TRAIN_FFN=1
export OPENWAM_BATCH_VIDEO_PREPROCESS=1 OPENWAM_COMPACT_TEXT_CONTEXT=1
exec bash scripts/train_cosmos3_droid.sh \
 training.num_epochs=null training.max_steps=36 training.save_steps=0 \
 training.save_full_states_for_resume=false training.lr_scheduler=null \
 training.output_path=/mnt/vepfs01/output/tiecheng.guo/openwam/outputs/cosmos3_droid_b16_ffn_checkpoint_benchmark
