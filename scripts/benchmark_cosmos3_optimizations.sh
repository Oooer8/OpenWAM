#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
variant="${1:?Select cache or compute}"
export OPENWAM_CHECKPOINT_EVERY_N_LAYER=2
export OPENWAM_COMPILE_NORM_GATE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPENWAM_TEXT_CACHE_DIR=/mnt/vepfs01/output/tiecheng.guo/openwam/work/cosmos3_text_cache
export TORCHINDUCTOR_COMPILE_THREADS=1 MAX_JOBS=1
export TORCHINDUCTOR_CACHE_DIR=/mnt/vepfs01/output/tiecheng.guo/openwam/work/torchinductor_cache
case "$variant" in
  cache)
    export OPENWAM_COMPILE_TRAIN_FFN=0 OPENWAM_BATCH_VIDEO_PREPROCESS=0 OPENWAM_COMPACT_TEXT_CONTEXT=0
    steps=24;;
  compute)
    export OPENWAM_COMPILE_TRAIN_FFN=1 OPENWAM_BATCH_VIDEO_PREPROCESS=1 OPENWAM_COMPACT_TEXT_CONTEXT=1
    test -f work/setup_logs/compact_text_context_validation.json
    steps=36;;
  *) echo 'Unknown benchmark variant'; exit 2;;
esac
exec bash scripts/train_cosmos3_droid.sh \
 training.num_epochs=null training.max_steps="$steps" training.save_steps=0 \
 training.save_full_states_for_resume=false training.lr_scheduler=null \
 training.output_path="/mnt/vepfs01/output/tiecheng.guo/openwam/outputs/cosmos3_droid_b16_${variant}_benchmark"
