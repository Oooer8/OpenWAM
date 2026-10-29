#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/mnt/vepfs01/output/tiecheng.guo/envs/openwam-alpha/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8 NNODES=1 NODE_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29621
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OPENWAM_LOCAL_METRICS=1 PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
# Keep the full stride-1, two-epoch data configuration. These flags only
# change execution; benchmark/validation records are in work/setup_logs/.
export OPENWAM_CHECKPOINT_EVERY_N_LAYER="${OPENWAM_CHECKPOINT_EVERY_N_LAYER:-2}"
export OPENWAM_TEXT_CACHE_DIR="${OPENWAM_TEXT_CACHE_DIR:-/mnt/vepfs01/output/tiecheng.guo/openwam/work/cosmos3_text_cache}"
export OPENWAM_BATCH_VIDEO_PREPROCESS="${OPENWAM_BATCH_VIDEO_PREPROCESS:-1}"
export OPENWAM_COMPACT_TEXT_CONTEXT="${OPENWAM_COMPACT_TEXT_CONTEXT:-1}"
export OPENWAM_COMPILE_TRAIN_FFN="${OPENWAM_COMPILE_TRAIN_FFN:-1}"
export OPENWAM_COMPILE_NORM_GATE="${OPENWAM_COMPILE_NORM_GATE:-1}"
export TORCHINDUCTOR_COMPILE_THREADS=1 MAX_JOBS=1
export TORCHINDUCTOR_CACHE_DIR=/mnt/vepfs01/output/tiecheng.guo/openwam/work/torchinductor_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
test -f work/setup_logs/foundation.ready
test -f work/cosmos3_droid_prepared/sample_check.json
test -f "$OPENWAM_TEXT_CACHE_DIR/validation.json"
python - <<'CHECK'
import json
from pathlib import Path
for name in ('training_optimization_numerics.json', 'compact_text_context_validation.json',
             'pixel_transport_check.json', 'norm_gate_compile_check.json'):
    result = json.loads((Path('work/setup_logs') / name).read_text())
    if result.get('status') != 'passed':
        raise RuntimeError(f'Optimization validation missing or failed: {name}')
CHECK
exec bash scripts/train.sh --config-name "${OPENWAM_TRAIN_CONFIG:-train_cosmos3_droid}" "$@"
