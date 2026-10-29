#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/mnt/vepfs01/output/tiecheng.guo/envs/openwam-alpha/bin:$PATH
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENWAM_LOCAL_METRICS=1 PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export OPENWAM_TEXT_CACHE_DIR=/mnt/vepfs01/output/tiecheng.guo/RoboDojo_openwam_prepared_v1/text_cache
export OPENWAM_CACHED_TEXT_ENCODER_CPU=1
# Existing bitwise-validated uint8 batch transport; full 512-token T5 context.
export OPENWAM_BATCH_VIDEO_PREPROCESS=1
export OPENWAM_COMPACT_TEXT_CONTEXT=0
export OPENWAM_COMPILE_TRAIN_FFN=0 OPENWAM_COMPILE_NORM_GATE=0
export OPENWAM_VAE_CHANNELS_LAST=0 OPENWAM_COMPILE_VAE_NORM=0
export OPENWAM_CHECKPOINT_EVERY_N_LAYER="${OPENWAM_CHECKPOINT_EVERY_N_LAYER:-1}"
export OPENWAM_SELECTIVE_CHECKPOINT=none OPENWAM_CHECKPOINT_REGION=block
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Respect explicit GPU selection, including during short verification runs.
export NPROC_PER_NODE="${NPROC_PER_NODE:-$(python -c 'import torch; print(torch.cuda.device_count())')}"
export NNODES="${NNODES:-1}" NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}" MASTER_PORT="${MASTER_PORT:-29631}"
python - <<'PY'
import json
from pathlib import Path
root = Path("/mnt/vepfs01/output/tiecheng.guo/RoboDojo_openwam_prepared_v1")
for path in ("manifest.json", "validation.json", "text_cache/validation.json"):
    result = json.loads((root / path).read_text())
    assert result["status"] in ("ready", "passed"), f"Offline validation not complete: {path}"
PY
exec bash scripts/train.sh --config-name train_robodojo_prepared "$@"
