#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/mnt/vepfs01/output/tiecheng.guo/envs/openwam-alpha/bin:$PATH
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
for ((attempt=0; attempt<180; attempt++)); do
    if [[ -f work/cosmos3_text_cache/manifest.json ]]; then break; fi
    if ! kill -0 3304742 2>/dev/null; then
        echo 'Precompute exited without a complete manifest'; exit 1
    fi
    sleep 5
done
test -f work/cosmos3_text_cache/manifest.json
python -u scripts/validate_cosmos3_text_cache.py --cache work/cosmos3_text_cache --prepared work/cosmos3_droid_prepared
python -u scripts/check_compact_text_context.py
