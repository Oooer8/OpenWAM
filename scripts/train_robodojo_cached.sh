#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
# Verified on 8 A800 GPUs, batch16/GA1: 1.930 s/update, 73.09 GiB peak allocated.
# Keep mixed-attention checkpointing; attention-only without block GC ran out of memory.
export OPENWAM_CHECKPOINT_PATTERN="${OPENWAM_CHECKPOINT_PATTERN:-100000}"
export OPENWAM_ZERO_OVERLAP="${OPENWAM_ZERO_OVERLAP:-1}"
export OPENWAM_EXACT_VAE_NORM=none
exec bash scripts/train_robodojo_prepared.sh --config-name train_robodojo_cached "$@"
