#!/usr/bin/env bash
# The same command runs once on every Volcano worker.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHON_BIN="${PYTHON_BIN:-${OPENWAM_PYTHON:-/mnt/vepfs01/output/tiecheng.guo/envs/openwam-alpha/bin/python}}"
export NNODES="${MLP_WORKER_NUM:-${NNODES:-4}}"
export NPROC_PER_NODE="${MLP_WORKER_GPU:-${NPROC_PER_NODE:-8}}"
export NODE_RANK="${MLP_ROLE_INDEX:-${NODE_RANK:-0}}"
export MASTER_ADDR="${MLP_WORKER_0_HOST:-${MASTER_ADDR:-}}"
export MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-29659}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}" TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export DRY_RUN="${DRY_RUN:-1}" ACTION="${ACTION:-run}"
exec "$PYTHON_BIN" -u "$ROOT/launch.py"
