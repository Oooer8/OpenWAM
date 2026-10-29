#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/mnt/vepfs01/output/tiecheng.guo/envs/openwam-alpha/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OPENWAM_LOCAL_METRICS=1 PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 NCCL_DEBUG=WARN
export OPENWAM_CHECKPOINT_EVERY_N_LAYER=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPENWAM_TEXT_CACHE_DIR=/mnt/vepfs01/output/tiecheng.guo/openwam/work/cosmos3_text_cache
export OPENWAM_COMPILE_TRAIN_FFN=1 OPENWAM_BATCH_VIDEO_PREPROCESS=1 OPENWAM_COMPACT_TEXT_CONTEXT=1
export TORCHINDUCTOR_COMPILE_THREADS=1 MAX_JOBS=1
export TORCHINDUCTOR_CACHE_DIR=/mnt/vepfs01/output/tiecheng.guo/openwam/work/torchinductor_cache
export OPENWAM_PROFILE_DIR=/mnt/vepfs01/output/tiecheng.guo/openwam/work/phase_profile_optimized_b16
exec torchrun --nnodes 1 --nproc_per_node 8 --node_rank 0 --master_addr 127.0.0.1 --master_port 29622 \
 scripts/profile_cosmos3_droid.py --config-name train_cosmos3_droid \
 training.num_epochs=null training.max_steps=24 training.save_steps=0 \
 training.save_full_states_for_resume=false training.lr_scheduler=null \
 training.output_path=/mnt/vepfs01/output/tiecheng.guo/openwam/outputs/cosmos3_droid_optimized_phase_profile
