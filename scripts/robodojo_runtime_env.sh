#!/usr/bin/env bash
# Rebuildable compiler/autotune files belong to this worker's local filesystem.
# No model, dataset, checkpoint, Python environment or user HOME is relocated.
export DS_ACCELERATOR=cuda
export TORCHINDUCTOR_COMPILE_THREADS=1 MAX_JOBS=1
if [[ -z ${OPENWAM_LOCAL_CACHE_ROOT:-} ]]; then
    export OPENWAM_LOCAL_CACHE_ROOT
    OPENWAM_LOCAL_CACHE_ROOT=$(mktemp -d "/tmp/openwam-runtime.${UID}.n${NODE_RANK:-0}.XXXXXX")
fi
mkdir -p -- "$OPENWAM_LOCAL_CACHE_ROOT"/{triton,extensions,inductor,cuda}
export TRITON_CACHE_DIR="$OPENWAM_LOCAL_CACHE_ROOT/triton"
export TORCH_EXTENSIONS_DIR="$OPENWAM_LOCAL_CACHE_ROOT/extensions"
export TORCHINDUCTOR_CACHE_DIR="$OPENWAM_LOCAL_CACHE_ROOT/inductor"
export CUDA_CACHE_PATH="$OPENWAM_LOCAL_CACHE_ROOT/cuda"
echo "PREFLIGHT: worker-local compiler caches: $OPENWAM_LOCAL_CACHE_ROOT"
