#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="/root/rac-reid"

export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"

export TMPDIR="$PROJECT/.runtime/tmp"
export XDG_CACHE_HOME="$PROJECT/.runtime/cache"
export TORCH_HOME="$PROJECT/.runtime/torch"
export YOLO_CONFIG_DIR="$PROJECT/.runtime/yolo-config"
export MPLCONFIGDIR="$PROJECT/.runtime/matplotlib"

export REID_CACHE_DIR="$PROJECT/.runtime/reid-cache"
export HF_HOME="$PROJECT/.runtime/reid-cache/huggingface"
export HF_HUB_CACHE="$PROJECT/.runtime/reid-cache/huggingface/hub"

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=""

mkdir -p \
    "$TMPDIR" \
    "$XDG_CACHE_HOME" \
    "$TORCH_HOME" \
    "$YOLO_CONFIG_DIR" \
    "$MPLCONFIGDIR" \
    "$REID_CACHE_DIR" \
    "$HF_HOME" \
    "$HF_HUB_CACHE"

cd "$PROJECT"

exec \
    "$PROJECT/.venv/bin/python" \
    "$PROJECT/scripts/web_app.py" \
    "$@"
