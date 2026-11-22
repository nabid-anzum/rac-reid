#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT="/root/rac-reid"

export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export TMPDIR="$PROJECT/.runtime/tmp"
export XDG_CACHE_HOME="$PROJECT/.runtime/cache"
export TORCH_HOME="$PROJECT/.runtime/torch"
export YOLO_CONFIG_DIR="$PROJECT/.runtime/yolo-config"
export MPLCONFIGDIR="$PROJECT/.runtime/matplotlib"
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=""

mkdir -p \
    "$PROJECT/.runtime/tmp" \
    "$PROJECT/.runtime/cache" \
    "$PROJECT/.runtime/torch" \
    "$PROJECT/.runtime/yolo-config" \
    "$PROJECT/.runtime/matplotlib"

cd "$PROJECT"

exec \
    "$PROJECT/.venv/bin/python" \
    "$PROJECT/scripts/web_app.py" \
    "$@"
