#!/usr/bin/env bash
# Official TransFusion-L trainer. Run from anywhere.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$ROOT/third_party/TransFusion"
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
cd "$REPO"
exec /home/kemove/anaconda3/envs/bevfusion/bin/python tools/train.py \
  "$ROOT/configs/transfusion_mini_voxel_L.py" \
  --gpus 1 \
  --work-dir "$ROOT/experiments/transfusion_voxel_L_mini" \
  "$@"
