#!/usr/bin/env bash
# Official TransFusion-L test and nuScenes evaluation.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$ROOT/third_party/TransFusion"
CKPT="${1:-$ROOT/experiments/transfusion_voxel_L_mini/latest.pth}"
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
cd "$REPO"
exec /home/kemove/anaconda3/envs/bevfusion/bin/python tools/test.py \
  "$ROOT/configs/transfusion_mini_voxel_L.py" \
  "$CKPT" \
  --eval bbox \
  --out "$ROOT/experiments/transfusion_voxel_L_mini/results.pkl"
