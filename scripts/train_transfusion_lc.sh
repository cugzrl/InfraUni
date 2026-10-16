#!/usr/bin/env bash
# Official TransFusion-LC trainer. The init checkpoint must already be converted.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$ROOT/third_party/TransFusion"
INIT="$ROOT/checkpoints/fusion_voxel0075_R50_mmdet3d.pth"
if [[ ! -f "$INIT" ]]; then
  echo "missing converted init: $INIT" >&2
  exit 1
fi
export PYTHONPATH="$REPO"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
cd "$REPO"
exec /home/kemove/anaconda3/envs/bevfusion/bin/python tools/train.py \
  "$ROOT/configs/transfusion_mini_voxel_LC.py" \
  --gpus 1 \
  --work-dir "$ROOT/experiments/transfusion_voxel_LC_mini" \
  "$@"
