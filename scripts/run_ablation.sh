#!/usr/bin/env bash
# B, C1, C2, C3 on nuScenes-mini, then the constant-velocity reference.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT:$ROOT/third_party/MC-DeTra:$ROOT/third_party/TransFusion"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
PY=/home/kemove/anaconda3/envs/bevfusion/bin/python
cd "$ROOT"
"$PY" tools/build_future_tracks.py
for variant in b c1 c2 c3; do
  rm -rf "$ROOT/experiments/ablation_${variant}_mini"
  "$PY" tools/train_joint.py --variant "$variant" --epochs 2 --workers 0
  "$PY" tools/eval_joint.py --checkpoint "$ROOT/experiments/ablation_${variant}_mini/epoch_2.pt"
done
"$PY" tools/eval_joint.py --variant cv
echo "ablation finished"
