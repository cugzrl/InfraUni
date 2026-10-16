#!/usr/bin/env bash
# Ablation train. Pass --variant b, c1, c2, or c3.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT:$ROOT/third_party/MC-DeTra:$ROOT/third_party/TransFusion"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
exec /home/kemove/anaconda3/envs/bevfusion/bin/python "$ROOT/tools/train_joint.py" "$@"
