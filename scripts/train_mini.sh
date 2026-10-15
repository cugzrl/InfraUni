#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/home/kemove/anaconda3/envs/infrauni/bin/python}"
"$PY" tools/train.py --config configs/nuscenes_mini.yaml "$@"
"$PY" tools/infer.py --config configs/nuscenes_mini.yaml \
  --ckpt experiments/nuscenes_mini/last.pt --split val --vis 6
