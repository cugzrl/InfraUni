#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/kemove/anaconda3/envs/bevfusion/bin/python
CKPT="${1:-experiments/mini_det/last.pt}"
"$PY" tools/infer.py --config configs/mini.yaml --checkpoint "$CKPT"
