#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/kemove/anaconda3/envs/bevfusion/bin/python
MODE="${1:-det}"
"$PY" tools/train.py --config configs/mini.yaml --mode "$MODE" --grad-check
