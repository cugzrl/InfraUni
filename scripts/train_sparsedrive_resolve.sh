#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="$ROOT/experiments/sparsedrive_camera_resolve_20261028"
mkdir -p "$RUN"
rm -f "$RUN/exit_code" "$RUN/finished_at"
exec >> "$RUN/console.log" 2>&1
trap 'status=$?; echo "$status" > "$RUN/exit_code"; date -Is > "$RUN/finished_at"' EXIT
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/third_party/SparseDrive/.runtime:$ROOT/third_party/SparseDrive:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
date -Is > "$RUN/started_at"
echo "$$" > "$RUN/launcher.pid"
cd "$ROOT"
/home/kemove/anaconda3/envs/bevspread/bin/python tools/train/train_sparsedrive_resolve.py --workers 4
