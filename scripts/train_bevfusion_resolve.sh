#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/kemove/devdata1/zrl/InfraUni
RUN=$ROOT/experiments/bevfusion_resolve_roi_v2_20261030
mkdir -p "$RUN"
exec 9>"$RUN/train.lock"
flock -n 9 || { echo "BEVFusion training is already active"; exit 75; }
cd "$ROOT"
export PATH=/home/kemove/anaconda3/envs/pcdet/bin:$PATH
export PYTHONPATH=$ROOT:$ROOT/third_party/RESOLVE
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
date -Is > "$RUN/started_at"
echo running > "$RUN/exit_code"
trap 's=$?; echo "$s" > "$RUN/exit_code"; date -Is > "$RUN/finished_at"' EXIT
python -u tools/train/train_bevfusion_resolve.py --config configs/resolve/bevfusion_roi_v2.yaml --output "$RUN" >> "$RUN/console.log" 2>&1 &
echo "$!" > "$RUN/train.pid"
wait "$!"
