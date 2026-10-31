#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/kemove/devdata1/zrl/InfraUni
RUN=$ROOT/experiments/resolve_roi_v2_20261029/cmt_extended_40e
export PATH=/home/kemove/anaconda3/envs/ccf/bin:$PATH
export PYTHONPATH=$ROOT/third_party/CMT:$ROOT/third_party/CMT/.runtime_deps:$ROOT
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONWARNINGS=ignore
mkdir -p "$RUN/eval"
checkpoint=${1:-$RUN/best_NDS_epoch_26.pth}
cd "$ROOT/third_party/CMT"
exec python -u tools/test.py projects/configs/resolve/cmt_resolve_r50_roi_v2_extend.py "$checkpoint" --out "$RUN/eval/predictions.pkl" --eval bbox --eval-options jsonfile_prefix="$RUN/eval/val"
