#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/kemove/devdata1/zrl/InfraUni
RUN=$ROOT/experiments/unibev_resolve_roi_v2_20261103
CHECKPOINT=${1:?Pass an absolute checkpoint path}
OUT=${2:-$RUN/eval_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$OUT"
export PATH=/home/kemove/anaconda3/envs/ccf/bin:$PATH
export PYTHONPATH=$ROOT/third_party/UniBEV/projects/UniBEV:$ROOT/third_party/CMT/.runtime_deps:$ROOT
export CUDA_VISIBLE_DEVICES=${UNIBEV_EVAL_GPU:-0} OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
cd "$ROOT/third_party/UniBEV"
python -u tools/test_resolve.py projects/UniBEV/configs/unibev/unibev_resolve_roi_v2.py "$CHECKPOINT" --eval bbox --out "$OUT/predictions.pkl" --eval-options jsonfile_prefix="$OUT/val" 2>&1 | tee "$OUT/eval.log"
