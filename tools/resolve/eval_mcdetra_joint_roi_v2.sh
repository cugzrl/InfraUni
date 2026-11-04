#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/kemove/devdata1/zrl/InfraUni
cd "$ROOT"
export PATH=/home/kemove/anaconda3/envs/ccf/bin:$PATH
export PYTHONPATH=$ROOT:$ROOT/third_party/MC-DeTra:$ROOT/third_party/CMT/.runtime_deps
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
exec python -u tools/eval/eval_mcdetra_joint_roi_v2.py "$@"
