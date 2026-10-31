#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/kemove/devdata1/zrl/InfraUni
export CMT_WORK_DIR=$ROOT/experiments/resolve_roi_v2_20261029/cmt_extended_40e
export CMT_CONFIG=projects/configs/resolve/cmt_resolve_r50_roi_v2_extend.py
if [ -z "${RESUME_CHECKPOINT:-}" ]; then
  if [ -f "$CMT_WORK_DIR/latest.pth" ]; then RESUME_CHECKPOINT=$CMT_WORK_DIR/latest.pth; else RESUME_CHECKPOINT=$CMT_WORK_DIR/resume_epoch20.pth; fi
fi
export RESUME_CHECKPOINT
exec bash "$ROOT/tools/resolve/train_cmt_roi_v2.sh"
