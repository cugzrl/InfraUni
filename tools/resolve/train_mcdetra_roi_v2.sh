#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/kemove/devdata1/zrl/InfraUni
RUN=$ROOT/experiments/resolve_roi_v2_20261029/mc-detra
mkdir -p "$RUN"
if [ -s "$RUN/train.pid" ] && kill -0 "$(cat "$RUN/train.pid")" 2>/dev/null; then echo "This training job is already active"; exit 75; fi
exec 9>"$RUN/train.lock"
flock -n 9 || exit 75
cd "$ROOT"
export PATH=/home/kemove/anaconda3/envs/ccf/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$ROOT:$ROOT/third_party/MC-DeTra:$ROOT/third_party/CMT/.runtime_deps
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export INFRAUNI_SHARED_PROTOCOL=1 INFRAUNI_SHARED_EVAL_DIR=$RUN/val
printf '%s\n' "$$" > "$RUN/launcher.pid"
date -Is > "$RUN/started_at"
printf "running\n" > "$RUN/exit_code"
trap 's=$?; printf "%s\n" "$s" > "$RUN/exit_code"; date -Is > "$RUN/finished_at"' EXIT
args=(--experiment-json "$ROOT/configs/resolve/mcdetra_roi_v2.json")
if [ -n "${RESUME_CHECKPOINT:-}" ]; then args+=(--resume-checkpoint "$RESUME_CHECKPOINT"); fi
python -u "$ROOT/tools/train/train_mcdetra_resolve_v2.py" "${args[@]}" &
printf '%s\n' "$!" > "$RUN/train.pid"
wait "$!"
