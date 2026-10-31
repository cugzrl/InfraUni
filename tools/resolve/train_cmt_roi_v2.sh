#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/kemove/devdata1/zrl/InfraUni
RUN=${CMT_WORK_DIR:-$ROOT/experiments/resolve_roi_v2_20261029/cmt}
mkdir -p "$RUN"
if [ -s "$RUN/train.pid" ] && kill -0 "$(cat "$RUN/train.pid")" 2>/dev/null; then echo "This training job is already active"; exit 75; fi
exec 9>"$RUN/train.lock"
flock -n 9 || exit 75
cd "$ROOT"
export PATH=/home/kemove/anaconda3/envs/ccf/bin:$PATH
export PYTHONPATH=$ROOT/third_party/CMT:$ROOT/third_party/CMT/.runtime_deps:$ROOT
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
printf '%s\n' "$$" > "$RUN/launcher.pid"
date -Is > "$RUN/started_at"
printf "running\n" > "$RUN/exit_code"
trap 's=$?; printf "%s\n" "$s" > "$RUN/exit_code"; date -Is > "$RUN/finished_at"' EXIT
cd "$ROOT/third_party/CMT"
args=("${CMT_CONFIG:-projects/configs/resolve/cmt_resolve_r50_roi_v2.py}" --work-dir "$RUN" --seed 0)
if [ -n "${RESUME_CHECKPOINT:-}" ]; then args+=(--resume-from "$RESUME_CHECKPOINT"); fi
python -u tools/train.py "${args[@]}" &
printf '%s\n' "$!" > "$RUN/train.pid"
wait "$!"
