#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/kemove/devdata1/zrl/InfraUni
RUN=$ROOT/experiments/unibev_resolve_roi_v2_20261103
mkdir -p "$RUN"
exec 9>"$RUN/train.lock"
flock -n 9 || { echo 'UniBEV training already running'; exit 75; }
export PATH=/home/kemove/anaconda3/envs/ccf/bin:$PATH
export PYTHONPATH=$ROOT/third_party/UniBEV/projects/UniBEV:$ROOT/third_party/CMT/.runtime_deps:$ROOT
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT/third_party/UniBEV"
printf '%s\n' "$$" > "$RUN/launcher.pid"
date -Is > "$RUN/started_at"
printf 'running\n' > "$RUN/exit_code"
trap 's=$?; printf "%s\n" "$s" > "$RUN/exit_code"; date -Is > "$RUN/finished_at"' EXIT
args=(projects/UniBEV/configs/unibev/unibev_resolve_roi_v2.py --work-dir "$RUN" --seed 0)
if [ -n "${RESUME_CHECKPOINT:-}" ]; then
 args+=(--resume-from "$RESUME_CHECKPOINT")
elif [ -f "$RUN/latest.pth" ]; then
 args+=(--resume-from "$RUN/latest.pth")
fi
python -u tools/train_resolve.py "${args[@]}" &
printf '%s\n' "$!" > "$RUN/train.pid"
wait "$!"
