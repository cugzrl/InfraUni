#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/kemove/devdata1/zrl/InfraUni
RUN=$ROOT/experiments/resolve_roi_v2_20261029/transfusion
mkdir -p "$RUN"
if [ -s "$RUN/train.pid" ] && kill -0 "$(cat "$RUN/train.pid")" 2>/dev/null; then echo "This training job is already active"; exit 75; fi
exec 9>"$RUN/train.lock"
flock -n 9 || exit 75
cd "$ROOT"
export PATH=/home/kemove/anaconda3/envs/pcdet/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$ROOT:$ROOT/third_party/RESOLVE
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
printf '%s\n' "$$" > "$RUN/launcher.pid"
date -Is > "$RUN/started_at"
printf "running\n" > "$RUN/exit_code"
trap 's=$?; printf "%s\n" "$s" > "$RUN/exit_code"; date -Is > "$RUN/finished_at"' EXIT
python -u "$ROOT/tools/train/train_transfusion.py" --cfg_file cfgs/infrauni/transfusion_roi_v2_allframes.yaml --batch_size 4 --epochs "${EPOCHS:-20}" --workers 4 --extra_tag roi_v2_20261029 --fix_random_seed --ckpt_save_interval 1 --max_ckpt_save_num 4 --num_epochs_to_eval 1 --logger_iter_interval 20 --wo_gpu_stat &
printf '%s\n' "$!" > "$RUN/train.pid"
wait "$!"
