#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN="$ROOT/experiments/resolve_transfusion_sequence_20261027"
mkdir -p "$RUN"
exec >> "$RUN/console.log" 2>&1
trap 'status=$?; echo "$status" > "$RUN/exit_code"; date -Is > "$RUN/finished_at"' EXIT
export CUDA_VISIBLE_DEVICES=GPU-bacc945a-d539-1c30-4e2c-691ee6493c87
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
date -Is > "$RUN/started_at"
echo "$$" > "$RUN/launcher.pid"
/home/kemove/anaconda3/envs/pcdet/bin/python "$ROOT/tools/train/train_transfusion.py" \
 --cfg_file cfgs/infrauni/transfusion_sequence.yaml --batch_size 4 --epochs 20 \
 --workers 4 --extra_tag resolve_sequence_20261027 --fix_random_seed \
 --ckpt_save_interval 5 --max_ckpt_save_num 4 --num_epochs_to_eval 1 \
 --logger_iter_interval 20 --wo_gpu_stat
