#!/usr/bin/env bash
# Create the infrauni env by copying the local openpcdet prefix.
# conda create --clone hits repo.anaconda.com and fails offline.
set -euo pipefail
SRC="${SRC:-/home/kemove/anaconda3/envs/openpcdet}"
DST="${DST:-/home/kemove/anaconda3/envs/infrauni}"
if [[ ! -x "$SRC/bin/python" ]]; then
  echo "missing source env: $SRC" >&2
  exit 1
fi
if [[ ! -x "$DST/bin/python" ]]; then
  rm -rf "$DST"
  cp -a "$SRC" "$DST"
fi
"$DST/bin/python" -c "import torch,cv2,yaml,scipy,matplotlib; print('infrauni', torch.__version__)"
