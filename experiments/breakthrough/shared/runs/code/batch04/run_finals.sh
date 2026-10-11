#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/../../../.."
PY=".venv/Scripts/python.exe"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
for spec in "repeat_bag 20261011" "repeat_bag 20261012" "repeat_bag 20261013" "diverse_bag 20261011" "diverse_bag 20261012" "diverse_bag 20261013"; do
  set -- $spec
  arm=$1; seed=$2
  echo "=== RUN $arm seed $seed $(date '+%H:%M:%S') ==="
  $PY -X utf8 runs/breakthrough_20261002/code/batch04/train_bag_v2.py \
    --arm "$arm" --seed "$seed" --lr 0.001 --epochs 12 --batch 32 \
    --attempt-dir "${arm}_s${seed}" \
    > "runs/breakthrough_20261002/batch04/log_${arm}_s${seed}.txt" 2>&1
  echo "=== DONE $arm seed $seed $(date '+%H:%M:%S') ==="
done
echo "ALL_FINALS_DONE"
