#!/usr/bin/env bash
# scripts/compare_deploy_all.sh — deployable models on the whole data/ecoli
# (552 images: train split for fitting, val split for checkpoint selection),
# run after the fold training has finished, then activated into models/.
#
#   bash scripts/compare_deploy_all.sh      # waits for runs/compare/DONE_rfdetr
#
# Budgets match the fold runs (same optimiser steps as the thesis recipe on
# 8x the images): YOLOv9-c 30 epochs batch 8, RF-DETR 25 epochs batch 1 x16
# accumulation with gradient checkpointing, 1,200 queries. Previous active
# weights are kept as models/<name>.bak-<unix_ts>.<ext>, like the app does.
# Markers: runs/compare/DONE_deploy or FAILED_deploy_<algo>.
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe
OUT=runs/compare
LOG=$OUT/deploy_all.log
mkdir -p "$OUT" models
echo "[$(date '+%F %T')] waiting for fold training" >> "$LOG"
while [ ! -e "$OUT/DONE_rfdetr" ] && [ ! -e "$OUT/FAILED_rfdetr" ]; do sleep 60; done
echo "[$(date '+%F %T')] start deployable runs" >> "$LOG"

activate() {   # activate <src> <target>
  local src=$1 target=$2 ts; ts=$(date +%s)
  if [ -e "$target" ]; then cp -p "$target" "${target%.*}.bak-$ts.${target##*.}"; fi
  cp -p "$src" "$target" && echo "[$(date '+%F %T')] activated $src -> $target" >> "$LOG"
}

if $PY -m training.train --data training/dataset.yaml --weights models/yolov9c.pt --epochs 30 --batch 8 \
     --imgsz 640 --device 0 --workers 2 --patience 0 --save-period -1 --max-det 1200 \
     --project "$PWD/runs/train" --name deploy_yolov9 >> "$LOG" 2>&1; then
  activate runs/train/deploy_yolov9/weights/best.pt models/best_yolov9c.pt
else
  touch "$OUT/FAILED_deploy_yolov9"; echo "[$(date '+%F %T')] deploy yolov9 FAILED" >> "$LOG"
fi

if $PY -m training.train_rfdetr --data training/dataset.yaml --weights medium --epochs 25 --batch 1 \
     --imgsz 640 --device 0 --workers 2 --patience 0 --num-queries 1200 --gradient-checkpointing \
     --project "$PWD/runs/train" --name deploy_rfdetr >> "$LOG" 2>&1; then
  activate runs/train/deploy_rfdetr/checkpoint_best_ema.pth models/best_rfdetr.pth
else
  touch "$OUT/FAILED_deploy_rfdetr"; echo "[$(date '+%F %T')] deploy rfdetr FAILED" >> "$LOG"
fi
touch "$OUT/DONE_deploy"
echo "[$(date '+%F %T')] deploy done" >> "$LOG"
