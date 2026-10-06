#!/usr/bin/env bash
# scripts/compare_train_all.sh — the full 5-fold run for both models, detached.
#
#   bash scripts/compare_train_all.sh            # from the repo root
#
# Trains YOLOv9-c on all folds first (fast), then RF-DETR, using the
# pre-expanded, pseudo-labelled fold views built by
#   scripts/compare_folds.py --train-from data/ecoli_x8
# (so compare_train.py --expand8 only scales the epoch budget; the trainers
# receive already-expanded data). Settings decided from the smoke runs on
# 2026-10-07: RF-DETR batch 1 x 16 accumulation + gradient checkpointing
# (2.9 GB, ~2 img/s on the RTX 5060 Laptop while the app backend holds 2 GB);
# YOLO batch 8. Progress: runs/compare/train_all.log; per-fold logs under
# runs/compare/<algo>/fold<k>_x8.log; marker files runs/compare/DONE_<algo>
# and runs/compare/DONE_ALL (or FAILED_<algo>) for pollers.
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe
OUT=runs/compare
LOG=$OUT/train_all.log
mkdir -p "$OUT"
rm -f "$OUT"/DONE_* "$OUT"/FAILED_*
echo "[$(date '+%F %T')] start" >> "$LOG"

if $PY scripts/compare_train.py --algo yolov9 --expand8 --folds 0 1 2 3 4 >> "$LOG" 2>&1; then
  touch "$OUT/DONE_yolov9"; echo "[$(date '+%F %T')] yolov9 done" >> "$LOG"
else
  touch "$OUT/FAILED_yolov9"; echo "[$(date '+%F %T')] yolov9 FAILED" >> "$LOG"
fi

if $PY scripts/compare_train.py --algo rfdetr --expand8 --batch 1 --gradient-checkpointing --folds 0 1 2 3 4 >> "$LOG" 2>&1; then
  touch "$OUT/DONE_rfdetr"; echo "[$(date '+%F %T')] rfdetr done" >> "$LOG"
else
  touch "$OUT/FAILED_rfdetr"; echo "[$(date '+%F %T')] rfdetr FAILED" >> "$LOG"
fi

touch "$OUT/DONE_ALL"
echo "[$(date '+%F %T')] all done" >> "$LOG"
