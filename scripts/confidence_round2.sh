#!/usr/bin/env bash
# scripts/confidence_round2.sh — self-training round 2 on fold 0, measured
# on fold 0's held-out human labels.
#
# Teacher = the best fold-0 YOLO so far (90-epoch run; it never saw fold 0's
# held-out images). Its detections at >= 0.25 are added where the labels have
# gaps and, with --refine, replace the coordinates of the human boxes they
# match — Ultralytics scales the class target by box IoU, so loose boxes cap
# confidence. Then retrain fold 0 for 29 epochs on the round-2 labels and
# score. Waits for the running confidence experiments first (they use the
# current fold views, which this rebuilds).
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe
LOG=runs/compare/confidence_round2.log
while [ ! -e runs/compare/DONE_confidence ]; do sleep 30; done
echo "[$(date '+%F %T')] start" >> "$LOG"
TEACHER=runs/compare/yolov9/fold0_x8_full_B_90ep/weights/best.pt

$PY scripts/make_pseudo_dataset.py --src data/ecoli_human --dst data/ecoli_x8_r2 --weights "$TEACHER" \
    --conf 0.25 --refine >> "$LOG" 2>&1 || { echo "pseudo r2 FAILED" >> "$LOG"; exit 1; }
$PY scripts/compare_folds.py --train-from data/ecoli_x8_r2 >> "$LOG" 2>&1 || { echo "folds FAILED" >> "$LOG"; exit 1; }
$PY scripts/compare_train.py --algo yolov9 --expand8 --patience 0 --folds 0 --name-suffix _r2 >> "$LOG" 2>&1 \
  && $PY scripts/confidence_eval.py --algo yolov9 --weights runs/compare/yolov9/fold0_x8_r2/weights/best.pt --label F_round2_refine >> "$LOG" 2>&1
$PY scripts/compare_train.py --algo yolov9 --expand8 --patience 0 --folds 0 --epochs 90 --name-suffix _r2_90ep >> "$LOG" 2>&1 \
  && $PY scripts/confidence_eval.py --algo yolov9 --weights runs/compare/yolov9/fold0_x8_r2_90ep/weights/best.pt --label G_round2_refine_90ep >> "$LOG" 2>&1

# Restore the round-1 fold views so later comparison reruns use the published data.
$PY scripts/compare_folds.py --train-from data/ecoli_x8 >> "$LOG" 2>&1
echo "[$(date '+%F %T')] done" >> "$LOG"
touch runs/compare/DONE_round2
