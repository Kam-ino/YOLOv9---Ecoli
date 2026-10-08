#!/usr/bin/env bash
# scripts/deploy_round2.sh — deployable models, self-training round 2.
#
# Round 2 on fold 0 doubled recall at score >= 0.60 (0.11 -> 0.22) and raised
# mAP50 (0.417 -> 0.445): the teacher's tight boxes replace the human box
# coordinates they match and fill label gaps. This applies the same step to
# the whole dataset with the deployable round-1 YOLO as teacher, then
# retrains both deployable models on it and activates them. Waits for the
# fold-0 round-2 experiment to finish (it rebuilds the fold views).
#
#   bash scripts/deploy_round2.sh
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe
OUT=runs/compare
LOG=$OUT/deploy_round2.log
DATA=data/ecoli_x8_r2d
EPOCHS_YOLO=${EPOCHS_YOLO:-90}
while [ ! -e "$OUT/DONE_round2" ]; do sleep 30; done
echo "[$(date '+%F %T')] start (yolo epochs $EPOCHS_YOLO)" >> "$LOG"
cp -p models/best_yolov9c.pt models/best_yolov9c.round1.pt 2>/dev/null
cp -p models/best_rfdetr.pth models/best_rfdetr.round1.pth 2>/dev/null

$PY scripts/make_pseudo_dataset.py --src data/ecoli_human --dst "$DATA" --weights models/best_yolov9c.round1.pt \
    --conf 0.25 --refine >> "$LOG" 2>&1 || { echo "pseudo FAILED" >> "$LOG"; touch "$OUT/FAILED_deploy_r2"; exit 1; }

activate() {   # activate <src> <target>
  local src=$1 target=$2 ts; ts=$(date +%s)
  if [ -e "$target" ]; then cp -p "$target" "${target%.*}.bak-$ts.${target##*.}"; fi
  cp -p "$src" "$target" && echo "[$(date '+%F %T')] activated $src -> $target" >> "$LOG"
}

if $PY -m training.train --data "$DATA/data.yaml" --weights models/yolov9c.pt --epochs "$EPOCHS_YOLO" --batch 8 \
     --imgsz 640 --device 0 --workers 2 --patience 0 --save-period -1 --max-det 1200 \
     --project "$PWD/runs/train" --name deploy_yolov9_r2 >> "$LOG" 2>&1; then
  activate runs/train/deploy_yolov9_r2/weights/best.pt models/best_yolov9c.pt
else
  touch "$OUT/FAILED_deploy_r2_yolov9"; echo "[$(date '+%F %T')] yolov9 r2 FAILED" >> "$LOG"
fi

if $PY -m training.train_rfdetr --data "$DATA/data.yaml" --weights medium --epochs 25 --batch 1 \
     --imgsz 640 --device 0 --workers 2 --patience 0 --num-queries 1200 --gradient-checkpointing \
     --project "$PWD/runs/train" --name deploy_rfdetr_r2 >> "$LOG" 2>&1; then
  activate runs/train/deploy_rfdetr_r2/checkpoint_best_ema.pth models/best_rfdetr.pth
else
  touch "$OUT/FAILED_deploy_r2_rfdetr"; echo "[$(date '+%F %T')] rfdetr r2 FAILED" >> "$LOG"
fi
touch "$OUT/DONE_deploy_r2"
echo "[$(date '+%F %T')] done" >> "$LOG"
