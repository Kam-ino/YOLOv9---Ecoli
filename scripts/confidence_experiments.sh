#!/usr/bin/env bash
# scripts/confidence_experiments.sh — which training change raises the
# confidence of correct detections without hurting accuracy? All on fold 0
# (train on its expanded, pseudo-labelled view; score on its 15 held-out
# originals with human labels). YOLOv9-c unless noted; ~15-45 min each.
#
#   A  baseline        29 ep, cls 0.5            (exists: fold0_x8_full)
#   B  longer          90 ep
#   C  cls gain 2.0    29 ep, cls 2.0
#   D  single class    29 ep, ecoli + cluster merged
#   E  longer + cls    90 ep, cls 2.0
#
# Results: runs/compare/confidence_eval.jsonl, summary in confidence_experiments.log.
set -u
cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
PY=.venv/Scripts/python.exe
LOG=runs/compare/confidence_experiments.log
while ! grep -q "exit=" logs/bench.log 2>/dev/null; do sleep 30; done      # let the benchmark finish first
echo "[$(date '+%F %T')] start" >> "$LOG"

run() {   # run <name> <extra compare_train args...>
  local name=$1; shift
  if ! $PY scripts/compare_train.py --algo yolov9 --expand8 --patience 0 --folds 0 --name-suffix "_$name" "$@" >> "$LOG" 2>&1; then
    echo "[$(date '+%F %T')] $name FAILED" >> "$LOG"; return 1; fi
  echo "[$(date '+%F %T')] $name trained" >> "$LOG"
}

$PY scripts/confidence_eval.py --algo yolov9 --weights runs/compare/yolov9/fold0_x8_full/weights/best.pt --label A_baseline_29ep >> "$LOG" 2>&1
$PY scripts/confidence_eval.py --algo rfdetr --weights runs/compare/rfdetr/fold0_x8_full/checkpoint_best_ema.pth --label A_rfdetr_25ep >> "$LOG" 2>&1

run full_B_90ep --epochs 90 && $PY scripts/confidence_eval.py --algo yolov9 --weights runs/compare/yolov9/fold0_x8_full_B_90ep/weights/best.pt --label B_90ep >> "$LOG" 2>&1
run full_C_cls2 --extra --cls 2.0 && $PY scripts/confidence_eval.py --algo yolov9 --weights runs/compare/yolov9/fold0_x8_full_C_cls2/weights/best.pt --label C_cls2 >> "$LOG" 2>&1
run full_D_single --extra --single-cls && $PY scripts/confidence_eval.py --algo yolov9 --weights runs/compare/yolov9/fold0_x8_full_D_single/weights/best.pt --label D_single_cls --single-cls >> "$LOG" 2>&1
run full_E_90ep_cls2 --epochs 90 --extra --cls 2.0 && $PY scripts/confidence_eval.py --algo yolov9 --weights runs/compare/yolov9/fold0_x8_full_E_90ep_cls2/weights/best.pt --label E_90ep_cls2 >> "$LOG" 2>&1

echo "[$(date '+%F %T')] done" >> "$LOG"
touch runs/compare/DONE_confidence
