"""
scripts/compare_train.py
========================
Train one algorithm on one or more folds of the comparison plan. Each
algorithm's recipe is pinned here so every fold (and every re-run) uses
identical settings.

    python scripts/compare_train.py --algo yolov9 --folds 0 1 2 3 4
    python scripts/compare_train.py --algo rfdetr --folds 0 --smoke     # 2 epochs, plumbing check

YOLOv9-c: training/train.py (the thesis exp_F recipe) with epochs 230,
patience 30 on inner-val fitness, batch 8, imgsz 640, max_det 1200, seed 0,
no periodic checkpoints.
RF-DETR:  training/train_rfdetr.py with epochs 200, patience 20 on inner-val
mAP50-95 (EMA), batch 4 (gradient accumulation to an effective 16), imgsz
640, 1,200 queries, seed 0.

Output: runs/compare/<algo>/fold<k>/ and runs/compare/<algo>/fold<k>.log.
``--project`` is passed *absolute* because Ultralytics rebases a relative
project path under its ``runs_dir`` setting (runs/detect/runs/…). Folds whose
best checkpoint already exists are skipped unless ``--force``.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "runs" / "compare"
BEST = {"yolov9": "weights/best.pt", "rfdetr": "checkpoint_best_ema.pth"}
MODULE = {"yolov9": "training.train", "rfdetr": "training.train_rfdetr"}
RECIPE = {
    "yolov9": {"epochs": 230, "patience": 30, "batch": 8, "weights": "models/yolov9c.pt"},
    "rfdetr": {"epochs": 200, "patience": 20, "batch": 4, "weights": "medium"},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--algo", required=True, choices=("yolov9", "rfdetr"))
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--smoke", action="store_true", help="2 epochs into fold<k>_smoke.")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-det", type=int, default=1200, help="YOLO val cap / RF-DETR queries.")
    ap.add_argument("--variant", default="medium")
    ap.add_argument("--gradient-checkpointing", action="store_true", help="RF-DETR only.")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="Anything after --extra goes to the training script verbatim.")
    a = ap.parse_args()

    r = RECIPE[a.algo]
    epochs = a.epochs or (2 if a.smoke else r["epochs"])
    batch = a.batch or r["batch"]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    for k in a.folds:
        name = f"fold{k}" + ("_smoke" if a.smoke else "")
        out = OUT_DIR / a.algo / name
        best = out / BEST[a.algo]
        if best.exists() and not a.force:
            print(f"[{a.algo} {name}] exists: {best} (use --force to retrain)")
            continue
        data = OUT_DIR / "folds" / f"fold{k}" / "data.yaml"
        if not data.exists():
            sys.exit(f"missing {data}: run scripts/compare_folds.py first")
        cmd = [sys.executable, "-m", MODULE[a.algo],
               "--data", str(data), "--weights", r["weights"],
               "--epochs", str(epochs), "--batch", str(batch), "--imgsz", str(a.imgsz),
               "--device", a.device, "--workers", str(a.workers),
               "--project", str(OUT_DIR / a.algo), "--name", name,
               "--patience", str(r["patience"])]
        if a.algo == "yolov9":
            cmd += ["--save-period", "-1", "--max-det", str(a.max_det)]
        else:
            cmd += ["--num-queries", str(a.max_det), "--variant", a.variant, "--seed", "0"]
            if a.gradient_checkpointing:
                cmd.append("--gradient-checkpointing")
        cmd += a.extra
        log = OUT_DIR / a.algo / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{a.algo} {name}] {' '.join(cmd)}\n  log: {log}", flush=True)
        t0 = time.time()
        with log.open("w", encoding="utf-8") as fh:
            fh.write("$ " + " ".join(cmd) + "\n")
            fh.flush()
            rc = subprocess.run(cmd, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
        mins = (time.time() - t0) / 60
        if rc != 0 or not best.exists():
            sys.exit(f"[{a.algo} {name}] FAILED rc={rc} after {mins:.1f} min; see {log}")
        print(f"[{a.algo} {name}] done in {mins:.1f} min -> {best}", flush=True)


if __name__ == "__main__":
    main()
