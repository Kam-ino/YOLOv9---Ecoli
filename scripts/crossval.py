"""
scripts/crossval.py
===================
K-fold cross-validation over the whole dataset (train + val pooled).

A 9-image val split cannot tell a real improvement from noise: the same
model moves ~0.04 mAP50 on it. Here every image is scored exactly once by
a model that never saw it, and the spread across folds says how big a
difference has to be before it means anything.

Each fold trains through ``training/train.py`` - the same recipe the app's
Train tab uses - and is scored two ways:
  * mAP50 / mAP50-95 from Ultralytics' validator (whole image at imgsz), and
  * recall of labelled objects through the app's own detector (tiling on),
    at conf >= 0.25, split by image source.

The dataset is snapshotted under runs/crossval/<tag>/ first, so labels can
be edited while this runs. Expect roughly 12 minutes per fold on a laptop GPU.

Usage (from the repo root):
    python scripts/crossval.py --tag baseline
    python scripts/crossval.py --tag smoke --folds 2 --epochs 2     # plumbing check

Learning curve - how much does more data help? Re-run the baseline's exact
folds on half of each training set, with the same number of training steps
(half the images -> twice the epochs):
    python scripts/crossval.py --tag half --folds-from runs/crossval/baseline \
        --train-fraction 0.5 --epochs 500
"""
import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.merge_duplicate_labels import IMG_EXTS, _xyxy, domain_of, load_labels  # noqa: E402


def label_of(img: Path) -> Path:
    """<root>/images/<split>/x.png -> <root>/labels/<split>/x.txt"""
    return img.parent.parent.parent / "labels" / img.parent.name / (img.stem + ".txt")


def subsample(images, fraction: float, seed: int):
    """Keep ``fraction`` of each image source. Nested: with the same seed, the
    images kept at 0.25 are also among those kept at 0.5."""
    if fraction >= 1.0:
        return list(images)
    kept = []
    for dom in ("microscope", "stained"):
        group = sorted((p for p in images if domain_of(p.name) == dom), key=lambda p: p.name)
        random.Random(seed).shuffle(group)
        kept += group[:max(1, round(len(group) * fraction))] if group else []
    return kept


def make_folds(images, k: int, seed: int):
    """Round-robin within each domain so every fold gets both image types."""
    rng = random.Random(seed)
    folds = [[] for _ in range(k)]
    for dom in ("microscope", "stained"):
        group = sorted(p for p in images if domain_of(p.name) == dom)
        rng.shuffle(group)
        for i, p in enumerate(group):
            folds[i % k].append(p)
    return folds


def recall_of_labelled(det, fold_dir: Path, conf: float):
    """Per domain: (labelled objects found, labelled objects) via the app detector."""
    out = {"microscope": [0, 0], "stained": [0, 0]}
    for img_path in sorted((fold_dir / "images" / "val").glob("*")):
        frame = cv2.imread(str(img_path))
        h, w = frame.shape[:2]
        lab = load_labels(fold_dir / "labels" / "val" / (img_path.stem + ".txt"))
        gt, gt_cls = _xyxy(lab) * np.array([w, h, w, h]), lab[:, 0].astype(int)
        used = np.zeros(len(gt), dtype=bool)
        dets = [d for d in det.predict(frame, tiled=True) if d.confidence >= conf]
        for d in sorted(dets, key=lambda d: -d.confidence):   # greedy, class-aware
            if not len(gt):
                break
            p = np.array(d.bbox)
            ix = np.clip(np.minimum(p[2], gt[:, 2]) - np.maximum(p[0], gt[:, 0]), 0, None)
            iy = np.clip(np.minimum(p[3], gt[:, 3]) - np.maximum(p[1], gt[:, 1]), 0, None)
            inter = ix * iy
            iou = inter / ((p[2] - p[0]) * (p[3] - p[1])
                           + (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]) - inter)
            iou[(gt_cls != d.class_id) | used] = 0
            if iou.max() >= 0.3:
                used[iou.argmax()] = True
        found, total = out[domain_of(img_path.name)]
        out[domain_of(img_path.name)] = [found + int(used.sum()), total + len(gt)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tag", required=True, help="Name for this experiment, e.g. baseline / relabelled.")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "ecoli")
    ap.add_argument("--weights", default="models/yolov9c.pt")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds-from", type=Path, default=None,
                    help="Reuse the held-out folds and label snapshot of an earlier run "
                         "(e.g. runs/crossval/baseline), so results pair fold by fold.")
    ap.add_argument("--train-fraction", type=float, default=1.0,
                    help="Train on this share of each fold's training images, per image "
                         "source. For learning curves; raise --epochs to match steps.")
    a = ap.parse_args()

    out = ROOT / "runs" / "crossval" / a.tag
    if out.exists():
        sys.exit(f"{out} already exists - pick another --tag or delete it.")

    # Same class list the app trains with.
    import yaml
    names = yaml.safe_load((ROOT / "training" / "dataset.yaml").read_text(encoding="utf-8"))["names"]

    def imgs(d: Path):
        return [p for p in d.glob("*") if p.suffix.lower() in IMG_EXTS]

    if a.folds_from:
        src = sorted(a.folds_from.glob("fold_*"), key=lambda d: int(d.name.split("_")[1]))
        if not src:
            sys.exit(f"No fold_* directories under {a.folds_from}")
        plan = [(imgs(d / "images" / "val"), imgs(d / "images" / "train")) for d in src]
    else:
        images = [p for s in ("train", "val", "test") for p in imgs(a.data / "images" / s)]
        plan = [(held, [p for p in images if p not in held])
                for held in make_folds(images, a.folds, a.seed)]
    plan = [(held, subsample(pool, a.train_fraction, a.seed)) for held, pool in plan]

    # Snapshot every fold now, so label edits made while this runs can't leak in.
    for k, (held_out, train) in enumerate(plan):
        fd = out / f"fold_{k}"
        for split, members in (("val", held_out), ("train", train)):
            (fd / "images" / split).mkdir(parents=True)
            (fd / "labels" / split).mkdir(parents=True)
            for p in members:
                shutil.copy2(p, fd / "images" / split / p.name)
                if label_of(p).exists():
                    shutil.copy2(label_of(p), fd / "labels" / split / (p.stem + ".txt"))
        (fd / "data.yaml").write_text(
            yaml.safe_dump({"path": fd.as_posix(), "train": "images/train",
                            "val": "images/val", "names": names}), encoding="utf-8")
    n_images = sum(len(h) for h, _ in plan)
    print(f"{n_images} images -> {len(plan)} folds; held out {[len(h) for h, _ in plan]}, "
          f"train {[len(t) for _, t in plan]} (snapshot in {out})")

    from ultralytics import YOLO
    from src.inference import YOLOv9Detector

    results = []
    for k in range(len(plan)):
        fd, name = out / f"fold_{k}", f"cv_{a.tag}_f{k}"
        subprocess.run(
            [sys.executable, "-m", "training.train", "--data", str(fd / "data.yaml"),
             "--weights", a.weights, "--epochs", str(a.epochs), "--patience", str(a.epochs),
             "--batch", str(a.batch), "--imgsz", str(a.imgsz), "--device", "0",
             "--workers", "2", "--name", name,
             # Only best.pt is used; periodic checkpoints cost ~190 MB each.
             "--save-period", "-1"],
            cwd=ROOT, check=True, stdout=open(fd / "train.log", "w", encoding="utf-8"),
            stderr=subprocess.STDOUT,
        )
        # Ultralytics nests run dirs differently per install; find it by name
        # (same approach as backend/app/training.py).
        best = max(ROOT.glob(f"**/{name}/weights/best.pt"), key=lambda p: p.stat().st_mtime)
        m = YOLO(str(best)).val(data=str(fd / "data.yaml"), imgsz=a.imgsz, batch=a.batch,
                                device=0, workers=0, plots=False, verbose=False,
                                project=str(fd), name="val", exist_ok=True)
        det = YOLOv9Detector(str(best), imgsz=a.imgsz, conf_threshold=a.conf, class_names=list(names.values()))
        r = {"fold": k, "map50": float(m.box.map50), "map": float(m.box.map),
             "recall": recall_of_labelled(det, fd, a.conf)}
        results.append(r)
        print(f"fold {k}: mAP50 {r['map50']:.3f}  mAP50-95 {r['map']:.3f}  recall {r['recall']}", flush=True)

    def pooled(dom):   # pooled over folds = every image counted once
        found = sum(r["recall"][d][0] for r in results for d in dom)
        total = sum(r["recall"][d][1] for r in results for d in dom)
        return found / max(total, 1)

    m50 = np.array([r["map50"] for r in results])
    m95 = np.array([r["map"] for r in results])
    summary = {
        "tag": a.tag, "folds": len(plan), "epochs": a.epochs, "images": n_images,
        "train_fraction": a.train_fraction, "train_images": [len(t) for _, t in plan],
        "map50_mean": float(m50.mean()), "map50_std": float(m50.std(ddof=1)) if len(m50) > 1 else 0.0,
        "map_mean": float(m95.mean()),
        "recall_all": pooled(("microscope", "stained")),
        "recall_microscope": pooled(("microscope",)), "recall_stained": pooled(("stained",)),
        "per_fold": results,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[{a.tag}] mAP50 {summary['map50_mean']:.3f} +/- {summary['map50_std']:.3f} across folds"
          f" | mAP50-95 {summary['map_mean']:.3f}")
    print(f"recall of labelled objects @conf>={a.conf}: all {summary['recall_all']:.3f}"
          f" | microscope {summary['recall_microscope']:.3f} | stained {summary['recall_stained']:.3f}")
    print("A later run beats this one only if its mAP50 gain is clearly larger than the +/- above.")


if __name__ == "__main__":
    main()
