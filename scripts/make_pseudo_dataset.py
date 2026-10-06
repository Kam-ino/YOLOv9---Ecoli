"""
scripts/make_pseudo_dataset.py
==============================
Build ``data/ecoli_x8/``: every image of ``data/ecoli`` in its 8 rotations /
mirrors (69 → 552), labelled with the human boxes (rotated along) **plus**
pseudo-labels — detections of a teacher model at ≥ ``--conf`` that do not
overlap an existing box (IoU < ``--overlap``, the rule the app's "Suggest
missing" uses). Which boxes are pseudo is recorded in ``pseudo.json`` so the
split can be undone or audited.

    python scripts/make_pseudo_dataset.py                       # teacher = models/best_yolov9c.pt
    python scripts/make_pseudo_dataset.py --conf 0.25 --tiled   # defaults shown

The output keeps the ``images/<split>`` + ``labels/<split>`` layout and
writes ``data.yaml``, so the app's trainers and the compare_* scripts can
point at it. Originals in ``data/ecoli`` are never touched.

Caveats for the paper: pseudo-labels inherit the teacher's errors and bias
any student towards it; evaluation must stay on the human labels of
unexpanded held-out images (scripts/compare_folds.py does that).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference import YOLOv9Detector  # noqa: E402
from training.dataset_view import IMG_EXTS, D4_SUFFIX, expand8, read_labels, write_labels  # noqa: E402


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def rows_to_xyxy(rows, w, h) -> np.ndarray:
    if not rows:
        return np.zeros((0, 4))
    r = np.array(rows, dtype=float)
    cx, cy, bw, bh = r[:, 1] * w, r[:, 2] * h, r[:, 3] * w, r[:, 4] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, default=ROOT / "data" / "ecoli")
    ap.add_argument("--dst", type=Path, default=ROOT / "data" / "ecoli_x8")
    ap.add_argument("--names", type=Path, default=ROOT / "training" / "dataset.yaml")
    ap.add_argument("--weights", default="models/best_yolov9c.pt", help="Teacher weights (YOLOv9).")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--overlap", type=float, default=0.3, help="Skip detections with IoU >= this to any box.")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--no-tiled", action="store_true", help="Whole-image inference only (app default is tiled).")
    ap.add_argument("--max-det", type=int, default=1200)
    a = ap.parse_args()

    import yaml
    names = yaml.safe_load(a.names.read_text(encoding="utf-8"))["names"]
    names = {int(k): v for k, v in names.items()} if isinstance(names, dict) else dict(enumerate(names))

    # 1. Expand: 8 orientations with rotated human labels.
    t0 = time.time()
    sources = {s: sorted(p for p in (a.src / "images" / s).glob("*")
                         if p.suffix.lower() in IMG_EXTS and D4_SUFFIX not in p.stem)   # originals only
               for s in ("train", "val", "test")}
    variants = []
    for split, imgs in sources.items():
        for img in imgs:
            variants += [(split, v) for v in expand8(img, a.dst / "images" / split, a.dst / "labels" / split)]
    print(f"expanded {sum(len(v) for v in sources.values())} images -> {len(variants)} in {time.time() - t0:.0f}s")

    # 2. Pseudo-label with the teacher.
    det = YOLOv9Detector(a.weights, imgsz=a.imgsz, conf_threshold=a.conf, iou_threshold=0.7,
                         class_names=[names[k] for k in sorted(names)], max_det=a.max_det)
    pseudo, n_human, n_pseudo, n_skipped = {}, 0, 0, 0
    t0 = time.time()
    for i, (split, img) in enumerate(variants):
        frame = cv2.imread(str(img))
        h, w = frame.shape[:2]
        lab = a.dst / "labels" / split / f"{img.stem}.txt"
        rows = read_labels(lab)
        n_human += len(rows)
        have = rows_to_xyxy(rows, w, h)
        dets = det.predict(frame, tiled=not a.no_tiled)
        added = []
        for d in sorted(dets, key=lambda d: -d.confidence):
            box = np.array([d.bbox])
            if len(have) and iou_xyxy(box, have).max() >= a.overlap:
                n_skipped += 1
                continue
            x1, y1, x2, y2 = d.bbox
            rows.append([d.class_id, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h])
            have = np.vstack([have, box])
            added.append({"row": len(rows) - 1, "conf": round(d.confidence, 4), "class_id": d.class_id})
        write_labels(lab, rows)
        if added:
            pseudo[f"{split}/{img.name}"] = added
            n_pseudo += len(added)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(variants)} images, {n_pseudo} pseudo boxes so far", flush=True)

    # 3. Manifest + yaml.
    (a.dst / "pseudo.json").write_text(json.dumps({
        "teacher": a.weights, "conf": a.conf, "overlap_iou": a.overlap, "imgsz": a.imgsz,
        "tiled": not a.no_tiled, "source": str(a.src), "n_images": len(variants),
        "n_human_boxes": n_human, "n_pseudo_boxes": n_pseudo, "n_detections_skipped_overlap": n_skipped,
        "images": pseudo}, indent=1), encoding="utf-8")
    (a.dst / "data.yaml").write_text(yaml.safe_dump({
        "path": a.dst.resolve().as_posix(), "train": "images/train", "val": "images/val", "test": "images/test",
        "nc": len(names), "names": {k: names[k] for k in sorted(names)},
        "expanded": True, "pseudo_labels": "pseudo.json"}, sort_keys=False), encoding="utf-8")
    print(f"{len(variants)} images: {n_human} human boxes (rotated) + {n_pseudo} pseudo boxes "
          f"({n_skipped} detections skipped as overlapping) in {time.time() - t0:.0f}s -> {a.dst}")


if __name__ == "__main__":
    main()
