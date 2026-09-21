"""
scripts/rank_label_gaps.py
==========================
Review queue for completing labels: which saved images most likely have
objects nobody boxed?

Runs the deployed model over the dataset exactly the way the Label tab's
"Suggest missing" button does (same weights, stored pixels, tiling) and
counts confident detections that overlap no existing box. Read-only.

Usage (from the repo root):
    python scripts/rank_label_gaps.py            # conf >= 0.25
    python scripts/rank_label_gaps.py --conf 0.4
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.merge_duplicate_labels import IMG_EXTS, _xyxy, load_labels  # noqa: E402
from src.config import load_config  # noqa: E402
from src.inference import YOLOv9Detector  # noqa: E402

RULES = """\
Labelling rules - apply the same way on every image:
  * ecoli          one box, tight around ONE cell you can tell apart from its neighbours.
  * ecoli_cluster  one box, tight around touching cells you can NOT separate.
                   No big box around an area of separate cells - box those cells instead.
  * Box every object in the frame. An unboxed cell teaches the model "background".
Do the val images first: every accuracy number is measured against them.
"""


def unmatched(pred: np.ndarray, gt: np.ndarray, thr: float = 0.3) -> int:
    """How many ``pred`` boxes (xyxy) overlap no ``gt`` box with IoU >= thr."""
    if len(gt) == 0:
        return len(pred)
    n = 0
    for p in pred:
        ix = np.clip(np.minimum(p[2], gt[:, 2]) - np.maximum(p[0], gt[:, 0]), 0, None)
        iy = np.clip(np.minimum(p[3], gt[:, 3]) - np.maximum(p[1], gt[:, 1]), 0, None)
        inter = ix * iy
        union = (p[2] - p[0]) * (p[3] - p[1]) + (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]) - inter
        n += int((inter / union).max() < thr)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "ecoli")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Match the Label tab's 'Min conf' slider.")
    a = ap.parse_args()

    cfg = load_config(a.config)
    det = YOLOv9Detector(
        cfg.model.weights, device=cfg.model.device, imgsz=cfg.model.imgsz,
        conf_threshold=a.conf, iou_threshold=cfg.model.iou_threshold,
        class_names=cfg.classes,
    )
    rows = []
    for split in ("val", "train", "test"):
        for img_path in sorted((a.data / "images" / split).glob("*")):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            lab = load_labels(a.data / "labels" / split / (img_path.stem + ".txt"))
            gt = _xyxy(lab) * np.array([w, h, w, h])
            # No CLAHE: stored pixels are what the model trained on (stream
            # captures are saved already enhanced). Same as "Suggest missing".
            pred =np.array([d.bbox for d in det.predict(frame, tiled=True)]).reshape(-1, 4)
            rows.append((split, img_path.name, len(gt), unmatched(pred, gt)))

    print(RULES)
    print(f"{'#':>3}  {'split':5} {'image':40} {'boxed':>6} {'suggested-missing':>18}")
    # val first, then most gaps first
    rows.sort(key=lambda r: (r[0] != "val", -r[3]))
    for i, (split, name, n_gt, n_miss) in enumerate(rows, 1):
        print(f"{i:>3}  {split:5} {name[:40]:40} {n_gt:>6} {n_miss:>18}")
    total = sum(r[3] for r in rows)
    print(f"\n{total} suggested-missing boxes across {len(rows)} images "
          f"({sum(r[2] for r in rows)} boxed today). In the Label tab: Edit -> "
          f"Suggest missing -> click away the wrong ones -> Save.")


if __name__ == "__main__":
    main()
