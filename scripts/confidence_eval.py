"""
scripts/confidence_eval.py
==========================
How confident is a detector on cells it actually finds? For one weights
file, on the held-out images of one fold (human labels, original
orientation): recall at score >= 0.25 and >= 0.60, the median score of the
detections matched to a GT box, and precision at 0.60 — plus mAP50 as the
accuracy check so a "more confident" model isn't just a worse one.

    python scripts/confidence_eval.py --algo yolov9 --weights runs/compare/yolov9/fold0_x8_full/weights/best.pt
    python scripts/confidence_eval.py --algo rfdetr --weights runs/compare/rfdetr/fold0_x8_full/checkpoint_best_ema.pth

Appends one JSON line per call to runs/compare/confidence_eval.jsonl.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.compare_eval import OUT_DIR, EXP_DIR, evaluate, ap_of, group, match_image  # noqa: E402
from src.inference import RFDETRDetector, YOLOv9Detector  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--algo", required=True, choices=("yolov9", "rfdetr"))
    ap.add_argument("--weights", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--label", default=None, help="Free-text tag for the results line.")
    ap.add_argument("--single-cls", action="store_true", help="Weights trained single-class: ignore GT class.")
    ap.add_argument("--tiled", action="store_true")
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()

    gt = json.loads((OUT_DIR / "gt_all.json").read_text(encoding="utf-8"))
    plan = json.loads((EXP_DIR / "folds.json").read_text(encoding="utf-8"))
    meta = {m["file_name"]: m for m in plan["images"]}
    ids = [im["id"] for im in gt["images"] if im["fold"] == a.fold]
    names = [c["name"] for c in sorted(gt["categories"], key=lambda c: c["id"])]
    data = ROOT / "data" / "ecoli_human"
    if a.single_cls:
        gt = {**gt, "annotations": [{**x, "category_id": 0} for x in gt["annotations"]]}

    if a.algo == "yolov9":
        det = YOLOv9Detector(a.weights, imgsz=a.imgsz, conf_threshold=0.001, iou_threshold=0.7,
                             class_names=names, max_det=1200, warmup=False)
    else:
        det = RFDETRDetector(a.weights, resolution=a.imgsz, num_queries=1200, conf_threshold=0.001,
                             class_names=names, warmup=False)
    preds = []
    for im in gt["images"]:
        if im["id"] not in ids:
            continue
        frame = cv2.imread(str(data / "images" / meta[im["file_name"]]["origin_split"] / im["file_name"]))
        for d in det.predict(frame, tiled=a.tiled):
            x1, y1, x2, y2 = d.bbox
            preds.append({"image_id": im["id"], "category_id": 0 if a.single_cls else d.class_id,
                          "bbox": [x1, y1, x2 - x1, y2 - y1], "score": d.confidence})
    gt_by, dt_by = group([x for x in gt["annotations"] if x["image_id"] in set(ids)]), group(preds)
    out = {"algo": a.algo, "weights": a.weights, "label": a.label or Path(a.weights).parent.parent.name,
           "fold": a.fold, "n_images": len(ids), "n_gt": sum(len(v) for v in gt_by.values())}
    for tau in (0.25, 0.6):
        tp = fp = fn = 0
        for i in ids:
            t, f, n, _ = match_image(gt_by.get(i, []), dt_by.get(i, []), tau)
            tp, fp, fn = tp + t, fp + f, fn + n
        out[f"recall@{tau}"] = tp / max(tp + fn, 1)
        out[f"precision@{tau}"] = tp / max(tp + fp, 1)
    matched_scores = []
    for i in ids:
        g, d = gt_by.get(i, []), sorted(dt_by.get(i, []), key=lambda x: -x["score"])
        if not g or not d:
            continue
        from scripts.compare_eval import iou_matrix
        m = iou_matrix(np.array([x["bbox"] for x in g], float), np.array([x["bbox"] for x in d], float))
        for r in range(len(g)):
            j = int(m[r].argmax())
            if m[r, j] >= 0.5 and d[j]["category_id"] == g[r]["category_id"]:
                matched_scores.append(d[j]["score"])
    out["median_score_of_found_cells"] = float(np.median(matched_scores)) if matched_scores else 0.0
    out["share_found_cells_ge_0.6"] = float(np.mean(np.array(matched_scores) >= 0.6)) if matched_scores else 0.0
    E = evaluate(gt, preds, img_ids=ids)
    out["mAP50"], out["mAP"] = ap_of(E, 0, 0), ap_of(E, 0)
    (OUT_DIR / "confidence_eval.jsonl").open("a", encoding="utf-8").write(json.dumps(out) + "\n")
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}))


if __name__ == "__main__":
    main()
