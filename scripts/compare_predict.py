"""
scripts/compare_predict.py
==========================
Cross-validated predictions for the YOLOv9 / RF-DETR comparison.

For every fold k, loads that fold's trained model and predicts its held-out
images (never seen in training or tuning) and its inner-val images (used
later to pick the confidence threshold). Every pooled image is therefore
predicted exactly once by a model that never saw it.

    python scripts/compare_predict.py --algo yolov9 [--nms-iou 0.7] [--tiled]
    python scripts/compare_predict.py --algo rfdetr

Weights per fold (override with --weights):
    runs/compare/yolov9/fold<k>/weights/best.pt
    runs/compare/rfdetr/fold<k>/checkpoint_best_ema.pth

Outputs (runs/compare/):
    preds_<tag>.json            COCO results over the held-out images, each with its fold
    preds_<tag>_valid_<k>.json  COCO results over fold k's inner val
    preds_<tag>.meta.json       weights md5s, thresholds, library versions, per-image latency

``--conf 0.001`` keeps the full score range so compare_eval.py can sweep
thresholds; ``--max-det 1200`` lifts both detectors' per-image cap above the
densest slide (1,043 cells). ``--nms-iou`` only matters for YOLOv9 (and for
the tile-merge NMS of either model when ``--tiled``); the sweep in
compare_crowding.py re-runs this script with several values.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference import RFDETRDetector, YOLOv9Detector  # noqa: E402
from training.dataset_view import names_of_yaml  # noqa: E402

OUT_DIR = ROOT / "runs" / "compare"
EXP_DIR = ROOT / "experiments" / "rfdetr_vs_yolov9"
BEST = {"yolov9": "weights/best.pt", "rfdetr": "checkpoint_best_ema.pth"}


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build(algo: str, weights: Path, a, names):
    common = dict(conf_threshold=a.conf, iou_threshold=a.nms_iou, class_names=names, device="auto")
    if algo == "yolov9":
        return YOLOv9Detector(str(weights), imgsz=a.imgsz, max_det=a.max_det, **common)
    return RFDETRDetector(str(weights), variant=a.variant, resolution=a.imgsz,
                          num_queries=a.max_det, **common)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--algo", required=True, choices=("yolov9", "rfdetr"))
    ap.add_argument("--weights", default=None,
                    help="Per-fold weights pattern with {k}, e.g. runs/compare/yolov9/fold{k}/weights/best.pt")
    ap.add_argument("--suffix", default="", help="Run-name suffix, e.g. _x8 for compare_train.py --expand8 runs.")
    ap.add_argument("--variant", default="medium", help="RF-DETR architecture of the checkpoints.")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--nms-iou", type=float, default=0.7, help="YOLO NMS IoU (thesis val setting).")
    ap.add_argument("--max-det", type=int, default=1200)
    ap.add_argument("--tiled", action="store_true", help="Native-resolution tiling, as the app does.")
    ap.add_argument("--tag", default=None, help="Output name; default <algo>[_tiled][_nms<iou>].")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "ecoli")
    a = ap.parse_args()

    tag = a.tag or (a.algo + ("_tiled" if a.tiled else "")
                    + (f"_nms{a.nms_iou:g}" if a.algo == "yolov9" and a.nms_iou != 0.7 else ""))
    pattern = a.weights or f"runs/compare/{a.algo}/fold{{k}}{a.suffix}/{BEST[a.algo]}"
    plan = json.loads((EXP_DIR / "folds.json").read_text(encoding="utf-8"))
    gt = json.loads((OUT_DIR / "gt_all.json").read_text(encoding="utf-8"))
    names = [c["name"] for c in sorted(gt["categories"], key=lambda c: c["id"])]
    by_id = {img["id"]: img for img in gt["images"]}
    meta_by_name = {m["file_name"]: m for m in plan["images"]}

    test_results, meta = [], {"algo": a.algo, "tag": tag, "args": vars(a), "folds": {}, "latency_ms": {}}
    for k in range(plan["folds"]):
        weights = ROOT / pattern.format(k=k)
        if not weights.is_file():
            sys.exit(f"missing weights for fold {k}: {weights}")
        det = build(a.algo, weights, a, names)
        meta["folds"][k] = {"weights": str(weights), "md5": md5(weights)}

        def run(image_ids):
            out = []
            for img_id in image_ids:
                img = by_id[img_id]
                path = a.data / "images" / meta_by_name[img["file_name"]]["origin_split"] / img["file_name"]
                frame = cv2.imread(str(path))
                assert frame is not None, path
                t0 = time.perf_counter()
                dets = det.predict(frame, tiled=a.tiled)
                meta["latency_ms"][img_id] = round((time.perf_counter() - t0) * 1000, 2)
                for d in dets:
                    x1, y1, x2, y2 = d.bbox
                    out.append({"image_id": img_id, "category_id": d.class_id,
                                "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                                "score": round(d.confidence, 5), "fold": k})
            return out

        test_ids = [i for i, img in by_id.items() if img["fold"] == k]
        valid_ids = [img["id"] for img in gt["images"] if k in meta_by_name[img["file_name"]]["inner_val_of"]]
        test_results += run(test_ids)
        valid_results = run(valid_ids)
        (OUT_DIR / f"preds_{tag}_valid_{k}.json").write_text(json.dumps(valid_results), encoding="utf-8")
        print(f"fold {k}: {len(test_ids)} test imgs -> {len([r for r in test_results if r['fold'] == k])} dets; "
              f"{len(valid_ids)} inner-val imgs → {len(valid_results)} dets  ({weights.name})", flush=True)

        del det
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    try:
        import torch
        meta["versions"] = {"torch": torch.__version__}
        if a.algo == "yolov9":
            import ultralytics
            meta["versions"]["ultralytics"] = ultralytics.__version__
        else:
            from importlib.metadata import version
            meta["versions"]["rfdetr"] = version("rfdetr")
    except Exception as exc:  # versions are nice-to-have
        meta["versions"] = {"error": repr(exc)}

    assert {r["image_id"] for r in test_results} <= set(by_id)
    (OUT_DIR / f"preds_{tag}.json").write_text(json.dumps(test_results), encoding="utf-8")
    (OUT_DIR / f"preds_{tag}.meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    covered = len({r["image_id"] for r in test_results})
    print(f"wrote preds_{tag}.json: {len(test_results)} detections on {covered}/{len(by_id)} images "
          f"(images with zero detections are legitimately absent)")


if __name__ == "__main__":
    main()
