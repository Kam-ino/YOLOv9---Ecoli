"""RFDETRDetector end to end on one real image (COCO-pretrained Medium).

Skips cleanly when ``rfdetr`` is not installed. Downloads the public
checkpoint on first run (~130 MB, cached under ~/.roboflow/models).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def test_rfdetr_adapter():
    try:
        import rfdetr  # noqa: F401
    except ImportError:
        print("rfdetr not installed — skipped")
        return
    import cv2
    from src.inference import RFDETRDetector

    img = next((ROOT / "data" / "ecoli" / "images" / "train").glob("*.png"))
    frame = cv2.imread(str(img))
    assert frame is not None and frame.ndim == 3

    # No fine-tuned file → COCO-pretrained; class names must come from the model.
    det = RFDETRDetector(weights_path="models/does-not-exist.pth", variant="medium",
                         resolution=640, num_queries=300, conf_threshold=0.05, warmup=False)
    assert det.algorithm == "rfdetr" and det.imgsz == 640 and det.max_det == 300
    assert len(det.class_names) >= 80, det.class_names[:5]

    dets = det.predict(frame, tiled=False)
    h, w = frame.shape[:2]
    assert len(dets) <= 300
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        assert -1 <= x1 <= x2 <= w + 1 and -1 <= y1 <= y2 <= h + 1, d
        assert 0.0 <= d.confidence <= 1.0 and d.class_name
    # Batched _raw keeps per-image alignment.
    raw = det._raw([frame, np.zeros((480, 640, 3), np.uint8)])
    assert len(raw) == 2 and raw[0][0].shape[1] == 4

    # Explicit class names override the model's.
    det.class_names = ["ecoli", "ecoli_cluster"]
    for d in det.predict(frame):
        assert d.class_name in ("ecoli", "ecoli_cluster") or d.class_name.startswith("cls_")
    print(f"test_rfdetr_adapter ok ({len(dets)} boxes on {img.name})")


if __name__ == "__main__":
    test_rfdetr_adapter()
