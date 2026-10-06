"""Shared detector base: Detection conversion and the tile merge, using a
fake backend so neither ultralytics nor rfdetr has to load."""
import numpy as np

from src.inference import InferenceError, _Detector


class Fake(_Detector):
    """Returns one centred box (class 1) per image/tile it is shown."""
    algorithm = "fake"

    def __init__(self):
        self.imgsz, self.conf, self.iou, self.max_det = 64, 0.1, 0.5, 100
        self.device, self.class_names = "cpu", ["a", "b"]
        self.calls = []

    def _raw(self, images):
        out = []
        for img in images:
            h, w = img.shape[:2]
            self.calls.append((h, w))
            out.append((
                np.array([[w * 0.4, h * 0.4, w * 0.6, h * 0.6]], dtype=np.float32),
                np.array([0.9], dtype=np.float32),
                np.array([1]),
            ))
        return out


def test_small_frame_single_pass():
    d = Fake()
    dets = d.predict(np.zeros((64, 64, 3), np.uint8), tiled=True)
    assert d.calls == [(64, 64)], d.calls            # no tiles for a small frame
    assert len(dets) == 1
    assert dets[0].class_name == "b" and dets[0].class_id == 1
    assert dets[0].bbox == (25.6, 25.6, 38.4, 38.4) or np.allclose(dets[0].bbox, (25.6, 25.6, 38.4, 38.4))
    assert abs(dets[0].confidence - 0.9) < 1e-6


def test_wide_frame_is_tiled_and_offset():
    d = Fake()
    frame = np.zeros((64, 200, 3), np.uint8)       # 200 > 1.5 * 64 → tiles
    dets = d.predict(frame, tiled=True)
    assert len(d.calls) > 1
    for det in dets:
        x1, y1, x2, y2 = det.bbox
        assert 0 <= x1 < x2 <= 200 and 0 <= y1 < y2 <= 64, det
    assert any(det.bbox[0] > 64 for det in dets), "tile boxes must be offset into frame coords"
    assert len(dets) <= d.max_det * 3
    assert d.predict(frame, tiled=False) and len(d.calls) > 0


def test_unknown_class_id_is_named():
    d = Fake()
    d.class_names = ["only"]
    det = d.predict(np.zeros((64, 64, 3), np.uint8))[0]
    assert det.class_name == "cls_1"


def test_rejects_bad_input():
    d = Fake()
    for bad in (np.zeros((0, 0, 3), np.uint8), np.zeros((64, 64), np.uint8)):
        try:
            d.predict(bad)
        except InferenceError:
            continue
        raise AssertionError(f"expected InferenceError for shape {bad.shape}")


if __name__ == "__main__":
    test_small_frame_single_pass()
    test_wide_frame_is_tiled_and_offset()
    test_unknown_class_id_is_named()
    test_rejects_bad_input()
    print("test_detector_base ok")
