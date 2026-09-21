"""Re-saving the same picture must extend its entry, not clone it.

Run from the repo root:  python -m tests.test_dataset_resave
"""
import tempfile
from pathlib import Path

import cv2
import numpy as np

from backend.app.dataset import DatasetStore
from backend.app.schemas import LabelBox


def _png(seed: int) -> bytes:
    pixels = np.random.default_rng(seed).integers(0, 255, (48, 64, 3), dtype=np.uint8)
    return cv2.imencode(".png", pixels)[1].tobytes()


def _box(cx: float) -> LabelBox:
    return LabelBox(class_id=0, cx=cx, cy=0.5, w=0.1, h=0.1)


def test_resave_extends_entry() -> None:
    with tempfile.TemporaryDirectory() as d:
        store = DatasetStore(Path(d))
        store.save(_png(1), [_box(.2), _box(.3)], "train", "img.png")
        again = store.save(_png(1), [_box(.3), _box(.6)], "val", "img.png")
        other = store.save(_png(2), [_box(.9)], "train", "img.png")

        # Same pixels: one entry, kept in its original split, repeated box skipped.
        assert (again.filename, again.split, again.num_boxes) == ("img.png", "train", 3)
        assert not list(Path(d, "images", "val").glob("*"))
        # Different pixels under the same name still get the collision suffix.
        assert other.filename == "img - 1.png"

        # An edit session sends the full list: a removed box must stay removed.
        edited = store.save(_png(1), [_box(.2)], "train", "img.png", replace=True)
        assert (edited.filename, edited.num_boxes) == ("img.png", 1)
        assert len(Path(d, "labels", "train", "img.txt").read_text().splitlines()) == 1


if __name__ == "__main__":
    test_resave_extends_entry()
    print("ok")
