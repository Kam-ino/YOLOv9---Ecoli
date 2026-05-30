"""Generate a synthetic 1280x720 test image and a 3-second test video.

Used for the smoke test when no USB microscope is plugged in. The
image contains random "blob" objects on a low-contrast background to
exercise the CLAHE + inference path. We don't expect yolov9c.pt to
detect anything meaningful here — we just want the pipeline to run.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def make_image(w: int = 1280, h: int = 720, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Low-contrast greyish background — mimics a microscopy field.
    img = (np.ones((h, w, 3), dtype=np.uint8) * 110)
    img += rng.integers(-15, 15, size=img.shape, dtype=np.int8).astype(np.uint8)

    # Sprinkle dim ellipsoidal blobs.
    for _ in range(60):
        cx = int(rng.integers(20, w - 20))
        cy = int(rng.integers(20, h - 20))
        rx = int(rng.integers(4, 14))
        ry = int(rng.integers(2, 8))
        angle = int(rng.integers(0, 180))
        # Slightly darker than the background.
        color = int(rng.integers(60, 95))
        cv2.ellipse(img, (cx, cy), (rx, ry), angle, 0, 360,
                    (color, color, color), thickness=-1, lineType=cv2.LINE_AA)

    # Soft blur so edges aren't perfectly sharp.
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def make_video(path: Path, frames: int = 90, fps: int = 30) -> None:
    img0 = make_image(seed=0)
    h, w = img0.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter at {path}")
    try:
        for i in range(frames):
            writer.write(make_image(seed=i))
    finally:
        writer.release()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="data", help="Output dir.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_path = out_dir / "synthetic_microscopy.jpg"
    cv2.imwrite(str(img_path), make_image())
    print(f"wrote {img_path}")

    vid_path = out_dir / "synthetic_microscopy.mp4"
    make_video(vid_path, frames=90, fps=30)
    print(f"wrote {vid_path}")
