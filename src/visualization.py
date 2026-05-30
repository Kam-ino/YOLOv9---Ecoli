"""
src/visualization.py
====================
Annotation overlays: bounding boxes with class/confidence labels and a
top-left HUD showing frame number, rolling FPS, and detection count.

All drawing is done in-place on the input frame for zero-copy speed in
the live loop. The same functions are reusable from a FastAPI endpoint
that wants to return an annotated JPEG/PNG.
"""
from typing import Iterable, Optional, Tuple

import cv2
import numpy as np


# Distinct, high-contrast BGR colours for up to 8 classes. Cycled for more.
_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (0, 255, 0),      # green
    (255, 128, 0),    # blue/cyan
    (0, 0, 255),      # red
    (255, 255, 0),    # cyan
    (255, 0, 255),    # magenta
    (0, 255, 255),    # yellow
    (255, 255, 255),  # white
    (128, 0, 255),    # pink
)


def color_for_class(class_id: int) -> Tuple[int, int, int]:
    """Stable colour assignment per class id."""
    return _PALETTE[class_id % len(_PALETTE)]


def draw_detections(frame: np.ndarray, detections: Iterable) -> np.ndarray:
    """Draw bounding boxes + ``class conf`` labels in-place.

    The label is rendered on a filled rectangle so it stays legible on
    busy or low-contrast microscopy backgrounds.

    Returns the same ``frame`` object for caller convenience.
    """
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        color = color_for_class(det.class_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{det.class_name} {det.confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1,
        )
        # Place the label above the box; if it would go off the top of
        # the frame, drop it inside the box instead.
        if y1 - th - baseline - 2 >= 0:
            top_left = (x1, y1 - th - baseline - 2)
            bot_right = (x1 + tw + 2, y1)
            text_org = (x1 + 1, y1 - baseline - 1)
        else:
            top_left = (x1, y1)
            bot_right = (x1 + tw + 2, y1 + th + baseline + 2)
            text_org = (x1 + 1, y1 + th + 1)

        cv2.rectangle(frame, top_left, bot_right, color, thickness=cv2.FILLED)
        cv2.putText(
            frame, label, text_org,
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )
    return frame


def draw_hud(
    frame: np.ndarray,
    fps: float,
    detection_count: int,
    frame_number: Optional[int] = None,
) -> np.ndarray:
    """Top-left HUD: frame number, FPS, per-frame detection count.

    Drawn on a translucent black backdrop so it reads against any image.
    """
    lines = []
    if frame_number is not None:
        lines.append(f"Frame: {frame_number}")
    lines.append(f"FPS: {fps:5.1f}")
    lines.append(f"Detections: {detection_count}")

    x, y = 10, 10
    pad = 6
    line_h = 22
    box_h = pad * 2 + line_h * len(lines)
    box_w = 200

    # Translucent backdrop for legibility against bright/textured frames.
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, dst=frame)

    for i, line in enumerate(lines):
        cv2.putText(
            frame, line,
            (x + pad, y + pad + line_h * (i + 1) - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA,
        )
    return frame
