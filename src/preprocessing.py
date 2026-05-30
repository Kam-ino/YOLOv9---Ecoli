"""
src/preprocessing.py
====================
Frame-level preprocessing for microscopy input.

Microscopy frames are typically low-contrast and unevenly illuminated.
CLAHE (Contrast Limited Adaptive Histogram Equalization) provides
much better local contrast than global histogram equalization, without
amplifying noise in flat regions — which matters because stained
bacteria often occupy only a small fraction of the field of view.
"""
import logging
from typing import Tuple

import cv2
import numpy as np


log = logging.getLogger(__name__)


def apply_clahe(
    frame: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: int = 8,
) -> np.ndarray:
    """Apply CLAHE to a BGR or grayscale frame and return the enhanced frame.

    For BGR input we operate on the L channel of LAB so we enhance
    luminance contrast without shifting colour — stain hues are
    diagnostic in microscopy and must be preserved.

    Args:
        frame:           Input image. ``HxW`` grayscale or ``HxWx3`` BGR.
        clip_limit:      Threshold for contrast limiting. Higher values
                         produce stronger enhancement and more noise.
        tile_grid_size:  Number of CLAHE tiles per axis.

    Returns:
        Enhanced frame with the same shape/dtype as the input. If the
        input shape is unrecognised the original frame is returned
        unchanged and a warning is logged (graceful degradation rather
        than crashing the live loop).
    """
    if frame is None or frame.size == 0:
        log.warning("apply_clahe received an empty frame; returning as-is.")
        return frame

    grid: Tuple[int, int] = (int(tile_grid_size), int(tile_grid_size))
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=grid)

    if frame.ndim == 2:
        return clahe.apply(frame)

    if frame.ndim == 3 and frame.shape[2] == 3:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        l_chan = clahe.apply(l_chan)
        return cv2.cvtColor(cv2.merge((l_chan, a_chan, b_chan)), cv2.COLOR_LAB2BGR)

    log.warning(
        "apply_clahe: unexpected frame shape %s; returning unchanged.",
        frame.shape,
    )
    return frame
