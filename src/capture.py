"""
src/capture.py
==============
USB microscope / video file capture wrapper.

Wraps :class:`cv2.VideoCapture` with context-manager semantics,
configurable requested resolution / FPS, and validation that the
source is actually open before the main loop starts trying to read.

Accepted sources:
    * ``int``                       — device index (0, 1, ...) for USB / built-in cams
    * ``str`` (file path)           — local video file for offline testing
    * ``str`` (``rtsp://``, ``http://``) — IP camera streams
"""
import logging
from typing import Optional, Tuple, Union

import cv2
import numpy as np


log = logging.getLogger(__name__)


class CaptureError(RuntimeError):
    """Raised when a video source cannot be opened or consistently fails to read."""


class VideoSource:
    """Context manager around :class:`cv2.VideoCapture`."""

    def __init__(
        self,
        source: Union[int, str],
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
    ):
        self._source = source
        self._req_width = width
        self._req_height = height
        self._req_fps = fps
        self._cap: Optional[cv2.VideoCapture] = None

    def __enter__(self) -> "VideoSource":
        # We let OpenCV pick the backend (CAP_ANY). On Windows the
        # DirectShow backend (CAP_DSHOW) is often faster for USB cams
        # but is ignored on Linux/macOS. Auto is the safest default;
        # advanced users can subclass and override.
        log.info("Opening capture source: %r", self._source)
        self._cap = cv2.VideoCapture(self._source)
        if not self._cap.isOpened():
            self._cap = None
            raise CaptureError(
                f"Could not open video source {self._source!r}. "
                "If this is a USB microscope, check that: (1) the device "
                "is plugged in, (2) the device index is correct (try 0, 1, 2 ...), "
                "(3) no other application is currently holding the camera, "
                "and (4) on Linux your user is in the 'video' group."
            )

        if self._req_width:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self._req_width))
        if self._req_height:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self._req_height))
        if self._req_fps:
            self._cap.set(cv2.CAP_PROP_FPS, float(self._req_fps))

        # Many USB cams silently ignore set() requests for unsupported
        # modes — log what we actually got so misconfiguration is visible.
        log.info(
            "Capture opened: requested %sx%s @ %s fps; actual %dx%d @ %.1f fps",
            self._req_width, self._req_height, self._req_fps,
            self.width, self.height, self.fps or 0.0,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            log.info("Capture released.")
        return False  # never suppress exceptions

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read one frame. Returns ``(ok, frame)`` exactly like cv2."""
        if self._cap is None:
            raise CaptureError("read() called on a closed VideoSource.")
        return self._cap.read()

    @property
    def width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if self._cap else 0

    @property
    def height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if self._cap else 0

    @property
    def fps(self) -> float:
        return float(self._cap.get(cv2.CAP_PROP_FPS)) if self._cap else 0.0

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
