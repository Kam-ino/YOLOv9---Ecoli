"""
backend/app/detector.py
=======================
Process-wide singleton wrapping :class:`src.inference.YOLOv9Detector`.

The detector is loaded once at FastAPI startup (the import + weights
load is several hundred MB / multiple seconds — doing it per-request
would be unusable). A threading.Lock guards the actual ``predict``
call because Ultralytics / PyTorch are not safe under concurrent calls
on a single model instance.
"""
import logging
from threading import Lock
from typing import Optional

from src.config import AppConfig, load_config
from src.inference import YOLOv9Detector


log = logging.getLogger(__name__)


class DetectorService:
    """Lazy singleton — :meth:`init` must be called once before use."""

    def __init__(self) -> None:
        self._cfg: Optional[AppConfig] = None
        self._detector: Optional[YOLOv9Detector] = None
        self._lock = Lock()

    def init(self, config_path: str) -> None:
        self._cfg = load_config(config_path)
        log.info("Initializing detector from %s", config_path)
        self._detector = YOLOv9Detector(
            weights_path=self._cfg.model.weights,
            device=self._cfg.model.device,
            imgsz=self._cfg.model.imgsz,
            conf_threshold=self._cfg.model.conf_threshold,
            iou_threshold=self._cfg.model.iou_threshold,
            class_names=self._cfg.classes,
        )
        log.info("Detector ready.")

    @property
    def is_ready(self) -> bool:
        return self._detector is not None

    @property
    def config(self) -> AppConfig:
        if self._cfg is None:
            raise RuntimeError("DetectorService.init() has not been called.")
        return self._cfg

    @property
    def detector(self) -> YOLOv9Detector:
        if self._detector is None:
            raise RuntimeError("DetectorService.init() has not been called.")
        return self._detector

    @property
    def lock(self) -> Lock:
        return self._lock


# Module-level singleton — imported by routes and the streaming generator.
service = DetectorService()
