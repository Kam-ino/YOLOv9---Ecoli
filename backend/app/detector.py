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
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from src.config import AppConfig, load_config
from src.inference import YOLOv9Detector


# Fall-back weights when the user resets the fine-tuned model.
_BASE_WEIGHTS = Path("models/yolov9c.pt")


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

    # ------------------------------------------------------------------

    def reset(self) -> Dict[str, Any]:
        """Move the fine-tuned weights to a .bak file and reload from base.

        Idempotent and recoverable:
          * If a trained checkpoint is at the configured ``model.weights``
            path, it's renamed to ``<stem>.bak-<unix_ts><suffix>`` next to
            the original so you can roll back manually.
          * The detector is rebuilt against ``models/yolov9c.pt`` (the
            pretrained COCO base). The badge in the UI will report 80
            classes after this, signalling "untrained".
          * Returns a small dict describing what changed so the UI can
            show a useful confirmation message.

        Held under ``self._lock`` so any in-flight ``predict()`` finishes
        before the swap, and no new inference starts until the new model
        is fully loaded.
        """
        if self._cfg is None:
            raise RuntimeError("DetectorService.init() has not been called.")

        with self._lock:
            current = Path(self._cfg.model.weights)
            backup_path: Optional[Path] = None
            if current.exists():
                ts = int(time.time())
                backup_path = current.with_name(
                    f"{current.stem}.bak-{ts}{current.suffix}"
                )
                current.rename(backup_path)
                log.info("Backed up %s → %s", current, backup_path)
            else:
                log.info("No trained weights at %s — nothing to back up.", current)

            base = _BASE_WEIGHTS
            if not base.exists():
                raise RuntimeError(
                    f"No base weights at {base}. Either run start.sh once "
                    f"to download yolov9c.pt, or restore the .bak file."
                )

            log.info("Reloading detector from base weights: %s", base)
            self._detector = YOLOv9Detector(
                weights_path=str(base),
                device=self._cfg.model.device,
                imgsz=self._cfg.model.imgsz,
                conf_threshold=self._cfg.model.conf_threshold,
                iou_threshold=self._cfg.model.iou_threshold,
                # IMPORTANT: pass None so the detector uses the *model's*
                # class names (COCO 80) — not the config's [ecoli] which
                # would mislabel every box.
                class_names=None,
            )

            return {
                "backup": str(backup_path) if backup_path else None,
                "active_weights": str(base),
                "classes": list(self._detector.class_names),
                "device": self._detector.device,
            }


# Module-level singleton — imported by routes and the streaming generator.
service = DetectorService()
