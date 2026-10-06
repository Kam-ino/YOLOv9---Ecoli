"""
backend/app/detector.py
=======================
Process-wide registry of detectors (one per algorithm) wrapping
:mod:`src.inference`.

The default detector (``model.algorithm`` in config) is loaded once at
FastAPI startup (the import + weights load is several hundred MB /
multiple seconds — doing it per-request would be unusable). The other
algorithm loads lazily the first time a request asks for it, so a YOLO-only
install never pays for RF-DETR. One re-entrant lock guards ``predict``
calls and model swaps: Ultralytics / PyTorch are not safe under concurrent
calls on a single model instance, and this is a single-user app.
"""
import importlib.util
import logging
import shutil
import time
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Optional

from src.config import ALGORITHMS, AppConfig, load_config
from src.inference import InferenceError, _Detector, build_detector


# Fall-back weights when the user resets the fine-tuned YOLO model. RF-DETR
# needs no file: its COCO-pretrained variant downloads on demand.
_BASE_WEIGHTS = Path("models/yolov9c.pt")


log = logging.getLogger(__name__)


class DetectorService:
    """Lazy singleton — :meth:`init` must be called once before use."""

    def __init__(self) -> None:
        self._cfg: Optional[AppConfig] = None
        self._detectors: Dict[str, _Detector] = {}
        self._errors: Dict[str, str] = {}      # last load failure per algorithm
        self._lock = RLock()

    def init(self, config_path: str) -> None:
        self._cfg = load_config(config_path)
        default = self._cfg.model.algorithm
        log.info("Initializing %s detector from %s", default, config_path)
        self._detectors[default] = self._build(default, self._weights_path(default))
        log.info("Detector ready.")

    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._cfg is not None and self._cfg.model.algorithm in self._detectors

    @property
    def config(self) -> AppConfig:
        if self._cfg is None:
            raise RuntimeError("DetectorService.init() has not been called.")
        return self._cfg

    @property
    def default_algorithm(self) -> str:
        return self.config.model.algorithm

    @property
    def detector(self) -> _Detector:
        """The default algorithm's detector (back-compat for older callers)."""
        return self.get(None)

    @property
    def lock(self) -> RLock:
        return self._lock

    # ------------------------------------------------------------------

    def resolve(self, algorithm: Optional[str]) -> str:
        """Normalise a request's ``algorithm`` (None → config default)."""
        algo = (algorithm or self.default_algorithm).strip().lower()
        if algo not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm {algorithm!r}; choose one of {ALGORITHMS}.")
        return algo

    def get(self, algorithm: Optional[str] = None) -> _Detector:
        """Return the detector for ``algorithm``, loading it on first use.

        Raises ``ValueError`` for an unknown name and
        :class:`src.inference.InferenceError` when it cannot be loaded.
        """
        algo = self.resolve(algorithm)
        det = self._detectors.get(algo)
        if det is not None:
            return det
        with self._lock:
            det = self._detectors.get(algo)       # re-check under the lock
            if det is None:
                log.info("Lazily loading %s detector.", algo)
                det = self._build(algo, self._weights_path(algo))
                self._detectors[algo] = det
            return det

    def available(self) -> Dict[str, Dict[str, Any]]:
        """Per-algorithm status for ``/api/health``."""
        out: Dict[str, Dict[str, Any]] = {}
        for algo in ALGORITHMS:
            det = self._detectors.get(algo)
            weights = self._weights_path(algo)
            out[algo] = {
                "available": self._installed(algo),
                "loaded": det is not None,
                "weights": str(weights),
                "weights_exists": weights.is_file(),
                "classes": list(det.class_names) if det else [],
                "device": det.device if det else None,
                "error": self._errors.get(algo),
            }
        return out

    # ------------------------------------------------------------------

    def _weights_path(self, algo: str) -> Path:
        cfg = self.config
        return Path(cfg.model.weights if algo == "yolov9" else cfg.rfdetr.weights)

    @staticmethod
    def _installed(algo: str) -> bool:
        pkg = "ultralytics" if algo == "yolov9" else "rfdetr"
        return importlib.util.find_spec(pkg) is not None

    def _build(self, algo: str, weights: Path, base: bool = False) -> _Detector:
        """Construct a detector; ``base=True`` loads the pretrained COCO model."""
        cfg = self.config
        if base:
            # COCO base → let the model's own 80 names through; the config's
            # [ecoli, ...] would mislabel every box.
            weights_arg = str(_BASE_WEIGHTS) if algo == "yolov9" else ""
            names = None
        else:
            weights_arg = str(weights)
            # Same rule for RF-DETR without a fine-tuned file: it serves the
            # COCO model, so the config's class list must not be applied.
            names = cfg.classes if (algo == "yolov9" or weights.is_file()) else None
        try:
            det = build_detector(algo, cfg, class_names=names, weights_path=weights_arg)
        except InferenceError as exc:
            self._errors[algo] = str(exc)
            raise
        self._errors.pop(algo, None)
        return det

    # ------------------------------------------------------------------

    def activate_weights(self, weights_path: str, algorithm: Optional[str] = None) -> Dict[str, Any]:
        """Copy freshly-trained weights over the active model path and
        hot-reload that algorithm's detector from them.

        Used to auto-deploy the best checkpoint of a finished training run:
          * The current active model (``model.weights`` / ``rfdetr.weights``
            in config) is backed up to ``<stem>.bak-<unix_ts><suffix>`` so you
            can roll back, unless it's already the same file.
          * ``weights_path`` is copied into the active path and the
            detector is rebuilt from it.

        Held under ``self._lock`` so any in-flight ``predict()`` finishes
        before the swap and no inference runs against a half-loaded model.
        """
        algo = self.resolve(algorithm)
        src = Path(weights_path)
        if not src.is_file():
            raise FileNotFoundError(f"No weights to activate at {src}.")

        with self._lock:
            target = self._weights_path(algo)
            target.parent.mkdir(parents=True, exist_ok=True)

            backup_path: Optional[Path] = None
            if target.exists() and target.resolve() != src.resolve():
                ts = int(time.time())
                backup_path = target.with_name(
                    f"{target.stem}.bak-{ts}{target.suffix}"
                )
                shutil.copy2(target, backup_path)
                log.info("Backed up active model %s → %s", target, backup_path)

            if target.resolve() != src.resolve():
                shutil.copy2(src, target)
            log.info("Activated trained %s weights %s → %s", algo, src, target)

            det = self._build(algo, target)
            self._detectors[algo] = det
            log.info("%s detector reloaded from activated weights.", algo)

            return {
                "algorithm": algo,
                "backup": str(backup_path) if backup_path else None,
                "active_weights": str(target),
                "classes": list(det.class_names),
                "device": det.device,
            }

    def reset(self, algorithm: Optional[str] = None) -> Dict[str, Any]:
        """Move the fine-tuned weights to a .bak file and reload from base.

        Idempotent and recoverable:
          * If a trained checkpoint is at the configured weights path, it's
            renamed to ``<stem>.bak-<unix_ts><suffix>`` next to the original
            so you can roll back manually.
          * The detector is rebuilt against the pretrained COCO base
            (``models/yolov9c.pt`` for YOLOv9; the downloaded RF-DETR variant
            otherwise). The badge in the UI will report 80 classes after
            this, signalling "untrained".
          * Returns a small dict describing what changed so the UI can
            show a useful confirmation message.

        Held under ``self._lock`` so any in-flight ``predict()`` finishes
        before the swap, and no new inference starts until the new model
        is fully loaded.
        """
        algo = self.resolve(algorithm)
        with self._lock:
            current = self._weights_path(algo)
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

            if algo == "yolov9" and not _BASE_WEIGHTS.exists():
                raise RuntimeError(
                    f"No base weights at {_BASE_WEIGHTS}. Either run start.sh once "
                    f"to download yolov9c.pt, or restore the .bak file."
                )

            log.info("Reloading %s detector from pretrained base.", algo)
            det = self._build(algo, current, base=True)
            self._detectors[algo] = det

            return {
                "algorithm": algo,
                "backup": str(backup_path) if backup_path else None,
                "active_weights": str(_BASE_WEIGHTS) if algo == "yolov9"
                else f"COCO-pretrained rf-detr-{self.config.rfdetr.variant}",
                "classes": list(det.class_names),
                "device": det.device,
            }


# Module-level singleton — imported by routes and the streaming generator.
service = DetectorService()
