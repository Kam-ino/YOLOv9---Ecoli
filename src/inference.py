"""
src/inference.py
================
YOLOv9 detector wrapper.

Loads YOLOv9 weights (PyTorch ``.pt`` or ONNX ``.onnx``) via Ultralytics
and exposes a uniform :meth:`YOLOv9Detector.predict` API that takes a
BGR numpy frame and returns a list of :class:`Detection` objects.

Library choice — Ultralytics vs WongKinYiu/yolov9
-------------------------------------------------
We use the Ultralytics package (``pip install ultralytics``) rather than
cloning the original WongKinYiu/yolov9 repo because:

* one entry point (``YOLO(path)``) transparently loads ``.pt`` / ``.onnx``
  / ``.engine`` / ``.openvino``, so we don't need a separate inference
  backend per export format;
* training, prediction, and export to ONNX/TensorRT share a single API
  — see ``training/train.py`` for the matching training entry point;
* it is actively maintained and integrates Albumentations, AMP, EMA and
  cosine LR by default.

The original WongKinYiu repo is only preferable when you need to modify
the architecture itself (custom heads, custom losses). This app does
inference and fine-tuning only, so Ultralytics is the right call.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np


log = logging.getLogger(__name__)


class InferenceError(RuntimeError):
    """Raised when the detector cannot be initialized or fails inference."""


@dataclass
class Detection:
    """A single object detection.

    Attributes:
        bbox:        ``(x1, y1, x2, y2)`` in *original* frame pixel coords.
        confidence:  Model confidence in ``[0, 1]``.
        class_id:    Integer class id, indexes into ``class_names``.
        class_name:  Human-readable class label.
    """
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


def _resolve_device(spec: str) -> str:
    """Convert a config ``device`` spec into a concrete torch device string."""
    if spec is None:
        spec = "auto"
    spec = str(spec).strip().lower()
    if spec not in ("auto", "cuda", "cpu") and not spec.startswith("cuda:"):
        log.warning("Unknown device spec %r — falling back to 'auto'.", spec)
        spec = "auto"

    if spec == "auto":
        try:
            import torch  # local import keeps cold-start light when not needed
            if torch.cuda.is_available():
                return "cuda:0"
        except ImportError:
            pass
        return "cpu"
    if spec == "cuda":
        return "cuda:0"
    return spec


class YOLOv9Detector:
    """Thin, frame-in / detections-out wrapper around ultralytics.YOLO."""

    def __init__(
        self,
        weights_path: str,
        device: str = "auto",
        imgsz: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_names: Optional[Sequence[str]] = None,
        warmup: bool = True,
    ):
        weights = Path(weights_path)
        if not weights.exists():
            raise InferenceError(
                f"Model weights not found: {weights}. "
                "Either fine-tune your own (see training/train.py) "
                "and place them here, or update model.weights in config.yaml."
            )

        try:
            from ultralytics import YOLO  # heavy import — keep inside __init__
        except ImportError as exc:
            raise InferenceError(
                "ultralytics is not installed. Run: pip install -r requirements.txt"
            ) from exc

        try:
            self.model = YOLO(str(weights))
        except Exception as exc:  # Ultralytics raises a variety of types
            raise InferenceError(
                f"Failed to load YOLOv9 model from {weights}: {exc}"
            ) from exc

        self.device = _resolve_device(device)
        self.imgsz = int(imgsz)
        self.conf = float(conf_threshold)
        self.iou = float(iou_threshold)

        # Prefer explicit class_names from config (authoritative). Fall
        # back to the model's embedded names (a dict on Ultralytics
        # models). Last resort: a single 'object' label so we never
        # crash on label lookups.
        model_names = getattr(self.model, "names", None)
        if class_names:
            self.class_names: List[str] = list(class_names)
        elif isinstance(model_names, dict):
            self.class_names = [model_names[i] for i in sorted(model_names)]
        elif isinstance(model_names, (list, tuple)):
            self.class_names = list(model_names)
        else:
            self.class_names = ["object"]

        log.info(
            "Detector ready: weights=%s backend=ultralytics device=%s "
            "imgsz=%d conf=%.2f iou=%.2f classes=%s",
            weights, self.device, self.imgsz, self.conf, self.iou,
            self.class_names,
        )

        if warmup:
            self._warmup()

    def _warmup(self) -> None:
        """Run one dummy inference so the first real frame isn't slow.

        Without warmup the first frame can take 1-3 seconds while CUDA
        kernels JIT-compile and ONNXRuntime / cuDNN populate caches —
        a visible hitch on a "live" feed.
        """
        try:
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            self.model.predict(
                dummy,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=False,
            )
            log.debug("Warmup inference complete.")
        except Exception as exc:  # non-fatal
            log.warning("Warmup inference failed (continuing): %s", exc)

    def predict(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a single BGR frame.

        Returns an empty list when there are no detections above
        ``conf_threshold`` after NMS. Raises :class:`InferenceError`
        for genuinely broken input or model failure.
        """
        if frame is None or frame.size == 0:
            raise InferenceError("Received empty frame for inference.")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise InferenceError(
                f"Expected HxWx3 BGR frame, got shape {frame.shape}."
            )

        try:
            results = self.model.predict(
                frame,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            raise InferenceError(f"model.predict raised: {exc}") from exc

        if not results:
            return []

        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return []

        # Ultralytics returns torch tensors on the inference device;
        # move to CPU + numpy for downstream visualization / JSON.
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        detections: List[Detection] = []
        for (x1, y1, x2, y2), conf, cid in zip(xyxy, confs, cls_ids):
            name = (
                self.class_names[cid]
                if 0 <= cid < len(self.class_names)
                else f"cls_{cid}"
            )
            detections.append(
                Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(conf),
                    class_id=int(cid),
                    class_name=name,
                )
            )
        return detections
