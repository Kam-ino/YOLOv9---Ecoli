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

# Ultralytics keeps at most 300 boxes per image by default; a dense stained
# slide holds more cells than that, so the tail was silently dropped.
MAX_DET = 1000
# Tiling: a frame whose long side exceeds TILE_TRIGGER x imgsz is also run as
# overlapping imgsz-sized crops at native resolution. Shrinking a 2000 px
# slide to 640 leaves rods only a few pixels long — below what the detector
# resolves. The overlap must exceed a single cell so each one is whole in
# at least one tile.
TILE_TRIGGER = 1.5
TILE_OVERLAP = 0.2
# A tile box ending this close to an interior tile edge is a cut-off object
# that a neighbouring tile sees whole: drop it.
TILE_EDGE_PX = 2


def tile_origins(size: int, tile: int, overlap: float = TILE_OVERLAP) -> List[int]:
    """Start offsets of evenly spaced, overlapping tiles covering ``size`` px."""
    if size <= tile:
        return [0]
    step = tile * (1.0 - overlap)
    n = int(np.ceil((size - tile) / step)) + 1
    return [int(round(o)) for o in np.linspace(0, size - tile, n)]


def keep_tile_box(
    box: Sequence[float], ox: int, oy: int, tile_w: int, tile_h: int,
    frame_w: int, frame_h: int,
) -> bool:
    """False for a box cut by an *interior* tile edge (frame borders are fine).

    ``box`` is tile-local ``(x1, y1, x2, y2)``; ``ox, oy`` is the tile origin.
    """
    x1, y1, x2, y2 = box
    e = TILE_EDGE_PX
    return not (
        (x1 <= e and ox > 0)
        or (y1 <= e and oy > 0)
        or (x2 >= tile_w - e and ox + tile_w < frame_w)
        or (y2 >= tile_h - e and oy + tile_h < frame_h)
    )


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

    def _raw(self, images: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Per image: ``(xyxy, conf, class_id)`` numpy arrays, post-NMS."""
        try:
            results = self.model.predict(
                images,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                max_det=MAX_DET,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            raise InferenceError(f"model.predict raised: {exc}") from exc
        # Ultralytics returns torch tensors on the inference device;
        # move to CPU + numpy for downstream visualization / JSON.
        return [
            (r.boxes.xyxy.cpu().numpy().reshape(-1, 4),
             r.boxes.conf.cpu().numpy(),
             r.boxes.cls.cpu().numpy().astype(int))
            for r in results
        ]

    def _tiled(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Whole frame (catches large clusters) + native-resolution tiles
        (catches small cells), merged with one class-aware NMS pass."""
        import torch
        from torchvision.ops import batched_nms

        h, w = frame.shape[:2]
        t = self.imgsz
        parts = self._raw([frame])
        origins = [(ox, oy) for oy in tile_origins(h, t) for ox in tile_origins(w, t)]
        for i in range(0, len(origins), 8):   # batches of 8 bound GPU memory
            chunk = origins[i:i + 8]
            crops = [np.ascontiguousarray(frame[oy:oy + t, ox:ox + t]) for ox, oy in chunk]
            for (ox, oy), crop, (b, c, k) in zip(chunk, crops, self._raw(crops)):
                th, tw = crop.shape[:2]
                keep = np.array(
                    [keep_tile_box(bb, ox, oy, tw, th, w, h) for bb in b], dtype=bool,
                )
                parts.append((b[keep] + np.array([ox, oy, ox, oy], dtype=b.dtype),
                              c[keep], k[keep]))

        xyxy = np.concatenate([p[0] for p in parts])
        confs = np.concatenate([p[1] for p in parts])
        cls_ids = np.concatenate([p[2] for p in parts])
        if len(xyxy) == 0:
            return xyxy, confs, cls_ids
        idx = batched_nms(
            torch.from_numpy(xyxy).float(), torch.from_numpy(confs).float(),
            torch.from_numpy(cls_ids), self.iou,
        ).numpy()[:MAX_DET * 3]
        return xyxy[idx], confs[idx], cls_ids[idx]

    def predict(self, frame: np.ndarray, tiled: bool = False) -> List[Detection]:
        """Run inference on a single BGR frame.

        ``tiled=True`` additionally runs native-resolution tiles when the
        frame is much larger than ``imgsz`` (see ``TILE_TRIGGER``). It costs
        one extra forward pass per tile, so still-image callers opt in and
        the live stream does not.

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

        if tiled and max(frame.shape[:2]) > TILE_TRIGGER * self.imgsz:
            xyxy, confs, cls_ids = self._tiled(frame)
        else:
            xyxy, confs, cls_ids = self._raw([frame])[0]

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
