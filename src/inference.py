"""
src/inference.py
================
Detector wrappers with one uniform API.

Two backends share the same frame-in / detections-out contract:

* :class:`YOLOv9Detector` — Ultralytics YOLOv9 (``.pt`` / ``.onnx``), NMS-based.
* :class:`RFDETRDetector` — Roboflow RF-DETR (DINOv2 + DETR decoder), NMS-free.

Both take a BGR numpy frame and return a list of :class:`Detection`; the
tiling logic for large slides lives once in :class:`_Detector`.

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

ALGORITHMS = ("yolov9", "rfdetr")
RFDETR_VARIANTS = ("nano", "small", "medium", "large")


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


RawBatch = List[Tuple[np.ndarray, np.ndarray, np.ndarray]]


class _Detector:
    """Shared frame-in / detections-out logic.

    Subclasses set ``imgsz``, ``conf``, ``iou``, ``max_det``, ``device``,
    ``class_names`` and implement :meth:`_raw`. Everything else — tiling,
    tile merging, the :class:`Detection` conversion — lives here once.
    """

    algorithm: str = "?"
    imgsz: int
    conf: float
    iou: float
    max_det: int
    device: str
    class_names: List[str]

    def _raw(self, images: List[np.ndarray]) -> RawBatch:
        """Per BGR image: ``(xyxy, conf, class_id)`` numpy arrays in image pixels."""
        raise NotImplementedError

    def _warmup(self) -> None:
        """Run one dummy inference so the first real frame isn't slow.

        Without warmup the first frame can take 1-3 seconds while CUDA
        kernels JIT-compile and ONNXRuntime / cuDNN populate caches —
        a visible hitch on a "live" feed.
        """
        try:
            self._raw([np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)])
            log.debug("Warmup inference complete.")
        except Exception as exc:  # non-fatal
            log.warning("Warmup inference failed (continuing): %s", exc)

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
        ).numpy()[:self.max_det * 3]
        return xyxy[idx], confs[idx], cls_ids[idx]

    def predict(self, frame: np.ndarray, tiled: bool = False) -> List[Detection]:
        """Run inference on a single BGR frame.

        ``tiled=True`` additionally runs native-resolution tiles when the
        frame is much larger than ``imgsz`` (see ``TILE_TRIGGER``). It costs
        one extra forward pass per tile, so still-image callers opt in and
        the live stream does not.

        Returns an empty list when there are no detections above
        ``conf_threshold``. Raises :class:`InferenceError` for genuinely
        broken input or model failure.
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


class YOLOv9Detector(_Detector):
    """Thin, frame-in / detections-out wrapper around ultralytics.YOLO."""

    algorithm = "yolov9"

    def __init__(
        self,
        weights_path: str,
        device: str = "auto",
        imgsz: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_names: Optional[Sequence[str]] = None,
        warmup: bool = True,
        max_det: int = MAX_DET,
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
        self.max_det = int(max_det)

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

    def _raw(self, images: List[np.ndarray]) -> RawBatch:
        """Per image: ``(xyxy, conf, class_id)`` numpy arrays, post-NMS."""
        try:
            results = self.model.predict(
                images,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
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


class RFDETRDetector(_Detector):
    """Frame-in / detections-out wrapper around Roboflow's ``rfdetr``.

    RF-DETR is NMS-free: the decoder emits ``num_queries`` candidate boxes
    per image and ``conf_threshold`` is the only filter. ``iou_threshold``
    is therefore unused for a single pass — it only drives the tile-merge
    NMS in :meth:`_Detector._tiled`, exactly as for YOLOv9.

    ``weights_path`` may point at a fine-tuned ``checkpoint_best_ema.pth``
    (see ``training/train_rfdetr.py``). When the file is missing the
    COCO-pretrained ``variant`` is loaded instead, so "reset to base" and
    a fresh install both work without a local file.
    """

    algorithm = "rfdetr"

    def __init__(
        self,
        weights_path: Optional[str],
        variant: str = "medium",
        resolution: int = 640,
        num_queries: int = 1200,
        device: str = "auto",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        class_names: Optional[Sequence[str]] = None,
        warmup: bool = True,
    ):
        try:
            import rfdetr  # heavy import — keep inside __init__
        except ImportError as exc:
            raise InferenceError(
                "rfdetr is not installed. Run: pip install -r requirements.txt"
            ) from exc

        variant = str(variant).strip().lower()
        if variant not in RFDETR_VARIANTS:
            raise InferenceError(
                f"Unknown RF-DETR variant {variant!r}; choose one of {RFDETR_VARIANTS}."
            )
        model_cls = getattr(rfdetr, f"RFDETR{variant.capitalize()}")

        self.device = _resolve_device(device)
        self.imgsz = int(resolution)
        self.conf = float(conf_threshold)
        self.iou = float(iou_threshold)
        self.max_det = int(num_queries)

        kwargs = dict(resolution=self.imgsz, device=self.device)
        weights = Path(weights_path) if weights_path else None
        if weights is not None and weights.is_file():
            # A fine-tuned checkpoint carries exactly num_queries query slots
            # (training/train_rfdetr.py expands the COCO base to that count).
            kwargs.update(pretrain_weights=str(weights), num_queries=self.max_det, num_select=self.max_det)
            source = str(weights)
        else:
            # The published COCO checkpoint has the library's default query
            # count; asking for more would fail to load, so keep its default.
            source = f"COCO-pretrained rf-detr-{variant}"
            log.info("No RF-DETR weights at %s — loading %s (default query count).", weights, source)

        try:
            self.model = model_cls(**kwargs)
        except Exception as exc:
            raise InferenceError(
                f"Failed to load RF-DETR ({source}): {exc}"
            ) from exc

        model_names = getattr(self.model, "class_names", None)
        if class_names:
            self.class_names: List[str] = list(class_names)
        elif model_names:
            self.class_names = list(model_names)
        else:
            self.class_names = ["object"]

        log.info(
            "Detector ready: weights=%s backend=rfdetr-%s device=%s "
            "resolution=%d queries=%d conf=%.2f classes=%s",
            source, variant, self.device, self.imgsz, self.max_det, self.conf,
            self.class_names,
        )

        if warmup:
            self._warmup()

    def _raw(self, images: List[np.ndarray]) -> RawBatch:
        """Per image: ``(xyxy, conf, class_id)`` numpy arrays. No NMS."""
        # rfdetr expects RGB; the rest of the app speaks BGR (cv2).
        rgb = [np.ascontiguousarray(img[:, :, ::-1]) for img in images]
        try:
            dets = self.model.predict(rgb, threshold=self.conf, include_source_image=False)
        except Exception as exc:
            raise InferenceError(f"rfdetr predict raised: {exc}") from exc
        if not isinstance(dets, (list, tuple)):
            dets = [dets]
        out: RawBatch = []
        for d in dets:
            xyxy = np.asarray(d.xyxy, dtype=np.float32).reshape(-1, 4)
            n = len(xyxy)
            conf = (np.asarray(d.confidence, dtype=np.float32)
                    if d.confidence is not None else np.ones(n, dtype=np.float32))
            cls = (np.asarray(d.class_id).astype(int)
                   if d.class_id is not None else np.zeros(n, dtype=int))
            out.append((xyxy, conf, cls))
        return out


def build_detector(algorithm: str, cfg, class_names: Optional[Sequence[str]] = None,
                   weights_path: Optional[str] = None, **overrides) -> _Detector:
    """Construct a detector for ``algorithm`` from an :class:`src.config.AppConfig`.

    ``weights_path`` overrides the configured path (used for hot-swaps and
    resets); ``overrides`` go straight to the constructor.
    """
    algorithm = str(algorithm).strip().lower()
    if algorithm == "yolov9":
        kw = dict(
            weights_path=weights_path or cfg.model.weights,
            device=cfg.model.device,
            imgsz=cfg.model.imgsz,
            conf_threshold=cfg.model.conf_threshold,
            iou_threshold=cfg.model.iou_threshold,
            class_names=class_names,
        )
        kw.update(overrides)
        return YOLOv9Detector(**kw)
    if algorithm == "rfdetr":
        kw = dict(
            weights_path=weights_path if weights_path is not None else cfg.rfdetr.weights,
            variant=cfg.rfdetr.variant,
            resolution=cfg.rfdetr.resolution,
            num_queries=cfg.rfdetr.num_queries,
            device=cfg.model.device,
            conf_threshold=cfg.model.conf_threshold,
            iou_threshold=cfg.model.iou_threshold,
            class_names=class_names,
        )
        kw.update(overrides)
        return RFDETRDetector(**kw)
    raise InferenceError(f"Unknown algorithm {algorithm!r}; choose one of {ALGORITHMS}.")
