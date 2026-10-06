"""
src/config.py
=============
Loads ``config.yaml`` into typed dataclass containers and validates the
fields the rest of the app depends on. Using dataclasses (rather than
pydantic) keeps the dependency surface small — the FastAPI layer that
will eventually wrap this code can pull these objects in without
pulling pydantic v1 / v2 compatibility into the picture.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import yaml


log = logging.getLogger(__name__)

ALGORITHMS = ("yolov9", "rfdetr")
RFDETR_VARIANTS = ("nano", "small", "medium", "large")


@dataclass
class ModelConfig:
    weights: str
    device: str
    imgsz: int
    conf_threshold: float
    iou_threshold: float
    # Default detector for callers that don't pass ?algorithm=. Both
    # detectors stay available regardless; this only picks the default.
    algorithm: str = "yolov9"


@dataclass
class RFDETRConfig:
    """RF-DETR detector settings (``rfdetr:`` block, optional in yaml).

    ``weights`` is the fine-tuned checkpoint the Train tab writes; when the
    file is missing the COCO-pretrained ``variant`` is served instead.
    ``resolution`` must be a multiple of 32 and should match training.
    ``num_queries`` caps detections per image (dense slides need >300).
    """
    weights: str = "models/best_rfdetr.pth"
    variant: str = "medium"
    resolution: int = 640
    num_queries: int = 1200


@dataclass
class CaptureConfig:
    source: Union[int, str]
    width: int
    height: int
    fps: int


@dataclass
class PreprocessingConfig:
    apply_clahe: bool
    clahe_clip_limit: float
    clahe_tile_grid_size: int


@dataclass
class LoggingConfig:
    level: str
    file: str


@dataclass
class OutputConfig:
    save_video: bool
    output_dir: str


@dataclass
class AppConfig:
    model: ModelConfig
    capture: CaptureConfig
    preprocessing: PreprocessingConfig
    classes: List[str]
    logging: LoggingConfig
    output: OutputConfig
    rfdetr: RFDETRConfig


def load_config(path: str) -> AppConfig:
    """Parse and validate ``config.yaml``.

    Raises:
        FileNotFoundError: if the path does not exist
        ValueError:        if the file is missing required sections / fields
                           or contains out-of-range values
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config not found at {cfg_path.resolve()}. "
            f"Either create config.yaml at the repo root or pass --config <path>."
        )

    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"{cfg_path} did not parse to a mapping (got {type(raw).__name__}).")

    try:
        model = ModelConfig(**raw["model"])
        capture = CaptureConfig(**raw["capture"])
        preproc = PreprocessingConfig(**raw["preprocessing"])
        logging_c = LoggingConfig(**raw["logging"])
        output = OutputConfig(**raw["output"])
        rfdetr = RFDETRConfig(**(raw.get("rfdetr") or {}))
        classes = list(raw.get("classes", []))
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Invalid config structure in {cfg_path}: {exc}. "
            f"Compare against the shipped config.yaml for the expected layout."
        ) from exc

    _validate(model, classes, rfdetr)

    log.debug("Config loaded from %s", cfg_path)
    return AppConfig(
        model=model,
        capture=capture,
        preprocessing=preproc,
        classes=classes,
        logging=logging_c,
        output=output,
        rfdetr=rfdetr,
    )


def _validate(model: ModelConfig, classes: List[str], rfdetr: RFDETRConfig) -> None:
    if not classes:
        raise ValueError("config.yaml must declare at least one entry under 'classes'.")

    if model.algorithm not in ALGORITHMS:
        raise ValueError(
            f"model.algorithm must be one of {ALGORITHMS} (got {model.algorithm!r})."
        )
    if rfdetr.variant not in RFDETR_VARIANTS:
        raise ValueError(
            f"rfdetr.variant must be one of {RFDETR_VARIANTS} (got {rfdetr.variant!r})."
        )
    # RF-DETR's DINOv2 backbone needs resolution % (patch_size * num_windows) == 0 (32).
    if rfdetr.resolution <= 0 or rfdetr.resolution % 32 != 0:
        raise ValueError(
            f"rfdetr.resolution must be a positive multiple of 32 (got {rfdetr.resolution})."
        )
    if rfdetr.num_queries < 1:
        raise ValueError(f"rfdetr.num_queries must be >= 1 (got {rfdetr.num_queries}).")

    # YOLO requires image dims to be multiples of the maximum stride (32).
    if model.imgsz <= 0 or model.imgsz % 32 != 0:
        raise ValueError(
            f"model.imgsz must be a positive multiple of 32 (got {model.imgsz})."
        )
    if not (0.0 < model.conf_threshold <= 1.0):
        raise ValueError(
            f"model.conf_threshold must be in (0, 1] (got {model.conf_threshold})."
        )
    if not (0.0 < model.iou_threshold <= 1.0):
        raise ValueError(
            f"model.iou_threshold must be in (0, 1] (got {model.iou_threshold})."
        )
