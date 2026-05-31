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
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

import yaml


log = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    weights: str
    device: str
    imgsz: int
    conf_threshold: float
    iou_threshold: float


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
class ClusterMergeConfig:
    """Post-processing pass that groups dense source-class detections
    into single target-class boxes (e.g. many ``ecoli`` boxes → one
    ``ecoli_cluster`` box). All fields optional with sensible defaults
    so old configs continue to load."""
    enabled: bool = False
    margin: float = 0.01            # fraction of max(image_w, image_h)
    min_size: int = 3               # minimum boxes per cluster
    source_class_name: str = "ecoli"
    target_class_name: str = "ecoli_cluster"


@dataclass
class AppConfig:
    model: ModelConfig
    capture: CaptureConfig
    preprocessing: PreprocessingConfig
    classes: List[str]
    logging: LoggingConfig
    output: OutputConfig
    cluster_merge: ClusterMergeConfig = field(default_factory=ClusterMergeConfig)


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
        classes = list(raw.get("classes", []))
        # cluster_merge is optional — old configs without the section
        # get the dataclass defaults (enabled=False, no behaviour change).
        cluster_merge = ClusterMergeConfig(**(raw.get("cluster_merge") or {}))
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Invalid config structure in {cfg_path}: {exc}. "
            f"Compare against the shipped config.yaml for the expected layout."
        ) from exc

    _validate(model, classes)

    log.debug("Config loaded from %s", cfg_path)
    return AppConfig(
        model=model,
        capture=capture,
        preprocessing=preproc,
        classes=classes,
        logging=logging_c,
        output=output,
        cluster_merge=cluster_merge,
    )


def _validate(model: ModelConfig, classes: List[str]) -> None:
    if not classes:
        raise ValueError("config.yaml must declare at least one entry under 'classes'.")

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
