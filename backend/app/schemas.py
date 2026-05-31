"""Pydantic response models for /api/* endpoints."""
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class DetectionDTO(BaseModel):
    bbox: Tuple[float, float, float, float] = Field(
        ..., description="(x1, y1, x2, y2) in original image pixel coords",
    )
    class_id: int
    class_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    detections: List[DetectionDTO]
    image_size: Tuple[int, int] = Field(..., description="(width, height)")
    inference_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    classes: List[str]


# ---------------------------------------------------------------------------
# Labeling / dataset endpoints
# ---------------------------------------------------------------------------

class LabelBox(BaseModel):
    """One annotation in YOLO format. All coords normalized to [0, 1]."""
    class_id: int = Field(..., ge=0)
    cx: float = Field(..., ge=0.0, le=1.0)
    cy: float = Field(..., ge=0.0, le=1.0)
    w: float = Field(..., gt=0.0, le=1.0)
    h: float = Field(..., gt=0.0, le=1.0)


class DatasetEntry(BaseModel):
    filename: str
    split: str
    image_url: str
    num_boxes: int
    created_at: float


class DatasetSplitStats(BaseModel):
    images: int
    boxes: int


class DatasetStats(BaseModel):
    classes: List[str]
    splits: Dict[str, DatasetSplitStats]
    totals: DatasetSplitStats


class DatasetEntryWithBoxes(BaseModel):
    """Returned by /api/dataset/find — entry metadata + the box list."""
    entry: DatasetEntry
    boxes: List[LabelBox]


class AddClassRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)


# ---------------------------------------------------------------------------
# Training endpoints
# ---------------------------------------------------------------------------

class TrainStartRequest(BaseModel):
    weights: str = "yolov9c.pt"
    epochs: int = Field(100, ge=1, le=2000)
    batch: int = Field(16, ge=1, le=256)
    imgsz: int = Field(640, ge=64, le=2048)
    device: str = "auto"          # "auto" | "cpu" | "0" | "0,1" | ...
    name: Optional[str] = None


class TrainingStatusResponse(BaseModel):
    state: str                    # idle | running | completed | failed | killed
    pid: Optional[int] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    return_code: Optional[int] = None
    name: Optional[str] = None
    command: Optional[List[str]] = None
    log_lines: List[str]
