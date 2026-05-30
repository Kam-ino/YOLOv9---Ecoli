"""Pydantic response models for /api/* endpoints."""
from typing import Dict, List, Tuple

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
