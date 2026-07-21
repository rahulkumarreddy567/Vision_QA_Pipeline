from typing import List
from pydantic import BaseModel, Field


class Detection(BaseModel):
    bbox: List[int] = Field(..., description="[x1, y1, x2, y2] pixel coordinates")
    detection_confidence: float
    defect_type: str
    classification_confidence: float


class PredictionResponse(BaseModel):
    filename: str
    num_defects: int
    inference_time_ms: float
    detections: List[Detection]


class HealthResponse(BaseModel):
    status: str
    yolo_loaded: bool
    resnet_loaded: bool
