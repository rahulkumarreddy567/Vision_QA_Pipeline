from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class Detection(BaseModel):
    bbox: List[int] = Field(..., description="[x1, y1, x2, y2] pixel coordinates")
    detection_confidence: float = Field(..., ge=0.0, le=1.0)
    defect_type: str
    classification_confidence: float = Field(..., ge=0.0, le=1.0)

    @field_validator("bbox")
    @classmethod
    def bbox_must_have_four_coords(cls, v: List[int]) -> List[int]:
        if len(v) != 4:
            raise ValueError("bbox must have exactly 4 coordinates [x1, y1, x2, y2]")
        return [int(c) for c in v]  # ensure ints even if floats come from YOLO


class PredictionResponse(BaseModel):
    filename: Optional[str]
    num_defects: int
    inference_time_ms: float
    detections: List[Detection]
    request_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    yolo_loaded: bool
    resnet_loaded: bool
    version: str = "1.0.0"
    mode: str = "production"
