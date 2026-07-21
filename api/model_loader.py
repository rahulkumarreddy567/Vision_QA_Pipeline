import os
from functools import lru_cache

from models.inference import DefectPipeline

YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "runs/detect/defect_yolo11/weights/best.pt")
RESNET_WEIGHTS = os.environ.get("RESNET_WEIGHTS", "models/resnet50_defect_classifier.pt")


@lru_cache(maxsize=1)
def get_pipeline() -> DefectPipeline:
    """Loads both models once and reuses across requests (lru_cache = singleton)."""
    return DefectPipeline(yolo_weights=YOLO_WEIGHTS, resnet_weights=RESNET_WEIGHTS)
