import os
import logging
from functools import lru_cache

from models.inference import DefectPipeline

logger = logging.getLogger("vision_qa")

YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "runs/detect/defect_yolo11/weights/best.pt")
RESNET_WEIGHTS = os.environ.get("RESNET_WEIGHTS", "models/resnet50_defect_classifier.pt")


@lru_cache(maxsize=1)
def get_pipeline() -> DefectPipeline:
    """Loads both models once and reuses across requests (lru_cache = singleton)."""
    logger.info("Loading pipeline — YOLO: %s | ResNet: %s", YOLO_WEIGHTS, RESNET_WEIGHTS)
    pipeline = DefectPipeline(yolo_weights=YOLO_WEIGHTS, resnet_weights=RESNET_WEIGHTS)
    logger.info(
        "Pipeline ready — fallback_mode=%s detector=%s classifier=%s",
        pipeline.fallback_mode,
        bool(pipeline.detector),
        bool(pipeline.classifier),
    )
    return pipeline


def reset_pipeline():
    """Clear the cached pipeline (useful for testing or hot-reload scenarios)."""
    get_pipeline.cache_clear()
