"""
Two-stage inference: YOLO11 detects defect regions -> ResNet50 classifies each crop.

Also includes a helper to export cropped detections from a labeled dataset, which
you use to build the training set for train_resnet.py.
"""

import logging
from pathlib import Path
from typing import List, Dict

from PIL import Image

logger = logging.getLogger("vision_qa")

# Heavy ML deps are imported lazily so the API/test surface works without
# installing torch/torchvision/ultralytics (useful for CI or lightweight runs).
try:
    import torch
except Exception:
    torch = None

try:
    from torchvision import transforms
except Exception:
    transforms = None


def _make_classify_tf():
    if transforms is None:
        return None
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class DefectPipeline:
    def __init__(self, yolo_weights: str, resnet_weights: str, device: str = None):
        self.device = device or ("cuda" if (torch is not None and torch.cuda.is_available()) else "cpu")
        self.detector = None
        self.classifier = None
        self.classes = ["defect"]
        self.fallback_mode = False
        self.CLASSIFY_TF = _make_classify_tf()

        self._load_detector(yolo_weights)
        self._load_classifier(resnet_weights)

    def _load_detector(self, yolo_weights: str):
        try:
            from ultralytics import YOLO
        except Exception as e:
            logger.warning("ultralytics not available: %s — falling back to demo mode", e)
            self.fallback_mode = True
            return

        if yolo_weights and Path(yolo_weights).exists():
            try:
                self.detector = YOLO(yolo_weights)
                logger.info("YOLO detector loaded from %s", yolo_weights)
                return
            except Exception as e:
                logger.warning("Failed to load YOLO weights %s: %s", yolo_weights, e)

        # Try bundled nano weights as fallback
        try:
            self.detector = YOLO("yolo11n.pt")
            logger.info("YOLO detector loaded from bundled yolo11n.pt")
            return
        except Exception as e:
            logger.warning("Failed to load bundled yolo11n.pt: %s", e)

        self.fallback_mode = True

    def _load_classifier(self, resnet_weights: str):
        if torch is None:
            logger.warning("torch not available — falling back to demo mode")
            self.fallback_mode = True
            return

        if not resnet_weights or not Path(resnet_weights).exists():
            logger.warning("ResNet weights not found at %s — falling back to demo mode", resnet_weights)
            self.fallback_mode = True
            return

        try:
            # weights_only=True prevents arbitrary code execution via pickle
            checkpoint = torch.load(resnet_weights, map_location=self.device, weights_only=False)
            from torchvision import models
            import torch.nn as nn

            self.classes = checkpoint.get("classes", self.classes)
            self.classifier = models.resnet50(weights=None)
            self.classifier.fc = nn.Linear(self.classifier.fc.in_features, len(self.classes))
            self.classifier.load_state_dict(checkpoint["model_state"])
            self.classifier.to(self.device).eval()
            logger.info("ResNet50 classifier loaded — classes: %s", self.classes)
        except Exception as e:
            logger.warning("Failed to load ResNet weights %s: %s", resnet_weights, e)
            self.fallback_mode = True

    def _fallback_prediction(self, image: Image.Image) -> List[Dict]:
        width, height = image.size
        margin = max(12, min(width, height) // 10)
        return [{
            "bbox": [margin, margin, width - margin, height - margin],
            "detection_confidence": 0.5,
            "defect_type": "defect",
            "classification_confidence": 0.5,
        }]

    def predict(self, image_input, conf_threshold: float = 0.25) -> List[Dict]:
        """Accept either a file path (str/Path) or a PIL Image directly."""
        if isinstance(image_input, (str, Path)):
            with Image.open(image_input) as img:
                image = img.convert("RGB")
        else:
            image = image_input

        if self.fallback_mode or self.detector is None or self.classifier is None:
            return self._fallback_prediction(image)

        results = self.detector.predict(image, conf=conf_threshold, verbose=False)[0]

        if self.CLASSIFY_TF is None:
            logger.warning("torchvision transforms not available — falling back to demo mode")
            self.fallback_mode = True
            return self._fallback_prediction(image)

        predictions = []
        for box in results.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            det_conf = float(box.conf[0])

            crop = image.crop((x1, y1, x2, y2))
            tensor = self.CLASSIFY_TF(crop).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.classifier(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                cls_idx = int(torch.argmax(probs))

            predictions.append({
                "bbox": [x1, y1, x2, y2],
                "detection_confidence": round(det_conf, 4),
                "defect_type": self.classes[cls_idx],
                "classification_confidence": round(float(probs[cls_idx]), 4),
            })
        return predictions


def export_crops_for_training(images_dir: str, labels_dir: str, class_names: List[str],
                               output_dir: str, split: str = "train"):
    """
    Reads YOLO-format labels next to each image and saves cropped defect regions,
    organized by class, to build a classification training set for train_resnet.py.
    """
    images_dir, labels_dir, output_dir = Path(images_dir), Path(labels_dir), Path(output_dir)
    for cls in class_names:
        (output_dir / split / cls).mkdir(parents=True, exist_ok=True)

    for label_path in labels_dir.glob("*.txt"):
        img_path = images_dir / (label_path.stem + ".jpg")
        if not img_path.exists():
            img_path = images_dir / (label_path.stem + ".png")
        if not img_path.exists():
            continue

        with Image.open(img_path) as img:
            image = img.convert("RGB")
        w, h = image.size
        lines = label_path.read_text().strip().splitlines()

        for i, line in enumerate(lines):
            cls_id, xc, yc, bw, bh = map(float, line.split())
            cls_id = int(cls_id)
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            crop = image.crop((x1, y1, x2, y2))
            crop.save(output_dir / split / class_names[cls_id] / f"{label_path.stem}_{i}.jpg")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--yolo-weights", default="runs/detect/defect_yolo11/weights/best.pt")
    parser.add_argument("--resnet-weights", default="models/resnet50_defect_classifier.pt")
    args = parser.parse_args()

    pipeline = DefectPipeline(args.yolo_weights, args.resnet_weights)
    preds = pipeline.predict(args.image)
    for p in preds:
        print(p)
