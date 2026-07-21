"""
Real-time inference on a webcam feed, a video file, or an RTSP camera stream —
the kind of live input a factory-floor camera would actually provide.

Usage:
    python models/realtime_inference.py --source 0                    # webcam
    python models/realtime_inference.py --source path/to/video.mp4    # video file
    python models/realtime_inference.py --source rtsp://camera-ip/stream  # IP camera

Draws bounding boxes + defect type + confidence on each frame, shows a live FPS
counter, and optionally saves an annotated output video.
"""

import argparse
import time
from collections import deque

import cv2
import torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
from ultralytics import YOLO

CLASSIFY_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class RealtimeDefectPipeline:
    def __init__(self, yolo_weights: str, resnet_weights: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.detector = YOLO(yolo_weights)

        checkpoint = torch.load(resnet_weights, map_location=self.device)
        self.classes = checkpoint["classes"]
        self.classifier = models.resnet50(weights=None)
        self.classifier.fc = nn.Linear(self.classifier.fc.in_features, len(self.classes))
        self.classifier.load_state_dict(checkpoint["model_state"])
        self.classifier.to(self.device).eval()

    def predict_frame(self, frame_bgr, conf_threshold: float = 0.25):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        results = self.detector.predict(image, conf=conf_threshold, verbose=False)[0]

        predictions = []
        for box in results.boxes:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            det_conf = float(box.conf[0])

            crop = image.crop((x1, y1, x2, y2))
            tensor = CLASSIFY_TF(crop).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.classifier(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                cls_idx = int(torch.argmax(probs))

            predictions.append({
                "bbox": (x1, y1, x2, y2),
                "detection_confidence": det_conf,
                "defect_type": self.classes[cls_idx],
                "classification_confidence": float(probs[cls_idx]),
            })
        return predictions


def draw_predictions(frame, predictions):
    for p in predictions:
        x1, y1, x2, y2 = p["bbox"]
        label = f"{p['defect_type']} {p['classification_confidence']:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0",
                         help="0 for default webcam, a file path, or an rtsp:// URL")
    parser.add_argument("--yolo-weights", default="runs/detect/defect_yolo11/weights/best.pt")
    parser.add_argument("--resnet-weights", default="models/resnet50_defect_classifier.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save-out", default=None, help="optional path to save annotated video")
    parser.add_argument("--frame-skip", type=int, default=1,
                         help="run inference every N frames (raise this to save compute on weak GPUs)")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    pipeline = RealtimeDefectPipeline(args.yolo_weights, args.resnet_weights)

    writer = None
    if args.save_out:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 20
        writer = cv2.VideoWriter(args.save_out, fourcc, fps_in, (w, h))

    fps_window = deque(maxlen=30)
    frame_idx = 0
    last_predictions = []

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.perf_counter()
        if frame_idx % args.frame_skip == 0:
            last_predictions = pipeline.predict_frame(frame, conf_threshold=args.conf)
        frame_idx += 1

        frame = draw_predictions(frame, last_predictions)

        fps_window.append(1.0 / max(time.perf_counter() - t0, 1e-6))
        fps = sum(fps_window) / len(fps_window)
        cv2.putText(frame, f"FPS: {fps:.1f} | defects: {len(last_predictions)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Vision QA - Real-Time Defect Detection", frame)
        if writer:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
