"""
Stage 1: train a YOLO11 model to localize defects.

MLflow tracking is built into Ultralytics — just point MLFLOW_TRACKING_URI at a
local folder or a remote MLflow server before running this script.

Usage:
    export MLFLOW_TRACKING_URI=./mlruns
    python models/train_yolo.py --data defect_data.yaml --epochs 100 --imgsz 640 --model yolo11s.pt
"""

import argparse
import os
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="defect_data.yaml")
    parser.add_argument("--model", default="yolo11s.pt",
                         help="yolo11n.pt (fastest) | yolo11s.pt (balanced) | yolo11m.pt (accurate)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="defect_yolo11")
    args = parser.parse_args()

    os.environ.setdefault("MLFLOW_TRACKING_URI", "./mlruns")
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "vision-qa-yolo")

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        patience=20,          # early stopping
        save=True,
        plots=True,
    )

    metrics = model.val()
    print("\n=== Validation metrics ===")
    print(f"mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision:    {metrics.box.mp:.4f}")
    print(f"Recall:       {metrics.box.mr:.4f}")

    best_weights = f"{args.project}/{args.name}/weights/best.pt"
    print(f"\nBest weights saved to: {best_weights}")
    print("Register this path in models/inference.py once you're happy with it.")


if __name__ == "__main__":
    main()
