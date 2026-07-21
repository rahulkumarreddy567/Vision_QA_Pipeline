"""
Stage 2: fine-tune a ResNet50 (ImageNet pretrained) to classify the defect type/severity
on the cropped regions that YOLO11 detected in Stage 1.

Expects a directory structure like:
    data/raw/crops/train/<class_name>/*.jpg
    data/raw/crops/val/<class_name>/*.jpg

If you don't have cropped images yet, run models/inference.py in "crop-export" mode
on your training set first (see that file for the helper function).

Usage:
    export MLFLOW_TRACKING_URI=./mlruns
    python models/train_resnet.py --crops-dir data/raw/crops --epochs 30
"""

import argparse
import os

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def build_model(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # freeze the backbone, fine-tune only the last block + classifier head
    for name, param in model.named_parameters():
        if not name.startswith("layer4") and not name.startswith("fc"):
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops-dir", default="data/raw/crops")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--out", default="models/resnet50_defect_classifier.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(os.path.join(args.crops_dir, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(args.crops_dir, "val"), transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    class_names = train_ds.classes
    model = build_model(num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "./mlruns"))
    mlflow.set_experiment("vision-qa-resnet50")

    with mlflow.start_run():
        mlflow.log_params({
            "epochs": args.epochs, "batch_size": args.batch, "lr": args.lr,
            "num_classes": len(class_names), "backbone": "resnet50_imagenet",
        })

        best_val_acc = 0.0
        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
            train_loss = running_loss / len(train_ds)

            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            val_acc = correct / max(total, 1)
            best_val_acc = max(best_val_acc, val_acc)

            print(f"Epoch {epoch+1}/{args.epochs} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")
            mlflow.log_metrics({"train_loss": train_loss, "val_acc": val_acc}, step=epoch)

        mlflow.log_metric("best_val_acc", best_val_acc)
        mlflow.pytorch.log_model(model, "model")
        torch.save({"model_state": model.state_dict(), "classes": class_names}, args.out)
        print(f"\nSaved classifier to {args.out} | best_val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()
