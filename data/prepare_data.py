"""
Dataset download + preparation helpers for the Vision QA Pipeline.

Usage:
    python data/prepare_data.py --dataset neu-det --output data/raw
    python data/prepare_data.py --dataset casting --output data/raw

This script does NOT ship the datasets (they belong to their original authors/hosts).
It documents where to get each one and converts annotations into YOLO format
(class x_center y_center width height, normalized 0-1) under <output>/images and
<output>/labels, split into train/val/test.
"""

import argparse
import shutil
import random
from pathlib import Path

DATASETS = {
    "neu-det": {
        "description": "NEU-DET steel surface defect dataset (6 classes, ~1800 images).",
        "source_hint": (
            "Download from Kaggle: 'NEU surface defect database' or the original "
            "Northeastern University release. Place raw images + XML (VOC-format) "
            "annotations under data/raw_src/neu-det/ before running this script."
        ),
        "classes": ["crazing", "inclusion", "patches", "pitted_surface",
                    "rolled-in_scale", "scratches"],
    },
    "casting": {
        "description": "Casting product defect dataset (submersible pump impeller).",
        "source_hint": (
            "Download from Kaggle: 'Real-life Industrial Dataset of Casting Product'. "
            "This one ships already split into defective/ok folders — treat it as a "
            "classification task for ResNet50, or manually box the defect region for "
            "YOLO if you want the two-stage pipeline."
        ),
        "classes": ["defective", "ok"],
    },
    "pcb": {
        "description": "PCB defect dataset (electronics manufacturing).",
        "source_hint": (
            "Download from Roboflow Universe: search 'PCB Defect Detection'. "
            "Roboflow lets you export directly in YOLO format — skip the VOC "
            "conversion step below if you use that export."
        ),
        "classes": ["missing_hole", "mouse_bite", "open_circuit",
                    "short", "spur", "spurious_copper"],
    },
}


def voc_to_yolo(xml_path, img_w, img_h, class_list):
    """Convert a single Pascal-VOC XML annotation to YOLO-format label lines."""
    import xml.etree.ElementTree as ET

    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text.strip()
        if cls_name not in class_list:
            continue
        cls_id = class_list.index(cls_name)
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)

        x_center = (xmin + xmax) / 2 / img_w
        y_center = (ymin + ymax) / 2 / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h
        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    return lines


def split_dataset(image_files, train_ratio=0.7, val_ratio=0.2, seed=42):
    random.seed(seed)
    shuffled = image_files.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS.keys(), required=True)
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--src", default=None,
                         help="Path to raw downloaded dataset (images + VOC XML). "
                              "Defaults to data/raw_src/<dataset>/")
    args = parser.parse_args()

    info = DATASETS[args.dataset]
    print(f"Dataset: {args.dataset}")
    print(f"  {info['description']}")
    print(f"  Source: {info['source_hint']}\n")

    src_dir = Path(args.src or f"data/raw_src/{args.dataset}")
    if not src_dir.exists():
        print(f"[!] Expected raw data at {src_dir} — download it first (see source hint above).")
        print("    This script only handles conversion + splitting once the raw files exist.")
        return

    out_dir = Path(args.output)
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    image_files = sorted(list(src_dir.glob("**/*.jpg")) + list(src_dir.glob("**/*.png")))
    if not image_files:
        print(f"[!] No images found under {src_dir}")
        return

    splits = split_dataset(image_files)
    from PIL import Image

    for split_name, files in splits.items():
        for img_path in files:
            xml_path = img_path.with_suffix(".xml")
            shutil.copy(img_path, out_dir / "images" / split_name / img_path.name)

            if xml_path.exists():
                with Image.open(img_path) as im:
                    w, h = im.size
                lines = voc_to_yolo(xml_path, w, h, info["classes"])
                label_path = out_dir / "labels" / split_name / (img_path.stem + ".txt")
                label_path.write_text("\n".join(lines))

    print(f"Done. Prepared {len(image_files)} images into {out_dir}")
    print("Update defect_data.yaml 'names' to match this dataset's classes if needed.")


if __name__ == "__main__":
    main()
