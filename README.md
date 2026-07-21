# Vision QA Pipeline — Manufacturing Defect Detection

A two-stage computer vision system for industrial quality control: **YOLO11** localizes
defects in an image, then a fine-tuned **ResNet50** classifies the defect type/severity
on the cropped region. Served via **FastAPI**, tracked with **MLflow**, containerized
with **Docker**, and deployed through a **GitHub Actions** CI/CD pipeline to AWS/Azure.

## Architecture

```
Image → YOLO11 (detect defect regions) → crop → ResNet50 (classify defect type)
                                                        │
                                                        ▼
                                          FastAPI /predict endpoint
                                                        │
                                    MLflow (experiment tracking + model registry)
                                                        │
                                Docker → GitHub Actions → AWS ECS / Azure Container Apps
```

## Repo structure

```
vision-qa-pipeline/
├── data/
│   └── prepare_data.py       # dataset download + YOLO-format conversion
├── models/
│   ├── train_yolo.py         # Stage 1: defect localization
│   ├── train_resnet.py       # Stage 2: defect classification
│   ├── inference.py          # combined two-stage inference pipeline (single image)
│   └── realtime_inference.py # live webcam/video/RTSP inference with overlay + FPS
├── api/
│   ├── main.py                # FastAPI app (/predict, /ws/stream, /stats, /live)
│   ├── schemas.py             # Pydantic request/response models
│   ├── model_loader.py        # loads models once at startup
│   └── static/live.html       # browser demo: webcam -> WebSocket -> live overlay
├── tests/
│   └── test_api.py
├── .github/workflows/
│   └── ci-cd.yml
├── defect_data.yaml            # YOLO dataset config
├── Dockerfile
├── docker-compose.yml          # api + mlflow server
├── requirements.txt
└── README.md
```

## Getting started

### 1. Set up environment
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Get a dataset
Pick one (see `data/prepare_data.py` for download helpers):
- **NEU-DET** (steel surface defects, 6 classes) — recommended starting point
- **Casting Product Defect Dataset** (Kaggle) — submersible pump impeller casting
- **PCB Defect Dataset** (Roboflow) — electronics

```bash
python data/prepare_data.py --dataset neu-det --output data/raw
```

### 3. Train Stage 1 — YOLO11 defect detector
```bash
python models/train_yolo.py --data defect_data.yaml --epochs 100 --imgsz 640
```

### 4. Train Stage 2 — ResNet50 defect classifier
```bash
python models/train_resnet.py --crops-dir data/raw/crops --epochs 30
```

### 5. Track experiments with MLflow
```bash
mlflow ui --backend-store-uri ./mlruns
# open http://localhost:5000
```

### 6. Run the API locally
```bash
uvicorn api.main:app --reload --port 8000
# POST an image to http://localhost:8000/predict
```

### 6b. Real-time inference — two options

**Option A: local webcam/video window (OpenCV)**
```bash
pip install opencv-python   # swap out opencv-python-headless first, they conflict
python models/realtime_inference.py --source 0                     # webcam
python models/realtime_inference.py --source path/to/line_video.mp4
python models/realtime_inference.py --source rtsp://camera-ip/stream --save-out out.mp4
```
Shows a live window with bounding boxes, defect type, and an FPS counter overlaid.
Use `--frame-skip N` to run inference every N frames if your GPU can't keep up with
the camera's native frame rate.

**Option B: browser-based live demo (WebSocket)**
```bash
uvicorn api.main:app --reload --port 8000
# open http://localhost:8000/live in a browser
```
This streams your webcam through a WebSocket (`/ws/stream`) to the API and draws
detections back over the live video feed — the more "demo-able" option since anyone
can open the link, no local Python/OpenCV setup needed on the client side.

Check `GET /stats` for rolling average latency and defects-per-frame — useful for
showing a recruiter real throughput numbers, not just a static accuracy metric.

### 7. Run with Docker
```bash
docker compose up --build
```

### 8. CI/CD
Push to `main` — GitHub Actions runs tests, builds the Docker image, and (once you
add your cloud credentials as repo secrets) deploys to AWS ECS or Azure Container Apps.
See `.github/workflows/ci-cd.yml`.

## Results (fill in after training)

| Metric | Value |
|---|---|
| YOLO11 mAP@0.5 | TBD |
| YOLO11 mAP@0.5:0.95 | TBD |
| ResNet50 classification accuracy | TBD |
| Inference latency (p50) | TBD ms |
| Inference latency (p95) | TBD ms |

## Resume bullet (fill in real numbers once trained)

> Built a two-stage computer vision pipeline (YOLO11 for defect localization, fine-tuned
> ResNet50 for defect classification) achieving **XX% mAP@0.5** on an industrial defect
> dataset. Served real-time predictions via a FastAPI/WebSocket endpoint processing
> live video at **XX FPS** with **sub-XXXms** latency; tracked experiments with MLflow,
> containerized with Docker, and deployed through a GitHub Actions CI/CD pipeline to AWS ECS.

## Notes on model choice

Ultralytics' newest release, **YOLO26** (early 2026), is NMS-free and lower-latency —
worth evaluating as a drop-in replacement for YOLO11 if you want to mention you compared
both in your README (`model = YOLO("yolo26n.pt")` — same API, no code changes needed).
