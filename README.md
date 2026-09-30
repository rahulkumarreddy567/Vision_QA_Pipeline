# Vision QA Pipeline — Manufacturing Defect Detection

[![CI/CD](https://github.com/rahulkumarreddy567/Vision_QA_Pipeline/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/rahulkumarreddy567/Vision_QA_Pipeline/actions)
[![Live Demo](https://img.shields.io/badge/demo-render.com-46E3B7?logo=render)](https://vision-qa-pipeline.onrender.com)

A production-grade two-stage computer vision system for industrial quality control.
**YOLO11** localizes defect regions; a fine-tuned **ResNet50** classifies each crop.
Served via **FastAPI**, monitored with **Prometheus**, containerized with **Docker**,
and deployed for free on **Render.com** via a **GitHub Actions CI/CD** pipeline.

## Live Demo

🌐 **[https://vision-qa-pipeline.onrender.com](https://vision-qa-pipeline.onrender.com)**

> Free tier spins down after 15 min of inactivity — first request may take ~30s to wake up.

| Route | Description |
|---|---|
| `GET /` | Landing page with drag-drop defect detection demo |
| `GET /live` | Upload image or stream webcam with live bounding-box overlay |
| `GET /docs` | Interactive Swagger / OpenAPI 3.1 docs |
| `POST /predict` | Upload an image → JSON detections |
| `GET /health` | Service health + model status |
| `GET /stats` | Rolling avg latency + defect rate |
| `GET /metrics` | Prometheus scrape endpoint |
| `WS /ws/stream` | WebSocket real-time frame stream |

## Architecture

```
Image / Webcam / RTSP stream
        │
        ▼
  YOLO11 (defect localization)
        │  bounding-box crops
        ▼
  ResNet50 (defect classification — type + confidence)
        │
        ▼
  FastAPI ──► POST /predict        (REST)
          ──► WS   /ws/stream      (WebSocket real-time)
          ──► GET  /metrics        (Prometheus)
          ──► GET  /stats          (JSON rolling stats)
          ──► GET  /               (Landing page)
          ──► GET  /live           (Browser demo)
        │
  Docker → GitHub Actions CI/CD → Render.com / AWS ECS
```

## Repo structure

```
vision-qa-pipeline/
├── api/
│   ├── main.py                # FastAPI app — all routes
│   ├── schemas.py             # Pydantic request/response models
│   ├── model_loader.py        # singleton pipeline loader (lru_cache)
│   └── static/
│       ├── index.html         # landing page with drag-drop demo
│       └── live.html          # upload + webcam live demo
├── models/
│   ├── inference.py           # two-stage inference pipeline
│   ├── realtime_inference.py  # local webcam/video/RTSP with OpenCV
│   ├── train_yolo.py          # Stage 1: YOLO11 defect detector
│   └── train_resnet.py        # Stage 2: ResNet50 classifier
├── data/
│   └── prepare_data.py        # dataset download + YOLO-format conversion
├── scripts/
│   └── smoke_predict.py       # quick smoke test (local or Render)
├── tests/
│   └── test_api.py
├── .github/workflows/
│   └── ci-cd.yml              # test → build → push → deploy → smoke test
├── defect_data.yaml           # YOLO dataset config (NEU-DET, 6 classes)
├── Dockerfile                 # multi-stage build, python:3.11-slim
├── docker-compose.yml         # api + mlflow server
├── render.yaml                # Render.com blueprint
└── requirements.txt
```

## Quickstart

### 1. Clone & install
```bash
git clone https://github.com/rahulkumarreddy567/Vision_QA_Pipeline.git
cd Vision_QA_Pipeline/vision-qa-pipeline
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the API (demo mode — no trained weights needed)
```bash
python -m uvicorn api.main:app --reload --port 8000
```
Open [http://localhost:8000](http://localhost:8000) — drag-drop any image to test.

### 3. Smoke test
```bash
# local
python scripts/smoke_predict.py

# against Render
python scripts/smoke_predict.py --url https://vision-qa-pipeline.onrender.com
```

### 4. Run tests
```bash
pytest tests/ -v
```

### 5. Run with Docker
```bash
docker compose up --build
# API:    http://localhost:8000
# MLflow: http://localhost:5000
```

## Training (optional — needed for production mode)

### Get a dataset
```bash
python data/prepare_data.py --dataset neu-det --output data/raw
```
Supported: `neu-det` (steel, 6 classes), `casting`, `pcb`.

### Train Stage 1 — YOLO11 detector
```bash
python models/train_yolo.py --data defect_data.yaml --epochs 100 --imgsz 640
```

### Train Stage 2 — ResNet50 classifier
```bash
python models/train_resnet.py --crops-dir data/raw/crops --epochs 30
```

### Set weights in environment
```bash
export YOLO_WEIGHTS=runs/detect/defect_yolo11/weights/best.pt
export RESNET_WEIGHTS=models/resnet50_defect_classifier.pt
python -m uvicorn api.main:app --port 8000
```

## Real-time inference (local webcam)
```bash
pip install opencv-python   # swap out opencv-python-headless
python models/realtime_inference.py --source 0                        # webcam
python models/realtime_inference.py --source path/to/video.mp4
python models/realtime_inference.py --source rtsp://camera-ip/stream
```

## Prometheus + Grafana monitoring
Add to `prometheus.yml`:
```yaml
scrape_configs:
  - job_name: vision-qa
    static_configs:
      - targets: ['localhost:8000']
```
Metrics exposed: `vision_qa_frames_processed`, `vision_qa_avg_latency_ms`,
`vision_qa_p95_latency_ms`, `vision_qa_avg_defects_per_frame`.

## CI/CD — GitHub Actions
Push to `main`:
1. Runs `pytest tests/`
2. Builds + pushes Docker image to Docker Hub (if secrets set)
3. Force-deploys to AWS ECS `vision-qa-cluster` in `eu-west-3` (if secrets set)
4. Smoke-tests the live Render URL

Required secrets: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`.

## Results

| Metric | Value |
|---|---|
| YOLO11 mAP@0.5 | TBD after training |
| ResNet50 accuracy | TBD after training |
| Inference latency p95 | TBD ms |
| Throughput (GPU) | TBD FPS |

## Key technologies
Python · PyTorch · YOLO11 · ResNet50 · FastAPI · WebSocket · Docker ·
GitHub Actions · AWS ECS · Render.com · Prometheus · MLflow · ONNX ·
OpenCV · REST API · CI/CD · Computer Vision · Deep Learning · Industry 4.0
