# Vision QA Pipeline — Manufacturing Defect Detection

[![CI/CD](https://github.com/YOUR_USERNAME/vision-qa-pipeline/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/YOUR_USERNAME/vision-qa-pipeline/actions)

A production-grade, two-stage computer vision system for industrial quality control
aligned with **ISO 9001 / Industry 4.0** manufacturing standards.
**YOLO11** localizes defect regions; a fine-tuned **ResNet50** classifies defect type/severity
on each crop. Served via **FastAPI**, monitored with **Prometheus + Grafana**,
experiment-tracked with **MLflow**, containerized with **Docker**, and deployed for free via
**Render.com** (or optionally to **AWS ECS eu-west-3**) through a **GitHub Actions CI/CD pipeline**.

## Architecture

```
Image / RTSP stream
    │
    ▼
YOLO11 (defect localization)
    │  bounding-box crops
    ▼
ResNet50 (defect classification — type + severity)
    │
    ▼
FastAPI  ──►  /predict (REST)
         ──►  /ws/stream (WebSocket, real-time)
         ──►  /metrics  (Prometheus scrape endpoint)
         ──►  /stats    (JSON rolling stats)
    │
MLflow (experiment tracking + model registry)
    │
Docker → GitHub Actions CI/CD → AWS ECS (eu-west-3, Paris)
```

## Repo structure

```
vision-qa-pipeline/
├── data/
│   └── prepare_data.py        # dataset download + YOLO-format conversion
├── models/
│   ├── train_yolo.py          # Stage 1: defect localization (YOLO11)
│   ├── train_resnet.py        # Stage 2: defect classification (ResNet50)
│   ├── inference.py           # two-stage inference pipeline (single image)
│   └── realtime_inference.py  # live webcam/video/RTSP inference + FPS overlay
├── api/
│   ├── main.py                # FastAPI app — /predict, /ws/stream, /metrics, /stats, /live
│   ├── schemas.py             # Pydantic request/response models
│   ├── model_loader.py        # singleton model loader (lru_cache)
│   └── static/live.html       # browser demo: webcam → WebSocket → live overlay
├── tests/
│   └── test_api.py
├── .github/workflows/
│   └── ci-cd.yml              # test → build → push → deploy to AWS ECS eu-west-3
├── defect_data.yaml            # YOLO dataset config (NEU-DET, 6 classes)
├── Dockerfile                  # multi-stage build, python:3.11-slim
├── docker-compose.yml          # api + mlflow server
├── requirements.txt
└── README.md
```

## Free Cloud Deployment (Render.com)

Render.com gives you a **free public HTTPS URL** with zero credit card required.

1. Push this repo to GitHub
2. Go to [dashboard.render.com](https://dashboard.render.com) → **New** → **Blueprint**
3. Connect your GitHub repo — Render reads `render.yaml` and deploys automatically
4. Your live URL: `https://vision-qa-pipeline.onrender.com`

> Note: Free tier spins down after 15 min of inactivity (cold start ~30s). Upgrade to Starter ($7/mo) for always-on.

Endpoints on your live URL:
- `GET  /health` — service health + model status
- `POST /predict` — upload an image, get JSON detections
- `GET  /docs` — interactive Swagger UI
- `GET  /live` — browser webcam demo
- `GET  /metrics` — Prometheus scrape endpoint
- `GET  /stats` — rolling latency + defect rate stats

### 1. Set up environment
```bash
python -m venv venv && source venv/bin/activate   # Linux/macOS
python -m venv venv && venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### 2. Get a dataset
Pick one (see `data/prepare_data.py` for download helpers):
- **NEU-DET** — steel surface defects, 6 classes, ~1800 images (recommended)
- **Casting Product Defect Dataset** (Kaggle) — submersible pump impeller casting
- **PCB Defect Dataset** (Roboflow) — electronics manufacturing

```bash
python data/prepare_data.py --dataset neu-det --output data/raw
```

### 3. Train Stage 1 — YOLO11 defect detector
```bash
export MLFLOW_TRACKING_URI=./mlruns
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
# REST:       POST http://localhost:8000/predict
# Prometheus: GET  http://localhost:8000/metrics
# Live demo:  GET  http://localhost:8000/live
```

## Local smoke test

After starting the API, you can quickly verify the service and do a sample prediction.

- Health check:

```bash
curl -sS http://127.0.0.1:8000/health | jq
```

- Sample image POST (one-liner using Python to generate a dummy JPEG and POST with httpx):

```bash
python - <<'PY'
import io, httpx
from PIL import Image
img = Image.new('RGB', (224,224), color=(120,120,120))
buf = io.BytesIO(); img.save(buf, format='JPEG'); buf.seek(0)
files = {'file': ('test.jpg', buf.getvalue(), 'image/jpeg')}
res = httpx.post('http://127.0.0.1:8000/predict', files=files)
print(res.status_code)
print(res.json())
PY
```

Notes:
- On Windows PowerShell, activate the venv with `venv\Scripts\Activate.ps1` or run the python executable directly from the `.venv` folder.
- If you don't have trained weights present, the API will run in `demo mode` and return a fallback detection.

## Committing local fixes

I made small runtime fixes to support running the API and tests without installing heavy ML packages during CI:

- `models/inference.py`: lazy-imports for `torch`, `torchvision`, and `ultralytics` so the API and tests can run in lightweight environments.
- `api/main.py`: static files served using package-relative paths so tests don't fail when run from a different CWD.

To commit these local changes:

```bash
git add -A
git commit -m "fix: lazy-load ML deps; make static path package-relative; update README"
```

### 6b. Real-time inference — two options

**Option A: local webcam/video window (OpenCV)**
```bash
pip install opencv-python   # swap out opencv-python-headless first
python models/realtime_inference.py --source 0                          # webcam
python models/realtime_inference.py --source path/to/line_video.mp4
python models/realtime_inference.py --source rtsp://camera-ip/stream --save-out out.mp4
```
Use `--frame-skip N` to run inference every N frames on weaker hardware.

**Option B: browser-based live demo (WebSocket)**
```bash
uvicorn api.main:app --reload --port 8000
# open http://localhost:8000/live
```
Streams webcam through `/ws/stream`, draws detections back over the live feed —
no client-side Python/OpenCV needed.

### 7. Prometheus + Grafana monitoring
The `/metrics` endpoint exposes Prometheus-compatible gauges:
- `vision_qa_frames_processed`
- `vision_qa_avg_latency_ms`
- `vision_qa_p95_latency_ms`
- `vision_qa_avg_defects_per_frame`

Add a scrape job to your `prometheus.yml`:
```yaml
scrape_configs:
  - job_name: vision-qa
    static_configs:
      - targets: ['localhost:8000']
```
Then import a Grafana dashboard pointing at that Prometheus datasource.

### 8. Run with Docker
```bash
docker compose up --build
# API:    http://localhost:8000
# MLflow: http://localhost:5000
```

### 9. CI/CD — GitHub Actions → AWS ECS (eu-west-3, Paris)
Push to `main`:
1. Runs `pytest tests/`
2. Builds + pushes Docker image to Docker Hub
3. Triggers a force-new-deployment on `vision-qa-cluster / vision-qa-api` in **eu-west-3**

Required repo secrets: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`.

### 10. ONNX export (edge deployment)
Export the trained YOLO11 model to ONNX for deployment on edge hardware (Jetson, Raspberry Pi CM4):
```bash
yolo export model=runs/detect/defect_yolo11/weights/best.pt format=onnx imgsz=640
```

## Results

| Metric | Value |
|---|---|
| YOLO11 mAP@0.5 | TBD |
| YOLO11 mAP@0.5:0.95 | TBD |
| ResNet50 classification accuracy | TBD |
| Inference latency p50 | TBD ms |
| Inference latency p95 | TBD ms |
| Throughput (GPU) | TBD FPS |

> Fill in after training. These numbers are what recruiters and ATS systems look for —
> even a single concrete metric (e.g. "92% mAP@0.5") dramatically increases callback rate.

## Resume bullet (fill in real numbers)

> Engineered a two-stage computer vision quality-inspection pipeline (YOLO11 + ResNet50)
> for industrial defect detection, achieving **XX% mAP@0.5** on the NEU-DET steel surface
> dataset. Deployed a FastAPI/WebSocket inference service processing live RTSP streams at
> **XX FPS** with **p95 latency < XXX ms**; integrated Prometheus metrics scraping and
> Grafana dashboards for real-time OEE monitoring; containerized with Docker and shipped
> via GitHub Actions CI/CD to **AWS ECS (eu-west-3)**. Experiment tracking via MLflow;
> ONNX export for edge deployment on factory-floor hardware.

## Key technologies (ATS keywords)

Python · PyTorch · YOLO11 · ResNet50 · FastAPI · WebSocket · Docker · GitHub Actions ·
AWS ECS · MLflow · Prometheus · Grafana · ONNX · OpenCV · REST API · CI/CD ·
Computer Vision · Deep Learning · Transfer Learning · Real-Time Inference ·
Manufacturing Quality Control · ISO 9001 · Industry 4.0 · OEE · Edge Deployment

## Notes on model variants

- **YOLO11n** — fastest, good for edge/Jetson deployment
- **YOLO11s** — balanced speed/accuracy (default)
- **YOLO11m** — highest accuracy, needs a proper GPU
- Ultralytics' **YOLO12** (2025) is NMS-free and lower-latency — worth benchmarking as a drop-in (`model = YOLO("yolo12n.pt")`, same API)
