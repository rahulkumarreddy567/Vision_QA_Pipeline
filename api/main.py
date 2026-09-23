import io
import os
import time
import uuid
import base64
import logging
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from api.schemas import PredictionResponse, HealthResponse, Detection
from api.model_loader import get_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("vision_qa")

# Pre-warm the pipeline on startup so the first request isn't slow
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_pipeline()
        logger.info("Pipeline pre-warmed successfully")
    except Exception as e:
        logger.warning("Pipeline pre-warm failed (will retry on first request): %s", e)
    yield

app = FastAPI(
    title="Vision QA Pipeline",
    description=(
        "Two-stage (YOLO11 + ResNet50) manufacturing defect detection API. "
        "Detects and classifies surface defects in real-time via REST and WebSocket."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))  # 10 MB default

_recent_latencies_ms: deque = deque(maxlen=200)
_recent_defect_counts: deque = deque(maxlen=200)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def _validate_image_file(filename: str, content_type: str):
    if not content_type or not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    ext = os.path.splitext(filename or "")[1].lower()
    if ext and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported image extension: {ext}")


def _decode_image(data: bytes) -> Image.Image:
    try:
        with io.BytesIO(data) as buf:
            return Image.open(buf).convert("RGB")
    except (UnidentifiedImageError, Exception):
        raise HTTPException(status_code=400, detail="Could not decode image")


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        pipeline = get_pipeline()
        status = "ok" if not pipeline.fallback_mode else "demo mode"
        return HealthResponse(
            status=status,
            yolo_loaded=bool(pipeline.detector),
            resnet_loaded=bool(pipeline.classifier),
            mode="demo" if pipeline.fallback_mode else "production",
        )
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return HealthResponse(status="models not loaded", yolo_loaded=False, resnet_loaded=False, mode="error")


@app.get("/stats")
def stats():
    if not _recent_latencies_ms:
        return {"frames_processed": 0}
    sorted_lat = sorted(_recent_latencies_ms)
    n = len(sorted_lat)
    return {
        "frames_processed": n,
        "avg_latency_ms": round(sum(sorted_lat) / n, 2),
        "p95_latency_ms": round(sorted_lat[max(int(n * 0.95) - 1, 0)], 2),
        "avg_defects_per_frame": round(sum(_recent_defect_counts) / len(_recent_defect_counts), 3),
    }


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics():
    n = len(_recent_latencies_ms)
    if n == 0:
        avg_lat, p95_lat, avg_defects = 0.0, 0.0, 0.0
    else:
        sorted_lat = sorted(_recent_latencies_ms)
        avg_lat = sum(_recent_latencies_ms) / n
        p95_lat = sorted_lat[max(int(n * 0.95) - 1, 0)]
        avg_defects = sum(_recent_defect_counts) / n
    lines = [
        "# HELP vision_qa_frames_processed Total frames processed (rolling window)",
        "# TYPE vision_qa_frames_processed gauge",
        f"vision_qa_frames_processed {n}",
        "# HELP vision_qa_avg_latency_ms Average inference latency in milliseconds",
        "# TYPE vision_qa_avg_latency_ms gauge",
        f"vision_qa_avg_latency_ms {avg_lat:.2f}",
        "# HELP vision_qa_p95_latency_ms P95 inference latency in milliseconds",
        "# TYPE vision_qa_p95_latency_ms gauge",
        f"vision_qa_p95_latency_ms {p95_lat:.2f}",
        "# HELP vision_qa_avg_defects_per_frame Average defects detected per frame",
        "# TYPE vision_qa_avg_defects_per_frame gauge",
        f"vision_qa_avg_defects_per_frame {avg_defects:.4f}",
    ]
    return "\n".join(lines) + "\n"


@app.get("/live")
def live_demo_page():
    return FileResponse(os.path.join(STATIC_DIR, "live.html"))


def _run_inference_on_image(image: Image.Image):
    pipeline = get_pipeline()
    start = time.perf_counter()
    detections = pipeline.predict(image)
    elapsed_ms = (time.perf_counter() - start) * 1000
    _recent_latencies_ms.append(elapsed_ms)
    _recent_defect_counts.append(len(detections))
    logger.info("inference completed: %d detections in %.1f ms", len(detections), elapsed_ms)
    return detections, elapsed_ms


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())[:8]
    _validate_image_file(file.filename or "", file.content_type or "")

    contents = await file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024*1024)} MB)")

    image = _decode_image(contents)
    detections, elapsed_ms = _run_inference_on_image(image)

    safe_filename = (file.filename or "unknown").replace("\n", "").replace("\r", "")
    logger.info("[%s] predict: file=%s defects=%d", request_id, safe_filename, len(detections))

    return PredictionResponse(
        filename=file.filename,
        num_defects=len(detections),
        inference_time_ms=round(elapsed_ms, 2),
        detections=[Detection(**d) for d in detections],
        request_id=request_id,
    )


@app.websocket("/ws/stream")
async def stream_predictions(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected: %s", websocket.client)
    try:
        while True:
            data = await websocket.receive_text()

            # strip data URL prefix if present
            if "," in data:
                data = data.split(",", 1)[1]

            try:
                frame_bytes = base64.b64decode(data)
            except Exception:
                await websocket.send_json({"error": "Invalid base64 data"})
                continue

            try:
                with io.BytesIO(frame_bytes) as buf:
                    image = Image.open(buf).convert("RGB")
                    image.load()  # force decode while buf is still open
            except Exception:
                await websocket.send_json({"error": "Could not decode frame"})
                continue

            try:
                detections, elapsed_ms = _run_inference_on_image(image)
            except Exception as e:
                logger.error("Inference error on WebSocket frame: %s", e)
                await websocket.send_json({"error": "Inference failed"})
                continue

            await websocket.send_json({
                "num_defects": len(detections),
                "inference_time_ms": round(elapsed_ms, 2),
                "detections": detections,
            })
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected: %s", websocket.client)
    except Exception as e:
        logger.error("Unexpected WebSocket error: %s", e)
