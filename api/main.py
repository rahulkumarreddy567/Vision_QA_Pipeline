import io
import time
import base64
from collections import deque

from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from api.schemas import PredictionResponse, HealthResponse, Detection
from api.model_loader import get_pipeline

app = FastAPI(
    title="Vision QA Pipeline",
    description="Two-stage (YOLO11 + ResNet50) manufacturing defect detection API",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory="api/static"), name="static")

# rolling window of recent inference latencies + defect counts, for a live /stats view
_recent_latencies_ms = deque(maxlen=200)
_recent_defect_counts = deque(maxlen=200)


@app.get("/health", response_model=HealthResponse)
def health():
    try:
        pipeline = get_pipeline()
        status = "ok" if not pipeline.fallback_mode else "demo mode"
        return HealthResponse(
            status=status,
            yolo_loaded=bool(pipeline.detector),
            resnet_loaded=bool(pipeline.classifier),
        )
    except Exception:
        return HealthResponse(status="models not loaded", yolo_loaded=False, resnet_loaded=False)


@app.get("/stats")
def stats():
    """Rolling real-time monitoring stats — avg latency + defect rate over the last N frames."""
    if not _recent_latencies_ms:
        return {"frames_processed": 0}
    return {
        "frames_processed": len(_recent_latencies_ms),
        "avg_latency_ms": round(sum(_recent_latencies_ms) / len(_recent_latencies_ms), 2),
        "p95_latency_ms": round(sorted(_recent_latencies_ms)[int(len(_recent_latencies_ms) * 0.95) - 1], 2),
        "avg_defects_per_frame": round(sum(_recent_defect_counts) / len(_recent_defect_counts), 3),
    }


@app.get("/live")
def live_demo_page():
    """Serves the browser demo page (webcam -> WebSocket -> live overlay)."""
    return FileResponse("api/static/live.html")


def _run_inference_on_image(image: Image.Image):
    pipeline = get_pipeline()
    start = time.perf_counter()
    tmp_path = "/tmp/_qa_frame.jpg"
    image.save(tmp_path)
    detections = pipeline.predict(tmp_path)
    elapsed_ms = (time.perf_counter() - start) * 1000

    _recent_latencies_ms.append(elapsed_ms)
    _recent_defect_counts.append(len(detections))
    return detections, elapsed_ms


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image")

    detections, elapsed_ms = _run_inference_on_image(image)

    return PredictionResponse(
        filename=file.filename,
        num_defects=len(detections),
        inference_time_ms=round(elapsed_ms, 2),
        detections=[Detection(**d) for d in detections],
    )


@app.websocket("/ws/stream")
async def stream_predictions(websocket: WebSocket):
    """
    Real-time streaming endpoint. Client sends base64-encoded JPEG frames (e.g. from
    a browser's getUserMedia + canvas, or an RTSP-consuming edge client) and receives
    JSON detections back for each frame, enabling live overlay rendering client-side.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # expects a data URL or raw base64 string
            if "," in data:
                data = data.split(",", 1)[1]
            frame_bytes = base64.b64decode(data)
            image = Image.open(io.BytesIO(frame_bytes)).convert("RGB")

            detections, elapsed_ms = _run_inference_on_image(image)

            await websocket.send_json({
                "num_defects": len(detections),
                "inference_time_ms": round(elapsed_ms, 2),
                "detections": detections,
            })
    except WebSocketDisconnect:
        pass
