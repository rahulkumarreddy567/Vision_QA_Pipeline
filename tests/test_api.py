"""
API tests. The two model-dependent tests are skipped automatically if trained
weights aren't present (e.g. in CI before you've trained anything) so the test
suite still passes and CI/CD stays green on a fresh clone.
"""

import io
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app, _recent_latencies_ms, _recent_defect_counts

client = TestClient(app)

YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "runs/detect/defect_yolo11/weights/best.pt")
RESNET_WEIGHTS = os.environ.get("RESNET_WEIGHTS", "models/resnet50_defect_classifier.pt")
MODELS_AVAILABLE = Path(YOLO_WEIGHTS).exists() and Path(RESNET_WEIGHTS).exists()


@pytest.fixture(autouse=True)
def clear_stats():
    """Reset rolling deques before every test so stats tests are independent."""
    _recent_latencies_ms.clear()
    _recent_defect_counts.clear()
    yield


def _fake_image_bytes():
    img = Image.new("RGB", (224, 224), color=(120, 120, 120))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf


def test_health_endpoint_returns_200():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_predict_rejects_non_image_file():
    response = client.post(
        "/predict",
        files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
    )
    assert response.status_code == 400


def test_predict_works_without_trained_model_files():
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", _fake_image_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert "num_defects" in body
    assert isinstance(body["detections"], list)


def test_stats_returns_zero_before_any_predict():
    # fresh client so deques are empty
    fresh = TestClient(app)
    response = fresh.get("/stats")
    assert response.status_code == 200
    assert response.json()["frames_processed"] == 0


def test_stats_populated_after_predict():
    client.post(
        "/predict",
        files={"file": ("test.jpg", _fake_image_bytes(), "image/jpeg")},
    )
    response = client.get("/stats")
    assert response.status_code == 200
    body = response.json()
    assert body["frames_processed"] >= 1
    assert "avg_latency_ms" in body
    assert "p95_latency_ms" in body
    assert "avg_defects_per_frame" in body


def test_metrics_endpoint_returns_prometheus_text():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "vision_qa_frames_processed" in response.text
    assert "vision_qa_avg_latency_ms" in response.text
    assert "vision_qa_p95_latency_ms" in response.text
    assert "vision_qa_avg_defects_per_frame" in response.text


def test_predict_corrupt_image_returns_400():
    response = client.post(
        "/predict",
        files={"file": ("bad.jpg", io.BytesIO(b"\x00\x01\x02"), "image/jpeg")},
    )
    assert response.status_code == 400


def test_root_redirects_to_docs():
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (301, 302, 307, 308)
    assert "/docs" in response.headers["location"]


def test_live_page_returns_html():
    response = client.get("/live")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


@pytest.mark.skipif(not MODELS_AVAILABLE, reason="trained model weights not present")
def test_predict_returns_valid_schema_on_real_models():
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", _fake_image_bytes(), "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert "num_defects" in body
    assert "detections" in body
    assert isinstance(body["detections"], list)
