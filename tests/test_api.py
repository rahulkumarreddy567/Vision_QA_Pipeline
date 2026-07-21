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

from api.main import app

client = TestClient(app)

YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "runs/detect/defect_yolo11/weights/best.pt")
RESNET_WEIGHTS = os.environ.get("RESNET_WEIGHTS", "models/resnet50_defect_classifier.pt")
MODELS_AVAILABLE = Path(YOLO_WEIGHTS).exists() and Path(RESNET_WEIGHTS).exists()


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
