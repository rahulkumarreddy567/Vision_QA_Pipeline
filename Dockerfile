# --- build stage: install deps into a venv ---
FROM python:3.11-slim AS build

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m venv /venv \
    && /venv/bin/pip install --no-cache-dir --upgrade pip \
    && /venv/bin/pip install --no-cache-dir -r requirements.txt

# --- runtime stage: copy only the venv + app code, keep image small ---
FROM python:3.11-slim AS runtime

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /venv /venv
ENV PATH="/venv/bin:$PATH"

COPY api/ ./api/
COPY models/ ./models/
COPY defect_data.yaml .

# Trained weights are expected to be mounted or copied in at deploy time —
# see docker-compose.yml for local dev, or bake them in for a production image:
# COPY runs/detect/defect_yolo11/weights/best.pt ./runs/detect/defect_yolo11/weights/best.pt
# COPY models/resnet50_defect_classifier.pt ./models/resnet50_defect_classifier.pt

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
