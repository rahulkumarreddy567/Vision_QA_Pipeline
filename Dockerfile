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
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r appuser && useradd -r -g appuser appuser

COPY --from=build /venv /venv
ENV PATH="/venv/bin:$PATH"

COPY api/ ./api/
COPY models/ ./models/
COPY defect_data.yaml .

# Bundle the nano YOLO weights so the app works in demo/fallback mode
# without any trained weights mounted at runtime.
COPY yolo11n.pt .

# Trained weights are expected to be mounted or copied in at deploy time —
# see docker-compose.yml for local dev, or bake them in for a production image:
# COPY runs/detect/defect_yolo11/weights/best.pt ./runs/detect/defect_yolo11/weights/best.pt
# COPY models/resnet50_defect_classifier.pt ./models/resnet50_defect_classifier.pt

RUN chown -R appuser:appuser /app
USER appuser

# Render.com injects $PORT (default 10000). Fall back to 8000 for local/Docker use.
ENV PORT=8000
EXPOSE $PORT

HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=60s \
    CMD python -c "import urllib.request,os; urllib.request.urlopen('http://localhost:'+os.environ.get('PORT','8000')+'/health')" || exit 1

CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 75
