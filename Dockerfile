# ──────────────────────────────────────────────────────────────────────────────
# Florentix AI — Render/Railway Production Dockerfile
# Lean TFLite-only runtime for plant disease inference
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# System deps for image processing (Pillow) and tflite-runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy and install Python deps first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend application code
COPY backend/ ./backend/

# Copy the TFLite model
COPY model/plant_model.tflite ./model/plant_model.tflite

# Health check for container orchestrators (Render, Railway, etc.)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Expose port
EXPOSE 8000

# Production server: single worker to keep RAM usage low on free tiers
# Use --timeout 120 for slow cold-start inference
CMD ["uvicorn", "backend.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "120", \
     "--limit-concurrency", "5"]
