# ---- stage 1: build the React frontend ----
FROM node:20-slim AS frontend
WORKDIR /fe
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- stage 2: python runtime ----
FROM python:3.11-slim
WORKDIR /app
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1

# Python deps (CPU torch — HF free tier has no GPU)
COPY requirements.txt pyproject.toml ./
COPY src/ ./src/
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt \
 && pip install -e .

# Pre-bake backbone weights so cold start is fast and offline
RUN python -c "import timm; timm.create_model('vit_base_patch32_224', pretrained=True, num_classes=0); \
from transformers import AutoModel, AutoTokenizer; \
AutoModel.from_pretrained('distilbert-base-uncased'); \
AutoTokenizer.from_pretrained('distilbert-base-uncased')"

COPY backend/ ./backend/
COPY --from=frontend /fe/dist ./frontend/dist
COPY deploy/assets/ ./deploy/assets/

ENV VS_CHECKPOINT=/app/deploy/assets/visionsearch.pt \
    VS_GALLERY_DIR=/app/deploy/assets/gallery \
    VS_IMAGES_DIR=/app/deploy/assets/images \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

EXPOSE 7860
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
