# VisionSearch Week 8 — Deploy & Document Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Package VisionSearch as a single-container app (FastAPI serving the built React frontend + search API), smoke-test the real Docker image locally, and finalize recruiter-ready docs + the Hugging Face deploy instructions.

**Architecture:** A multi-stage Dockerfile builds the frontend (`node`) then runs everything from one Python image: FastAPI serves `/search`, `/images`, and the static `dist/` on `$PORT` (HF default 7860). Model weights are pre-baked into the image so cold start is offline/fast. Deploy assets (trained checkpoint + a lean ~500-image gallery + FAISS index) live under `deploy/assets/` (gitignored on GitHub; pushed to the HF Space via LFS).

**Tech Stack:** Docker (multi-stage), FastAPI/uvicorn, Node build, Hugging Face Spaces (Docker SDK).

## Global Constraints

- One container, one process: `uvicorn backend.main:app` on `$PORT` (default **7860**, HF's default).
- **CPU torch** in the image (HF free tier has no GPU): install from the cpu wheel index.
- **Pre-bake** ViT-B/32 + DistilBERT weights at build time; set `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` at runtime.
- Asset paths come from env (`VS_CHECKPOINT`, `VS_GALLERY_DIR`, `VS_IMAGES_DIR`) so the container points at baked assets.
- Demo gallery = **500** test images (~70 MB) to keep the image lean.
- `deploy/assets/` is **gitignored** (size + Flickr licensing); shipped to HF via LFS, present locally for the Docker build.
- Don't break the 32 passing tests. The actual HF push is the **user's** step (their account/token).

---

### Task 1: Env-configurable asset paths

**Files:**
- Modify: `src/visionsearch/config.py` (add `gallery_dir`, `images_dir`, `checkpoint_path` from env)
- Modify: `backend/search_service.py` (use `CONFIG.gallery_dir`, `CONFIG.checkpoint_path`)
- Modify: `backend/main.py` (use `CONFIG.images_dir`)
- Modify: `scripts/build_index.py` (use `CONFIG.checkpoint_path`)
- Modify: `tests/test_config.py` (assert the new fields exist)

**Interfaces:**
- `CONFIG.gallery_dir: Path`, `CONFIG.images_dir: Path`, `CONFIG.checkpoint_path: Path` — each overridable by the matching `VS_*` env var.

- [ ] **Step 1: Extend `tests/test_config.py`** (add a failing assertion)

```python
def test_config_has_deploy_paths():
    from pathlib import Path
    assert isinstance(CONFIG.gallery_dir, Path)
    assert isinstance(CONFIG.images_dir, Path)
    assert isinstance(CONFIG.checkpoint_path, Path)
```

- [ ] **Step 2: Run, expect failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_config.py -q`
Expected: AttributeError on `CONFIG.gallery_dir`.

- [ ] **Step 3: Update `src/visionsearch/config.py`** — add `import os` and the fields

```python
"""Single source of truth for paths, hyperparameters, and device."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _detect_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _env_path(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value) if value else default


@dataclass(frozen=True)
class Config:
    # Hyperparameters
    embed_dim: int = 256          # shared embedding space dimension
    image_size: int = 224         # keep at 224 — memory grows quadratically
    # Paths
    data_dir: Path = field(default_factory=lambda: _ROOT / "data")
    checkpoint_dir: Path = field(default_factory=lambda: _ROOT / "checkpoints")
    # Deploy-overridable asset paths (env wins; defaults point at local training output)
    checkpoint_path: Path = field(
        default_factory=lambda: _env_path("VS_CHECKPOINT", _ROOT / "checkpoints" / "visionsearch.pt"))
    gallery_dir: Path = field(
        default_factory=lambda: _env_path("VS_GALLERY_DIR", _ROOT / "data" / "flickr30k" / "gallery"))
    images_dir: Path = field(
        default_factory=lambda: _env_path("VS_IMAGES_DIR", _ROOT / "data" / "flickr30k" / "images"))
    # Runtime
    device: str = field(default_factory=_detect_device)


CONFIG = Config()
```

- [ ] **Step 4: Point the backend + index builder at the config paths**

In `backend/search_service.py`: replace `GALLERY_DIR = CONFIG.data_dir / "flickr30k" / "gallery"` usage and the checkpoint load:
```python
        ckpt = torch.load(CONFIG.checkpoint_path, map_location=self.device)
        ...
        self.index = ImageIndex.load(CONFIG.gallery_dir)
```
(remove the module-level `GALLERY_DIR`).

In `backend/main.py`: replace `IMAGES = CONFIG.data_dir / "flickr30k" / "images"` with `IMAGES = CONFIG.images_dir`.

In `scripts/build_index.py`: replace `torch.load(CONFIG.checkpoint_dir / "visionsearch.pt", ...)` with `torch.load(CONFIG.checkpoint_path, ...)`.

- [ ] **Step 5: Run full test suite, expect all green**

Run: `.venv/Scripts/python.exe -m pytest -q`
Expected: 33 passed (32 + the new config test). Backend tests still pass via default paths.

- [ ] **Step 6: Commit + push**

```bash
git add src/visionsearch/config.py backend/search_service.py backend/main.py scripts/build_index.py tests/test_config.py
git commit -m "feat: env-configurable asset paths (deploy-ready)"
git push origin main
```

---

### Task 2: Assemble lean deploy assets

**Files:**
- Create: `scripts/prepare_deploy.py`
- Modify: `.gitignore` (ignore `deploy/assets/`)

**Interfaces:**
- Produces `deploy/assets/{visionsearch.pt, gallery/gallery.faiss, gallery/gallery_meta.json, images/*.jpg}`.

- [ ] **Step 1: Add `deploy/assets/` to `.gitignore`**

Append under a new heading:
```
# Deploy assets (large + dataset licensing; shipped to HF Space via LFS, not GitHub)
deploy/assets/
```

- [ ] **Step 2: Write `scripts/prepare_deploy.py`**

```python
"""Assemble a lean, self-contained deploy gallery under deploy/assets/.

Embeds the first N test images, builds a FAISS index, and copies those images +
the trained checkpoint. Run before building the Docker image.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from visionsearch.config import CONFIG
from visionsearch.data.flickr30k import load_annotations
from visionsearch.data.transforms import build_transform
from visionsearch.index.faiss_index import ImageIndex
from visionsearch.models.dual_encoder import DualEncoder

DATA = CONFIG.data_dir / "flickr30k"
SRC_IMAGES = DATA / "images"
ASSETS = Path(__file__).resolve().parents[1] / "deploy" / "assets"


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery-size", type=int, default=500)
    args = ap.parse_args()

    anns = load_annotations(DATA / "flickr_annotations_30k.csv", split="test")[: args.gallery_size]
    model = DualEncoder().to(CONFIG.device)
    ckpt = torch.load(CONFIG.checkpoint_path, map_location=CONFIG.device)
    model.image_head.load_state_dict(ckpt["image_head"])
    model.text_head.load_state_dict(ckpt["text_head"])
    model.eval()
    tfm = build_transform(train=False)

    (ASSETS / "images").mkdir(parents=True, exist_ok=True)
    embeds, filenames = [], []
    for i in range(0, len(anns), 64):
        chunk = anns[i:i + 64]
        px = torch.stack([tfm(Image.open(SRC_IMAGES / a.filename).convert("RGB")) for a in chunk])
        embeds.append(model.encode_image(px.to(CONFIG.device)).cpu().numpy())
        for a in chunk:
            shutil.copy(SRC_IMAGES / a.filename, ASSETS / "images" / a.filename)
            filenames.append(a.filename)
    ImageIndex.build(np.concatenate(embeds), filenames).save(ASSETS / "gallery")
    shutil.copy(CONFIG.checkpoint_path, ASSETS / "visionsearch.pt")
    size_mb = sum(f.stat().st_size for f in (ASSETS / "images").glob("*.jpg")) / 1e6
    print(f"deploy assets ready: {len(filenames)} images (~{size_mb:.0f} MB) -> {ASSETS}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it**

Run: `.venv/Scripts/python.exe scripts/prepare_deploy.py --gallery-size 500`
Expected: `deploy assets ready: 500 images (~70 MB) -> .../deploy/assets`.

- [ ] **Step 4: Commit + push** (script + gitignore only; assets are ignored)

```bash
git add scripts/prepare_deploy.py .gitignore
git commit -m "feat: prepare_deploy.py — assemble lean deploy gallery"
git push origin main
```

---

### Task 3: Dockerfile + HF Space config

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`
- Create: `deploy/README_hf.md`

- [ ] **Step 1: Write `Dockerfile`** (multi-stage)

```dockerfile
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
```

- [ ] **Step 2: Write `.dockerignore`**

```
.venv/
**/__pycache__/
.git/
data/
checkpoints/
runs/
wandb/
frontend/node_modules/
frontend/dist/
docs/
*.md
!deploy/README_hf.md
.pytest_cache/
```

- [ ] **Step 3: Write `deploy/README_hf.md`** (the HF Space README with frontmatter)

```markdown
---
title: VisionSearch
emoji: 🔍
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# VisionSearch

Semantic text-to-image search — a from-scratch contrastive dual-encoder (CLIP-style).
Type a sentence, get matching images. Trained on Flickr30k; this Space serves a
500-image demo gallery via FAISS. Source: https://github.com/Samikshabatra/VisionSearch
```

- [ ] **Step 4: Commit + push**

```bash
git add Dockerfile .dockerignore deploy/README_hf.md
git commit -m "feat: single-container Dockerfile + HF Space config"
git push origin main
```

---

### Task 4: Build + smoke-test the real Docker image

- [ ] **Step 1: Build the image** (multi-stage; pre-bakes weights — several minutes)

Run: `docker build -t visionsearch:latest .`
Expected: completes; final image built.

- [ ] **Step 2: Run the container**

Run: `docker run -d --name vs -p 7860:7860 visionsearch:latest`
Expected: container id printed; `docker logs vs` shows uvicorn startup.

- [ ] **Step 3: Smoke-test all endpoints** (wait for startup, then hit them)

```bash
for i in $(seq 1 30); do curl -s http://localhost:7860/health >/dev/null 2>&1 && break; sleep 2; done
curl -s http://localhost:7860/health                       # {"status":"ok","gallery_size":500}
curl -s -X POST http://localhost:7860/search -H "Content-Type: application/json" \
  -d '{"query":"a dog on the beach","k":3}'                # ranked results
curl -s -o /dev/null -w "GET / -> %{http_code}\n" http://localhost:7860/   # 200 (index.html)
```
Expected: health gallery_size=500; search returns ranked images; `/` serves the SPA (200).

- [ ] **Step 4: Stop + clean up the container**

Run: `docker rm -f vs`
(Keep the image `visionsearch:latest` for the user.)

- [ ] **Step 5: No commit** (verification only) — record the smoke-test result in the Week-8 summary.

---

### Task 5: Final docs polish

**Files:**
- Modify: `README.md` (architecture diagram, deploy section, mark Week 8, "Deploy it yourself" steps)

- [ ] **Step 1: Add a "Deploy" section to `README.md`** documenting the HF push

```markdown
## Deploy (Hugging Face Spaces)

Single container — FastAPI serves the built frontend + search API.

```bash
python scripts/prepare_deploy.py --gallery-size 500   # assemble deploy/assets
docker build -t visionsearch .                         # build + smoke-test locally
docker run -p 7860:7860 visionsearch

# To publish on Hugging Face (Docker SDK Space):
#  1. Create a Space (SDK: Docker), use deploy/README_hf.md as its README.md
#  2. git lfs track "deploy/assets/**"; git add -f deploy/assets
#  3. push the repo (code + LFS assets) to the Space remote
```
```

- [ ] **Step 2: Replace the ASCII flow in "How it works" with a clearer architecture note** (keep it honest; the ASCII diagram is fine — just ensure the FastAPI+FAISS+React serving path is described in the Deploy section).

- [ ] **Step 3: Mark Week 8 done** in the roadmap table and bump the week badge to "Week 8 of 8 — complete". Update status badge from `in%20progress` to `shipped` (or `complete`).

- [ ] **Step 4: Commit + push**

```bash
git add README.md
git commit -m "docs: deploy instructions + finalize README (Week 8)"
git push origin main
```

---

## Self-Review Notes

- **Spec coverage (Week 8):** deploy to HF Spaces → Dockerfile + HF config + instructions (Tasks 3,5); confirm it works → real Docker build + smoke test (Task 4); README with architecture, design decisions, results table (already), how-to-run → Task 5 + existing README. Demo GIF is noted as optional (can't record headless) — the qualitative PNG + live Space serve that role.
- **Honesty:** the actual HF push is the user's step (account/token); the plan stops at a locally-verified image + documented steps.
- **No placeholders:** all code/files complete. Asset paths are env-driven so the same backend runs locally (defaults) and in the container (VS_* env).
- **Type consistency:** `CONFIG.checkpoint_path/gallery_dir/images_dir` used identically across backend, build_index, prepare_deploy; container env vars match those names.
```
