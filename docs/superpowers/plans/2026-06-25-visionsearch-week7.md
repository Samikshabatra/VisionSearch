# VisionSearch Week 7 — Search App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A working end-to-end search app — precomputed gallery embeddings in a FAISS index, a FastAPI `/search` API, and the React frontend wired to live results.

**Architecture:** `index/faiss_index.py` wraps a FAISS `IndexFlatIP` (inner product = cosine on our L2-normalized embeds). `scripts/build_index.py` embeds the test-split gallery once and saves the index + filename metadata. `backend/` (FastAPI) loads the model + index, exposes `POST /search` (text → embed → FAISS → ranked images), `GET /health`, and `GET /images/{name}`. The React frontend (scaffolded Week 1) swaps mock data for a real `fetch('/search')`, with a Vite dev proxy to the backend.

**Tech Stack:** faiss-cpu, FastAPI, uvicorn, React + Vite, pytest (TestClient).

## Global Constraints

- Embeddings are **L2-normalized** → use `faiss.IndexFlatIP` (inner product = cosine).
- Gallery = the **test split** (1000 images the model never trained on — honest demo).
- Backend port **8020** locally (8000/8010 are taken by other projects on this machine).
- Dev: Vite proxies `/search`, `/images`, `/health` → `http://localhost:8020` (no CORS needed).
- Load model from `checkpoints/visionsearch.pt`; query path needs only `encode_text`.
- `backend/` is a package (`__init__.py`); imports the installed `visionsearch`.
- Use `.venv/Scripts/python.exe`. Push after the week. Index files live under `data/` (gitignored).

---

### Task 1: FAISS index wrapper

**Files:**
- Create: `src/visionsearch/index/faiss_index.py`
- Create: `tests/test_faiss_index.py`

**Interfaces:**
- Produces: `ImageIndex` with:
  - `ImageIndex.build(embeds: np.ndarray[N,D], filenames: list[str]) -> ImageIndex`
  - `.search(query: np.ndarray[D] | [Q,D], k: int) -> tuple[scores, indices]`
  - `.filenames: list[str]`; `.save(dir: Path)`; `ImageIndex.load(dir: Path) -> ImageIndex`

- [ ] **Step 1: Write failing tests** (`tests/test_faiss_index.py`)

```python
import numpy as np

from visionsearch.index.faiss_index import ImageIndex


def _norm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def test_build_and_search_finds_self():
    rng = np.random.default_rng(0)
    embeds = _norm(rng.standard_normal((20, 16)).astype("float32"))
    idx = ImageIndex.build(embeds, [f"{i}.jpg" for i in range(20)])
    scores, ids = idx.search(embeds[3], k=5)
    assert ids[0][0] == 3                 # an item retrieves itself first
    assert scores[0][0] > scores[0][1]    # scores sorted descending

def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    embeds = _norm(rng.standard_normal((10, 8)).astype("float32"))
    names = [f"img{i}.jpg" for i in range(10)]
    ImageIndex.build(embeds, names).save(tmp_path)
    loaded = ImageIndex.load(tmp_path)
    assert loaded.filenames == names
    assert loaded.search(embeds[0], k=1)[1][0][0] == 0
```

- [ ] **Step 2: Run, expect failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_faiss_index.py -q`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `src/visionsearch/index/faiss_index.py`**

```python
"""FAISS index over L2-normalized image embeddings (inner product = cosine)."""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

_INDEX_FILE = "gallery.faiss"
_META_FILE = "gallery_meta.json"


class ImageIndex:
    def __init__(self, index: "faiss.Index", filenames: list[str]) -> None:
        self.index = index
        self.filenames = filenames

    @classmethod
    def build(cls, embeds: np.ndarray, filenames: list[str]) -> "ImageIndex":
        embeds = np.ascontiguousarray(embeds, dtype="float32")
        index = faiss.IndexFlatIP(embeds.shape[1])     # inner product on unit vectors = cosine
        index.add(embeds)
        return cls(index, list(filenames))

    def search(self, query: np.ndarray, k: int = 10):
        q = np.ascontiguousarray(query, dtype="float32")
        if q.ndim == 1:
            q = q[None, :]
        return self.index.search(q, k)                  # (scores[Q,k], indices[Q,k])

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / _INDEX_FILE))
        (directory / _META_FILE).write_text(json.dumps({"filenames": self.filenames}))

    @classmethod
    def load(cls, directory: Path) -> "ImageIndex":
        directory = Path(directory)
        index = faiss.read_index(str(directory / _INDEX_FILE))
        meta = json.loads((directory / _META_FILE).read_text())
        return cls(index, meta["filenames"])
```

- [ ] **Step 4: Run tests, expect pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_faiss_index.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit + push**

```bash
git add src/visionsearch/index/faiss_index.py tests/test_faiss_index.py
git commit -m "feat: FAISS ImageIndex (build/search/save/load)"
git push origin main
```

---

### Task 2: Build the gallery index

**Files:**
- Create: `scripts/build_index.py`

**Interfaces:**
- Consumes: `DualEncoder` + checkpoint, `ImageIndex`. Produces `data/flickr30k/gallery/{gallery.faiss, gallery_meta.json}`.

- [ ] **Step 1: Write `scripts/build_index.py`**

```python
"""Embed the test-split gallery once and build a FAISS index (the demo's precompute step)."""
from __future__ import annotations

import torch
from PIL import Image

from visionsearch.config import CONFIG
from visionsearch.data.flickr30k import load_annotations
from visionsearch.data.transforms import build_transform
from visionsearch.index.faiss_index import ImageIndex
from visionsearch.models.dual_encoder import DualEncoder

DATA = CONFIG.data_dir / "flickr30k"
IMAGES = DATA / "images"


@torch.no_grad()
def main() -> None:
    anns = load_annotations(DATA / "flickr_annotations_30k.csv", split="test")
    model = DualEncoder().to(CONFIG.device)
    ckpt = torch.load(CONFIG.checkpoint_dir / "visionsearch.pt", map_location=CONFIG.device)
    model.image_head.load_state_dict(ckpt["image_head"])
    model.text_head.load_state_dict(ckpt["text_head"])
    model.eval()
    tfm = build_transform(train=False)

    embeds, filenames = [], []
    for i in range(0, len(anns), 64):
        chunk = anns[i:i + 64]
        px = torch.stack([tfm(Image.open(IMAGES / a.filename).convert("RGB")) for a in chunk])
        embeds.append(model.encode_image(px.to(CONFIG.device)).cpu().numpy())
        filenames += [a.filename for a in chunk]
    import numpy as np
    embeds = np.concatenate(embeds)

    out = DATA / "gallery"
    ImageIndex.build(embeds, filenames).save(out)
    print(f"indexed {len(filenames)} images -> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `.venv/Scripts/python.exe scripts/build_index.py`
Expected: `indexed 1000 images -> .../data/flickr30k/gallery`; creates `gallery.faiss` + `gallery_meta.json`.

- [ ] **Step 3: Commit + push** (script only; index files are under gitignored `data/`)

```bash
git add scripts/build_index.py
git commit -m "feat: build FAISS gallery index from test split"
git push origin main
```

---

### Task 3: FastAPI backend

**Files:**
- Create: `backend/__init__.py`
- Create: `backend/search_service.py`
- Create: `backend/main.py`
- Create: `tests/test_backend.py`

**Interfaces:**
- `SearchService.search(query: str, k: int) -> list[dict]` → `[{filename, url, score}]`.
- App routes: `GET /health`, `POST /search {query, k}`, `GET /images/{name}`.

- [ ] **Step 1: Write `backend/search_service.py`**

```python
"""Loads the trained model + FAISS gallery and answers text queries."""
from __future__ import annotations

import torch
from transformers import AutoTokenizer

from visionsearch.config import CONFIG
from visionsearch.index.faiss_index import ImageIndex
from visionsearch.models.dual_encoder import DualEncoder

GALLERY_DIR = CONFIG.data_dir / "flickr30k" / "gallery"


class SearchService:
    def __init__(self) -> None:
        self.device = CONFIG.device
        self.model = DualEncoder().to(self.device)
        ckpt = torch.load(CONFIG.checkpoint_dir / "visionsearch.pt", map_location=self.device)
        self.model.image_head.load_state_dict(ckpt["image_head"])
        self.model.text_head.load_state_dict(ckpt["text_head"])
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.index = ImageIndex.load(GALLERY_DIR)

    @torch.no_grad()
    def search(self, query: str, k: int = 12) -> list[dict]:
        tokens = self.tokenizer([query], return_tensors="pt").to(self.device)
        q = self.model.encode_text(tokens["input_ids"], tokens["attention_mask"]).cpu().numpy()
        scores, ids = self.index.search(q, k)
        out = []
        for score, idx in zip(scores[0], ids[0]):
            name = self.index.filenames[int(idx)]
            out.append({"filename": name, "url": f"/images/{name}", "score": round(float(score), 4)})
        return out

    @property
    def gallery_size(self) -> int:
        return len(self.index.filenames)
```

- [ ] **Step 2: Write `backend/main.py`**

```python
"""FastAPI app: /search, /health, /images, and (in prod) the built frontend."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from visionsearch.config import CONFIG
from backend.search_service import SearchService

IMAGES = CONFIG.data_dir / "flickr30k" / "images"
FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"

app = FastAPI(title="VisionSearch")
_service: SearchService | None = None


def get_service() -> SearchService:
    global _service
    if _service is None:
        _service = SearchService()
    return _service


class SearchRequest(BaseModel):
    query: str
    k: int = 12


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "gallery_size": get_service().gallery_size}


@app.post("/search")
def search(req: SearchRequest) -> dict:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="empty query")
    return {"results": get_service().search(req.query, req.k)}


@app.get("/images/{name}")
def image(name: str) -> FileResponse:
    path = IMAGES / name
    if not path.exists():
        raise HTTPException(status_code=404, detail="not found")
    return FileResponse(path)


# In production the built frontend is served from the same origin.
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
```

- [ ] **Step 3: Write `backend/__init__.py`** (empty) and `tests/test_backend.py`

```python
# backend/__init__.py  -> empty file
```

```python
# tests/test_backend.py
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_health_reports_gallery():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["gallery_size"] == 1000


def test_search_returns_ranked_results():
    r = client.post("/search", json={"query": "a dog on the beach", "k": 5})
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 5
    assert all("filename" in x and "url" in x for x in results)
    scores = [x["score"] for x in results]
    assert scores == sorted(scores, reverse=True)   # ranked


def test_empty_query_rejected():
    assert client.post("/search", json={"query": "  "}).status_code == 400
```

- [ ] **Step 4: Run backend tests** (requires the index from Task 2)

Run: `.venv/Scripts/python.exe -m pytest tests/test_backend.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit + push**

```bash
git add backend/__init__.py backend/search_service.py backend/main.py tests/test_backend.py
git commit -m "feat: FastAPI search backend (/search, /health, /images)"
git push origin main
```

---

### Task 4: Wire the React frontend to live search

**Files:**
- Modify: `frontend/vite.config.js` (dev proxy)
- Modify: `frontend/src/App.jsx` (real fetch)
- Delete: `frontend/src/mockResults.js`

**Interfaces:**
- Frontend calls `POST /search` and renders `{filename, url, score}` results from the live backend.

- [ ] **Step 1: Add a dev proxy to `frontend/vite.config.js`**

```js
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const backend = "http://localhost:8020";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/search": backend,
      "/images": backend,
      "/health": backend,
    },
  },
});
```

- [ ] **Step 2: Rewrite `frontend/src/App.jsx`** to fetch live results

```jsx
import { useState } from "react";

export default function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleSearch(e) {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError("");
    try {
      const res = await fetch("/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, k: 12 }),
      });
      if (!res.ok) throw new Error(`search failed (${res.status})`);
      const data = await res.json();
      setResults(data.results);
    } catch (err) {
      setError(err.message);
      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 px-6 py-10">
      <header className="max-w-3xl mx-auto text-center mb-8">
        <h1 className="text-4xl font-bold tracking-tight">VisionSearch</h1>
        <p className="text-neutral-400 mt-2">Type a sentence, get the matching images.</p>
      </header>

      <form onSubmit={handleSearch} className="max-w-2xl mx-auto flex gap-2 mb-10">
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="two people on a beach at sunset"
          className="flex-1 rounded-lg bg-neutral-900 border border-neutral-700 px-4 py-3 outline-none focus:border-indigo-500"
        />
        <button
          disabled={loading}
          className="rounded-lg bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 px-6 py-3 font-medium"
        >
          {loading ? "Searching…" : "Search"}
        </button>
      </form>

      {error && <p className="text-center text-red-400 mb-6">{error}</p>}

      <div className="max-w-5xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-4">
        {results.map((r) => (
          <figure
            key={r.filename}
            className="rounded-lg overflow-hidden bg-neutral-900 border border-neutral-800"
          >
            <img src={r.url} alt="" className="w-full h-40 object-cover" />
            <figcaption className="px-3 py-2 text-sm text-neutral-400">score {r.score}</figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Delete the mock module**

```bash
rm frontend/src/mockResults.js
```

- [ ] **Step 4: Verify the frontend builds**

Run: `cd frontend && npm run build`
Expected: builds with no errors (no remaining import of mockResults).

- [ ] **Step 5: Live end-to-end smoke test** — start backend, query it, confirm real results

```bash
# terminal-equivalent: start backend in background, hit it, stop it
.venv/Scripts/python.exe -m uvicorn backend.main:app --port 8020 &
# wait for startup, then:
curl -s -X POST http://localhost:8020/search -H "Content-Type: application/json" -d '{"query":"a dog on the beach","k":3}'
```
Expected: JSON with 3 results (filenames + /images/ urls + descending scores). Stop the server after.

- [ ] **Step 6: Commit + push**

```bash
git add frontend/vite.config.js frontend/src/App.jsx
git rm frontend/src/mockResults.js
git commit -m "feat: wire React frontend to live /search backend"
git push origin main
```

---

## Self-Review Notes

- **Spec coverage (Week 7):** precompute gallery embeddings + FAISS index → Tasks 1–2; query box returns top-k → Tasks 3–4; similarity scores shown on results → Task 4 (figcaption). All roadmap Week-7 bullets covered. (Example-query chips are a nice-to-have deferred to Week 8 polish.)
- **Precompute technique** (deferred from Week 5) lands here in `build_index.py` — its natural home.
- **No placeholders:** all code complete. The backend test is a real integration test against the built index (Task 2 must run first).
- **Type consistency:** `ImageIndex.search` returns `(scores, indices)` used consistently in `build_index`/`SearchService`; result dict keys `{filename, url, score}` match what `App.jsx` renders; port 8020 consistent across vite proxy + smoke test.
```
