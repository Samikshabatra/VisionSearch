# VisionSearch Week 1 — Foundations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the VisionSearch repo with a GPU-verified CUDA PyTorch environment, an installable ML package skeleton, an interview-ready theory note, and (stretch) a React frontend shell.

**Architecture:** Dedicated `.venv` with CUDA (cu128) PyTorch for the RTX 5060. An installable `src/visionsearch/` package holds the ML code (config first; data/models/loss land in later weeks). FastAPI `backend/` and React+Vite+Tailwind `frontend/` are scaffolded as empty shells this week. Docs hold the teaching deliverable.

**Tech Stack:** Python 3.11.7, PyTorch cu128, FAISS (later), FastAPI (later), React + Vite + Tailwind, pytest, Git/GitHub.

## Global Constraints

- Python **3.11.7** (existing on machine).
- PyTorch must be the **CUDA cu128 build** — RTX 5060 is Blackwell **sm_120**; cu126/CPU wheels will not run on it. Install via `--index-url https://download.pytorch.org/whl/cu128`.
- VRAM budget **8 GB** — every later design choice (frozen backbone, AMP, grad-accum, 224×224) serves this.
- Use the venv's interpreter directly (`.venv/Scripts/python.exe`) — do NOT rely on shell `activate` persisting between tool calls.
- **Scaffold + TEACH:** explain each ML concept as it is introduced; the theory note must be interview-ready in plain language.
- **Push to GitHub after each big update** — remote `origin` = `https://github.com/Samikshabatra/VisionSearch`, branch `main`. Already initialized and pushed once (spec).
- Commit messages end with the `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` trailer.

---

### Task 1: CUDA PyTorch environment, GPU-verified

**Files:**
- Create: `.venv/` (gitignored)
- Create: `requirements.txt`
- Create: `scripts/check_env.py`

**Interfaces:**
- Produces: a working `.venv` whose `python` reports `torch.cuda.is_available() == True` on the RTX 5060; `scripts/check_env.py` exits 0 only when the GPU is usable.

- [ ] **Step 1: Create the dedicated venv**

```bash
cd "/c/Users/Samiksha Batra/Desktop/VisionSearch"
python -m venv .venv
.venv/Scripts/python.exe --version   # expect: Python 3.11.7
```

- [ ] **Step 2: Write `requirements.txt`** (torch installed separately via the cu128 index)

```text
# Core ML — torch/torchvision installed separately from the cu128 index (see README)
numpy
pillow
# Backbones & data
transformers
timm
# Vector search
faiss-cpu
# Training utilities
tensorboard
tqdm
# API (used from Week 7; pinned here so the env is stable)
fastapi
uvicorn[standard]
# Dev
pytest
```

- [ ] **Step 3: Install CUDA PyTorch first, then the rest**

```bash
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

- [ ] **Step 4: Write `scripts/check_env.py`** (this script IS the test for this task)

```python
"""GPU/CUDA smoke test for VisionSearch. Exits 0 only if the RTX 5060 is usable.

Run any time the environment is in doubt:
    .venv/Scripts/python.exe scripts/check_env.py
"""
import sys


def main() -> int:
    import torch

    print(f"torch        : {torch.__version__}")
    print(f"cuda built   : {torch.version.cuda}")
    available = torch.cuda.is_available()
    print(f"cuda runtime : {available}")
    if not available:
        print("FAIL: CUDA not available — install the cu128 build (see requirements notes).")
        return 1

    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"device       : {name} ({total_gb:.1f} GB)")

    # Actually run a tensor op on the GPU — availability alone is not proof.
    x = torch.randn(1024, 1024, device="cuda")
    y = (x @ x).sum().item()
    print(f"matmul on GPU: ok (checksum={y:.1f})")
    print("PASS: GPU is usable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run it and verify the GPU is actually used**

```bash
.venv/Scripts/python.exe scripts/check_env.py
```
Expected: prints the RTX 5060 line, `matmul on GPU: ok`, `PASS`, exit 0.
If it FAILs on `cuda runtime : False`, the wrong wheel installed — reinstall torch from the cu128 index (Step 3).

- [ ] **Step 6: Commit and push**

```bash
git add requirements.txt scripts/check_env.py
git commit -m "feat: CUDA cu128 env + GPU smoke test"
git push origin main
```

---

### Task 2: Installable ML package skeleton + config

**Files:**
- Create: `pyproject.toml`
- Create: `src/visionsearch/__init__.py`
- Create: `src/visionsearch/config.py`
- Create: `tests/test_config.py`
- Create: `src/visionsearch/{data,models,train,eval,index}/__init__.py` (empty package markers)

**Interfaces:**
- Produces: `from visionsearch.config import CONFIG` — a frozen dataclass with `device: str` (`"cuda"`/`"cpu"`), `embed_dim: int = 256`, `image_size: int = 224`, `data_dir: Path`, `checkpoint_dir: Path`. Later weeks import `CONFIG` for all paths/hyperparams.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from pathlib import Path

from visionsearch.config import CONFIG


def test_config_has_core_fields():
    assert CONFIG.embed_dim == 256
    assert CONFIG.image_size == 224
    assert CONFIG.device in ("cuda", "cpu")
    assert isinstance(CONFIG.data_dir, Path)
    assert isinstance(CONFIG.checkpoint_dir, Path)
```

- [ ] **Step 2: Write `pyproject.toml`** (editable install target)

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "visionsearch"
version = "0.1.0"
description = "Semantic text-to-image search via a from-scratch contrastive dual-encoder"
requires-python = ">=3.11"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 3: Write `src/visionsearch/config.py`**

```python
"""Single source of truth for paths, hyperparameters, and device."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _detect_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@dataclass(frozen=True)
class Config:
    # Hyperparameters
    embed_dim: int = 256          # shared embedding space dimension
    image_size: int = 224         # keep at 224 — memory grows quadratically
    # Paths
    data_dir: Path = field(default_factory=lambda: _ROOT / "data")
    checkpoint_dir: Path = field(default_factory=lambda: _ROOT / "checkpoints")
    # Runtime
    device: str = field(default_factory=_detect_device)


CONFIG = Config()
```

- [ ] **Step 4: Create the empty subpackage markers and root `__init__.py`**

```bash
printf '"""VisionSearch: contrastive text-to-image retrieval."""\n__version__ = "0.1.0"\n' > src/visionsearch/__init__.py
for d in data models train eval index; do
  mkdir -p "src/visionsearch/$d"
  printf '' > "src/visionsearch/$d/__init__.py"
done
```

- [ ] **Step 5: Editable-install the package and run the test**

```bash
.venv/Scripts/python.exe -m pip install -e .
.venv/Scripts/python.exe -m pytest tests/test_config.py -v
```
Expected: 1 passed; `CONFIG.device == "cuda"` on this machine.

- [ ] **Step 6: Commit and push**

```bash
git add pyproject.toml src/visionsearch tests/test_config.py
git commit -m "feat: installable visionsearch package + config single-source-of-truth"
git push origin main
```

---

### Task 3: Interview-ready theory note (the TEACH deliverable)

**Files:**
- Create: `docs/theory-note.md`

**Interfaces:**
- Produces: a plain-language reference the user can recite in interviews. No code dependency.

- [ ] **Step 1: Write `docs/theory-note.md`** covering, each in 2–4 sentences of plain language with a one-line "interview answer":
  - **Contrastive learning** — pull matching image–caption pairs together, push mismatched pairs apart.
  - **Dual-encoder vs cross-encoder** — why dual encoders allow precomputing image vectors and scaling search to millions; cross-encoders are accurate but too slow for retrieval.
  - **Shared embedding space & cosine similarity** — both encoders map into one space; rank by cosine.
  - **Symmetric InfoNCE loss** — softmax over in-batch similarities, averaged over the image→text and text→image directions; the other items in the batch are the negatives.
  - **Learnable temperature** — scales the logits, controlling how sharply positives separate from negatives; learned, often clamped.
  - **Why freeze the image backbone** — memory + data efficiency on 8 GB, and it mirrors how real VLMs train the alignment, not the encoder.
  - **Recall@K** — for a query, is the correct image in the top K; always reported vs the raw-CLIP baseline.
  - **8 GB efficiency moves** — frozen backbone + precomputed features, AMP, gradient accumulation, 224×224, optional LoRA.

  Each bullet must end with a bolded **Interview answer:** one-liner. No placeholders.

- [ ] **Step 2: Read it back for accuracy and plain language**

Verify: no jargon left unexplained; every "Interview answer" is one sentence; matches the spec's §2/§5/§6/§7.

- [ ] **Step 3: Commit and push**

```bash
git add docs/theory-note.md
git commit -m "docs: interview-ready theory note (CLIP, InfoNCE, dual-encoder)"
git push origin main
```

---

### Task 4 (STRETCH — only if Tasks 1–3 done with time left): React + Vite + Tailwind frontend shell

**Files:**
- Create: `frontend/` (Vite React scaffold)
- Modify: `frontend/src/App.jsx` — search box + mock results grid
- Create: `frontend/src/mockResults.js`

**Interfaces:**
- Produces: a frontend that runs with `npm run dev` and shows a search box + a grid of mock result cards (placeholder images + similarity scores). No backend wired yet.

- [ ] **Step 1: Scaffold Vite React app**

```bash
cd "/c/Users/Samiksha Batra/Desktop/VisionSearch"
npm create vite@latest frontend -- --template react
cd frontend && npm install
npm install -D tailwindcss @tailwindcss/vite
```

- [ ] **Step 2: Wire Tailwind v4 via the Vite plugin** in `frontend/vite.config.js`

```js
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
});
```

Replace `frontend/src/index.css` contents with:

```css
@import "tailwindcss";
```

- [ ] **Step 3: Mock results data** in `frontend/src/mockResults.js`

```js
// Placeholder results until the FastAPI /search endpoint exists (Week 7).
export const MOCK_RESULTS = Array.from({ length: 8 }, (_, i) => ({
  id: i,
  url: `https://picsum.photos/seed/visionsearch${i}/400/300`,
  score: (0.92 - i * 0.04).toFixed(3),
}));
```

- [ ] **Step 4: Search box + results grid** in `frontend/src/App.jsx`

```jsx
import { useState } from "react";
import { MOCK_RESULTS } from "./mockResults";

export default function App() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState([]);

  function handleSearch(e) {
    e.preventDefault();
    // TODO(Week 7): POST query to FastAPI /search. For now, show mock results.
    setResults(MOCK_RESULTS);
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
        <button className="rounded-lg bg-indigo-600 hover:bg-indigo-500 px-6 py-3 font-medium">
          Search
        </button>
      </form>

      <div className="max-w-5xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-4">
        {results.map((r) => (
          <figure key={r.id} className="rounded-lg overflow-hidden bg-neutral-900 border border-neutral-800">
            <img src={r.url} alt="" className="w-full h-40 object-cover" />
            <figcaption className="px-3 py-2 text-sm text-neutral-400">score {r.score}</figcaption>
          </figure>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Run the dev server and verify it renders**

```bash
cd frontend && npm run dev
```
Expected: dev server starts; opening the URL shows the VisionSearch header, search box, and (after clicking Search) an 8-image mock grid with scores. Stop the server (Ctrl-C) after verifying.

- [ ] **Step 6: Commit and push**

```bash
cd "/c/Users/Samiksha Batra/Desktop/VisionSearch"
git add frontend
git commit -m "feat: React+Vite+Tailwind frontend shell with mock results"
git push origin main
```

---

## Self-Review Notes

- **Spec coverage (Week-1 scope, spec §8):** venv+CUDA torch verified → Task 1; repo skeleton + pyproject + commit/push → Tasks 1–2 (git already init'd with spec); theory note → Task 3; check_env.py → Task 1; React shell stretch → Task 4. All §8 items covered.
- **Later-week modules (spec §4)** intentionally NOT built this week — only their empty package markers (Task 2 Step 4), so future weeks drop in cleanly.
- **No placeholders:** all code shown in full. The `TODO(Week 7)` comments are intentional forward-markers, not plan gaps.
- **Type consistency:** `CONFIG` fields (`embed_dim`, `image_size`, `device`, `data_dir`, `checkpoint_dir`) are defined once in Task 2 and used by name in the test; no later Week-1 task depends on undefined symbols.
