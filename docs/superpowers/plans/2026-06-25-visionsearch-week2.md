# VisionSearch Week 2 — Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A clean Flickr30k data pipeline — download, parse, a PyTorch `Dataset` with 224×224 transforms and DistilBERT tokenization, leakage-free splits, and a batch visualization.

**Architecture:** No `datasets` library (pandas is blocked by Windows Application Control). We fetch raw files with `huggingface_hub` and parse the annotations CSV with stdlib `csv`/`ast`. The `Dataset` returns `(image_tensor, caption_str, img_id)`; a `collate_fn` tokenizes each batch with DistilBERT so padding is per-batch. Splits come from the dataset's own `split` column (standard Karpathy split).

**Tech Stack:** huggingface_hub, torchvision transforms, transformers (DistilBERT tokenizer), PIL, pytest.

## Global Constraints

- **Do NOT import or install `pandas` or the `datasets` library** — pandas' DLL is blocked by Smart App Control on this machine. Parse CSV with stdlib `csv` + `ast.literal_eval`.
- Data source: **`nlphuji/flickr30k`** (public, no auth). `flickr30k-images.zip` + `flickr_annotations_30k.csv`.
- CSV columns: `raw` (list-literal of 5 captions), `sentids`, `split` (`train`/`val`/`test`), `filename`, `img_id`.
- Images normalized at **224×224**; ImageNet mean/std for now (revisit when backbone is chosen in Week 3).
- DistilBERT tokenizer `distilbert-base-uncased`, `max_length=40` (captions are short).
- Use `.venv/Scripts/python.exe` directly. Push to GitHub after the week's work.
- Tests must NOT depend on the 4 GB download — use a tiny generated fixture.

---

### Task 1: Annotations module (download + parse, no pandas)

**Files:**
- Create: `src/visionsearch/data/flickr30k.py`
- Create: `tests/conftest.py`
- Create: `tests/test_annotations.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Annotation` with `filename: str`, `captions: list[str]`, `img_id: int`, `split: str`.
  - `load_annotations(csv_path: Path, split: str | None = None) -> list[Annotation]`.
  - `download_flickr30k(data_dir: Path) -> tuple[Path, Path]` returning `(images_dir, csv_path)` (downloads + unzips; idempotent).

- [ ] **Step 1: Write the fixture** (`tests/conftest.py`) — a tiny CSV + 3 generated images, no network

```python
import ast
import csv
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> dict:
    """A 3-image Flickr30k-shaped dataset on disk (no network, no 4GB download)."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    rows = [
        {"filename": "a.jpg", "split": "train", "img_id": 0,
         "raw": ["a red square", "a crimson box", "red shape", "scarlet", "red"]},
        {"filename": "b.jpg", "split": "train", "img_id": 1,
         "raw": ["a green field", "grass", "green area", "meadow", "green"]},
        {"filename": "c.jpg", "split": "test", "img_id": 2,
         "raw": ["a blue sky", "blue", "sky", "azure", "clear sky"]},
    ]
    colors = {"a.jpg": (220, 20, 20), "b.jpg": (20, 200, 20), "c.jpg": (20, 20, 220)}
    for fn, color in colors.items():
        Image.new("RGB", (64, 48), color).save(images_dir / fn)

    csv_path = tmp_path / "ann.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["raw", "sentids", "split", "filename", "img_id"])
        for r in rows:
            w.writerow([str(r["raw"]), str([0, 1, 2, 3, 4]), r["split"], r["filename"], r["img_id"]])
    return {"images_dir": images_dir, "csv_path": csv_path}
```

- [ ] **Step 2: Write failing tests** (`tests/test_annotations.py`)

```python
from visionsearch.data.flickr30k import Annotation, load_annotations


def test_load_parses_captions(tiny_dataset):
    anns = load_annotations(tiny_dataset["csv_path"])
    assert len(anns) == 3
    a = anns[0]
    assert isinstance(a, Annotation)
    assert a.filename == "a.jpg"
    assert a.captions == ["a red square", "a crimson box", "red shape", "scarlet", "red"]
    assert a.img_id == 0


def test_filter_by_split(tiny_dataset):
    train = load_annotations(tiny_dataset["csv_path"], split="train")
    test = load_annotations(tiny_dataset["csv_path"], split="test")
    assert {a.filename for a in train} == {"a.jpg", "b.jpg"}
    assert {a.filename for a in test} == {"c.jpg"}


def test_splits_are_leakage_free(tiny_dataset):
    train = {a.filename for a in load_annotations(tiny_dataset["csv_path"], split="train")}
    test = {a.filename for a in load_annotations(tiny_dataset["csv_path"], split="test")}
    assert train.isdisjoint(test)
```

- [ ] **Step 3: Run, expect failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_annotations.py -v`
Expected: ImportError / module not found.

- [ ] **Step 4: Implement `src/visionsearch/data/flickr30k.py`**

```python
"""Flickr30k acquisition + annotation parsing — no pandas, no `datasets` library.

pandas' compiled DLL is blocked by Windows Application Control on this machine,
so we fetch raw files via huggingface_hub and parse the CSV with the stdlib.
"""
from __future__ import annotations

import ast
import csv
import zipfile
from dataclasses import dataclass
from pathlib import Path

_REPO = "nlphuji/flickr30k"
_CSV_FILE = "flickr_annotations_30k.csv"
_ZIP_FILE = "flickr30k-images.zip"


@dataclass(frozen=True)
class Annotation:
    filename: str
    captions: list[str]
    img_id: int
    split: str


def load_annotations(csv_path: Path, split: str | None = None) -> list[Annotation]:
    """Parse the Flickr30k annotations CSV into Annotation records.

    `split` filters to "train" / "val" / "test" when given.
    """
    out: list[Annotation] = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if split is not None and row["split"] != split:
                continue
            out.append(
                Annotation(
                    filename=row["filename"],
                    captions=list(ast.literal_eval(row["raw"])),
                    img_id=int(row["img_id"]),
                    split=row["split"],
                )
            )
    return out


def download_flickr30k(data_dir: Path) -> tuple[Path, Path]:
    """Download + unzip Flickr30k into data_dir. Idempotent. Returns (images_dir, csv_path)."""
    from huggingface_hub import hf_hub_download

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir = data_dir / "images"

    csv_cached = hf_hub_download(_REPO, _CSV_FILE, repo_type="dataset")
    csv_path = data_dir / _CSV_FILE
    if not csv_path.exists():
        csv_path.write_bytes(Path(csv_cached).read_bytes())

    if not images_dir.exists() or not any(images_dir.glob("*.jpg")):
        zip_cached = hf_hub_download(_REPO, _ZIP_FILE, repo_type="dataset")
        images_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_cached) as z:
            # The zip stores files under flickr30k-images/<name>.jpg — flatten into images/.
            for member in z.namelist():
                if member.endswith(".jpg"):
                    target = images_dir / Path(member).name
                    if not target.exists():
                        with z.open(member) as src, open(target, "wb") as dst:
                            dst.write(src.read())
    return images_dir, csv_path
```

- [ ] **Step 5: Run tests, expect pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_annotations.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit + push**

```bash
git add src/visionsearch/data/flickr30k.py tests/conftest.py tests/test_annotations.py
git commit -m "feat: Flickr30k acquisition + annotation parsing (no pandas)"
git push origin main
```

---

### Task 2: Transforms + Dataset + collate

**Files:**
- Create: `src/visionsearch/data/transforms.py`
- Create: `src/visionsearch/data/dataset.py`
- Create: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `Annotation`, `load_annotations` (Task 1); `CONFIG.image_size`.
- Produces:
  - `build_transform(train: bool, image_size: int = 224) -> Callable` (PIL → Tensor[3,H,W]).
  - `Flickr30kDataset(images_dir, annotations, transform, train=True)` → `__getitem__` returns `(image: Tensor[3,224,224], caption: str, img_id: int)`. Train samples a random caption; eval uses caption[0].
  - `make_collate_fn(tokenizer, max_length=40) -> Callable` producing batch dict with keys `pixel_values [B,3,224,224]`, `input_ids [B,L]`, `attention_mask [B,L]`, `img_id [B]`.

- [ ] **Step 1: Write failing tests** (`tests/test_dataset.py`)

```python
import torch
from transformers import AutoTokenizer

from visionsearch.data.dataset import Flickr30kDataset, make_collate_fn
from visionsearch.data.flickr30k import load_annotations
from visionsearch.data.transforms import build_transform


def _ds(tiny_dataset, train=True):
    anns = load_annotations(tiny_dataset["csv_path"])
    return Flickr30kDataset(
        images_dir=tiny_dataset["images_dir"],
        annotations=anns,
        transform=build_transform(train=train, image_size=32),
        train=train,
    )


def test_dataset_len_and_item_shape(tiny_dataset):
    ds = _ds(tiny_dataset)
    assert len(ds) == 3
    img, caption, img_id = ds[0]
    assert img.shape == (3, 32, 32)
    assert isinstance(caption, str) and caption
    assert isinstance(img_id, int)


def test_eval_caption_is_deterministic(tiny_dataset):
    ds = _ds(tiny_dataset, train=False)
    assert ds[0][1] == "a red square"  # caption[0] in eval mode


def test_collate_batches_and_tokenizes(tiny_dataset):
    ds = _ds(tiny_dataset)
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    collate = make_collate_fn(tok, max_length=16)
    batch = collate([ds[0], ds[1], ds[2]])
    assert batch["pixel_values"].shape == (3, 3, 32, 32)
    assert batch["input_ids"].shape[0] == 3
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["img_id"].tolist() == [0, 1, 2]
    assert batch["input_ids"].dtype == torch.long
```

- [ ] **Step 2: Run, expect failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_dataset.py -v`
Expected: import error.

- [ ] **Step 3: Implement `src/visionsearch/data/transforms.py`**

```python
"""Image transforms. ImageNet normalization (revisit when backbone is chosen, Week 3)."""
from __future__ import annotations

from typing import Callable

from torchvision import transforms

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def build_transform(train: bool, image_size: int = 224) -> Callable:
    if train:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
```

- [ ] **Step 4: Implement `src/visionsearch/data/dataset.py`**

```python
"""Flickr30k PyTorch Dataset + a tokenizing collate function."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset

from .flickr30k import Annotation


class Flickr30kDataset(Dataset):
    """One item per image. Train mode samples a random caption; eval uses caption[0].

    Returns (image_tensor[3,H,W], caption_str, img_id).
    """

    def __init__(self, images_dir: Path, annotations: list[Annotation],
                 transform: Callable, train: bool = True):
        self.images_dir = Path(images_dir)
        self.annotations = annotations
        self.transform = transform
        self.train = train

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int):
        ann = self.annotations[idx]
        image = Image.open(self.images_dir / ann.filename).convert("RGB")
        image = self.transform(image)
        caption = random.choice(ann.captions) if self.train else ann.captions[0]
        return image, caption, ann.img_id


def make_collate_fn(tokenizer, max_length: int = 40) -> Callable:
    """Collate (image, caption, img_id) tuples; tokenize captions per-batch with DistilBERT."""

    def collate(batch):
        images, captions, img_ids = zip(*batch)
        pixel_values = torch.stack(images, dim=0)
        tokens = tokenizer(
            list(captions),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "img_id": torch.tensor(img_ids, dtype=torch.long),
        }

    return collate
```

- [ ] **Step 5: Run tests, expect pass** (first run downloads the DistilBERT tokenizer — small)

Run: `.venv/Scripts/python.exe -m pytest tests/test_dataset.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit + push**

```bash
git add src/visionsearch/data/transforms.py src/visionsearch/data/dataset.py tests/test_dataset.py
git commit -m "feat: Flickr30k Dataset, transforms, and tokenizing collate"
git push origin main
```

---

### Task 3: Download the real dataset + smoke-test a real batch

**Files:**
- Create: `scripts/download_data.py`

**Interfaces:**
- Consumes: `download_flickr30k`, `load_annotations`, `Flickr30kDataset`, `build_transform`, `make_collate_fn`, `CONFIG`.

- [ ] **Step 1: Write `scripts/download_data.py`**

```python
"""Download Flickr30k and print a summary. Run once: pulls ~4GB on first run."""
from visionsearch.config import CONFIG
from visionsearch.data.flickr30k import download_flickr30k, load_annotations


def main() -> None:
    data_dir = CONFIG.data_dir / "flickr30k"
    print(f"Downloading Flickr30k into {data_dir} (first run pulls ~4GB)...")
    images_dir, csv_path = download_flickr30k(data_dir)
    n_images = sum(1 for _ in images_dir.glob("*.jpg"))
    print(f"images: {n_images} in {images_dir}")
    for split in ("train", "val", "test"):
        print(f"  {split:5s}: {len(load_annotations(csv_path, split=split))} images")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the download** (the big step — ~4GB, several minutes)

Run: `.venv/Scripts/python.exe scripts/download_data.py`
Expected: `images: 31783 ...`, then `train: 29000`, `val: 1014`, `test: 1000`.

- [ ] **Step 3: Smoke-test a real batch end-to-end**

```bash
.venv/Scripts/python.exe - <<'PY'
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from visionsearch.config import CONFIG
from visionsearch.data.flickr30k import load_annotations
from visionsearch.data.dataset import Flickr30kDataset, make_collate_fn
from visionsearch.data.transforms import build_transform

data_dir = CONFIG.data_dir / "flickr30k"
anns = load_annotations(data_dir / "flickr_annotations_30k.csv", split="train")
ds = Flickr30kDataset(data_dir / "images", anns, build_transform(train=True), train=True)
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=make_collate_fn(tok))
b = next(iter(dl))
print("pixel_values", b["pixel_values"].shape, "input_ids", b["input_ids"].shape)
PY
```
Expected: `pixel_values torch.Size([8, 3, 224, 224]) input_ids torch.Size([8, L])`.

- [ ] **Step 4: Commit + push**

```bash
git add scripts/download_data.py
git commit -m "feat: dataset download script + real-batch smoke test"
git push origin main
```

---

### Task 4: Visualize a batch (the Week-2 deliverable artifact)

**Files:**
- Create: `scripts/visualize_batch.py`

**Interfaces:**
- Consumes: the dataset stack from Tasks 1–3. Produces `docs/sample_batch.png`.

- [ ] **Step 1: Write `scripts/visualize_batch.py`** (saves a grid PNG; headless — no notebook needed)

```python
"""Save a grid of sample images with their captions to docs/sample_batch.png."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from visionsearch.config import CONFIG
from visionsearch.data.flickr30k import load_annotations
from visionsearch.data.dataset import Flickr30kDataset
from visionsearch.data.transforms import build_transform

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def main() -> None:
    data_dir = CONFIG.data_dir / "flickr30k"
    anns = load_annotations(data_dir / "flickr_annotations_30k.csv", split="train")
    ds = Flickr30kDataset(data_dir / "images", anns[:8], build_transform(train=False), train=False)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (img, caption, _) in zip(axes.flat, ds):
        ax.imshow((img * _STD + _MEAN).clamp(0, 1).permute(1, 2, 0).numpy())
        ax.set_title(caption[:50] + ("…" if len(caption) > 50 else ""), fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    out = CONFIG.data_dir.parent / "docs" / "sample_batch.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Install matplotlib (visualization-only dep) and add to requirements**

```bash
.venv/Scripts/python.exe -m pip install matplotlib
```
Append `matplotlib` to `requirements.txt` under a `# Visualization` comment.

- [ ] **Step 3: Run it and verify the artifact**

Run: `.venv/Scripts/python.exe scripts/visualize_batch.py`
Expected: prints `saved .../docs/sample_batch.png`; the PNG shows 8 real images with captions.

- [ ] **Step 4: Commit + push** (the PNG is a deliverable, so it IS committed despite data/ being gitignored)

```bash
git add scripts/visualize_batch.py requirements.txt docs/sample_batch.png
git commit -m "feat: batch visualization (Week 2 deliverable)"
git push origin main
```

---

## Self-Review Notes

- **Spec coverage (Week 2):** download → Task 1/3; Dataset+DataLoader+transforms+tokenization → Task 2; leakage-free split → Task 1 (`test_splits_are_leakage_free`, and dataset-native splits); visualize a batch → Task 4. All roadmap Week-2 bullets covered.
- **No pandas / no `datasets`:** enforced in Global Constraints; CSV parsed with stdlib only.
- **Tests independent of 4GB download:** Tasks 1–2 use the `tiny_dataset` fixture; the real download is isolated to Task 3.
- **Type consistency:** `Annotation` fields and the collate batch keys (`pixel_values`, `input_ids`, `attention_mask`, `img_id`) are defined once and reused verbatim in tests and the visualize/smoke scripts.
