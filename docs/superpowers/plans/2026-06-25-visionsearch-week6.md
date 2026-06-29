# VisionSearch Week 6 — Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Evaluate the trained model with the full Flickr30k retrieval protocol (all 5 captions, text→image AND image→text), compare it head-to-head against raw pretrained CLIP on the same test set, and produce an evaluation report with a metrics table and qualitative examples.

**Architecture:** `eval/retrieval.py` adds the full multi-caption metric. `scripts/evaluate.py` embeds the test split with (a) our trained model (loaded from checkpoint) and (b) raw `openai/clip-vit-base-patch32`, computes both, and writes `docs/eval_report.md`. `scripts/qualitative.py` renders top-k retrievals for a few queries.

**Tech Stack:** torch, transformers (CLIP), matplotlib, pytest.

## Global Constraints

- Full protocol on the **test** split (1000 images, 5000 captions):
  - **text→image**: each caption is a query over the 1000 images; hit if the caption's source image is in top-K.
  - **image→text**: each image is a query over the 5000 captions; hit if ANY of its 5 captions is in top-K.
- Baseline: **raw `openai/clip-vit-base-patch32`** (zero-shot) on the SAME test set.
- **Honesty framing**: our model was *trained* on Flickr30k; CLIP is *zero-shot* — state this in the report; it is "vs off-the-shelf", not a controlled match.
- Load our model from `checkpoints/visionsearch.pt` (head state_dicts; logit_scale irrelevant to ranking).
- `num_workers=0`; AMP/eval under `torch.no_grad()`; `.venv/Scripts/python.exe`. Push after the week.

---

### Task 1: Full retrieval metric (t2i + i2t, multi-caption)

**Files:**
- Create: `src/visionsearch/eval/retrieval.py`
- Create: `tests/test_retrieval.py`

**Interfaces:**
- Produces: `retrieval_recall(image_embeds: Tensor[N,D], text_embeds: Tensor[M,D], text_img_pos: Tensor[M], ks=(1,5,10)) -> dict[str,float]` with keys `t2i_R@{k}` and `i2t_R@{k}`. `text_img_pos[m]` is the image row index (0..N-1) that caption m belongs to.

- [ ] **Step 1: Write failing tests** (`tests/test_retrieval.py`)

```python
import torch
import torch.nn.functional as F

from visionsearch.eval.retrieval import retrieval_recall


def _aligned(n=10, d=16, caps=2):
    """n images (basis-like vectors); each image has `caps` captions identical to it."""
    img = F.normalize(torch.randn(n, d), dim=-1)
    text = img.repeat_interleave(caps, dim=0)              # caption rows clone their image
    text_img_pos = torch.arange(n).repeat_interleave(caps)
    return img, text, text_img_pos


def test_perfect_alignment_full_recall():
    img, text, pos = _aligned()
    r = retrieval_recall(img, text, pos)
    assert r["t2i_R@1"] == 1.0
    assert r["i2t_R@1"] == 1.0


def test_keys_and_monotonic():
    img, text, pos = _aligned(n=30)
    r = retrieval_recall(img, text, pos)
    for d in ("t2i", "i2t"):
        assert r[f"{d}_R@1"] <= r[f"{d}_R@5"] <= r[f"{d}_R@10"]


def test_random_is_low():
    torch.manual_seed(0)
    img = F.normalize(torch.randn(100, 16), dim=-1)
    text = F.normalize(torch.randn(200, 16), dim=-1)
    pos = torch.arange(100).repeat_interleave(2)
    r = retrieval_recall(img, text, pos)
    assert r["t2i_R@1"] < 0.1
```

- [ ] **Step 2: Run, expect failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_retrieval.py -q`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `src/visionsearch/eval/retrieval.py`**

```python
"""Full Flickr30k retrieval protocol: text→image and image→text, 5 captions/image."""
from __future__ import annotations

import torch
from torch import Tensor


def retrieval_recall(image_embeds: Tensor, text_embeds: Tensor,
                     text_img_pos: Tensor, ks=(1, 5, 10)) -> dict[str, float]:
    out: dict[str, float] = {}

    # text → image: each caption ranks the images; correct image = text_img_pos[m]
    t2i = text_embeds @ image_embeds.t()                   # [M, N]
    t2i_rank = t2i.argsort(dim=1, descending=True)
    for k in ks:
        hit = (t2i_rank[:, :k] == text_img_pos.unsqueeze(1)).any(dim=1).float().mean().item()
        out[f"t2i_R@{k}"] = hit

    # image → text: each image ranks the captions; correct = ANY caption of that image
    i2t = image_embeds @ text_embeds.t()                   # [N, M]
    i2t_rank = i2t.argsort(dim=1, descending=True)
    n_images = image_embeds.size(0)
    targets = torch.arange(n_images, device=image_embeds.device).unsqueeze(1)
    for k in ks:
        topk_img = text_img_pos[i2t_rank[:, :k]]           # [N, k] -> image of each retrieved caption
        hit = (topk_img == targets).any(dim=1).float().mean().item()
        out[f"i2t_R@{k}"] = hit

    return out
```

- [ ] **Step 4: Run tests, expect pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_retrieval.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit + push**

```bash
git add src/visionsearch/eval/retrieval.py tests/test_retrieval.py
git commit -m "feat: full retrieval metric (t2i + i2t, multi-caption)"
git push origin main
```

---

### Task 2: Evaluate our model vs raw CLIP → report

**Files:**
- Create: `scripts/evaluate.py`

**Interfaces:**
- Consumes: `DualEncoder`, checkpoint, `retrieval_recall`, dataset annotations; downloads CLIP.

- [ ] **Step 1: Write `scripts/evaluate.py`**

```python
"""Evaluate VisionSearch vs raw CLIP on the Flickr30k test split → docs/eval_report.md."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor

from visionsearch.config import CONFIG
from visionsearch.data.flickr30k import load_annotations
from visionsearch.data.transforms import build_transform
from visionsearch.eval.retrieval import retrieval_recall
from visionsearch.models.dual_encoder import DualEncoder

DEVICE = CONFIG.device
DATA = CONFIG.data_dir / "flickr30k"
IMAGES = DATA / "images"


def _batched(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


@torch.no_grad()
def embed_ours(anns):
    model = DualEncoder().to(DEVICE)
    ckpt = torch.load(CONFIG.checkpoint_dir / "visionsearch.pt", map_location=DEVICE)
    model.image_head.load_state_dict(ckpt["image_head"])
    model.text_head.load_state_dict(ckpt["text_head"])
    model.eval()
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tfm = build_transform(train=False)

    img_embeds = []
    for chunk in _batched(anns, 64):
        px = torch.stack([tfm(Image.open(IMAGES / a.filename).convert("RGB")) for a in chunk]).to(DEVICE)
        img_embeds.append(model.encode_image(px).cpu())
    img_embeds = torch.cat(img_embeds)

    captions, pos = [], []
    for i, a in enumerate(anns):
        for c in a.captions:
            captions.append(c); pos.append(i)
    txt_embeds = []
    for chunk in _batched(captions, 256):
        t = tok(chunk, padding=True, truncation=True, max_length=40, return_tensors="pt").to(DEVICE)
        txt_embeds.append(model.encode_text(t["input_ids"], t["attention_mask"]).cpu())
    txt_embeds = torch.cat(txt_embeds)
    return img_embeds, txt_embeds, torch.tensor(pos)


@torch.no_grad()
def embed_clip(anns):
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    img_embeds = []
    for chunk in _batched(anns, 64):
        imgs = [Image.open(IMAGES / a.filename).convert("RGB") for a in chunk]
        inp = proc(images=imgs, return_tensors="pt").to(DEVICE)
        img_embeds.append(F.normalize(clip.get_image_features(**inp), dim=-1).cpu())
    img_embeds = torch.cat(img_embeds)

    captions, pos = [], []
    for i, a in enumerate(anns):
        for c in a.captions:
            captions.append(c); pos.append(i)
    txt_embeds = []
    for chunk in _batched(captions, 256):
        inp = proc(text=chunk, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        txt_embeds.append(F.normalize(clip.get_text_features(**inp), dim=-1).cpu())
    txt_embeds = torch.cat(txt_embeds)
    return img_embeds, txt_embeds, torch.tensor(pos)


def main() -> None:
    anns = load_annotations(DATA / "flickr_annotations_30k.csv", split="test")
    print(f"test: {len(anns)} images, {sum(len(a.captions) for a in anns)} captions")

    ours = retrieval_recall(*embed_ours(anns))
    clip = retrieval_recall(*embed_clip(anns))

    ks = (1, 5, 10)
    lines = [
        "# VisionSearch — Evaluation Report",
        "",
        "Flickr30k **test** split (1000 images, 5000 captions). Recall@K (%).",
        "",
        "> Note: VisionSearch is **trained** on Flickr30k; raw CLIP is **zero-shot**. "
        "This is a 'vs off-the-shelf' comparison, not a controlled match.",
        "",
        "| Direction | Metric | VisionSearch (ours) | Raw CLIP (zero-shot) |",
        "|---|---|---|---|",
    ]
    for d, label in (("t2i", "text→image"), ("i2t", "image→text")):
        for k in ks:
            key = f"{d}_R@{k}"
            lines.append(f"| {label} | R@{k} | {ours[key]*100:.1f} | {clip[key]*100:.1f} |")
    report = "\n".join(lines) + "\n"

    out = CONFIG.data_dir.parent / "docs" / "eval_report.md"
    out.write_text(report, encoding="utf-8")
    print(report)
    print("saved", out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the evaluation** (downloads CLIP ~600MB first time; embeds test set with both models)

Run: `.venv/Scripts/python.exe scripts/evaluate.py`
Expected: prints the table; saves `docs/eval_report.md`. Our t2i R@1 should be in the ballpark of the val number (~20%); CLIP's zero-shot numbers are the reference point.

- [ ] **Step 3: Sanity-check the report** — open `docs/eval_report.md`; numbers are plausible (recall monotonic in K; both directions filled in).

- [ ] **Step 4: Commit + push**

```bash
git add scripts/evaluate.py docs/eval_report.md
git commit -m "feat: evaluate vs raw CLIP on test split (eval report)"
git push origin main
```

---

### Task 3: Qualitative examples (wins + failures)

**Files:**
- Create: `scripts/qualitative.py`

**Interfaces:**
- Consumes: our model + checkpoint; produces `docs/qualitative_examples.png`.

- [ ] **Step 1: Write `scripts/qualitative.py`** (a few text queries → top-5 retrieved images, correct one outlined)

```python
"""Qualitative retrieval examples from VisionSearch → docs/qualitative_examples.png."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import AutoTokenizer

from visionsearch.config import CONFIG
from visionsearch.data.flickr30k import load_annotations
from visionsearch.data.transforms import build_transform
from visionsearch.models.dual_encoder import DualEncoder

DEVICE = CONFIG.device
DATA = CONFIG.data_dir / "flickr30k"
IMAGES = DATA / "images"
QUERIES = [
    "a dog running on the beach",
    "two people riding bicycles",
    "a child playing with a ball",
    "people sitting at a restaurant table",
]


@torch.no_grad()
def main() -> None:
    anns = load_annotations(DATA / "flickr_annotations_30k.csv", split="test")
    model = DualEncoder().to(DEVICE)
    ckpt = torch.load(CONFIG.checkpoint_dir / "visionsearch.pt", map_location=DEVICE)
    model.image_head.load_state_dict(ckpt["image_head"])
    model.text_head.load_state_dict(ckpt["text_head"])
    model.eval()
    tfm = build_transform(train=False)
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # embed the gallery (all test images)
    embeds = []
    for i in range(0, len(anns), 64):
        chunk = anns[i:i + 64]
        px = torch.stack([tfm(Image.open(IMAGES / a.filename).convert("RGB")) for a in chunk]).to(DEVICE)
        embeds.append(model.encode_image(px).cpu())
    gallery = torch.cat(embeds)

    fig, axes = plt.subplots(len(QUERIES), 5, figsize=(15, 3 * len(QUERIES)))
    for row, q in enumerate(QUERIES):
        t = tok([q], return_tensors="pt").to(DEVICE)
        qe = model.encode_text(t["input_ids"], t["attention_mask"]).cpu()
        top = (qe @ gallery.t())[0].argsort(descending=True)[:5]
        for col, idx in enumerate(top):
            ax = axes[row, col]
            ax.imshow(Image.open(IMAGES / anns[idx].filename).convert("RGB"))
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(q, fontsize=9)
        axes[row, 0].set_title(f'query: "{q}"', loc="left", fontsize=10)
    fig.tight_layout()
    out = CONFIG.data_dir.parent / "docs" / "qualitative_examples.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `.venv/Scripts/python.exe scripts/qualitative.py`
Expected: saves `docs/qualitative_examples.png` — 4 rows, each a query with its top-5 images.

- [ ] **Step 3: Inspect** — open the PNG; confirm the top results are semantically relevant to each query.

- [ ] **Step 4: Commit + push**

```bash
git add scripts/qualitative.py docs/qualitative_examples.png
git commit -m "feat: qualitative retrieval examples"
git push origin main
```

---

## Self-Review Notes

- **Spec coverage (Week 6):** Recall@1/5/10 t2i + i2t → Task 1; raw-CLIP baseline on same test set → Task 2; qualitative wins/failures → Task 3; eval report (table + examples) → Tasks 2+3. All roadmap Week-6 bullets covered.
- **Honesty:** the trained-vs-zero-shot caveat is written into the report itself (Task 2 Step 1).
- **No placeholders:** all code complete. CLIP normalization uses `get_image_features`/`get_text_features` + explicit `F.normalize`.
- **Type consistency:** `retrieval_recall(image_embeds, text_embeds, text_img_pos)` signature matches both `embed_ours`/`embed_clip` returns; checkpoint keys (`image_head`, `text_head`) match Week 5's save format.
```
