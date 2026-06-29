# VisionSearch Week 5 — Full Training & Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Train the dual-encoder on the full Flickr30k train split, evaluate validation Recall@K each epoch, checkpoint the best model, and produce loss + recall curves.

**Architecture:** Direct training (frozen backbones run live each step) — benchmarked at ~3 min/epoch on the RTX 5060 with batch 32 + AMP, so feature precomputation is unnecessary here (it returns in Week 7 for the FAISS gallery). A minimal `eval/recall.py` computes text→image Recall@K on the val split. `fit` gains an `on_epoch_end` callback so a single persistent optimizer spans all epochs while we evaluate + checkpoint between them.

**Tech Stack:** torch, transformers, matplotlib, tensorboard, pytest.

## Global Constraints

- Direct training; backbones frozen (Weeks 3–4). Effective batch via `--accum` (e.g. 32×4=128).
- **Checkpoint the best model by validation Recall@1** (text→image).
- Recall@K uses the simple index-aligned protocol (1 caption/image, eval transform). The full
  5-captions-per-image protocol + raw-CLIP baseline is Week 6.
- Keep a SINGLE optimizer across epochs (don't recreate it per epoch — that resets AdamW state).
- AMP on cuda only; `num_workers=0` (Windows DataLoader-worker pitfall).
- Use `.venv/Scripts/python.exe`. Push after the week. Checkpoints + runs/ stay gitignored; curves PNG is committed.

---

### Task 1: Recall@K evaluation

**Files:**
- Create: `src/visionsearch/eval/recall.py`
- Create: `tests/test_recall.py`

**Interfaces:**
- Produces:
  - `recall_at_k(image_embeds: Tensor[N,D], text_embeds: Tensor[N,D], ks=(1,5,10)) -> dict[str,float]` —
    text→image retrieval; correct image for text i is image i. Returns `{"R@1":..,"R@5":..,"R@10":..}`.
  - `@torch.no_grad() encode_val(model, loader, device) -> tuple[Tensor[N,D], Tensor[N,D]]` returning
    `(image_embeds, text_embeds)` aligned by dataset order.

- [ ] **Step 1: Write failing tests** (`tests/test_recall.py`)

```python
import torch
import torch.nn.functional as F

from visionsearch.eval.recall import recall_at_k


def test_perfect_alignment_is_full_recall():
    e = F.normalize(torch.randn(20, 16), dim=-1)
    r = recall_at_k(e, e)  # image i == text i exactly
    assert r["R@1"] == 1.0 and r["R@5"] == 1.0 and r["R@10"] == 1.0


def test_random_is_low_recall():
    torch.manual_seed(0)
    img = F.normalize(torch.randn(100, 16), dim=-1)
    txt = F.normalize(torch.randn(100, 16), dim=-1)
    r = recall_at_k(img, txt)
    assert r["R@1"] < 0.2  # chance-level for 100 items


def test_recall_monotonic_in_k():
    torch.manual_seed(1)
    img = F.normalize(torch.randn(50, 16), dim=-1)
    txt = F.normalize(torch.randn(50, 16), dim=-1)
    r = recall_at_k(img, txt)
    assert r["R@1"] <= r["R@5"] <= r["R@10"]
```

- [ ] **Step 2: Run, expect failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_recall.py -q`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `src/visionsearch/eval/recall.py`**

```python
"""Recall@K for text→image retrieval (the metric we optimize and report)."""
from __future__ import annotations

import torch
from torch import Tensor


def recall_at_k(image_embeds: Tensor, text_embeds: Tensor, ks=(1, 5, 10)) -> dict[str, float]:
    """Index-aligned protocol: the correct image for text i is image i.

    For each text query we rank all images by cosine similarity and check whether
    the matching image lands in the top K.
    """
    sims = text_embeds @ image_embeds.t()                  # [N_text, N_img]
    targets = torch.arange(sims.size(0), device=sims.device)
    ranking = sims.argsort(dim=1, descending=True)         # most similar first
    out: dict[str, float] = {}
    for k in ks:
        topk = ranking[:, :k]
        hit = (topk == targets.unsqueeze(1)).any(dim=1).float().mean().item()
        out[f"R@{k}"] = hit
    return out


@torch.no_grad()
def encode_val(model, loader, device) -> tuple[Tensor, Tensor]:
    """Embed a validation loader into aligned (image_embeds, text_embeds)."""
    model.eval()
    imgs, txts = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        image_embeds, text_embeds = model(batch)
        imgs.append(image_embeds.cpu())
        txts.append(text_embeds.cpu())
    model.train()
    return torch.cat(imgs), torch.cat(txts)
```

- [ ] **Step 4: Run tests, expect pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_recall.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit + push**

```bash
git add src/visionsearch/eval/recall.py tests/test_recall.py
git commit -m "feat: Recall@K (text->image) eval"
git push origin main
```

---

### Task 2: Epoch callback in fit + training CLI with val recall + best checkpoint

**Files:**
- Modify: `src/visionsearch/train/trainer.py` (add `on_epoch_end` callback)
- Modify: `tests/test_trainer.py` (add callback-fires test)
- Modify: `scripts/train.py` (per-epoch val recall, best checkpoint, curves)

**Interfaces:**
- `fit(..., on_epoch_end: Callable[[int], None] | None = None)` — called after each epoch with the
  epoch index; the single optimizer persists across all epochs.

- [ ] **Step 1: Add a failing callback test to `tests/test_trainer.py`**

```python
def test_on_epoch_end_fires_each_epoch():
    batch = {
        "pixel_values": torch.randn(4, 3, 224, 224),
        "input_ids": torch.randint(0, 1000, (4, 8)),
        "attention_mask": torch.ones(4, 8, dtype=torch.long),
    }
    model = DualEncoder()
    loss_fn = ContrastiveLoss()
    seen = []
    fit(model, loss_fn, [batch], epochs=3, device=CONFIG.device, use_amp=False,
        on_epoch_end=lambda e: seen.append(e))
    assert seen == [0, 1, 2]
```

- [ ] **Step 2: Run, expect failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_trainer.py::test_on_epoch_end_fires_each_epoch -q`
Expected: TypeError (unexpected keyword `on_epoch_end`).

- [ ] **Step 3: Add the callback to `fit`** in `src/visionsearch/train/trainer.py`

Add the parameter to the signature:
```python
    log_dir: str | None = None,
    max_steps: int | None = None,
    on_epoch_end=None,
) -> list[float]:
```
And at the end of each epoch's inner loop (after the `for i, batch ...` loop body, before the
`if max_steps ... break` epoch-level check), call it:
```python
        if on_epoch_end is not None:
            on_epoch_end(epoch)
```
Change the outer loop to enumerate epochs:
```python
    for epoch in range(epochs):
        for i, batch in enumerate(batches):
            ...
        if on_epoch_end is not None:
            on_epoch_end(epoch)
        if max_steps is not None and step >= max_steps:
            break
```

- [ ] **Step 4: Run callback test + existing overfit test, expect pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_trainer.py -q`
Expected: 2 passed.

- [ ] **Step 5: Rewrite `scripts/train.py`** to evaluate val recall per epoch and checkpoint the best

```python
"""Train VisionSearch on the full Flickr30k train split with val-recall checkpointing.

Usage: .venv/Scripts/python.exe scripts/train.py --epochs 5 --batch-size 32 --accum 4
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from visionsearch.config import CONFIG
from visionsearch.data.dataset import Flickr30kDataset, make_collate_fn
from visionsearch.data.flickr30k import load_annotations
from visionsearch.data.transforms import build_transform
from visionsearch.eval.recall import encode_val, recall_at_k
from visionsearch.models.dual_encoder import DualEncoder
from visionsearch.train.loss import ContrastiveLoss
from visionsearch.train.trainer import fit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    data_dir = CONFIG.data_dir / "flickr30k"
    csv = data_dir / "flickr_annotations_30k.csv"
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    collate = make_collate_fn(tok)

    train_ds = Flickr30kDataset(data_dir / "images", load_annotations(csv, "train"),
                                build_transform(train=True), train=True)
    val_ds = Flickr30kDataset(data_dir / "images", load_annotations(csv, "val"),
                              build_transform(train=False), train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                            num_workers=0, collate_fn=collate)

    model, loss_fn = DualEncoder(), ContrastiveLoss()
    CONFIG.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt = CONFIG.checkpoint_dir / "visionsearch.pt"

    recall_history: list[dict] = []
    best = {"R@1": -1.0}

    def on_epoch_end(epoch: int) -> None:
        nonlocal best
        img_e, txt_e = encode_val(model, val_loader, CONFIG.device)
        rec = recall_at_k(img_e, txt_e)
        recall_history.append(rec)
        print(f"epoch {epoch}: val {rec}")
        if rec["R@1"] > best["R@1"]:
            best = rec
            torch.save({
                "image_head": model.image_head.state_dict(),
                "text_head": model.text_head.state_dict(),
                "logit_scale": loss_fn.logit_scale.detach().cpu(),
                "val_recall": rec,
            }, ckpt)
            print(f"  saved best -> {ckpt}")

    loss_history = fit(model, loss_fn, train_loader, epochs=args.epochs, lr=args.lr,
                       device=CONFIG.device, accum_steps=args.accum,
                       use_amp=(CONFIG.device == "cuda"), log_dir="runs/train",
                       max_steps=args.max_steps, on_epoch_end=on_epoch_end)

    # curves: loss (per step) + val R@1/5/10 (per epoch)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
    a1.plot(loss_history); a1.set_title("Train InfoNCE loss"); a1.set_xlabel("step")
    if recall_history:
        for k in ("R@1", "R@5", "R@10"):
            a2.plot([r[k] for r in recall_history], marker="o", label=k)
        a2.set_title("Val recall (text->image)"); a2.set_xlabel("epoch"); a2.legend()
    fig.tight_layout()
    out = CONFIG.data_dir.parent / "docs" / "training_curves.png"
    fig.savefig(out, dpi=110)
    print(f"best val {best}; curves -> {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Smoke-test the CLI** (1 epoch capped at a few steps; just checks the wiring + curve)

Run: `.venv/Scripts/python.exe scripts/train.py --epochs 1 --batch-size 8 --accum 1 --max-steps 3`
Expected: prints an `epoch 0: val {...}` line, saves checkpoint + `docs/training_curves.png`.

- [ ] **Step 7: Commit + push**

```bash
git add src/visionsearch/train/trainer.py tests/test_trainer.py scripts/train.py
git commit -m "feat: per-epoch val-recall checkpointing + training curves"
git push origin main
```

---

### Task 3: Full training run (the deliverable)

**Files:**
- Modify: `docs/training_curves.png` (regenerated by the real run)

- [ ] **Step 1: Run full training** (~3 min/epoch + val; runs in the background)

Run: `.venv/Scripts/python.exe scripts/train.py --epochs 5 --batch-size 32 --accum 4 --lr 1e-3`
Expected: 5 `epoch N: val {...}` lines with R@1 rising above chance (≈0.1% for 1014 imgs, so any
R@1 in the double-digit % is a real signal); best checkpoint saved; `docs/training_curves.png` shows
falling loss + rising recall.

- [ ] **Step 2: Inspect the curves** — open `docs/training_curves.png`; confirm loss falls and val recall climbs then plateaus.

- [ ] **Step 3: Record the final numbers** — note best val R@1/5/10 (used as the Week-6 result vs the CLIP baseline).

- [ ] **Step 4: Commit + push** (curves are the deliverable; checkpoint stays gitignored)

```bash
git add docs/training_curves.png
git commit -m "feat: full training run — loss + val recall curves (Week 5 deliverable)"
git push origin main
```

---

## Self-Review Notes

- **Spec coverage (Week 5):** train on full dataset → Task 3; tune lr/temp/batch via CLI flags → Task 2; OOM playbook documented in constraints (lower batch→seq→res); checkpoint best by val recall → Task 1+2; logged runs → TensorBoard `runs/train` + curves PNG. All roadmap Week-5 bullets covered.
- **Optimizer persistence:** the `on_epoch_end` callback keeps a single optimizer across epochs (vs recreating per epoch) — explicitly addressed.
- **No placeholders:** all code complete; `--max-steps` enables a fast smoke test before the real run.
- **Type consistency:** `recall_at_k`/`encode_val` signatures match their use in `train.py`; checkpoint dict keys (`image_head`, `text_head`, `logit_scale`) match Week 4's format, plus `val_recall`.
```
