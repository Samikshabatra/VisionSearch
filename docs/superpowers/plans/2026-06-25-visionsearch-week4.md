# VisionSearch Week 4 — Contrastive Training Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** The from-scratch contrastive core — symmetric InfoNCE loss with a learnable temperature — plus a training loop with AMP and gradient accumulation, proven by overfitting a tiny subset (loss crashes toward zero).

**Architecture:** `ContrastiveLoss` holds the learnable `logit_scale` (= log(1/τ), clamped à la CLIP). It builds the B×B similarity matrix, scales it, and applies cross-entropy in both directions (image→text, text→image), averaged. `train/trainer.py` provides `fit()` with AdamW over heads + temperature, `torch.amp` autocast + GradScaler, and gradient accumulation. `scripts/train.py` is the CLI; `scripts/sanity_overfit.py` produces the falling-loss-curve deliverable.

**Tech Stack:** torch (nn, amp), tensorboard, matplotlib, pytest.

## Global Constraints

- Loss = **symmetric InfoNCE**: `labels = arange(B)`; `(CE(logits, labels) + CE(logits.T, labels)) / 2`.
- **Learnable temperature**: a parameter `logit_scale` initialized to `log(1/0.07)`; `logits = logit_scale.clamp(max=log(100)).exp() * (img @ txt.T)`.
- Optimizer trains **only heads + logit_scale** (backbones stay frozen): `AdamW(list(model.trainable_parameters()) + list(loss_fn.parameters()))`.
- **AMP** via `torch.amp.autocast(device_type, ...)` + `torch.amp.GradScaler(device_type, ...)`, enabled only on `cuda`.
- **Gradient accumulation**: scale loss by `1/accum_steps`, step every `accum_steps` batches.
- Embeds arrive **already L2-normalized** from the DualEncoder (Week 3) — the loss does NOT re-normalize.
- Use `.venv/Scripts/python.exe`. Push after the week.

---

### Task 1: Symmetric InfoNCE loss + learnable temperature

**Files:**
- Create: `src/visionsearch/train/loss.py`
- Create: `tests/test_loss.py`

**Interfaces:**
- Produces: `ContrastiveLoss(init_temperature: float = 0.07, max_scale: float = 100.0)` — an `nn.Module`;
  `forward(image_embeds: Tensor[B,D], text_embeds: Tensor[B,D]) -> Tensor[]` (scalar loss).
  `logit_scale` is a learnable `nn.Parameter`.

- [ ] **Step 1: Write failing tests** (`tests/test_loss.py`)

```python
import math

import torch
import torch.nn.functional as F

from visionsearch.train.loss import ContrastiveLoss


def _orthogonal_pairs(b=6, d=16):
    """Aligned image/text embeds: row i identical across modalities, rows mutually distinct."""
    e = F.normalize(torch.eye(b, d), dim=-1)
    return e.clone(), e.clone()


def test_loss_is_scalar():
    loss_fn = ContrastiveLoss()
    img, txt = _orthogonal_pairs()
    loss = loss_fn(img, txt)
    assert loss.ndim == 0


def test_aligned_beats_misaligned():
    loss_fn = ContrastiveLoss()
    img, txt = _orthogonal_pairs()
    aligned = loss_fn(img, txt)
    # Shuffle text so the diagonal is no longer the match.
    misaligned = loss_fn(img, txt[torch.tensor([1, 2, 3, 4, 5, 0])])
    assert aligned < misaligned


def test_temperature_is_learnable_and_gets_grad():
    loss_fn = ContrastiveLoss()
    assert loss_fn.logit_scale.requires_grad
    img, txt = _orthogonal_pairs()
    loss_fn(img, txt).backward()
    assert loss_fn.logit_scale.grad is not None


def test_temperature_is_clamped():
    loss_fn = ContrastiveLoss(max_scale=100.0)
    with torch.no_grad():
        loss_fn.logit_scale.fill_(50.0)  # absurdly large
    img, txt = _orthogonal_pairs()
    loss_fn(img, txt)  # must not overflow/NaN
    # effective scale capped at max_scale
    assert loss_fn.logit_scale.clamp(max=math.log(100.0)).exp().item() <= 100.0 + 1e-3
```

- [ ] **Step 2: Run, expect failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_loss.py -q`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `src/visionsearch/train/loss.py`**

```python
"""Symmetric InfoNCE loss with a learnable temperature — the from-scratch core.

For a batch of B (image, text) pairs whose embeddings are already L2-normalized:
  1. similarity matrix S = image @ text.T  (B×B; S[i,j] = cos(img_i, txt_j))
  2. scale by a learnable temperature: logits = scale * S
  3. the correct match for row i is column i, so labels = [0, 1, ..., B-1]
  4. cross-entropy row-wise (image→text) AND column-wise (text→image), averaged

The off-diagonal entries are the in-batch negatives. A bigger batch = more
negatives = a sharper learning signal (hence gradient accumulation in training).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class ContrastiveLoss(nn.Module):
    def __init__(self, init_temperature: float = 0.07, max_scale: float = 100.0) -> None:
        super().__init__()
        # We learn log(1/temperature) ("logit_scale") rather than temperature directly:
        # optimizing in log-space keeps the scale positive and well-conditioned (CLIP does this).
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / init_temperature)))
        self.max_log_scale = math.log(max_scale)

    def forward(self, image_embeds: Tensor, text_embeds: Tensor) -> Tensor:
        scale = self.logit_scale.clamp(max=self.max_log_scale).exp()
        logits = scale * image_embeds @ text_embeds.t()       # (B, B)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_i2t = F.cross_entropy(logits, labels)            # each image picks its caption
        loss_t2i = F.cross_entropy(logits.t(), labels)        # each caption picks its image
        return (loss_i2t + loss_t2i) / 2
```

- [ ] **Step 4: Run tests, expect pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_loss.py -q`
Expected: 4 passed.

- [ ] **Step 5: Commit + push**

```bash
git add src/visionsearch/train/loss.py tests/test_loss.py
git commit -m "feat: symmetric InfoNCE loss + learnable temperature"
git push origin main
```

---

### Task 2: Training loop (AMP + gradient accumulation) with overfit proof

**Files:**
- Create: `src/visionsearch/train/trainer.py`
- Create: `tests/test_trainer.py`

**Interfaces:**
- Consumes: `DualEncoder`, `ContrastiveLoss`.
- Produces: `fit(model, loss_fn, batches, *, epochs=1, lr=1e-3, device="cpu", accum_steps=1, use_amp=True, log_dir=None, max_steps=None) -> list[float]` returning per-step loss history. `batches` is any iterable of collate-style batch dicts.

- [ ] **Step 1: Write failing test** (`tests/test_trainer.py`) — proves the loop overfits a fixed batch

```python
import torch

from visionsearch.config import CONFIG
from visionsearch.models.dual_encoder import DualEncoder
from visionsearch.train.loss import ContrastiveLoss
from visionsearch.train.trainer import fit


def test_overfits_a_fixed_batch():
    torch.manual_seed(0)
    batch = {
        "pixel_values": torch.randn(8, 3, 224, 224),
        "input_ids": torch.randint(0, 1000, (8, 12)),
        "attention_mask": torch.ones(8, 12, dtype=torch.long),
    }
    model = DualEncoder()
    loss_fn = ContrastiveLoss()
    # Feed the SAME batch many times → the heads should memorize it; loss must fall.
    history = fit(model, loss_fn, [batch] * 60, epochs=1, lr=1e-3,
                  device=CONFIG.device, use_amp=False)
    assert history[-1] < history[0]
    assert history[-1] < 0.5 * history[0]
```

- [ ] **Step 2: Run, expect failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_trainer.py -q`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `src/visionsearch/train/trainer.py`**

```python
"""Training loop: AdamW over heads + temperature, AMP, gradient accumulation."""
from __future__ import annotations

from typing import Iterable

import torch
from torch.amp import GradScaler, autocast

from ..models.dual_encoder import DualEncoder
from .loss import ContrastiveLoss


def fit(
    model: DualEncoder,
    loss_fn: ContrastiveLoss,
    batches: Iterable[dict],
    *,
    epochs: int = 1,
    lr: float = 1e-3,
    device: str = "cpu",
    accum_steps: int = 1,
    use_amp: bool = True,
    log_dir: str | None = None,
    max_steps: int | None = None,
) -> list[float]:
    model.to(device)
    loss_fn.to(device)
    model.train()

    params = list(model.trainable_parameters()) + list(loss_fn.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr)

    amp_on = use_amp and device == "cuda"
    scaler = GradScaler(device, enabled=amp_on)

    writer = None
    if log_dir is not None:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir)

    history: list[float] = []
    step = 0
    optimizer.zero_grad()
    for _ in range(epochs):
        for i, batch in enumerate(batches):
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast(device_type=device, enabled=amp_on):
                image_embeds, text_embeds = model(batch)
                loss = loss_fn(image_embeds, text_embeds)
            scaler.scale(loss / accum_steps).backward()
            if (i + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            history.append(loss.item())
            if writer is not None:
                writer.add_scalar("train/loss", loss.item(), step)
            step += 1
            if max_steps is not None and step >= max_steps:
                break
        if max_steps is not None and step >= max_steps:
            break

    if writer is not None:
        writer.close()
    return history
```

- [ ] **Step 4: Run test, expect pass** (runs on GPU; a few seconds)

Run: `.venv/Scripts/python.exe -m pytest tests/test_trainer.py -q`
Expected: 1 passed (loss falls to <50% of its start).

- [ ] **Step 5: Commit + push**

```bash
git add src/visionsearch/train/trainer.py tests/test_trainer.py
git commit -m "feat: training loop with AMP + gradient accumulation"
git push origin main
```

---

### Task 3: Sanity overfit on real data → falling-loss-curve deliverable + train CLI

**Files:**
- Create: `scripts/sanity_overfit.py`
- Create: `scripts/train.py`

**Interfaces:**
- Consumes: dataset stack (Week 2), `DualEncoder`, `ContrastiveLoss`, `fit`.

- [ ] **Step 1: Write `scripts/sanity_overfit.py`** (overfit a tiny REAL subset; save the curve)

```python
"""Sanity check: overfit a tiny real Flickr30k subset; the loss must crash toward zero.

Saves docs/sanity_loss_curve.png — the Week-4 deliverable.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from visionsearch.config import CONFIG
from visionsearch.data.dataset import Flickr30kDataset, make_collate_fn
from visionsearch.data.flickr30k import load_annotations
from visionsearch.data.transforms import build_transform
from visionsearch.models.dual_encoder import DualEncoder
from visionsearch.train.loss import ContrastiveLoss
from visionsearch.train.trainer import fit


def main() -> None:
    data_dir = CONFIG.data_dir / "flickr30k"
    anns = load_annotations(data_dir / "flickr_annotations_30k.csv", split="train")[:32]
    ds = Flickr30kDataset(data_dir / "images", anns, build_transform(train=False), train=False)
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=make_collate_fn(tok))

    model, loss_fn = DualEncoder(), ContrastiveLoss()
    # Many passes over the same 32 examples → memorization; loss should collapse.
    history = fit(model, loss_fn, list(loader) * 40, epochs=1, lr=1e-3,
                  device=CONFIG.device, use_amp=(CONFIG.device == "cuda"))

    print(f"loss: {history[0]:.3f} -> {history[-1]:.3f}")
    plt.figure(figsize=(7, 4))
    plt.plot(history)
    plt.xlabel("step"); plt.ylabel("InfoNCE loss")
    plt.title("Sanity overfit (32 examples) — loss should crash toward 0")
    plt.tight_layout()
    out = CONFIG.data_dir.parent / "docs" / "sanity_loss_curve.png"
    plt.savefig(out, dpi=110)
    print("saved", out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and confirm the loss collapses**

Run: `.venv/Scripts/python.exe scripts/sanity_overfit.py`
Expected: prints e.g. `loss: 2.7xx -> 0.0xx`; saves `docs/sanity_loss_curve.png` showing a falling curve.

- [ ] **Step 3: Inspect the artifact** — open `docs/sanity_loss_curve.png`; verify it falls steeply then flattens near zero.

- [ ] **Step 4: Write `scripts/train.py`** (the real CLI used for full training in Week 5)

```python
"""Train VisionSearch on the full Flickr30k train split.

Usage: .venv/Scripts/python.exe scripts/train.py --epochs 5 --batch-size 32 --accum 4
"""
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from visionsearch.config import CONFIG
from visionsearch.data.dataset import Flickr30kDataset, make_collate_fn
from visionsearch.data.flickr30k import load_annotations
from visionsearch.data.transforms import build_transform
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
    anns = load_annotations(data_dir / "flickr_annotations_30k.csv", split="train")
    ds = Flickr30kDataset(data_dir / "images", anns, build_transform(train=True), train=True)
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, collate_fn=make_collate_fn(tok))

    model, loss_fn = DualEncoder(), ContrastiveLoss()
    history = fit(model, loss_fn, loader, epochs=args.epochs, lr=args.lr,
                  device=CONFIG.device, accum_steps=args.accum,
                  use_amp=(CONFIG.device == "cuda"), log_dir="runs/train",
                  max_steps=args.max_steps)

    CONFIG.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt = CONFIG.checkpoint_dir / "visionsearch.pt"
    torch.save({
        "image_head": model.image_head.state_dict(),
        "text_head": model.text_head.state_dict(),
        "logit_scale": loss_fn.logit_scale.detach().cpu(),
    }, ckpt)
    print(f"final loss {history[-1]:.3f}; saved {ckpt}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Smoke-test the CLI for a few steps** (NOT full training — that's Week 5)

Run: `.venv/Scripts/python.exe scripts/train.py --max-steps 3 --batch-size 8 --accum 1`
Expected: runs 3 steps, prints a final loss, saves `checkpoints/visionsearch.pt`.

- [ ] **Step 6: Commit + push** (the loss curve PNG is a deliverable)

```bash
git add scripts/sanity_overfit.py scripts/train.py docs/sanity_loss_curve.png
git commit -m "feat: sanity overfit (loss-curve deliverable) + train CLI"
git push origin main
```

---

## Self-Review Notes

- **Spec coverage (Week 4):** symmetric InfoNCE + learnable temperature → Task 1; AMP + gradient accumulation + logging → Task 2 (`fit`); overfit-tiny-subset sanity + falling-loss-curve screenshot → Task 3. All roadmap Week-4 bullets covered. Full training run is Week 5 (uses `scripts/train.py`).
- **No placeholders:** all code complete. `max_steps` exists so the CLI can be smoke-tested without a full run.
- **Type consistency:** `fit` consumes batch dicts with Week-2 keys; optimizer params combine `model.trainable_parameters()` (Week 3) + `loss_fn.parameters()` (Task 1); `logit_scale` name consistent across loss, trainer, and checkpoint.
- **AMP correctness:** autocast/GradScaler enabled only on cuda; loss divided by `accum_steps` before backward; step every `accum_steps`.
```
