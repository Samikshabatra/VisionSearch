# VisionSearch Week 3 — Model Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A dual-encoder that maps images and text into a shared L2-normalized 256-d space, built on frozen ViT-B/32 + DistilBERT backbones with from-scratch projection heads.

**Architecture:** `ImageEncoder` (frozen timm `vit_base_patch32_224`, 768-d) and `TextEncoder` (frozen DistilBERT, mean-pooled 768-d) are borrowed and frozen. `ProjectionHead` (from scratch — a 2-layer MLP) maps each 768-d feature into the shared 256-d space. `DualEncoder` wires them together and L2-normalizes the outputs so cosine similarity is a dot product. The learnable temperature and loss are Week 4.

**Tech Stack:** timm, transformers (DistilBERT), torch.nn, pytest.

## Global Constraints

- Image backbone: **`vit_base_patch32_224`** via timm, `pretrained=True, num_classes=0` → (B, 768). **Frozen.**
- Text backbone: **`distilbert-base-uncased`** via `transformers.AutoModel` → `last_hidden_state`, **mean-pooled over `attention_mask`** → (B, 768). **Frozen.**
- Shared embedding dim = **`CONFIG.embed_dim` (256)**. Outputs are **L2-normalized** (ranking is cosine = dot product).
- Projection heads are the ONLY trainable parameters this week → implement as small MLPs (Linear→GELU→Linear + LayerNorm), because frozen features need more adaptation than a bare linear projection.
- Backbones must stay in `eval()` even when the model is in train mode (dropout off) — override `train()`.
- Use `.venv/Scripts/python.exe`. Tests load cached pretrained weights (already downloaded). Push after the week.

---

### Task 1: Frozen encoders (image + text)

**Files:**
- Create: `src/visionsearch/models/encoders.py`
- Create: `tests/test_encoders.py`

**Interfaces:**
- Produces:
  - `ImageEncoder()` with `.out_dim == 768`; `forward(pixel_values: Tensor[B,3,224,224]) -> Tensor[B,768]`.
  - `TextEncoder()` with `.out_dim == 768`; `forward(input_ids, attention_mask) -> Tensor[B,768]`.
  - Both frozen (all backbone params `requires_grad == False`); backbone stays in eval mode under `.train()`.

- [ ] **Step 1: Write failing tests** (`tests/test_encoders.py`)

```python
import torch

from visionsearch.models.encoders import ImageEncoder, TextEncoder


def test_image_encoder_shape_and_frozen():
    enc = ImageEncoder()
    assert enc.out_dim == 768
    out = enc(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 768)
    assert all(not p.requires_grad for p in enc.parameters())


def test_image_backbone_stays_eval_in_train_mode():
    enc = ImageEncoder()
    enc.train()  # put module in train mode
    assert not enc.backbone.training  # backbone must remain eval (dropout off)


def test_text_encoder_shape_and_frozen():
    enc = TextEncoder()
    assert enc.out_dim == 768
    input_ids = torch.randint(0, 100, (2, 7))
    attention_mask = torch.ones(2, 7, dtype=torch.long)
    out = enc(input_ids, attention_mask)
    assert out.shape == (2, 768)
    assert all(not p.requires_grad for p in enc.parameters())
```

- [ ] **Step 2: Run, expect failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_encoders.py -q`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `src/visionsearch/models/encoders.py`**

```python
"""Borrowed, frozen backbones. We train only the projection heads (Week 3) on top.

Image: timm ViT-B/32 (768-d). Text: DistilBERT, mean-pooled over the mask (768-d).
"""
from __future__ import annotations

import timm
import torch
from torch import nn
from transformers import AutoModel

_TEXT_MODEL = "distilbert-base-uncased"


def _freeze(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad_(False)
    module.eval()


class ImageEncoder(nn.Module):
    """Frozen ViT-B/32 feature extractor → (B, 768)."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "vit_base_patch32_224", pretrained=True, num_classes=0
        )
        self.out_dim = self.backbone.num_features  # 768
        _freeze(self.backbone)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()  # keep frozen backbone in eval (no dropout)
        return self

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.backbone(pixel_values)


class TextEncoder(nn.Module):
    """Frozen DistilBERT, mean-pooled over the attention mask → (B, 768)."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(_TEXT_MODEL)
        self.out_dim = self.backbone.config.dim  # 768
        _freeze(self.backbone)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts
```

- [ ] **Step 4: Run tests, expect pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_encoders.py -q`
Expected: 3 passed.

- [ ] **Step 5: Commit + push**

```bash
git add src/visionsearch/models/encoders.py tests/test_encoders.py
git commit -m "feat: frozen ViT-B/32 + DistilBERT encoders"
git push origin main
```

---

### Task 2: Projection heads (from scratch)

**Files:**
- Create: `src/visionsearch/models/heads.py`
- Create: `tests/test_heads.py`

**Interfaces:**
- Produces: `ProjectionHead(in_dim: int, embed_dim: int, hidden_dim: int | None = None)`;
  `forward(x: Tensor[B,in_dim]) -> Tensor[B,embed_dim]`. All parameters trainable.

- [ ] **Step 1: Write failing tests** (`tests/test_heads.py`)

```python
import torch

from visionsearch.models.heads import ProjectionHead


def test_projection_shape():
    head = ProjectionHead(in_dim=768, embed_dim=256)
    out = head(torch.randn(4, 768))
    assert out.shape == (4, 256)


def test_projection_is_trainable():
    head = ProjectionHead(in_dim=768, embed_dim=256)
    assert all(p.requires_grad for p in head.parameters())
    # gradient actually flows
    out = head(torch.randn(2, 768)).sum()
    out.backward()
    assert any(p.grad is not None for p in head.parameters())
```

- [ ] **Step 2: Run, expect failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_heads.py -q`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `src/visionsearch/models/heads.py`**

```python
"""From-scratch projection head: maps a frozen backbone feature into the shared space.

A 2-layer MLP (not a bare linear) because the backbones are frozen — the head is the
only place the image/text features can be adapted to align with each other.
"""
from __future__ import annotations

from torch import nn, Tensor


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        hidden_dim = hidden_dim or embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.net(x))
```

- [ ] **Step 4: Run tests, expect pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_heads.py -q`
Expected: 2 passed.

- [ ] **Step 5: Commit + push**

```bash
git add src/visionsearch/models/heads.py tests/test_heads.py
git commit -m "feat: from-scratch MLP projection head"
git push origin main
```

---

### Task 3: DualEncoder (shared space, L2-normalized)

**Files:**
- Create: `src/visionsearch/models/dual_encoder.py`
- Create: `tests/test_dual_encoder.py`

**Interfaces:**
- Consumes: `ImageEncoder`, `TextEncoder` (Task 1), `ProjectionHead` (Task 2), `CONFIG.embed_dim`.
- Produces: `DualEncoder(embed_dim: int = CONFIG.embed_dim)` with:
  - `encode_image(pixel_values) -> Tensor[B,embed_dim]` (L2-normalized)
  - `encode_text(input_ids, attention_mask) -> Tensor[B,embed_dim]` (L2-normalized)
  - `forward(batch: dict) -> tuple[Tensor, Tensor]` returning `(image_embeds, text_embeds)` from a
    collate batch dict (keys `pixel_values`, `input_ids`, `attention_mask`).
  - `trainable_parameters()` yields only head params.

- [ ] **Step 1: Write failing tests** (`tests/test_dual_encoder.py`)

```python
import torch

from visionsearch.config import CONFIG
from visionsearch.models.dual_encoder import DualEncoder


def _fake_batch(b=3, seq=7):
    return {
        "pixel_values": torch.randn(b, 3, 224, 224),
        "input_ids": torch.randint(0, 100, (b, seq)),
        "attention_mask": torch.ones(b, seq, dtype=torch.long),
    }


def test_forward_shapes_and_shared_dim():
    model = DualEncoder()
    img, txt = model(_fake_batch())
    assert img.shape == (3, CONFIG.embed_dim)
    assert txt.shape == (3, CONFIG.embed_dim)


def test_outputs_are_l2_normalized():
    model = DualEncoder()
    img, txt = model(_fake_batch())
    assert torch.allclose(img.norm(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(txt.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_only_heads_are_trainable():
    model = DualEncoder()
    trainable = [p for p in model.parameters() if p.requires_grad]
    head_params = list(model.image_head.parameters()) + list(model.text_head.parameters())
    # every trainable param belongs to a head; backbones contribute none
    assert sum(p.numel() for p in trainable) == sum(p.numel() for p in head_params)
```

- [ ] **Step 2: Run, expect failure**

Run: `.venv/Scripts/python.exe -m pytest tests/test_dual_encoder.py -q`
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `src/visionsearch/models/dual_encoder.py`**

```python
"""Dual-encoder: frozen backbones + from-scratch heads → shared L2-normalized space."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from ..config import CONFIG
from .encoders import ImageEncoder, TextEncoder
from .heads import ProjectionHead


class DualEncoder(nn.Module):
    def __init__(self, embed_dim: int = CONFIG.embed_dim) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_head = ProjectionHead(self.image_encoder.out_dim, embed_dim)
        self.text_head = ProjectionHead(self.text_encoder.out_dim, embed_dim)

    def encode_image(self, pixel_values: Tensor) -> Tensor:
        feats = self.image_encoder(pixel_values)
        return F.normalize(self.image_head(feats), dim=-1)

    def encode_text(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        feats = self.text_encoder(input_ids, attention_mask)
        return F.normalize(self.text_head(feats), dim=-1)

    def forward(self, batch: dict) -> tuple[Tensor, Tensor]:
        image_embeds = self.encode_image(batch["pixel_values"])
        text_embeds = self.encode_text(batch["input_ids"], batch["attention_mask"])
        return image_embeds, text_embeds

    def trainable_parameters(self):
        yield from self.image_head.parameters()
        yield from self.text_head.parameters()
```

- [ ] **Step 4: Run tests, expect pass**

Run: `.venv/Scripts/python.exe -m pytest tests/test_dual_encoder.py -q`
Expected: 3 passed.

- [ ] **Step 5: Real-batch forward smoke test** (uses the actual dataset + model on GPU)

```bash
.venv/Scripts/python.exe - <<'PY'
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from visionsearch.config import CONFIG
from visionsearch.data.flickr30k import load_annotations
from visionsearch.data.dataset import Flickr30kDataset, make_collate_fn
from visionsearch.data.transforms import build_transform
from visionsearch.models.dual_encoder import DualEncoder

data_dir = CONFIG.data_dir / "flickr30k"
anns = load_annotations(data_dir / "flickr_annotations_30k.csv", split="train")
ds = Flickr30kDataset(data_dir / "images", anns, build_transform(train=True), train=True)
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
dl = DataLoader(ds, batch_size=8, shuffle=True, collate_fn=make_collate_fn(tok))
batch = {k: v.to(CONFIG.device) for k, v in next(iter(dl)).items()}

model = DualEncoder().to(CONFIG.device)
img, txt = model(batch)
sim = img @ txt.t()  # 8x8 cosine similarity matrix
print("device:", CONFIG.device)
print("image_embeds", tuple(img.shape), "text_embeds", tuple(txt.shape))
print("similarity matrix", tuple(sim.shape), "diag mean", sim.diag().mean().item())
PY
```
Expected: `image_embeds (8, 256) text_embeds (8, 256)`, `similarity matrix (8, 8)` on `cuda`.
(The diagonal won't be high yet — the heads are untrained. That's Week 4's job.)

- [ ] **Step 6: Commit + push**

```bash
git add src/visionsearch/models/dual_encoder.py tests/test_dual_encoder.py
git commit -m "feat: DualEncoder into shared L2-normalized space"
git push origin main
```

---

## Self-Review Notes

- **Spec coverage (Week 3):** load frozen backbones → Task 1; projection heads (from scratch) → Task 2; dual-encoder forward + shape-assertion tests → Task 3 (+ real-batch GPU smoke test). All roadmap Week-3 bullets covered.
- **Temperature/loss deferred to Week 4** by design — Week 3 produces embeddings only.
- **No placeholders:** all code complete. The `@torch.no_grad()` on encoder forwards is intentional (frozen backbones never need grad), and is compatible with training the heads (grad starts at the head inputs).
- **Type consistency:** batch dict keys (`pixel_values`, `input_ids`, `attention_mask`) match Week 2's `make_collate_fn` output; `out_dim`/`embed_dim` names consistent across encoders, heads, and DualEncoder.
```
