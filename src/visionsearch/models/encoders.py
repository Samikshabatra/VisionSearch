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
