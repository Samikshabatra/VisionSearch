"""Dual-encoder: frozen backbones + from-scratch heads → shared L2-normalized space."""
from __future__ import annotations

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
