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
