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
