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
