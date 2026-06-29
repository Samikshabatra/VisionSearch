"""FAISS index over L2-normalized image embeddings (inner product = cosine)."""
from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

_INDEX_FILE = "gallery.faiss"
_META_FILE = "gallery_meta.json"


class ImageIndex:
    def __init__(self, index: "faiss.Index", filenames: list[str]) -> None:
        self.index = index
        self.filenames = filenames

    @classmethod
    def build(cls, embeds: np.ndarray, filenames: list[str]) -> "ImageIndex":
        embeds = np.ascontiguousarray(embeds, dtype="float32")
        index = faiss.IndexFlatIP(embeds.shape[1])     # inner product on unit vectors = cosine
        index.add(embeds)
        return cls(index, list(filenames))

    def search(self, query: np.ndarray, k: int = 10):
        q = np.ascontiguousarray(query, dtype="float32")
        if q.ndim == 1:
            q = q[None, :]
        return self.index.search(q, k)                  # (scores[Q,k], indices[Q,k])

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / _INDEX_FILE))
        (directory / _META_FILE).write_text(json.dumps({"filenames": self.filenames}))

    @classmethod
    def load(cls, directory: Path) -> "ImageIndex":
        directory = Path(directory)
        index = faiss.read_index(str(directory / _INDEX_FILE))
        meta = json.loads((directory / _META_FILE).read_text())
        return cls(index, meta["filenames"])
