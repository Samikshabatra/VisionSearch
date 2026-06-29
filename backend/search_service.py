"""Loads the trained model + FAISS gallery and answers text queries."""
from __future__ import annotations

import torch
from transformers import AutoTokenizer

from visionsearch.config import CONFIG
from visionsearch.index.faiss_index import ImageIndex
from visionsearch.models.dual_encoder import DualEncoder


class SearchService:
    def __init__(self) -> None:
        self.device = CONFIG.device
        self.model = DualEncoder().to(self.device)
        ckpt = torch.load(CONFIG.checkpoint_path, map_location=self.device)
        self.model.image_head.load_state_dict(ckpt["image_head"])
        self.model.text_head.load_state_dict(ckpt["text_head"])
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.index = ImageIndex.load(CONFIG.gallery_dir)

    @torch.no_grad()
    def search(self, query: str, k: int = 12) -> list[dict]:
        tokens = self.tokenizer([query], return_tensors="pt").to(self.device)
        q = self.model.encode_text(tokens["input_ids"], tokens["attention_mask"]).cpu().numpy()
        scores, ids = self.index.search(q, k)
        out = []
        for score, idx in zip(scores[0], ids[0]):
            name = self.index.filenames[int(idx)]
            out.append({"filename": name, "url": f"/images/{name}", "score": round(float(score), 4)})
        return out

    @property
    def gallery_size(self) -> int:
        return len(self.index.filenames)
