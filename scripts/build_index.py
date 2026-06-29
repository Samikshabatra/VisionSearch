"""Embed the test-split gallery once and build a FAISS index (the demo's precompute step)."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from visionsearch.config import CONFIG
from visionsearch.data.flickr30k import load_annotations
from visionsearch.data.transforms import build_transform
from visionsearch.index.faiss_index import ImageIndex
from visionsearch.models.dual_encoder import DualEncoder

DATA = CONFIG.data_dir / "flickr30k"
IMAGES = DATA / "images"


@torch.no_grad()
def main() -> None:
    anns = load_annotations(DATA / "flickr_annotations_30k.csv", split="test")
    model = DualEncoder().to(CONFIG.device)
    ckpt = torch.load(CONFIG.checkpoint_dir / "visionsearch.pt", map_location=CONFIG.device)
    model.image_head.load_state_dict(ckpt["image_head"])
    model.text_head.load_state_dict(ckpt["text_head"])
    model.eval()
    tfm = build_transform(train=False)

    embeds, filenames = [], []
    for i in range(0, len(anns), 64):
        chunk = anns[i:i + 64]
        px = torch.stack([tfm(Image.open(IMAGES / a.filename).convert("RGB")) for a in chunk])
        embeds.append(model.encode_image(px.to(CONFIG.device)).cpu().numpy())
        filenames += [a.filename for a in chunk]
    embeds = np.concatenate(embeds)

    out = DATA / "gallery"
    ImageIndex.build(embeds, filenames).save(out)
    print(f"indexed {len(filenames)} images -> {out}")


if __name__ == "__main__":
    main()
