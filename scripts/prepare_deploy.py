"""Assemble a lean, self-contained deploy gallery under deploy/assets/.

Embeds the first N test images, builds a FAISS index, and copies those images +
the trained checkpoint. Run before building the Docker image.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from visionsearch.config import CONFIG
from visionsearch.data.flickr30k import load_annotations
from visionsearch.data.transforms import build_transform
from visionsearch.index.faiss_index import ImageIndex
from visionsearch.models.dual_encoder import DualEncoder

DATA = CONFIG.data_dir / "flickr30k"
SRC_IMAGES = DATA / "images"
ASSETS = Path(__file__).resolve().parents[1] / "deploy" / "assets"


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery-size", type=int, default=500)
    args = ap.parse_args()

    anns = load_annotations(DATA / "flickr_annotations_30k.csv", split="test")[: args.gallery_size]
    model = DualEncoder().to(CONFIG.device)
    ckpt = torch.load(CONFIG.checkpoint_path, map_location=CONFIG.device)
    model.image_head.load_state_dict(ckpt["image_head"])
    model.text_head.load_state_dict(ckpt["text_head"])
    model.eval()
    tfm = build_transform(train=False)

    (ASSETS / "images").mkdir(parents=True, exist_ok=True)
    embeds, filenames = [], []
    for i in range(0, len(anns), 64):
        chunk = anns[i:i + 64]
        px = torch.stack([tfm(Image.open(SRC_IMAGES / a.filename).convert("RGB")) for a in chunk])
        embeds.append(model.encode_image(px.to(CONFIG.device)).cpu().numpy())
        for a in chunk:
            shutil.copy(SRC_IMAGES / a.filename, ASSETS / "images" / a.filename)
            filenames.append(a.filename)
    ImageIndex.build(np.concatenate(embeds), filenames).save(ASSETS / "gallery")
    shutil.copy(CONFIG.checkpoint_path, ASSETS / "visionsearch.pt")
    size_mb = sum(f.stat().st_size for f in (ASSETS / "images").glob("*.jpg")) / 1e6
    print(f"deploy assets ready: {len(filenames)} images (~{size_mb:.0f} MB) -> {ASSETS}")


if __name__ == "__main__":
    main()
