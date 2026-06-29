"""Qualitative retrieval examples from VisionSearch -> docs/qualitative_examples.png."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image
from transformers import AutoTokenizer

from visionsearch.config import CONFIG
from visionsearch.data.flickr30k import load_annotations
from visionsearch.data.transforms import build_transform
from visionsearch.models.dual_encoder import DualEncoder

DEVICE = CONFIG.device
DATA = CONFIG.data_dir / "flickr30k"
IMAGES = DATA / "images"
QUERIES = [
    "a dog running on the beach",
    "two people riding bicycles",
    "a child playing with a ball",
    "people sitting at a restaurant table",
]


@torch.no_grad()
def main() -> None:
    anns = load_annotations(DATA / "flickr_annotations_30k.csv", split="test")
    model = DualEncoder().to(DEVICE)
    ckpt = torch.load(CONFIG.checkpoint_dir / "visionsearch.pt", map_location=DEVICE)
    model.image_head.load_state_dict(ckpt["image_head"])
    model.text_head.load_state_dict(ckpt["text_head"])
    model.eval()
    tfm = build_transform(train=False)
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # embed the gallery (all test images)
    embeds = []
    for i in range(0, len(anns), 64):
        chunk = anns[i:i + 64]
        px = torch.stack([tfm(Image.open(IMAGES / a.filename).convert("RGB")) for a in chunk]).to(DEVICE)
        embeds.append(model.encode_image(px).cpu())
    gallery = torch.cat(embeds)

    fig, axes = plt.subplots(len(QUERIES), 5, figsize=(15, 3 * len(QUERIES)))
    for row, q in enumerate(QUERIES):
        t = tok([q], return_tensors="pt").to(DEVICE)
        qe = model.encode_text(t["input_ids"], t["attention_mask"]).cpu()
        top = (qe @ gallery.t())[0].argsort(descending=True)[:5]
        for col, idx in enumerate(top):
            ax = axes[row, col]
            ax.imshow(Image.open(IMAGES / anns[idx].filename).convert("RGB"))
            ax.axis("off")
        axes[row, 0].set_title(f'query: "{q}"  (top-5 ->)', loc="left", fontsize=11)
    fig.tight_layout()
    out = CONFIG.data_dir.parent / "docs" / "qualitative_examples.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
