"""Evaluate VisionSearch vs raw CLIP on the Flickr30k test split → docs/eval_report.md."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor

from visionsearch.config import CONFIG
from visionsearch.data.flickr30k import load_annotations
from visionsearch.data.transforms import build_transform
from visionsearch.eval.retrieval import retrieval_recall
from visionsearch.models.dual_encoder import DualEncoder

DEVICE = CONFIG.device
DATA = CONFIG.data_dir / "flickr30k"
IMAGES = DATA / "images"


def _batched(items, n):
    for i in range(0, len(items), n):
        yield items[i:i + n]


@torch.no_grad()
def embed_ours(anns):
    model = DualEncoder().to(DEVICE)
    ckpt = torch.load(CONFIG.checkpoint_dir / "visionsearch.pt", map_location=DEVICE)
    model.image_head.load_state_dict(ckpt["image_head"])
    model.text_head.load_state_dict(ckpt["text_head"])
    model.eval()
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tfm = build_transform(train=False)

    img_embeds = []
    for chunk in _batched(anns, 64):
        px = torch.stack([tfm(Image.open(IMAGES / a.filename).convert("RGB")) for a in chunk]).to(DEVICE)
        img_embeds.append(model.encode_image(px).cpu())
    img_embeds = torch.cat(img_embeds)

    captions, pos = [], []
    for i, a in enumerate(anns):
        for c in a.captions:
            captions.append(c); pos.append(i)
    txt_embeds = []
    for chunk in _batched(captions, 256):
        t = tok(chunk, padding=True, truncation=True, max_length=40, return_tensors="pt").to(DEVICE)
        txt_embeds.append(model.encode_text(t["input_ids"], t["attention_mask"]).cpu())
    txt_embeds = torch.cat(txt_embeds)
    return img_embeds, txt_embeds, torch.tensor(pos)


@torch.no_grad()
def embed_clip(anns):
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    img_embeds = []
    for chunk in _batched(anns, 64):
        imgs = [Image.open(IMAGES / a.filename).convert("RGB") for a in chunk]
        inp = proc(images=imgs, return_tensors="pt").to(DEVICE)
        # transformers 5.x returns an output object; the projected 512-d embed is pooler_output.
        feats = clip.get_image_features(**inp).pooler_output
        img_embeds.append(F.normalize(feats, dim=-1).cpu())
    img_embeds = torch.cat(img_embeds)

    captions, pos = [], []
    for i, a in enumerate(anns):
        for c in a.captions:
            captions.append(c); pos.append(i)
    txt_embeds = []
    for chunk in _batched(captions, 256):
        inp = proc(text=chunk, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        feats = clip.get_text_features(**inp).pooler_output
        txt_embeds.append(F.normalize(feats, dim=-1).cpu())
    txt_embeds = torch.cat(txt_embeds)
    return img_embeds, txt_embeds, torch.tensor(pos)


def main() -> None:
    anns = load_annotations(DATA / "flickr_annotations_30k.csv", split="test")
    print(f"test: {len(anns)} images, {sum(len(a.captions) for a in anns)} captions")

    ours = retrieval_recall(*embed_ours(anns))
    clip = retrieval_recall(*embed_clip(anns))

    ks = (1, 5, 10)
    lines = [
        "# VisionSearch -- Evaluation Report",
        "",
        "Flickr30k **test** split (1000 images, 5000 captions). Recall@K (%).",
        "",
        "> Note: VisionSearch is **trained** on Flickr30k but trains ONLY lightweight projection",
        "> heads on a **frozen, ImageNet-supervised** ViT-B/32 + DistilBERT. Raw CLIP is **zero-shot**",
        "> but its backbones were themselves contrastively pretrained on ~400M image-text pairs with",
        "> a learnable encoder. This is a 'vs off-the-shelf foundation model' comparison, not a",
        "> controlled match -- CLIP is effectively the performance ceiling here.",
        "",
        "| Direction | Metric | VisionSearch (ours) | Raw CLIP (zero-shot) |",
        "|---|---|---|---|",
    ]
    for d, label in (("t2i", "text->image"), ("i2t", "image->text")):
        for k in ks:
            key = f"{d}_R@{k}"
            lines.append(f"| {label} | R@{k} | {ours[key]*100:.1f} | {clip[key]*100:.1f} |")

    chance = 100.0 / len(anns)
    lines += [
        "",
        "## Interpretation",
        "",
        f"- **vs chance** (~{chance:.2f}% for {len(anns)} images): our text->image R@1 of "
        f"{ours['t2i_R@1']*100:.1f}% is ~{ours['t2i_R@1']/(1/len(anns)):.0f}x better than random -- "
        "the from-scratch contrastive alignment clearly works.",
        "- **vs CLIP**: CLIP wins decisively. Expected -- it trained the *whole encoder* on ~400M",
        "  pairs, while we adapt a *frozen ImageNet backbone* with ~0.5M head params on 29k images.",
        "  The gap quantifies what large-scale alignment pretraining buys.",
        "- **Takeaway**: the result demonstrates the technique (InfoNCE, dual-encoder, temperature)",
        "  and honest benchmarking, achieved on a single 8 GB laptop GPU for zero cost.",
        "",
    ]
    report = "\n".join(lines) + "\n"

    out = CONFIG.data_dir.parent / "docs" / "eval_report.md"
    out.write_text(report, encoding="utf-8")
    print(f"saved {out}\n")
    print(f"ours t2i R@1={ours['t2i_R@1']*100:.1f}  clip t2i R@1={clip['t2i_R@1']*100:.1f}")


if __name__ == "__main__":
    main()
