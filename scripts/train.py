"""Train VisionSearch on the full Flickr30k train split.

Usage: .venv/Scripts/python.exe scripts/train.py --epochs 5 --batch-size 32 --accum 4
"""
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from visionsearch.config import CONFIG
from visionsearch.data.dataset import Flickr30kDataset, make_collate_fn
from visionsearch.data.flickr30k import load_annotations
from visionsearch.data.transforms import build_transform
from visionsearch.models.dual_encoder import DualEncoder
from visionsearch.train.loss import ContrastiveLoss
from visionsearch.train.trainer import fit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    data_dir = CONFIG.data_dir / "flickr30k"
    anns = load_annotations(data_dir / "flickr_annotations_30k.csv", split="train")
    ds = Flickr30kDataset(data_dir / "images", anns, build_transform(train=True), train=True)
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, collate_fn=make_collate_fn(tok))

    model, loss_fn = DualEncoder(), ContrastiveLoss()
    history = fit(model, loss_fn, loader, epochs=args.epochs, lr=args.lr,
                  device=CONFIG.device, accum_steps=args.accum,
                  use_amp=(CONFIG.device == "cuda"), log_dir="runs/train",
                  max_steps=args.max_steps)

    CONFIG.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt = CONFIG.checkpoint_dir / "visionsearch.pt"
    torch.save({
        "image_head": model.image_head.state_dict(),
        "text_head": model.text_head.state_dict(),
        "logit_scale": loss_fn.logit_scale.detach().cpu(),
    }, ckpt)
    print(f"final loss {history[-1]:.3f}; saved {ckpt}")


if __name__ == "__main__":
    main()
