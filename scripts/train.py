"""Train VisionSearch on the full Flickr30k train split with val-recall checkpointing.

Usage: .venv/Scripts/python.exe scripts/train.py --epochs 5 --batch-size 32 --accum 4
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from visionsearch.config import CONFIG
from visionsearch.data.dataset import Flickr30kDataset, make_collate_fn
from visionsearch.data.flickr30k import load_annotations
from visionsearch.data.transforms import build_transform
from visionsearch.eval.recall import encode_val, recall_at_k
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
    csv = data_dir / "flickr_annotations_30k.csv"
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    collate = make_collate_fn(tok)

    train_ds = Flickr30kDataset(data_dir / "images", load_annotations(csv, "train"),
                                build_transform(train=True), train=True)
    val_ds = Flickr30kDataset(data_dir / "images", load_annotations(csv, "val"),
                              build_transform(train=False), train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False,
                            num_workers=0, collate_fn=collate)

    model, loss_fn = DualEncoder(), ContrastiveLoss()
    CONFIG.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt = CONFIG.checkpoint_dir / "visionsearch.pt"

    recall_history: list[dict] = []
    best = {"R@1": -1.0}

    def on_epoch_end(epoch: int) -> None:
        nonlocal best
        img_e, txt_e = encode_val(model, val_loader, CONFIG.device)
        rec = recall_at_k(img_e, txt_e)
        recall_history.append(rec)
        print(f"epoch {epoch}: val {rec}")
        if rec["R@1"] > best["R@1"]:
            best = rec
            torch.save({
                "image_head": model.image_head.state_dict(),
                "text_head": model.text_head.state_dict(),
                "logit_scale": loss_fn.logit_scale.detach().cpu(),
                "val_recall": rec,
            }, ckpt)
            print(f"  saved best -> {ckpt}")

    loss_history = fit(model, loss_fn, train_loader, epochs=args.epochs, lr=args.lr,
                       device=CONFIG.device, accum_steps=args.accum,
                       use_amp=(CONFIG.device == "cuda"), log_dir="runs/train",
                       max_steps=args.max_steps, on_epoch_end=on_epoch_end)

    # curves: loss (per step) + val R@1/5/10 (per epoch)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
    a1.plot(loss_history); a1.set_title("Train InfoNCE loss"); a1.set_xlabel("step")
    if recall_history:
        for k in ("R@1", "R@5", "R@10"):
            a2.plot([r[k] for r in recall_history], marker="o", label=k)
        a2.set_title("Val recall (text->image)"); a2.set_xlabel("epoch"); a2.legend()
    fig.tight_layout()
    out = CONFIG.data_dir.parent / "docs" / "training_curves.png"
    fig.savefig(out, dpi=110)
    print(f"best val {best}; curves -> {out}")


if __name__ == "__main__":
    main()
