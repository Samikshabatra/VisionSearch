"""Sanity check: overfit a tiny real Flickr30k subset; the loss must crash toward zero.

Saves docs/sanity_loss_curve.png — the Week-4 deliverable.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    data_dir = CONFIG.data_dir / "flickr30k"
    anns = load_annotations(data_dir / "flickr_annotations_30k.csv", split="train")[:32]
    ds = Flickr30kDataset(data_dir / "images", anns, build_transform(train=False), train=False)
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=make_collate_fn(tok))

    model, loss_fn = DualEncoder(), ContrastiveLoss()
    # Many passes over the same 32 examples → memorization; loss should collapse.
    history = fit(model, loss_fn, list(loader) * 40, epochs=1, lr=1e-3,
                  device=CONFIG.device, use_amp=(CONFIG.device == "cuda"))

    print(f"loss: {history[0]:.3f} -> {history[-1]:.3f}")
    plt.figure(figsize=(7, 4))
    plt.plot(history)
    plt.xlabel("step"); plt.ylabel("InfoNCE loss")
    plt.title("Sanity overfit (32 examples) — loss should crash toward 0")
    plt.tight_layout()
    out = CONFIG.data_dir.parent / "docs" / "sanity_loss_curve.png"
    plt.savefig(out, dpi=110)
    print("saved", out)


if __name__ == "__main__":
    main()
