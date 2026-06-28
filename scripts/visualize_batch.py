"""Save a grid of sample images with their captions to docs/sample_batch.png."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from visionsearch.config import CONFIG
from visionsearch.data.dataset import Flickr30kDataset
from visionsearch.data.flickr30k import load_annotations
from visionsearch.data.transforms import build_transform

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def main() -> None:
    data_dir = CONFIG.data_dir / "flickr30k"
    anns = load_annotations(data_dir / "flickr_annotations_30k.csv", split="train")
    ds = Flickr30kDataset(data_dir / "images", anns[:8], build_transform(train=False), train=False)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (img, caption, _) in zip(axes.flat, ds):
        ax.imshow((img * _STD + _MEAN).clamp(0, 1).permute(1, 2, 0).numpy())
        ax.set_title(caption[:50] + ("…" if len(caption) > 50 else ""), fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    out = CONFIG.data_dir.parent / "docs" / "sample_batch.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
