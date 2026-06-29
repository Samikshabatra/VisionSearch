import torch
import torch.nn.functional as F

from visionsearch.eval.retrieval import retrieval_recall


def _aligned(n=10, d=16, caps=2):
    """n images (basis-like vectors); each image has `caps` captions identical to it."""
    img = F.normalize(torch.randn(n, d), dim=-1)
    text = img.repeat_interleave(caps, dim=0)              # caption rows clone their image
    text_img_pos = torch.arange(n).repeat_interleave(caps)
    return img, text, text_img_pos


def test_perfect_alignment_full_recall():
    img, text, pos = _aligned()
    r = retrieval_recall(img, text, pos)
    assert r["t2i_R@1"] == 1.0
    assert r["i2t_R@1"] == 1.0


def test_keys_and_monotonic():
    img, text, pos = _aligned(n=30)
    r = retrieval_recall(img, text, pos)
    for d in ("t2i", "i2t"):
        assert r[f"{d}_R@1"] <= r[f"{d}_R@5"] <= r[f"{d}_R@10"]


def test_random_is_low():
    torch.manual_seed(0)
    img = F.normalize(torch.randn(100, 16), dim=-1)
    text = F.normalize(torch.randn(200, 16), dim=-1)
    pos = torch.arange(100).repeat_interleave(2)
    r = retrieval_recall(img, text, pos)
    assert r["t2i_R@1"] < 0.1
