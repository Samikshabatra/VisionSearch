import torch
import torch.nn.functional as F

from visionsearch.eval.recall import recall_at_k


def test_perfect_alignment_is_full_recall():
    e = F.normalize(torch.randn(20, 16), dim=-1)
    r = recall_at_k(e, e)  # image i == text i exactly
    assert r["R@1"] == 1.0 and r["R@5"] == 1.0 and r["R@10"] == 1.0


def test_random_is_low_recall():
    torch.manual_seed(0)
    img = F.normalize(torch.randn(100, 16), dim=-1)
    txt = F.normalize(torch.randn(100, 16), dim=-1)
    r = recall_at_k(img, txt)
    assert r["R@1"] < 0.2  # chance-level for 100 items


def test_recall_monotonic_in_k():
    torch.manual_seed(1)
    img = F.normalize(torch.randn(50, 16), dim=-1)
    txt = F.normalize(torch.randn(50, 16), dim=-1)
    r = recall_at_k(img, txt)
    assert r["R@1"] <= r["R@5"] <= r["R@10"]
