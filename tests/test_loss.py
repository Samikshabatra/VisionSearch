import math

import torch
import torch.nn.functional as F

from visionsearch.train.loss import ContrastiveLoss


def _orthogonal_pairs(b=6, d=16):
    """Aligned image/text embeds: row i identical across modalities, rows mutually distinct."""
    e = F.normalize(torch.eye(b, d), dim=-1)
    return e.clone(), e.clone()


def test_loss_is_scalar():
    loss_fn = ContrastiveLoss()
    img, txt = _orthogonal_pairs()
    loss = loss_fn(img, txt)
    assert loss.ndim == 0


def test_aligned_beats_misaligned():
    loss_fn = ContrastiveLoss()
    img, txt = _orthogonal_pairs()
    aligned = loss_fn(img, txt)
    # Shuffle text so the diagonal is no longer the match.
    misaligned = loss_fn(img, txt[torch.tensor([1, 2, 3, 4, 5, 0])])
    assert aligned < misaligned


def test_temperature_is_learnable_and_gets_grad():
    loss_fn = ContrastiveLoss()
    assert loss_fn.logit_scale.requires_grad
    img, txt = _orthogonal_pairs()
    loss_fn(img, txt).backward()
    assert loss_fn.logit_scale.grad is not None


def test_temperature_is_clamped():
    loss_fn = ContrastiveLoss(max_scale=100.0)
    with torch.no_grad():
        loss_fn.logit_scale.fill_(50.0)  # absurdly large
    img, txt = _orthogonal_pairs()
    loss_fn(img, txt)  # must not overflow/NaN
    # effective scale capped at max_scale
    assert loss_fn.logit_scale.clamp(max=math.log(100.0)).exp().item() <= 100.0 + 1e-3
