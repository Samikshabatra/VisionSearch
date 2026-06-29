import torch

from visionsearch.config import CONFIG
from visionsearch.models.dual_encoder import DualEncoder
from visionsearch.train.loss import ContrastiveLoss
from visionsearch.train.trainer import fit


def test_overfits_a_fixed_batch():
    torch.manual_seed(0)
    batch = {
        "pixel_values": torch.randn(8, 3, 224, 224),
        "input_ids": torch.randint(0, 1000, (8, 12)),
        "attention_mask": torch.ones(8, 12, dtype=torch.long),
    }
    model = DualEncoder()
    loss_fn = ContrastiveLoss()
    # Feed the SAME batch many times → the heads should memorize it; loss must fall.
    history = fit(model, loss_fn, [batch] * 60, epochs=1, lr=1e-3,
                  device=CONFIG.device, use_amp=False)
    assert history[-1] < history[0]
    assert history[-1] < 0.5 * history[0]


def test_on_epoch_end_fires_each_epoch():
    batch = {
        "pixel_values": torch.randn(4, 3, 224, 224),
        "input_ids": torch.randint(0, 1000, (4, 8)),
        "attention_mask": torch.ones(4, 8, dtype=torch.long),
    }
    model = DualEncoder()
    loss_fn = ContrastiveLoss()
    seen = []
    fit(model, loss_fn, [batch], epochs=3, device=CONFIG.device, use_amp=False,
        on_epoch_end=lambda e: seen.append(e))
    assert seen == [0, 1, 2]
