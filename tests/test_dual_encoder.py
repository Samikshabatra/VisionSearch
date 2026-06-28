import torch

from visionsearch.config import CONFIG
from visionsearch.models.dual_encoder import DualEncoder


def _fake_batch(b=3, seq=7):
    return {
        "pixel_values": torch.randn(b, 3, 224, 224),
        "input_ids": torch.randint(0, 100, (b, seq)),
        "attention_mask": torch.ones(b, seq, dtype=torch.long),
    }


def test_forward_shapes_and_shared_dim():
    model = DualEncoder()
    img, txt = model(_fake_batch())
    assert img.shape == (3, CONFIG.embed_dim)
    assert txt.shape == (3, CONFIG.embed_dim)


def test_outputs_are_l2_normalized():
    model = DualEncoder()
    img, txt = model(_fake_batch())
    assert torch.allclose(img.norm(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(txt.norm(dim=-1), torch.ones(3), atol=1e-5)


def test_only_heads_are_trainable():
    model = DualEncoder()
    trainable = [p for p in model.parameters() if p.requires_grad]
    head_params = list(model.image_head.parameters()) + list(model.text_head.parameters())
    # every trainable param belongs to a head; backbones contribute none
    assert sum(p.numel() for p in trainable) == sum(p.numel() for p in head_params)
