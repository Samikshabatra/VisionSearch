import torch

from visionsearch.models.encoders import ImageEncoder, TextEncoder


def test_image_encoder_shape_and_frozen():
    enc = ImageEncoder()
    assert enc.out_dim == 768
    out = enc(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 768)
    assert all(not p.requires_grad for p in enc.parameters())


def test_image_backbone_stays_eval_in_train_mode():
    enc = ImageEncoder()
    enc.train()  # put module in train mode
    assert not enc.backbone.training  # backbone must remain eval (dropout off)


def test_text_encoder_shape_and_frozen():
    enc = TextEncoder()
    assert enc.out_dim == 768
    input_ids = torch.randint(0, 100, (2, 7))
    attention_mask = torch.ones(2, 7, dtype=torch.long)
    out = enc(input_ids, attention_mask)
    assert out.shape == (2, 768)
    assert all(not p.requires_grad for p in enc.parameters())
