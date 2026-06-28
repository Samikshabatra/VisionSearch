import torch

from visionsearch.models.heads import ProjectionHead


def test_projection_shape():
    head = ProjectionHead(in_dim=768, embed_dim=256)
    out = head(torch.randn(4, 768))
    assert out.shape == (4, 256)


def test_projection_is_trainable():
    head = ProjectionHead(in_dim=768, embed_dim=256)
    assert all(p.requires_grad for p in head.parameters())
    # gradient actually flows
    out = head(torch.randn(2, 768)).sum()
    out.backward()
    assert any(p.grad is not None for p in head.parameters())
