"""Recall@K for text→image retrieval (the metric we optimize and report)."""
from __future__ import annotations

import torch
from torch import Tensor


def recall_at_k(image_embeds: Tensor, text_embeds: Tensor, ks=(1, 5, 10)) -> dict[str, float]:
    """Index-aligned protocol: the correct image for text i is image i.

    For each text query we rank all images by cosine similarity and check whether
    the matching image lands in the top K.
    """
    sims = text_embeds @ image_embeds.t()                  # [N_text, N_img]
    targets = torch.arange(sims.size(0), device=sims.device)
    ranking = sims.argsort(dim=1, descending=True)         # most similar first
    out: dict[str, float] = {}
    for k in ks:
        topk = ranking[:, :k]
        hit = (topk == targets.unsqueeze(1)).any(dim=1).float().mean().item()
        out[f"R@{k}"] = hit
    return out


@torch.no_grad()
def encode_val(model, loader, device) -> tuple[Tensor, Tensor]:
    """Embed a validation loader into aligned (image_embeds, text_embeds)."""
    model.eval()
    imgs, txts = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        image_embeds, text_embeds = model(batch)
        imgs.append(image_embeds.cpu())
        txts.append(text_embeds.cpu())
    model.train()
    return torch.cat(imgs), torch.cat(txts)
