"""Full Flickr30k retrieval protocol: text→image and image→text, 5 captions/image."""
from __future__ import annotations

import torch
from torch import Tensor


def retrieval_recall(image_embeds: Tensor, text_embeds: Tensor,
                     text_img_pos: Tensor, ks=(1, 5, 10)) -> dict[str, float]:
    out: dict[str, float] = {}

    # text → image: each caption ranks the images; correct image = text_img_pos[m]
    t2i = text_embeds @ image_embeds.t()                   # [M, N]
    t2i_rank = t2i.argsort(dim=1, descending=True)
    for k in ks:
        hit = (t2i_rank[:, :k] == text_img_pos.unsqueeze(1)).any(dim=1).float().mean().item()
        out[f"t2i_R@{k}"] = hit

    # image → text: each image ranks the captions; correct = ANY caption of that image
    i2t = image_embeds @ text_embeds.t()                   # [N, M]
    i2t_rank = i2t.argsort(dim=1, descending=True)
    n_images = image_embeds.size(0)
    targets = torch.arange(n_images, device=image_embeds.device).unsqueeze(1)
    for k in ks:
        topk_img = text_img_pos[i2t_rank[:, :k]]           # [N, k] -> image of each retrieved caption
        hit = (topk_img == targets).any(dim=1).float().mean().item()
        out[f"i2t_R@{k}"] = hit

    return out
