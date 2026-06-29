# VisionSearch -- Evaluation Report

Flickr30k **test** split (1000 images, 5000 captions). Recall@K (%).

> Note: VisionSearch is **trained** on Flickr30k but trains ONLY lightweight projection
> heads on a **frozen, ImageNet-supervised** ViT-B/32 + DistilBERT. Raw CLIP is **zero-shot**
> but its backbones were themselves contrastively pretrained on ~400M image-text pairs with
> a learnable encoder. This is a 'vs off-the-shelf foundation model' comparison, not a
> controlled match -- CLIP is effectively the performance ceiling here.

| Direction | Metric | VisionSearch (ours) | Raw CLIP (zero-shot) |
|---|---|---|---|
| text->image | R@1 | 19.2 | 58.8 |
| text->image | R@5 | 48.1 | 83.4 |
| text->image | R@10 | 61.1 | 90.1 |
| image->text | R@1 | 25.0 | 79.5 |
| image->text | R@5 | 55.9 | 95.0 |
| image->text | R@10 | 67.9 | 98.1 |

## Interpretation

- **vs chance** (~0.10% for 1000 images): our text->image R@1 of 19.2% is ~192x better than random -- the from-scratch contrastive alignment clearly works.
- **vs CLIP**: CLIP wins decisively. Expected -- it trained the *whole encoder* on ~400M
  pairs, while we adapt a *frozen ImageNet backbone* with ~0.5M head params on 29k images.
  The gap quantifies what large-scale alignment pretraining buys.
- **Takeaway**: the result demonstrates the technique (InfoNCE, dual-encoder, temperature)
  and honest benchmarking, achieved on a single 8 GB laptop GPU for zero cost.

