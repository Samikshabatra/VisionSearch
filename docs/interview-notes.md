# VisionSearch — Interview Notes

Your cheat-sheet for talking about this project. Lead with the **30-second pitch**, then go as deep as
the interviewer wants. The golden rule: **be honest about the numbers and frame the limits as
understanding, not excuses.**

---

## 1. 30-second pitch

> "VisionSearch is a semantic text-to-image search engine — you type a sentence and it returns matching
> images. It's a from-scratch contrastive **dual-encoder**: I take frozen pretrained backbones (a ViT for
> images, DistilBERT for text), and I train two projection heads with a **symmetric InfoNCE loss** and a
> **learnable temperature** to align text and images in one embedding space. At search time I embed the
> query and rank a **FAISS** index of precomputed image vectors by cosine similarity. It's a real product
> — FastAPI backend, React frontend, deployed in a single container — trained on one 8 GB laptop GPU for
> zero cost."

This one paragraph hits: the product, the technique, the from-scratch part, the systems work, and the
constraint story.

---

## 2. How it works (draw this if there's a whiteboard)

```
text query ──► DistilBERT (frozen) ──► text head (trained) ──► 256-d ─┐
                                                                       ├─ cosine similarity ─► ranked images
image ───────► ViT-B/32 (frozen) ───► image head (trained) ─► 256-d ──┘   (FAISS over precomputed vectors)
```

- **Dual-encoder**: image and text are encoded *separately* into the same 256-d space.
- The backbones are **borrowed and frozen**; the **heads + loss + temperature are mine**.
- Image vectors are **precomputed once** into a FAISS index, so search is one query-embed + a fast
  inner-product lookup.

---

## 3. Core concepts — one-line answers

- **Contrastive learning?** Train matching (image, caption) pairs to sit close in vector space and
  mismatched pairs to sit far apart, using the natural pairing as the label.
- **Dual-encoder vs cross-encoder?** Dual encoders embed each side separately so image vectors can be
  precomputed and searched instantly; cross-encoders are more accurate but must re-process every
  query–image pair, too slow for retrieval at scale.
- **Symmetric InfoNCE?** Build a B×B similarity matrix for the batch, scale it, and apply cross-entropy
  in both directions (image→text and text→image) with the matching pair on the diagonal as the target;
  the other items in the batch are the negatives.
- **In-batch negatives / why batch size matters?** Every other item in the batch is a negative, so a
  bigger batch = more negatives = sharper signal. I use **gradient accumulation** to simulate a larger
  batch on 8 GB.
- **Learnable temperature?** It scales the similarity logits to control how sharply positives separate
  from negatives. I learn `log(1/τ)` (keeps it positive and well-conditioned) and clamp it, following CLIP.
- **Why freeze the image backbone?** Memory + data efficiency on 8 GB, and it mirrors how real
  vision-language models train the *alignment*, not the encoder. Freezing also lets me precompute image
  features once.
- **Why an MLP projection head (not linear)?** Because the backbone is frozen, the head is the *only*
  place features can adapt — so it needs more capacity than CLIP's bare linear projection (CLIP can use
  linear because it trains the whole encoder).
- **Recall@K?** For a text query, is the correct image in the top K (reported at K=1,5,10)? Always next
  to a baseline so the number is interpretable.
- **Why FAISS / how does it scale?** It indexes the precomputed image vectors so top-k lookup stays fast
  as the gallery grows — the standard retrieval pattern.

---

## 4. The hard questions (rehearse these)

**"CLIP beats your model badly. Why is this worth anything?"**
> "Right — CLIP gets ~59% R@1, mine gets ~19%. That gap is expected and I can explain it precisely: CLIP
> trained its *entire encoder* contrastively on ~400M image–text pairs; I adapted a *frozen,
> ImageNet-supervised* backbone with ~0.5M trainable head parameters on 29k images on a laptop. The point
> wasn't to beat a foundation model — it was to build and understand the contrastive alignment mechanism
> end-to-end, and to benchmark it *honestly* against the off-the-shelf ceiling. My model is still ~190×
> better than chance and retrieves semantically correct images."

**"Why does 'a girl coding' return bad results?"**
> "Out-of-distribution. Flickr30k is 2014 everyday photos — it has essentially no computer/coding images,
> so there's no correct answer to retrieve. It's a dataset/distribution limit, not a model bug; the model
> actually maps it to 'a person doing a focused activity,' which is the closest concept available."

**"How would you make it better?"**
> "In order of impact: (1) use a **CLIP-pretrained image backbone** instead of ImageNet ViT — the features
> are already alignment-friendly; (2) **unfreeze the backbone** with gradient caching for large-batch
> contrastive training on limited VRAM; (3) **hard-negative mining**; (4) more/broader data (COCO).
> Each is a concrete, known lever."

**"What was the hardest engineering problem?"**
> Pick one you actually hit: the Windows DataLoader-worker hang (fixed with num_workers=0), the
> macOS AppleDouble files doubling the image count, the transformers 5.x API change for CLIP features,
> or the Tailwind plugin that was silently never wired (caught only by re-reading the build output).
> These show debugging + "the build passing ≠ it does what I intended."

**"What did the 8 GB budget force you to do?"**
> "Freeze the backbone and precompute features, mixed-precision (AMP), gradient accumulation to keep the
> effective batch large, and cap images at 224×224. Concrete efficiency engineering, not just smaller numbers."

---

## 5. Engineering decisions & tradeoffs (the "why")

- **Frozen backbone + precompute** — the load-bearing efficiency choice; enabled training on 8 GB.
- **Measured before optimizing** — I benchmarked training at ~3 min/epoch before deciding *not* to build a
  feature-precompute pipeline for training (YAGNI); precompute earned its place later, for the FAISS gallery.
- **Provider/encoder modularity** — encoders, heads, loss, index are separate units, each independently
  tested (32 passing tests, TDD throughout).
- **Honest evaluation** — full Flickr30k protocol (5 captions, both directions) and a raw-CLIP baseline on
  the same test set; I report the gap rather than hiding it.
- **Single-container deploy** — FastAPI serves the built React app + the search API, so the demo is one URL.

---

## 6. Results (say these honestly)

Flickr30k test split, text→image Recall@K (%):

| | R@1 | R@5 | R@10 |
|---|---|---|---|
| VisionSearch (ours, trained heads) | 19.2 | 48.1 | 61.1 |
| Raw CLIP (zero-shot, the ceiling) | 58.8 | 83.4 | 90.1 |
| Chance | ~0.1 | ~0.5 | ~1.0 |

"~190× better than chance at R@1; CLIP is the ceiling and the gap is the cost of not having 400M pairs."

---

## 7. Tech stack rationale (one-liners)

- **PyTorch** — industry standard, CUDA for the RTX 5060.
- **timm / Hugging Face** — open pretrained backbones (ViT-B/32, DistilBERT).
- **FAISS** — fast similarity search over precomputed vectors.
- **FastAPI + React** — a real product UI, not a notebook; deployable as one container.
- **Flickr30k** — ~31k images × 5 captions; the right size for an 8 GB budget.

---

## 8. What I'd build next (shows forward-thinking)

Image-to-image search ("more like this", reuses the same index), a multilingual text encoder (queries in
Indian languages), unfreezing the backbone, and swapping in a COCO subset to show how scores move with
more data.
