# VisionSearch — Theory Note

> Plain-language reference for the concepts behind VisionSearch. Each section explains the idea,
> then gives a one-line **Interview answer** you can say out loud. Read this before Week 3 (model)
> and Week 4 (loss) — the code will make far more sense once these click.

---

## 1. Contrastive learning

Instead of teaching the model "what is in this image" (classification), we teach it **which things
belong together**. We show it many (image, caption) pairs. For each image, its own caption is the
*positive*; every other caption in the batch is a *negative*. Training pulls each image close to its
own caption in vector space and pushes it away from all the others. After enough of this, "meaning"
becomes geometry: similar things end up near each other.

Why it's powerful here: we never need hand-labeled classes. The caption that *already came with* the
image is the supervision signal. This is "self-supervised-ish" learning from naturally paired data —
the same idea that scaled CLIP to 400M pairs.

**Interview answer:** *Contrastive learning trains a model to put matching pairs close together and
mismatched pairs far apart in a shared vector space, using the natural pairing (image ↔ caption) as
the label.*

---

## 2. Dual-encoder vs cross-encoder

There are two ways to score how well a text matches an image:

- **Cross-encoder:** feed the text *and* the image into one model together; it outputs a match score.
  Very accurate, because the two can interact at every layer. But to search a library of N images you
  must re-run the model N times *per query* — far too slow for retrieval at scale.
- **Dual-encoder (what we build):** encode the image and the text **separately** into vectors, then
  compare with a cheap cosine similarity. The big win: image vectors can be **precomputed once** and
  stored. At query time you encode only the query and do fast vector math against the whole library.

Retrieval needs speed over millions of items, so dual-encoders win. Cross-encoders are often used as a
*reranker* on a small shortlist — but that's beyond this project's scope.

**Interview answer:** *Dual-encoders embed image and text separately so image vectors can be
precomputed and searched instantly; cross-encoders are more accurate but must process every query–image
pair from scratch, which is too slow for retrieval.*

---

## 3. Shared embedding space & cosine similarity

Both encoders output vectors of the **same dimension** (we use 256). They're trained so a caption's
vector and its image's vector point in nearly the **same direction**. To rank, we use **cosine
similarity** — the cosine of the angle between two vectors — which ignores length and measures only
direction. We L2-normalize the vectors first, so cosine similarity becomes a simple dot product.

Why direction, not distance: after normalization every vector lies on a unit sphere, and "how aligned
are these two meanings" is naturally an angle.

**Interview answer:** *Both encoders map into one shared 256-d space; we L2-normalize and rank by
cosine similarity, so retrieval is just a dot product between the query vector and each image vector.*

---

## 4. Symmetric InfoNCE loss

This is the heart of the from-scratch work. For a batch of B (image, text) pairs:

1. Compute a **B×B similarity matrix** — every image against every text.
2. Scale it by the temperature (next section). These scaled similarities are the **logits**.
3. The diagonal entries are the *correct* matches; everything off-diagonal is a negative.
4. Apply **cross-entropy row-wise** (image → its text) and **column-wise** (text → its image), then
   **average the two directions**. That symmetry is why it's called *symmetric* InfoNCE.

Intuitively: for each image, it's a classification problem where the "classes" are the B captions in
the batch, and the right answer is the caption on the diagonal. The **in-batch negatives** (all the
other captions) are what make it work — more negatives = a sharper, more discriminative signal. That's
exactly why batch size matters so much for contrastive learning, and why we use gradient accumulation
to simulate a bigger batch on 8 GB.

**Interview answer:** *Symmetric InfoNCE builds a similarity matrix for the batch and applies
cross-entropy in both directions (image→text and text→image) with the matching pair on the diagonal as
the target; the other items in the batch serve as negatives.*

---

## 5. Learnable temperature

The similarity matrix is divided by a small number τ (temperature) before the softmax. Temperature
controls **how sharp** the probability distribution is:

- Small τ → logits get large → softmax becomes peaky → the model is pushed to separate the positive
  *very* sharply from negatives (strong gradients, but can be unstable).
- Large τ → softer distribution → gentler separation.

Following CLIP, we don't hand-pick τ — we make it a **learnable parameter** (in practice we learn
`log τ` and clamp it, e.g. so the effective scale can't blow up). The model finds its own optimal
sharpness during training.

**Interview answer:** *Temperature scales the similarity logits before softmax to control how sharply
positives separate from negatives; we learn it (as log-temperature, clamped) rather than fixing it.*

---

## 6. Why freeze the image backbone

The image encoder (a ViT or ResNet pretrained on ImageNet-scale data) already produces excellent visual
features. We **freeze** it and train only the projection heads + text side. Two reasons:

- **Efficiency on 8 GB:** a frozen backbone needs no gradients and no optimizer state for its millions
  of parameters, and we can **precompute every image's features exactly once** and reuse them — training
  becomes light and fast.
- **It mirrors real VLMs:** large vision-language models typically train the *alignment*, not the
  encoder from scratch. Starting from strong pretrained features and aligning them is the correct,
  data-efficient choice — especially without a data-center.

Unfreezing it (with the gradient-cache trick for large-batch contrastive training) is a Week-8+ stretch
goal, not the default.

**Interview answer:** *Freezing the image backbone saves memory and data by reusing strong pretrained
features and precomputing them once; it also matches how real vision-language models train the
alignment rather than the encoder.*

---

## 7. Recall@K (and why the baseline matters)

Retrieval quality is measured by **Recall@K**: for a given text query, does the correct image appear in
the top K results?

- **Recall@1** — correct image is the very top result.
- **Recall@5 / @10** — correct image is somewhere in the top 5 / top 10.

A single number means nothing in isolation. We always report our trained model **next to raw, untrained
pretrained CLIP** on the *same* test set. If our alignment beats off-the-shelf CLIP, the delta is
concrete evidence the from-scratch training actually did something — this is the strongest thing to show
in an interview.

**Interview answer:** *Recall@K asks whether the correct image is in the top K for a query (reported at
K=1,5,10), always compared against a raw-CLIP baseline so the number is interpretable.*

---

## 8. The 8 GB efficiency story

Everything above is shaped by one constraint: a single 8 GB GPU. The concrete moves:

| Move | What it buys |
|---|---|
| Freeze backbone + **precompute** image features | No image-side gradients/optimizer state; embed each image once |
| **Mixed precision (AMP)** fp16/bf16 | ~Halves activation memory; uses Blackwell tensor cores |
| **Gradient accumulation** | Simulates a large batch (e.g. 32 × 4 = 128) for more negatives |
| **224×224 images** | Memory grows with resolution², so resist going higher |
| **LoRA** on text encoder (optional) | Parameter-efficient fine-tuning keeps optimizer memory tiny |

OOM playbook, in order: lower batch → shorten sequence length → lower resolution.

**Interview answer:** *On 8 GB I freeze and precompute image features, use mixed precision and gradient
accumulation to keep effective batch large, and cap resolution at 224 — concrete efficiency engineering,
not just smaller numbers.*

---

## One-paragraph summary (the elevator version)

VisionSearch is a **dual-encoder** trained with **contrastive learning**: a frozen image encoder and a
DistilBERT text encoder each get a small **projection head** into a shared 256-d space, and **symmetric
InfoNCE** with a **learnable temperature** pulls each image close to its caption and away from the rest
of the batch. At search time I embed the query, rank a **FAISS** index of precomputed image vectors by
cosine similarity, and return the top results. I freeze the backbone and use mixed precision + gradient
accumulation to train the whole thing on a single 8 GB GPU, and I prove it works by beating raw CLIP on
**Recall@K**.
