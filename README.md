<div align="center">

# 🔍 VisionSearch

### Semantic text-to-image search — *"type a sentence, get the matching images."*

A **from-scratch contrastive dual-encoder** (CLIP-style) that learns to align text and images
in a shared embedding space, then retrieves images by natural-language query.

<!-- Status badges -->
![Status](https://img.shields.io/badge/status-in%20progress-yellow)
![Week](https://img.shields.io/badge/roadmap-Week%207%20of%208-blue)
![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-cu128-ee4c2c)
![GPU](https://img.shields.io/badge/GPU-RTX%205060%20(8GB)-76b900)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

---

## What it does

You type a sentence — *"two people on a beach at sunset"* — and VisionSearch returns the images
that best match it, ranked by similarity. No tags, no filenames, no metadata: it understands the
**meaning** of the query and the **content** of the images.

This is the same mechanism behind modern retrieval-augmented (RAG) systems: embed everything into
one vector space, then rank by similarity. Here it's applied to **multimodal** (text ↔ image) retrieval.

## How it works

A **dual-encoder**: one tower turns an image into a vector, another turns a text query into a vector,
and both land in the **same shared embedding space**. They're trained contrastively so a caption sits
*close* to its own image and *far* from every other image.

```
   TEXT QUERY                                          IMAGE LIBRARY
"two people on a beach…"                          (thousands of images)
        │                                                   │
   Text encoder (DistilBERT)                    Image encoder (ViT/ResNet, FROZEN)
        │  + projection head ◄── trained ──►  + projection head │
        ▼                                                   ▼
   text vector ─────────►  SHARED EMBEDDING SPACE  ◄──── image vectors
                                    │                 (precomputed → FAISS index)
                          cosine similarity
                                    ▼
                            ranked results
```

### The part that's built from scratch

The borrowed backbones are **frozen** — what gets trained is the alignment, which is exactly how real
vision-language models are built:

- **Projection heads** that map both encoders into the shared space
- The **symmetric InfoNCE loss** — pulls each image–caption pair together, pushes apart all other
  pairs in the batch (the in-batch negatives)
- A **learnable temperature** that controls how sharply positives separate from negatives

> Knowing *not* to pretrain a backbone on a laptop — and to train only the alignment — is itself the
> design decision this project demonstrates.

## Tech stack

| Layer | Choice |
|---|---|
| Framework | PyTorch (CUDA **cu128**, for the Blackwell RTX 5060) |
| Backbones | ViT-B/32 or ResNet-50 (image, frozen) · DistilBERT (text) — Hugging Face / timm |
| Dataset | Flickr30k (~31k images, 5 captions each) |
| Vector search | FAISS (CPU) over precomputed image vectors |
| Backend | FastAPI (`POST /search`) |
| Frontend | React + Vite + Tailwind |
| Deploy | Hugging Face Spaces (single container) |

## Training on 8 GB of VRAM

Contrastive learning wants large batches (more in-batch negatives = a sharper signal), which 8 GB
resists. The efficiency moves that resolve that tension:

- **Freeze the image backbone** and **precompute image features once** — the biggest win
- **Mixed precision (AMP, fp16/bf16)** — ~halves activation memory
- **Gradient accumulation** — simulate a big batch (e.g. 32 × 4 = 128 effective)
- **Images at 224×224** — resolution drives memory quadratically
- **Optional LoRA** on the text encoder

## How success is measured

**Recall@K** (K = 1, 5, 10): for a text query, is the correct image in the top K? Always reported
**next to raw pretrained CLIP** on the same test set — the baseline delta is what makes the number mean
something.

## Results

Flickr30k **test** split (1000 images, 5000 captions), text→image Recall@K (%):

| Metric | VisionSearch (ours) | Raw CLIP (zero-shot) | Chance |
|---|---|---|---|
| R@1 | 19.2 | 58.8 | ~0.1 |
| R@5 | 48.1 | 83.4 | ~0.5 |
| R@10 | 61.1 | 90.1 | ~1.0 |

Honest read: we train **only ~0.5M projection-head parameters** on a **frozen, ImageNet-supervised**
backbone with 29k images on a single 8 GB laptop GPU. Raw CLIP trained its *entire encoder* on ~400M
image–text pairs, so it's the performance ceiling — not a controlled match. Our model is **~190× better
than chance** at R@1 and retrieves semantically correct images for free-text queries (see
[qualitative examples](docs/qualitative_examples.png)), demonstrating that the from-scratch contrastive
alignment works. Full report: [docs/eval_report.md](docs/eval_report.md).

## Roadmap

`environment → data → model → loss/training → evaluate vs baseline → app → deploy → document`

| Week | Milestone | Status |
|---|---|---|
| 1 | Foundations: CUDA env, repo skeleton, theory note | ✅ done |
| 2 | Flickr30k data pipeline (Dataset, transforms, leakage-free splits) | ✅ done |
| 3 | Model: frozen encoders + projection heads + dual-encoder forward | ✅ done |
| 4 | Symmetric InfoNCE loss + AMP + gradient accumulation | ✅ done |
| 5 | Full training & tuning | ✅ done |
| 6 | Evaluation: Recall@K vs CLIP baseline | ✅ done |
| 7 | Search app: FAISS index + FastAPI + React UI | ✅ done |
| 8 | Deploy to Hugging Face Spaces + polish docs | ⏳ next |

## Project structure

```
VisionSearch/
├── src/visionsearch/        # installable ML package
│   ├── config.py            # single source of truth (paths, hyperparams, device)
│   ├── data/                # Flickr30k Dataset + transforms        [Week 2]
│   ├── models/              # frozen encoders, projection heads, dual-encoder  [Week 3]
│   ├── train/               # symmetric InfoNCE + temperature        [Week 4]
│   ├── eval/                # Recall@K + CLIP baseline               [Week 6]
│   └── index/               # FAISS build/query                      [Week 7]
├── backend/                 # FastAPI search API                     [Week 7]
├── frontend/                # React + Vite + Tailwind                [Week 7]
├── scripts/check_env.py     # GPU/CUDA smoke test
├── docs/                    # theory note, design spec, plans
└── tests/
```

## Getting started

> Requires Python 3.11 and an NVIDIA GPU. The CUDA wheel below targets Blackwell (sm_120 / RTX 50-series);
> for other GPUs use the matching index from [pytorch.org](https://pytorch.org/get-started/locally/).

```bash
git clone https://github.com/Samikshabatra/VisionSearch
cd VisionSearch

python -m venv .venv
# Windows:  .venv\Scripts\activate     Linux/macOS:  source .venv/bin/activate

# Install CUDA PyTorch first, then the rest
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .

# Verify the GPU is usable
python scripts/check_env.py
```

### Run the search app (after training)

```bash
# 1. get data + train + build the gallery index
python scripts/download_data.py
python scripts/train.py --epochs 5 --batch-size 32 --accum 4
python scripts/build_index.py

# 2. start the API (serves /search, /images, and the built frontend)
python -m uvicorn backend.main:app --port 8020

# 3. dev frontend (separate terminal; proxies to the API on :8020)
cd frontend && npm install && npm run dev
```

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">
<sub>Built as a placement project to demonstrate contrastive learning, embeddings, and vector retrieval — the foundation of modern RAG.</sub>
</div>
