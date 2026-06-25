# VisionSearch — Design Spec

**Date:** 2026-06-25
**Status:** Approved (design); implementation plan to follow
**Source roadmap:** `VisionSearch_Roadmap.pdf`

> "Type a sentence, get the matching images."

---

## 1. Goal

A deployed **semantic text→image search engine** backed by a **from-scratch contrastive dual-encoder** the user trains herself. A user types a natural-language query ("two people on a beach at sunset") and gets a ranked grid of matching images. The defensible, interview-grade core is the contrastive alignment — projection heads, symmetric InfoNCE loss, and a learnable temperature — trained on top of pretrained, frozen backbones.

This is a **placement / learning project**. Success is measured both by the artifact (a live demo + documented repo with a baseline comparison) and by the user's ability to *explain* the contrastive core in interviews. Working mode is **scaffold + teach**: Claude builds structure and explains every key ML concept; the user must come away able to defend it.

## 2. What we are building

A **dual-encoder** model:
- An **image encoder** (pretrained, **frozen**) + a projection head → image vector.
- A **text encoder** (DistilBERT) + a projection head → text vector.
- Both land in a **shared embedding space**, trained so a caption sits close to its image and far from all others.
- At search time: embed the query once, rank the image library by cosine similarity via a **FAISS** index over precomputed image vectors.

**The from-scratch part (hers):** the projection heads, the contrastive training loop, the **symmetric InfoNCE loss**, and the **learnable temperature**. Backbones are borrowed and frozen — which mirrors how real vision-language models train the alignment, not the encoder. This is a deliberate senior-level signal stated in the README.

## 3. Tech stack

| Layer | Choice | Note |
|---|---|---|
| Framework | PyTorch (**CUDA / cu128**) | RTX 5060 is Blackwell (sm_120) → needs cu128 wheels |
| Image backbone | timm / HF, frozen | ViT-B/32 or ResNet-50 |
| Text backbone | DistilBERT (HF) | optional LoRA fine-tune (stretch) |
| Dataset | Flickr30k | ~31k images, 5 captions each — fits 8 GB |
| Vector search | FAISS (CPU) | ANN over precomputed image vectors |
| Experiment logs | TensorBoard | loss / recall curves → README screenshots |
| **Backend** | **FastAPI** | `POST /search`, `/health`; serves built frontend |
| **Frontend** | **React + Vite + Tailwind** | proper UI — search box + results grid (NOT Streamlit/Gradio, per user) |
| Deploy | Hugging Face Spaces (Docker) | single container: FastAPI serves the built frontend → one demo link |
| VCS | Git + GitHub | `https://github.com/Samikshabatra/VisionSearch`; push after each big update |

**Why a real frontend (not Gradio/Streamlit):** user explicitly wants a polished UI as a differentiator. The single-container deploy (FastAPI serving the Vite `dist/`) preserves the "one clickable demo link" advantage while looking like a real product. The frontend swap does **not** touch the ML core.

## 4. Architecture & module boundaries

```
VisionSearch/
├── data/                       # Flickr30k — gitignored; tiny sample committed
├── src/visionsearch/           # installable ML package (pip install -e .)
│   ├── config.py               # paths, hyperparams, device — single source of truth
│   ├── data/dataset.py         # Flickr30k Dataset + transforms + tokenize     [Wk2]
│   ├── models/encoders.py      # frozen image backbone + text encoder wrappers [Wk3]
│   ├── models/heads.py         # projection heads — FROM SCRATCH                [Wk3]
│   ├── models/dual_encoder.py  # forward pass into shared space                [Wk3]
│   ├── train/loss.py           # symmetric InfoNCE + learnable temperature      [Wk4]
│   ├── eval/recall.py          # Recall@1/5/10 + raw-CLIP baseline              [Wk6]
│   └── index/faiss_index.py    # build / query FAISS                            [Wk7]
├── backend/                    # FastAPI: POST /search, /health, serve frontend dist
├── frontend/                   # React + Vite + Tailwind (search box + results grid)
├── scripts/check_env.py        # GPU/CUDA smoke test                            [Wk1]
├── scripts/train.py            # training entrypoint                            [Wk5]
├── scripts/precompute_embeddings.py  # build demo gallery vectors + index       [Wk7]
├── notebooks/                  # exploration / visualization
├── docs/                       # theory note, decisions, eval report
├── tests/                      # shape-assertion tests
├── pyproject.toml  requirements.txt  README.md  .gitignore
```

**Design principle — wall off borrowed from from-scratch.** `models/encoders.py` (frozen, borrowed) is kept separate from `models/heads.py` + `train/loss.py` (hers). That separation *is* the interview narrative and keeps each unit independently testable.

**Single source of truth.** `config.py` holds paths, hyperparameters, device, embedding dim — nothing hardcoded across modules.

## 5. Fitting 8 GB VRAM

The "efficient training" interview story, applied as concrete settings:
- **Freeze the image backbone; precompute image features once** (biggest win — no image-side gradients/optimizer state).
- **Mixed precision (AMP fp16/bf16)** — ~halves activation memory, uses Blackwell tensor cores.
- **Gradient accumulation** — e.g. 32 real × 4 accum = 128 effective batch (contrastive learning needs many in-batch negatives).
- **Images at 224×224** — resolution drives memory quadratically.
- **Optional LoRA on the text encoder** — keeps optimizer memory tiny if fine-tuning.

OOM playbook: lower batch → sequence length → resolution, in that order.

## 6. Success metrics

**Recall@K (K = 1, 5, 10)** for text→image (and image→text) on a leakage-free test split, **always reported next to raw pretrained CLIP** on the same test set. The baseline delta is the strongest interview evidence. Plus qualitative win/failure examples.

## 7. Build order (8 weeks)

`environment → data → model → loss/training → evaluate vs baseline → app → deploy → document`. Each week finishes before the next starts. Weeks map 1:1 to the module table in §4.

## 8. This session's scope — Week 1 foundations

1. **Dedicated venv** + **CUDA PyTorch (cu128)** installed and **verified running a tensor on the RTX 5060** (`torch.cuda.is_available() == True`). Resolves the current CPU-only `torch 2.12.0+cpu`.
2. **Repo skeleton** (§4) + `git init` + `.gitignore` (data/, venv/, checkpoints, `__pycache__`, node_modules, dist) + `pyproject.toml` + clean first commit + **push to GitHub**.
3. **`docs/theory-note.md`** — CLIP, dual-encoder, InfoNCE, temperature, why-freeze-backbone, dual-vs-cross-encoder, in plain interview-ready language (the "teach" deliverable).
4. **`scripts/check_env.py`** — re-runnable GPU/CUDA smoke test.
5. **Stretch (if time):** minimal React + Vite + Tailwind frontend shell with mock results, so the UI is visibly started.

Out of scope this session: dataset download, training, FAISS, real inference. Those are Weeks 2–8, each its own milestone.

## 9. Pitfalls to avoid (from roadmap)

- Leaving it undeployed (a live link beats a better laptop-only model).
- Skipping the raw-CLIP baseline comparison.
- Tiny batches with no gradient accumulation (starves contrastive learning).
- Caption leakage across train/val/test splits (inflates scores; interview red flag).
- Scope creep (no captioning/VQA now — ship retrieval first).
- Cranking image resolution (fastest path to OOM for little gain).

## 10. Stretch goals (after it ships)

Unfreeze the backbone via gradient-cache; image-to-image search; multilingual/Indian-language text encoder; zero-shot classification from the embeddings; swap Flickr30k → COCO subset to show data scaling.
