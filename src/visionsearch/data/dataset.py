"""Flickr30k PyTorch Dataset + a tokenizing collate function."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset

from .flickr30k import Annotation


class Flickr30kDataset(Dataset):
    """One item per image. Train mode samples a random caption; eval uses caption[0].

    Returns (image_tensor[3,H,W], caption_str, img_id).
    """

    def __init__(self, images_dir: Path, annotations: list[Annotation],
                 transform: Callable, train: bool = True):
        self.images_dir = Path(images_dir)
        self.annotations = annotations
        self.transform = transform
        self.train = train

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int):
        ann = self.annotations[idx]
        image = Image.open(self.images_dir / ann.filename).convert("RGB")
        image = self.transform(image)
        caption = random.choice(ann.captions) if self.train else ann.captions[0]
        return image, caption, ann.img_id


def make_collate_fn(tokenizer, max_length: int = 40) -> Callable:
    """Collate (image, caption, img_id) tuples; tokenize captions per-batch with DistilBERT."""

    def collate(batch):
        images, captions, img_ids = zip(*batch)
        pixel_values = torch.stack(images, dim=0)
        tokens = tokenizer(
            list(captions),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "img_id": torch.tensor(img_ids, dtype=torch.long),
        }

    return collate
