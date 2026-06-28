"""Image transforms. ImageNet normalization (revisit when backbone is chosen, Week 3)."""
from __future__ import annotations

from typing import Callable

from torchvision import transforms

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def build_transform(train: bool, image_size: int = 224) -> Callable:
    if train:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_MEAN, _STD),
        ])
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])
