import csv
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> dict:
    """A 3-image Flickr30k-shaped dataset on disk (no network, no 4GB download)."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    rows = [
        {"filename": "a.jpg", "split": "train", "img_id": 0,
         "raw": ["a red square", "a crimson box", "red shape", "scarlet", "red"]},
        {"filename": "b.jpg", "split": "train", "img_id": 1,
         "raw": ["a green field", "grass", "green area", "meadow", "green"]},
        {"filename": "c.jpg", "split": "test", "img_id": 2,
         "raw": ["a blue sky", "blue", "sky", "azure", "clear sky"]},
    ]
    colors = {"a.jpg": (220, 20, 20), "b.jpg": (20, 200, 20), "c.jpg": (20, 20, 220)}
    for fn, color in colors.items():
        Image.new("RGB", (64, 48), color).save(images_dir / fn)

    csv_path = tmp_path / "ann.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["raw", "sentids", "split", "filename", "img_id"])
        for r in rows:
            w.writerow([str(r["raw"]), str([0, 1, 2, 3, 4]), r["split"], r["filename"], r["img_id"]])
    return {"images_dir": images_dir, "csv_path": csv_path}
