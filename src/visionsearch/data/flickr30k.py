"""Flickr30k acquisition + annotation parsing — no pandas, no `datasets` library.

pandas' compiled DLL is blocked by Windows Application Control on this machine,
so we fetch raw files via huggingface_hub and parse the CSV with the stdlib.
"""
from __future__ import annotations

import ast
import csv
import zipfile
from dataclasses import dataclass
from pathlib import Path

_REPO = "nlphuji/flickr30k"
_CSV_FILE = "flickr_annotations_30k.csv"
_ZIP_FILE = "flickr30k-images.zip"


@dataclass(frozen=True)
class Annotation:
    filename: str
    captions: list[str]
    img_id: int
    split: str


def load_annotations(csv_path: Path, split: str | None = None) -> list[Annotation]:
    """Parse the Flickr30k annotations CSV into Annotation records.

    `split` filters to "train" / "val" / "test" when given.
    """
    out: list[Annotation] = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if split is not None and row["split"] != split:
                continue
            out.append(
                Annotation(
                    filename=row["filename"],
                    captions=list(ast.literal_eval(row["raw"])),
                    img_id=int(row["img_id"]),
                    split=row["split"],
                )
            )
    return out


def download_flickr30k(data_dir: Path) -> tuple[Path, Path]:
    """Download + unzip Flickr30k into data_dir. Idempotent. Returns (images_dir, csv_path)."""
    from huggingface_hub import hf_hub_download

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    images_dir = data_dir / "images"

    csv_cached = hf_hub_download(_REPO, _CSV_FILE, repo_type="dataset")
    csv_path = data_dir / _CSV_FILE
    if not csv_path.exists():
        csv_path.write_bytes(Path(csv_cached).read_bytes())

    if not images_dir.exists() or not any(images_dir.glob("*.jpg")):
        zip_cached = hf_hub_download(_REPO, _ZIP_FILE, repo_type="dataset")
        images_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_cached) as z:
            # The zip stores files under flickr30k-images/<name>.jpg — flatten into images/.
            # The archive was built on macOS, so skip AppleDouble junk (._name.jpg, __MACOSX/).
            for member in z.namelist():
                name = Path(member).name
                if name.endswith(".jpg") and not name.startswith("._") and "__MACOSX" not in member:
                    target = images_dir / name
                    if not target.exists():
                        with z.open(member) as src, open(target, "wb") as dst:
                            dst.write(src.read())
    return images_dir, csv_path
