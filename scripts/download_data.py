"""Download Flickr30k and print a summary. Run once: pulls ~4GB on first run."""
from visionsearch.config import CONFIG
from visionsearch.data.flickr30k import download_flickr30k, load_annotations


def main() -> None:
    data_dir = CONFIG.data_dir / "flickr30k"
    print(f"Downloading Flickr30k into {data_dir} (first run pulls ~4GB)...")
    images_dir, csv_path = download_flickr30k(data_dir)
    n_images = sum(1 for _ in images_dir.glob("*.jpg"))
    print(f"images: {n_images} in {images_dir}")
    for split in ("train", "val", "test"):
        print(f"  {split:5s}: {len(load_annotations(csv_path, split=split))} images")


if __name__ == "__main__":
    main()
