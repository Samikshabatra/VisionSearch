import torch
from transformers import AutoTokenizer

from visionsearch.data.dataset import Flickr30kDataset, make_collate_fn
from visionsearch.data.flickr30k import load_annotations
from visionsearch.data.transforms import build_transform


def _ds(tiny_dataset, train=True):
    anns = load_annotations(tiny_dataset["csv_path"])
    return Flickr30kDataset(
        images_dir=tiny_dataset["images_dir"],
        annotations=anns,
        transform=build_transform(train=train, image_size=32),
        train=train,
    )


def test_dataset_len_and_item_shape(tiny_dataset):
    ds = _ds(tiny_dataset)
    assert len(ds) == 3
    img, caption, img_id = ds[0]
    assert img.shape == (3, 32, 32)
    assert isinstance(caption, str) and caption
    assert isinstance(img_id, int)


def test_eval_caption_is_deterministic(tiny_dataset):
    ds = _ds(tiny_dataset, train=False)
    assert ds[0][1] == "a red square"  # caption[0] in eval mode


def test_collate_batches_and_tokenizes(tiny_dataset):
    ds = _ds(tiny_dataset)
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    collate = make_collate_fn(tok, max_length=16)
    batch = collate([ds[0], ds[1], ds[2]])
    assert batch["pixel_values"].shape == (3, 3, 32, 32)
    assert batch["input_ids"].shape[0] == 3
    assert batch["attention_mask"].shape == batch["input_ids"].shape
    assert batch["img_id"].tolist() == [0, 1, 2]
    assert batch["input_ids"].dtype == torch.long
