from visionsearch.data.flickr30k import Annotation, load_annotations


def test_load_parses_captions(tiny_dataset):
    anns = load_annotations(tiny_dataset["csv_path"])
    assert len(anns) == 3
    a = anns[0]
    assert isinstance(a, Annotation)
    assert a.filename == "a.jpg"
    assert a.captions == ["a red square", "a crimson box", "red shape", "scarlet", "red"]
    assert a.img_id == 0


def test_filter_by_split(tiny_dataset):
    train = load_annotations(tiny_dataset["csv_path"], split="train")
    test = load_annotations(tiny_dataset["csv_path"], split="test")
    assert {a.filename for a in train} == {"a.jpg", "b.jpg"}
    assert {a.filename for a in test} == {"c.jpg"}


def test_splits_are_leakage_free(tiny_dataset):
    train = {a.filename for a in load_annotations(tiny_dataset["csv_path"], split="train")}
    test = {a.filename for a in load_annotations(tiny_dataset["csv_path"], split="test")}
    assert train.isdisjoint(test)
