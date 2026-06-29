import numpy as np

from visionsearch.index.faiss_index import ImageIndex


def _norm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def test_build_and_search_finds_self():
    rng = np.random.default_rng(0)
    embeds = _norm(rng.standard_normal((20, 16)).astype("float32"))
    idx = ImageIndex.build(embeds, [f"{i}.jpg" for i in range(20)])
    scores, ids = idx.search(embeds[3], k=5)
    assert ids[0][0] == 3                 # an item retrieves itself first
    assert scores[0][0] > scores[0][1]    # scores sorted descending


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    embeds = _norm(rng.standard_normal((10, 8)).astype("float32"))
    names = [f"img{i}.jpg" for i in range(10)]
    ImageIndex.build(embeds, names).save(tmp_path)
    loaded = ImageIndex.load(tmp_path)
    assert loaded.filenames == names
    assert loaded.search(embeds[0], k=1)[1][0][0] == 0
