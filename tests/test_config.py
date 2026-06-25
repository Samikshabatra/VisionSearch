from pathlib import Path

from visionsearch.config import CONFIG


def test_config_has_core_fields():
    assert CONFIG.embed_dim == 256
    assert CONFIG.image_size == 224
    assert CONFIG.device in ("cuda", "cpu")
    assert isinstance(CONFIG.data_dir, Path)
    assert isinstance(CONFIG.checkpoint_dir, Path)
