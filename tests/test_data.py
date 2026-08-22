from pathlib import Path

from data_scarcity import config


def test_config_paths_are_initialized():
    assert isinstance(config.PROJ_ROOT, Path)
    assert config.DATA_DIR == config.PROJ_ROOT / "data"
    assert config.MODELS_DIR == config.PROJ_ROOT / "models"
