import os
from pathlib import Path

from neural_de.utils.model_manager import ModelManager
from neural_de.utils.twe_logger import get_logger

ENHANCER = "desnow"
REQUIRED_MODEL = "prenet_latest.pth"
MODEL_DIRECTORY = Path(os.path.expanduser("~")) / ".neuralde" / ENHANCER

TESTS_DIR = Path(__file__).parent.parent.parent.resolve()
MODEL_MANAGER = ModelManager(enhancer=ENHANCER, required_model=REQUIRED_MODEL, logger=get_logger())


def test_model_validity():
    checksums = MODEL_MANAGER._load_checksums()
    assert REQUIRED_MODEL in checksums
    assert MODEL_MANAGER._is_model_valid()
