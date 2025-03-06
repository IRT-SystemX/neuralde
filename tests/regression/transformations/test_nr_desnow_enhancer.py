import os
from pathlib import Path

import cv2
import numpy as np
from sympy import reduced_totient

from neural_de.transformations import DeSnowEnhancer

TESTS_DIR = Path(__file__).parent.parent.parent.resolve()

ENHANCER = "desnow"
REQUIRED_MODEL = "prenet_latest.pth"


MODEL_CACHE_DIRECTORY = Path(os.path.expanduser("~")) / ".neuralde" / ENHANCER
TARGET_MODEL = MODEL_CACHE_DIRECTORY / REQUIRED_MODEL


class TestDeSnowEnhancer:
    def test_transform(self):
        # assert TARGET_MODEL.is_file(), "Model not available"
        origin_path = Path(TESTS_DIR / "regression/data/desnow_enhancer/in.png")
        origin_image = cv2.imread(str(origin_path))
        origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
        result_path = Path(TESTS_DIR / 'regression/data/desnow_enhancer/out.png')
        result_image = cv2.imread(str(result_path))
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

        enhancer = DeSnowEnhancer()
        purified = enhancer.transform(np.array([origin_image]))[0]

        assert np.mean(abs((purified).astype(np.uint8) - result_image.astype(np.uint8))) < 0.01
