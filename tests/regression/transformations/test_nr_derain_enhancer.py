import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from neural_de.transformations import DeRainEnhancer

TESTS_FOLDER = Path(__file__).parent.parent.parent.resolve()

ENHANCER = "derain"
REQUIRED_MODEL = "derain_checkpoint.pth"

MODEL_CACHE_DIRECTORY = Path(os.path.expanduser("~")) / ".neuralde" / ENHANCER
TARGET_MODEL = MODEL_CACHE_DIRECTORY / REQUIRED_MODEL


class TestDeRainEnhancer:
    def test_transform(self):
        # assert TARGET_MODEL.is_file()
        orig_path = Path(TESTS_FOLDER / 'regression/data/derain_enhancer/in.jpeg')
        orig_img = cv2.imread(str(orig_path))
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        result_path = Path(TESTS_FOLDER / 'regression/data/derain_enhancer/out.png')
        result_img = cv2.imread(str(result_path))
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        purifier = DeRainEnhancer()
        purified = purifier.transform([orig_img])[0]
        cropped_h = result_img.shape[0]
        cropped_w = result_img.shape[1]
        purified_same_format = (purified * 255).astype(np.uint8)[:cropped_h, :cropped_w]
        assert purified.shape == orig_img.shape
        # The image can have difference at the border, so we only check the mean channel
        # difference is less than 0.1
        assert np.mean(abs(purified_same_format.astype(np.float32) -
                           result_img.astype(np.float32))) < 0.1
