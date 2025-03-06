from neural_de.transformations import NightImageEnhancer
import numpy as np
from pathlib import Path
import cv2

TESTS_FOLDER = Path(__file__).parent.parent.parent.resolve()


class TestNightEnhancer:
    def test_transform(self):
        orig_path = Path(TESTS_FOLDER / 'regression/data/night2day/in.png')
        orig_img = cv2.imread(str(orig_path))
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        result_path = Path(TESTS_FOLDER / 'regression/data/night2day/out.png')
        result_img = cv2.imread(str(result_path))
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        purifier = NightImageEnhancer()
        purified = purifier.transform([orig_img])[0]

        assert np.mean(abs((purified * 255).astype(np.uint8) - result_img.astype(np.uint8))) < 0.01
