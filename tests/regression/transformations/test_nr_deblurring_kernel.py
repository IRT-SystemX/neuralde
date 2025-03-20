from neural_de.transformations.kernel_deblurring_enhancer import KernelDeblurringEnhancer
import numpy as np
from pathlib import Path
import cv2

TESTS_FOLDER = Path(__file__).parent.parent.parent.resolve()


class TestKernelDeblurringEnhancer:
    def test_transform(self):
        orig_path = Path(TESTS_FOLDER / "regression/data/kernel_deblurring/in.jpg")
        orig_img = cv2.imread(str(orig_path))
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        result_path = Path(TESTS_FOLDER / 'regression/data/kernel_deblurring/out.png')
        result_img = cv2.imread(str(result_path))
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        enhancer = KernelDeblurringEnhancer()
        purified = enhancer.transform([orig_img])[0]

        assert np.array_equal(purified, result_img)
