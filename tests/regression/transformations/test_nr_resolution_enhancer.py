from neural_de.transformations.resolution_enhancer import ResolutionEnhancer
import numpy as np
from pathlib import Path
import cv2

TESTS_FOLDER = Path(__file__).parent.parent.parent.resolve()


class TestResolutionShift:
    def test_transform(self):
        res_shift = ResolutionEnhancer(device="cpu")
        orig_path = Path(TESTS_FOLDER / 'regression/data/resolution_enhancer/in.png')
        orig_img = cv2.imread(str(orig_path))
        upsampled_path = Path(TESTS_FOLDER / 'regression/data/resolution_enhancer/out.png')
        upsampled_img = cv2.imread(str(upsampled_path))

        # verify upsampling
        out = res_shift.transform([orig_img], target_shape=(orig_img.shape[0] * 2,
                                                            orig_img.shape[1] * 2))
        assert np.allclose(np.round(out[0] * 255).astype(np.uint8),
                              upsampled_img[:orig_img.shape[0] * 2, :orig_img.shape[1] * 2], rtol=.01)
