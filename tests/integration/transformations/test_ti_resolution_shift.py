from neural_de.transformations.resolution_enhancer import ResolutionEnhancer
import numpy as np


class TestResolutionShift:
    def test_transform(self):
        res_shift = ResolutionEnhancer(device="cpu")

        # realistic downsampling call
        img_batch = np.zeros((3, 10, 10, 3))
        out = res_shift.transform(img_batch, target_shape=(5, 5))
        assert len(out) == 3
        assert out[0].shape == (5, 5, 3)
        assert np.sum(out) <= 0.5

        # realistic upsampling call
        img_batch = [np.full((30, 30, 3), 42)]
        out = res_shift.transform(img_batch, target_shape=(60, 60, 3))
        assert len(out) == 1
        assert out[0].shape == (60, 60, 3)  # closest superior power of two
        assert np.sum(np.round(out * 255)) == 42 * 60 * 60 * 3
