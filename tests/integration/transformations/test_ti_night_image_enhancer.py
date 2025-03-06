from neural_de.transformations import NightImageEnhancer
import numpy as np

class TestNightImageEnhancer:
    def test_transform(self):
        enhancer = NightImageEnhancer()

        # realistic downsampling call
        img_batch = np.zeros((3,32,32,3))
        out = enhancer.transform(img_batch)
        assert len(out) == 3
        assert out[0].shape == (32, 32, 3)
        assert np.sum(out) > 0

