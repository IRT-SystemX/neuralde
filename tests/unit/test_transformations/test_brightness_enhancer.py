from neural_de.transformations.brightness_enhancer import BrightnessEnhancer
from neural_de.external.nplie.nplie import NPLIE
import numpy as np

class TestBrightnessEnhancer:

    def test_transform_brightness(self):
        #Images source
        image = np.ones(shape=(645, 432, 3), dtype=np.float32)
        image[:,:,0] *= 0.147
        image[:,:,1] *= 0.369
        image[:,:,2] *= 0.258
        image2 = np.ones(shape=(546, 342, 3), dtype=np.float32)
        image2[:, :, 0] *= 0.147
        image2[:, :, 1] *= 0.369
        image2[:, :, 2] *= 0.258
        # Image target
        src = NPLIE(image)
        #Test
        bright_ehn = BrightnessEnhancer()
        transformed_image = bright_ehn.transform([image])
        assert np.max(np.abs(transformed_image[0] - src)) < 0.01
        #Test type of result
        assert isinstance(transformed_image[0], np.ndarray)
        #Test dimensions of the images
        assert (transformed_image[0].shape == image.shape)
        #Batch test
        transformed_image2 = bright_ehn.transform([image, image2])
        assert (len(transformed_image2) == 2)
        assert (transformed_image2[1].shape == image2.shape)