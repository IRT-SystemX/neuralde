from neural_de.transformations.de_rain_enhancer import DeRainEnhancer
import numpy as np


def test_derainenhancer():
    # check shape is correct
    model = DeRainEnhancer()
    batch_img = np.ndarray((3, 50, 50, 3))
    purified_image = model.transform(batch_img)
    assert purified_image.shape == (3, 50, 50, 3)
