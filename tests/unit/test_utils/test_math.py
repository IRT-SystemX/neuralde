from neural_de.utils.math import get_pad_value, crop_image
from pytest import raises
import numpy as np


def test_get_pad_value():
    with raises(ValueError):
        _ = get_pad_value(0, 0)
    assert get_pad_value(64, 64) == 0
    assert get_pad_value(128, 64) == 0
    assert get_pad_value(3, 32) == 29


def test_crop_image():
    img = np.zeros((10, 10))
    img[:3] = 1
    img[7:] = 1
    ref_img = img[2:8, 2:8]

    res_img = crop_image(img, .4)

    np.testing.assert_array_equal(ref_img, res_img)
