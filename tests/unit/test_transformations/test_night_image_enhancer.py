from neural_de.transformations.night_image_enhancer import NightImageEnhancer
import numpy as np
from pytest import raises
from unittest.mock import MagicMock


class TestNightImageEnhancer:
    def test_init(self):
        enhancer = NightImageEnhancer()
        assert hasattr(enhancer, "_s2_model")
        assert enhancer._s2_model is not None
        assert enhancer._pipeline is None
        assert enhancer._config.variant == "S-2"

    def test_init_pipeline(self):
        enhancer = NightImageEnhancer()
        enhancer._preprocessing_size = [64, 64]
        enhancer._init_pipeline()
        assert enhancer._pipeline is not None
        assert enhancer._pipeline.input.shape == (None, 64, 64, 3)

    def test_preprocessing(self):
        # test list of image already multiple of 64 : no padding required
        images = [np.full((64, 64, 3), 42), np.ones((64, 64, 3))]
        images, padh, padw = NightImageEnhancer._preprocessing(images)
        assert images.shape == (2, 64, 64, 3)
        assert images[0, -1, -1, 2] == 42 / 255
        assert images[1, 0, 0, 0] == 1 / 255
        assert images[1, -1, 0, 0] == 1 / 255
        assert padh == 0
        assert padw == 0
        # test np.array of 3 images : each should be padded to a multiple of 64
        images = np.zeros((3, 32, 65, 1))
        images, padh, padw = NightImageEnhancer._preprocessing(images)
        assert images.shape == (3, 64, 128, 1)
        assert np.sum(images) == 0
        assert padh == 32
        assert padw == 63

    def test_transform(self):
        res_shift = NightImageEnhancer()

        # empty input check
        img_batch = []
        with raises(ValueError):
            _ = res_shift.transform(img_batch)
        # images with less than 32 pixel in h or w should raise an error
        img_batch = np.zeros((3, 31, 32, 3))
        with raises(ValueError):
            _ = res_shift.transform(img_batch)
        img_batch = np.zeros((3, 32, 31, 3))
        with raises(ValueError):
            _ = res_shift.transform(img_batch)
        # batch with different images shape should raise an error
        img_batch = [np.zeros((64, 64, 3)), np.zeros((64, 64, 3)), np.zeros((64, 65, 3))]
        with raises(ValueError):
            _ = res_shift.transform(img_batch)
        # content not an image or with incorrect number of dimensions should raise an error
        with raises(ValueError):
            _ = res_shift.transform([["not_an_image"]])
        with raises(ValueError):
            _ = res_shift.transform(np.zeros((3, 3, 3, 3, 3)))
        with raises(ValueError):
            _ = res_shift.transform(np.zeros((3, 3)))

        # realistic night enhance call
        img_batch = np.zeros((3, 32, 32, 3))
        res_shift._preprocessing = MagicMock(return_value=(np.zeros((3, 64, 64, 3)), 32, 32))
        res_shift._pipeline = MagicMock()
        res_shift._pipeline.predict = MagicMock(return_value=np.zeros((1, 1, 3, 64, 64, 3)))
        res_shift._init_pipeline = MagicMock()
        out = res_shift.transform(img_batch)
        assert len(out) == 3
        assert out[0].shape == (32, 32, 3)
        assert res_shift._pipeline.predict.call_count == 1
        assert res_shift._preprocessing.call_count == 1
        assert res_shift._init_pipeline.call_count == 1
