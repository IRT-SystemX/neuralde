from neural_de.transformations.resolution_enhancer import ResolutionEnhancer
from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor
from unittest.mock import MagicMock
from pytest import raises
import torch
import numpy as np


class TestResolutionShift:

    def test___init__(self):
        res_shift = ResolutionEnhancer(device="cpu")
        assert hasattr(res_shift, "_logger")
        res_shift._logger = MagicMock(return_type=None)
        assert hasattr(res_shift, "_device")
        assert res_shift._device == "cpu"
        assert res_shift._model is None
        assert res_shift._processor is None
        res_shift._logger.assert_not_called()

        with raises(TypeError):
            ResolutionEnhancer(device="gpu")

    def test__init_nn(self):
        devices = ["cpu"]
        # test all available devices
        if torch.cuda.is_available():
            devices.append("cuda")
        for device in devices:
            res_shift = ResolutionEnhancer(device=device)
            assert res_shift._model is None
            res_shift._init_nn()
            assert isinstance(res_shift._model, Swin2SRForImageSuperResolution)
            assert isinstance(res_shift._processor, Swin2SRImageProcessor)

    def test_intermediate_sampling(self):
        res_shift = ResolutionEnhancer(device="cpu")
        images = [np.zeros((1, 1, 1)),
                  np.zeros((42, 42, 42), dtype=np.uint8),
                  np.full((1800, 1200, 3), 42, dtype=float),
                  np.full((20, 10, 1), 0), np.ones((5, 5, 3))]
        ratios = [(2,2), (21,21), (400,300), (40,10), (50,50)]
        impossible_ratios = [(1,1), (1,1), (0,0),(1,1), (0,0)]

        for i, img in enumerate(images):
            img_shifted = res_shift._intermediate_sampling(img, ratios[i])
            assert img_shifted.shape[0] == int(ratios[i][0]//2)
            assert img_shifted.shape[1] == int(ratios[i][1]//2)
            with raises(ValueError):
                _ = res_shift._intermediate_sampling(img, impossible_ratios[i])

    def test_upsample(self):
        res_shift = ResolutionEnhancer(device="cpu")
        images_batchs = [np.zeros((1, 5, 5, 1)),
                  np.zeros((1, 42, 42, 3), dtype=np.uint8),
                  np.full((3, 8, 16, 3), 42, dtype=float),
                  np.full((4, 20, 10, 1), 0), np.ones((1, 5, 5, 3))]

        for i, imgs in enumerate(images_batchs):
            img_shifted = res_shift._upsample(imgs)
            assert img_shifted.shape[0] == imgs.shape[0]
            assert img_shifted.shape[1] > imgs.shape[1] * 2
            assert img_shifted.shape[1] <= imgs.shape[1] * 2 + 16  # padding may raise the shape up to 16 pixels
            assert img_shifted.shape[2] <= imgs.shape[2] * 2 + 16
            assert img_shifted.shape[2] > imgs.shape[2] * 2
            assert img_shifted.shape[3] == 3


    def test_transform(self):
        res_shift = ResolutionEnhancer(device="cpu")


        # empty input check
        img_batch = []
        with raises(ValueError):
            _ = res_shift.transform(img_batch, target_shape=(5, 5))
        # negative ratio should raise an error
        img_batch = np.zeros((1,3,3,3))
        with raises(ValueError):
            _ = res_shift.transform(img_batch, target_shape=(0, 0))
        # Invalid ratio parameter should raise a type error
        with raises(TypeError):
            _ = res_shift.transform(img_batch, target_shape="not_a_valid_shape")
        with raises(TypeError):
            _ = res_shift.transform(img_batch, target_shape=1)
        # content not an image or with incorrect number of dimensions should raise an error
        with raises(ValueError):
            _ = res_shift.transform([["not_an_image"]], target_shape=(3,3))
        with raises(ValueError):
            _ = res_shift.transform(np.zeros((3,3,3,3,3)), target_shape=(3,3))
        with raises(ValueError):
            _ = res_shift.transform(np.zeros((3,3)), target_shape=(5,5))

        # realistic upsampling call
        img_batch = np.zeros((3, 10, 10, 3))
        res_shift._upsample = MagicMock(return_value=np.zeros((3, 16, 16, 3)))
        res_shift._intermediate_sampling = MagicMock(return_value=np.zeros((7, 6, 3)))
        out = res_shift.transform(img_batch, target_shape=(15, 12))
        assert len(out) == 3
        assert out[0].shape == (15, 12, 3)
        assert res_shift._upsample.call_count == 1
        assert res_shift._intermediate_sampling.call_count == 3

        # realistic inplace call
        img_batch = [np.zeros((3, 3, 1))]
        res_shift._intermediate_sampling = MagicMock(return_value=np.zeros((1, 1, 1)))
        res_shift._upsample = MagicMock(return_value=np.zeros((1, 3, 3, 1)))
        out = res_shift.transform(img_batch, target_shape=(3, 3))
        assert len(out) == 1
        assert out[0].shape == (3, 3, 1)
        res_shift._intermediate_sampling.assert_called_once()
        res_shift._upsample.assert_called_once()
