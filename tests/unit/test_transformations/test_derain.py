import os
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch
from pytest import raises

from neural_de.transformations.de_rain_enhancer import DeRainEnhancer

TESTS_DIR = Path(__file__).parent.parent.parent.resolve()

ENHANCER = "derain"
REQUIRED_MODEL = "derain_checkpoint.pth"

MODEL_CACHE_DIRECTORY = Path(os.path.expanduser("~")) / ".neuralde" / ENHANCER
TARGET_MODEL = MODEL_CACHE_DIRECTORY / REQUIRED_MODEL


class TestDeRainEnhancer:
    def test_init(self):
        enhancer = DeRainEnhancer()
        assert hasattr(enhancer, "_logger")
        assert hasattr(enhancer, "_device")
        assert enhancer._device == "cpu"
        assert enhancer._resnet is not None

    def test_transform(self):
        res_shift = DeRainEnhancer()

        # empty input check
        img_batch = []
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
        img_batch = np.zeros((3, 30, 30, 3))
        res_shift._resnet = MagicMock(return_value=(torch.from_numpy(np.zeros((3, 3, 32, 32))),))
        out = res_shift.transform(img_batch)
        assert len(out) == 3
        assert out[0].shape == img_batch.shape[1:]
        assert res_shift._resnet.call_count == 1
        assert np.min(out) >= 0
        assert np.max(out) <= 1
