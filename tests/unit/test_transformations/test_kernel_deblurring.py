from neural_de.transformations.kernel_deblurring_enhancer import KernelDeblurringEnhancer
from neural_de.transformations.kernel_deblurring_enhancer import _KERNELS
import numpy as np
from pytest import raises
from unittest.mock import MagicMock


class TestNightImageEnhancer:
    def test_init(self):
        # incorrect kernel name should raise a valueerror
        with raises(ValueError):
            _ = KernelDeblurringEnhancer(kernel="low")
        # preset kernels
        for kernel in _KERNELS.keys():
            enhancer = KernelDeblurringEnhancer(kernel=kernel)
            assert hasattr(enhancer, "_sharpen_kernel")
            assert np.array_equal(enhancer._sharpen_kernel, _KERNELS[kernel])
            assert np.sum(enhancer._sharpen_kernel) == 1
        # custom
        with raises(ValueError):
            _ = KernelDeblurringEnhancer(custom_kernel="not an array")
        with raises(ValueError):
            _ = KernelDeblurringEnhancer(custom_kernel=[[],[],[2]])
        custom_kernel = [[3, 2, 1], [0.2, 0, 3], [2, 1, -0.5]]
        enhancer = KernelDeblurringEnhancer(custom_kernel=custom_kernel)
        assert np.array_equal(enhancer._sharpen_kernel, np.array(custom_kernel))


    def test_transform(self):
        res_shift = KernelDeblurringEnhancer(kernel="high")

        # empty input check
        img_batch = []
        with raises(ValueError):
            _ = res_shift.transform(img_batch)

        # content not an image or with incorrect number of dimensions should raise an error
        with raises(ValueError):
            _ = res_shift.transform([["not_an_image"]])
        with raises(ValueError):
            _ = res_shift.transform(np.zeros((3, 3, 3, 3, 3)))
        with raises(ValueError):
            _ = res_shift.transform(np.zeros((3, 3)))

        # realistic deblurring call
        for kernel in _KERNELS.keys():
            res_shift = KernelDeblurringEnhancer(kernel=kernel)
            img_batch = [np.full((3, 3, 3), 42.5), np.full((3, 3, 3), 42.5), np.zeros((3,2,1))]
            out = res_shift.transform(img_batch)
            assert len(out) == 3
            assert out[0].shape == (3, 3, 3)
            assert not np.array_equal(out[0], out[2])
            assert np.array_equal(img_batch[1], img_batch[0])
            assert np.array_equal(img_batch[0], out[0])
            assert np.sum(out[2]) == 0
