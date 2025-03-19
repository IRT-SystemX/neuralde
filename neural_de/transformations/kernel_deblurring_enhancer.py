"""
Simple wrapper to share experimental results on working params for a Deblurring Kernel
"""
import logging
from typing import Union

import cv2
import numpy as np

from neural_de.transformations.transformation import BaseTransformation
from neural_de.utils.twe_logger import log_and_raise

# Preset Kernels
_KERNELS = {
    "medium": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    "high": np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
}


class KernelDeblurringEnhancer(BaseTransformation):
    """
    Kernel Deblurring image transformation based on OpenCv implementation. Provides pre-set
    filter of medium and high intensity.

    Args:
        kernel: ``high`` or ``medium``: use a pre-set kernel with high or medium intensity.
        custom: Optional, custom kernel to use. It can be any non empty 2D matrix. If provided,
            the value of `kernel` will not be used.
        logger: It is recommended to use the Confiance logger, obtainable with
            neural_de.utils.get_logger(...). If None, one logging with stdout will be provided.
    """

    def __init__(self, kernel: str = "high", custom_kernel: Union[list, np.ndarray] = None,
                 logger: logging.Logger = None):
        super().__init__(logger)
        if custom_kernel is not None:
            try:
                self._sharpen_kernel = np.array(custom_kernel)
            except ValueError:
                log_and_raise(self._logger, ValueError, "Could not convert parameter custom_kernel "
                                                        "to a valid array")
            if len(self._sharpen_kernel.shape) != 2:
                log_and_raise(self._logger, ValueError, "Parameter custom_kernel should have "
                                                        "exactly two dimensions")
            self._logger.info("Custom weights loaded")
        elif kernel in _KERNELS:
            self._sharpen_kernel = _KERNELS[kernel]
        else:
            log_and_raise(self._logger, ValueError, f"Either a valid kernel type (one of "
                                                    f"{_KERNELS.keys()}) or custom_kernel should be"
                                                    f" provided")

    def transform(self, images: Union[list[np.ndarray], np.ndarray]):
        """
        Deblur a batch of images using a Kernel-based method.

        Args:
          images: Batch of images. Each image should be of a ``np.ndarray`` of target_shape *(h,w,
            channels)*. Images dimensions do not need to be the same across the batch.
        Returns:
          The same images with less blurr.
        """
        self._check_batch_validity(images)
        return [cv2.filter2D(src=image, kernel=self._sharpen_kernel, ddepth=-1)
                for image in images]
