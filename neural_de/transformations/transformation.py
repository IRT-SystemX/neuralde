"""
Parent class for any transformation method.

"""
from __future__ import annotations

import logging
from typing import Union
import numpy as np
from neural_de.utils.twe_logger import log_and_raise, get_logger
from neural_de.utils.validation import is_batch_valid, is_device_valid


class BaseTransformation:
    """
    Parent class for any transformation methods of the library.
    Provides the methods for logging and input validation.

    Args:
        logger: logging.logger. It is recommended to use the Confiance one, obtainable with
            neural_de.utils.get_logger(...)
    """
    def __init__(self, logger: logging.Logger = None):
        if logger is None:
            self._logger = get_logger()
        else:
            self._logger = logger

    def _check_batch_validity(self, images: Union[list, np.ndarray], same_dim: bool = False):
        """
        Check if the batch of images provided by the user conforms to the expected standards.
        Raises an error if it does not.

        Args:
            images: list / batch of images to validate.
            same_dim: Check if all images have the same dimension (Optional).
        Returns:
            None
        """
        is_valid, reason = is_batch_valid(images, same_dim=same_dim)
        if not is_valid:
            log_and_raise(self._logger, ValueError,
                          "Parameter images is not a valid input batch:" + reason)

    def check_device_validity(self, device: str = "cpu"):
        """
        Check if the selected device is valid.
        Args:
            device: str - cpu / cuda
        Returns:
            None
        """
        if not is_device_valid(device):
            log_and_raise(self._logger, TypeError,
                          f"Device {device} is not a valid Pytorch device")
