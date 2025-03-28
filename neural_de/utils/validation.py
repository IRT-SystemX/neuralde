"""
This module contains a collection of utility functions to verify that the inputs of the
transformation methods are valid.
"""
from typing import Union

import numpy as np
import torch


def is_device_valid(device: str) -> bool:
    """
    Verify if ``device`` is a valid torch device.

    Args:
        device: the device we want to check the validity
    Returns:
        True if the input string is a valid torch device, false if not.
    """
    # Device may be custom or provided by other libraries (like when using torch with tpu), so we
    # check if it is accepted by torch instead of trying a matching regexp.
    try:
        torch.device(device)
    except RuntimeError:
        return False
    return True


def is_batch_valid(image_batch: Union[list, np.ndarray], same_dim=False) -> (bool, str):
    """
    Check if ``images`` is a valid batch of image: either a list of array numpy of dimension
    3 (h,w, c), or a single numpy array of dimension 4 (batch_size, h, w, c).

    Args:
        image_batch: Input to check.
        same_dim: If True, will also verify that every image has the same dimension
    Returns:
        Tuple of (bool,string): The bool is True if the input is a valid batch of images, and
            False otherwise. The string is empty if the bool is True, if not it contains the reason
            of the non validation.
    """
    if isinstance(image_batch, list):
        # Each image should be of dimension 3
        if len(image_batch) == 0:
            return False, "Batch has no images"
        for i, img in enumerate(image_batch):
            if not isinstance(img, np.ndarray) or len(img.shape) != 3:
                return False, f"Image at index {i} is not a valid image." \
                              f" It should be a np.ndarray of 3 dimensions)"
            if same_dim and img.shape != image_batch[0].shape:
                return False, f"All batch images should have the same dimension, but the image at" \
                              f" index {i} does not have the same as the previous ones."
        return True, ""
    if isinstance(image_batch, np.ndarray):
        # Image batch should be of dimension 4
        if image_batch.shape[0] == 0:
            return False, "Batch has no images"
        if len(image_batch.shape) == 4:
            return True, ""
    return False, "Batch is expected to be provided either as a list or as a numpy array"


def is_power_of_two(value: int) -> bool:
    """
    Check if the integer ``value`` is a power of 2.

    Args:
        value: Integer to check

    Returns:
        True if value is a power of 2, else otherwise.
    """
    return (value & (value - 1) == 0) and value != 0
