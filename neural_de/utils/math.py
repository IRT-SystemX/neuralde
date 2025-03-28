"""
Collection of utility math-based utility methods
"""
import numpy as np


def get_pad_value(dim: int, ratio: int) -> int:
    """
    Return the number of pixel to add to *dim* so that it will be a multiple of *ratio*

    Args:
        dim: Actual number of pixel in a dimension
        ratio: Value *dim* should be a multiple of
    Returns:
        Number of pixel to pad
    """
    if ratio <= 0:
        raise ValueError("Ratio parameter should be strictly > 0")
    return ((dim + ratio - 1) // ratio) * ratio - dim


def is_scaled(image: np.ndarray) -> bool:
    """
    Checks to see whether the pixel values have already been rescaled to [0, 1].

    Args:
        image: image batch
    Returns:
        True if image_batch is already scaled
    """
    if image.dtype == np.uint8:
        return False

    # It's possible the image has pixel values in [0, 255] but is of floating type
    return np.min(image) >= 0 and np.max(image) <= 1


def crop_image(image: np.ndarray, ratio: float) -> np.ndarray:
    """
        Return cropped image depends on a ratio of its size
        Ratio = .3 => crop 30% of image size

        Args:
            image: original image
            ratio: Value in rage [0., 1.[
        Returns:
            Cropped image from its center
    """
    if ratio < 0 or ratio >= 1:
        raise ValueError("Ratio parameter should be >= 0 and < 1")

    x, y = image.shape[0], image.shape[1]
    ratio = ratio / 2

    return image[int(x * ratio):int(x * (1 - ratio)), int(y * ratio):int(y * (1 - ratio))]
