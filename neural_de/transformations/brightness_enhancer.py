"""
Image brightness enhancement method.
"""
from typing import Union
import numpy as np
from neural_de.transformations.transformation import BaseTransformation
from neural_de.external.nplie.nplie import NPLIE
from neural_de.utils.math import is_scaled


class BrightnessEnhancer(BaseTransformation):
    """

    BaseTransformation method for image brightness change.
    It uses NPLIE-based method for brightness enhancement, and Opencv for transforming the
    image.

    Example :

    See the notebook `examples/BrightnessEnhancer_example.ipynb` for more usage details.

    1- Import the class

    . code-block:: python

    from neural_de.transformations import BrightnessEnhancer

    2- Create an instance of BrightnessEnhancer.

    . code-block:: python

    bright_ehn = BrightnessEnhancer()

    3- Apply the brightness change to a batch of images to a given shape

    . code-block:: python

    out_images = bright_ehn.transform(images)

    Args:
        logger: It is recommended to use the Confiance logger, obtainable with
        neural_de.utils.get_logger(...).
        If None, one logging with stdout will be provided.
    """

    def enhance_brightness(self, image):
        """
        Args:
            Image: numpy array format with float32 dtype.
        Returns:
            Image numpy array format with float32 dtype.
        """
        image = image.astype(np.float32)
        if not is_scaled(image):
            self._logger.info("Image normalized as between [0;1]")
            image /= 255.
        return NPLIE(image)

    def transform(self, images: Union[list[np.ndarray], np.ndarray]):
        """
        Improve brightness a batch of images using a NPLIE-based method.

        Args:
          images: Batch of images. Each image should be of a ``np.ndarray`` of target_shape *(h,w,
            channels)*. Images dimensions do not need to be the same across the batch.
        Returns:
          The same images with improved brightness.
        """
        self._check_batch_validity(images)
        return [self.enhance_brightness(image) for image in images]
