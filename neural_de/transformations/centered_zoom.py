"""
Simple wrapper to share experimental results on working params for a CenteredZoom
"""
import logging
from typing import Union

import numpy as np

from neural_de.utils.math import crop_image
from neural_de.transformations.transformation import BaseTransformation


class CenteredZoom(BaseTransformation):
    """
    CenteredZoom image transformation based on a numpy implementation.
    Given a batch of 3-channels image of size width x height, return the centered tile of size
    width*keep_ratio x height*keep_ratio.
    This transformation does not perform any resolution enhancement of the returned content.
    See ResolutionEnhancer to perform both crop and resolution enhancement.

    Args:
        keep_ratio: The proportion of the input image we keep. Must be in ]0,1[.
        logger: It is recommended to use the Confiance logger, obtainable with
            neural_de.utils.get_logger(...). If None, one logging with stdout will be provided.
    """

    def __init__(self, keep_ratio: float, logger: logging.Logger = None):
        super().__init__(logger)
        self._keep_ratio = keep_ratio

    def transform(self, images: Union[list[np.ndarray], np.ndarray]):
        """
        Apply CenteredZoom to a batch of images using numpy slicing method.

        Args:
          images: Batch of images. Each image should be of a ``np.ndarray`` of target_shape *(h,w,
            channels)*. Images dimensions do not need to be the same across the batch.
        Returns:
          The images zoomed to a given ratio in respect to its center.
        """
        self._check_batch_validity(images)
        return [crop_image(image=image, ratio=1 - self._keep_ratio) for image in images]

    def transform_with_annotations(self, images: Union[list[np.ndarray], np.ndarray], bbox: list):
        """
        Transform bounding boxes to the reference in the new zoomed image.
        Args:
            images: Batch of images. Each image should be of a ``np.ndarray`` of target_shape *(h,w,
            channels)*. Images dimensions do not need to be the same across the batch.
            bbox: list of list of list [batch_dim, nb_object_per_image, [x1, y1, x2, y2]]
            with the x1, y1, x2, y2 bounding box position in the original image.

        Returns:
            The images zoomed to a given ratio in respect to its center.
            The list of ``np.ndarray`` with the x1, y1, x2, y2 bounding box position in the zoomed
             image.
        """
        transformed_images = self.transform(images)
        transformed_bbox = []
        for ind, image in enumerate(images):
            crop_x = image.shape[0] * ((1 - self._keep_ratio) / 2)
            crop_y = image.shape[1] * ((1 - self._keep_ratio) / 2)
            new_bbox = [[lp[0] - crop_x, lp[1] - crop_y, lp[2] - crop_x, lp[3] - crop_y] for lp in bbox[ind]]
            transformed_bbox.append(np.array(new_bbox))
        return transformed_images, transformed_bbox
