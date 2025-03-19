"""
Implementation of the ResolutionEnhancer method.

Attributes:
    UPSCALE_MODEL: version of the transformer model used for image upscaling
"""

import logging
from typing import Union
import cv2
import numpy as np
from torch import no_grad
from transformers import Swin2SRForImageSuperResolution
from transformers import Swin2SRImageProcessor
from neural_de.transformations.centered_zoom import CenteredZoom
from neural_de.utils.twe_logger import log_and_raise
from neural_de.transformations.transformation import BaseTransformation


UPSCALE_MODEL: str = "caidas/swin2SR-classical-sr-x2-64"


class ResolutionEnhancer(BaseTransformation):
    """
    BaseTransformation method for image resolution change.
    It uses neural-based method for resolution enhancement, and Opencv for diminishing the
    resolution.

    Example :
        See the notebook `examples/ResolutionEnhancer_example.ipynb` for more usage details.

        1- Import the class

        .. code-block:: python

                from neural_de.transformations import ResolutionEnhancer

        2- Create an instance of ResolutionEnhancer.
        ``device ="Cuda"`` is recommended if you have a gpu and torch with cuda enabled.

        .. code-block:: python

                res_shift = ResolutionEnhancer(device= "cpu")

        3- Apply the resolution change to a batch of images to a given shape

        .. code-block:: python

                out_images = res_shift.transform(images, ratio=2)

    Args:
        device: Any torch-compatible device string.
        logger: It is recommended to use the Confiance logger, obtainable with
            neural_de.utils.get_logger(...). If None, one logging with stdout will be provided.
    """
    def __init__(self, device: str = 'cpu', logger: logging.Logger = None):
        super().__init__(logger)

        self.check_device_validity(device)
        self._device = device
        self._processor = None
        self._model = None
        self._logger.info("ResolutionEnhancer Initialized ")

    def _init_nn(self) -> None:
        """
        Initialise the Swin2SR neural network used for image upsampling.
        """
        self._processor = Swin2SRImageProcessor(do_pad=True)
        self._model = Swin2SRForImageSuperResolution.from_pretrained(UPSCALE_MODEL) \
            .to(self._device)
        self._logger.info("Swin2 model loaded to %s", self._device)

    def _intermediate_sampling(self, image: np.ndarray, shape: tuple) -> np.ndarray:
        """Uses **OpenCv** resize to get the image resolution to half the size of the final
        target_shape.

        Args:
          image: Image to resize.
          shape: Target size
        Returns:
          Resized image with half the size of target size
        """
        image = image.astype(np.uint8)
        n_cols = shape[1] // 2
        n_rows = shape[0] // 2
        if n_cols < 1 or n_rows < 1:
            log_and_raise(self._logger, ValueError,
                          "Target target_shape is too small : no pixel on at least one dimension")
        return cv2.resize(image, (n_cols, n_rows), interpolation=cv2.INTER_LINEAR)

    def _upsample(self, images: np.ndarray) -> np.ndarray:
        """Uses a **SwinTransformer** to raise the resolution of image by a factor 2.

        Args:
          images: Batch of identically shaped images to resize.
        Returns:
          Resized image
        """
        if self._processor is None:
            self._init_nn()

        pixel_values = self._processor(images, return_tensors="pt").pixel_values
        with no_grad():
            outputs = self._model(pixel_values.to(self._device))

        output = outputs.reconstruction.data.float().cpu().clamp_(0, 1).numpy()
        output = np.moveaxis(output, source=1, destination=-1)
        # output = (output * 255.0).round().astype(np.uint8)
        return output

    def transform(self, images: Union[list[np.ndarray], np.ndarray],
                  target_shape: Union[list, tuple], crop_ratio: float = 0.) -> np.ndarray:
        """
        Modify the resolution of a batch of images to a given target_shape.

        Args:
          images: Batch of images. Each image should be of a ``np.ndarray`` of target_shape *(h,w,
            channels)*
          target_shape:  New resolution (h,w) in pixel.
          crop_ratio: image cropping ratio (range in [0., 1.[)
        Returns:
          Images with new resolution.
        """
        self._check_batch_validity(images)
        if not isinstance(target_shape, tuple):
            try:
                target_shape = tuple(target_shape)
            except TypeError:
                log_and_raise(self._logger, TypeError,
                              "Type tuple expected for parameter *target_shape*")

        if target_shape[0] <= 0 or target_shape[1] <= 0:
            log_and_raise(self._logger, ValueError, "Target dimensions should be > 0")

        if crop_ratio == 0:
            intermediate_images = np.array(
                [self._intermediate_sampling(img, target_shape) for img in images])
        else:
            centered_zoom_transformer = CenteredZoom(keep_ratio=1 - crop_ratio)
            transformed_img = centered_zoom_transformer.transform(images)
            intermediate_images = np.array(
                [self._intermediate_sampling(img, target_shape) for img in transformed_img])
        return self._upsample(intermediate_images)[:, :target_shape[0], :target_shape[1]]
