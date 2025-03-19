"""
Night to day enhancer - Maxim based implementation
"""

import logging
from dataclasses import dataclass, asdict
from typing import Union
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from neural_de.transformations.transformation import BaseTransformation
from neural_de.external.maxim_tf.create_maxim_model import Model
from neural_de.external.maxim_tf.maxim.configs import MAXIM_CONFIGS
from neural_de.utils.math import get_pad_value, is_scaled
from neural_de.utils.twe_logger import log_and_raise


# To avoid having all the ram locked
physical_devices = tf.config.list_physical_devices('GPU')
for gpu_instance in physical_devices:
    tf.config.experimental.set_memory_growth(gpu_instance, True)

# Static configuration : the method is only valid for these parameters.
_NIGHT_ENHANCER_MODEL = "https://tfhub.dev/sayakpaul/maxim_s-2_enhancement_fivek/1"
_S2_PADDING = 64


@dataclass
class NightConfig:
    """
    Static Enhancer configuration
    """
    variant: str = "S-2"
    dropout_rate: float = 0.0
    num_outputs: int = 3
    use_bias: bool = True
    num_supervision_scales: int = 3


class NightImageEnhancer(BaseTransformation):
    """
    Provides Night to Day image transformation using the MAXIM model.

    Args:
        logger: It is recommended to use the Confiance logger, obtainable with
            neural_de.utils.get_logger(...). If None, one logging with stdout will be provided.
    """

    def __init__(self, device: str = 'cpu', logger: logging.Logger = None):
        super().__init__(logger)
        if device == 'cuda':
            self._device = tf.config.list_logical_devices('GPU')[0].name
        else:
            self._device = device
        self._s2_model = tf.keras.models.load_model(hub.resolve(_NIGHT_ENHANCER_MODEL))
        self._logger.info('Model %s loaded', _NIGHT_ENHANCER_MODEL)
        self._preprocessing_size = None
        self._pipeline = None
        self._config = NightConfig()

    def _init_pipeline(self):
        """
        Initialize the MAXIM model and pipeline.
        """
        with tf.device(self._device):
            configs = MAXIM_CONFIGS.get("S-2")
            configs.update(asdict(self._config))
            configs.update({"input_resolution": self._preprocessing_size})
            self._pipeline = Model(**configs)
            self._pipeline.set_weights(self._s2_model.get_weights())

    @staticmethod
    def _preprocessing(images: np.ndarray):
        """
        Preprocess an image batch for the MAXIM model :
            -  normalize
            -  pad (with reflection) so that the image dimension are a multiple of *S2_PADDING* if
               they are not

        Args:
            images: image batch to preprocess.
        """
        images = np.asarray(images)
        # normalize only if necessary
        if not is_scaled(images[0]):
            images = images / 255
        padh = get_pad_value(images.shape[1], ratio=_S2_PADDING)
        padw = get_pad_value(images.shape[2], ratio=_S2_PADDING)
        padded_images = tf.pad(images, [(0, 0), (padh // 2, padh - padh // 2),
                                        (padw // 2, padw - padw // 2), (0, 0)], mode="REFLECT")
        return padded_images, padh, padw

    def transform(self, images: Union[list[np.ndarray], np.ndarray]):
        """
        Transform a batch of night image into "day images", ie the same image but looking
        as if taken in daylight.

        Args:
          images: Batch of images. Each image should be of a ``np.ndarray`` of target_shape *(h,w,
            channels)*.
        Returns:
          The same images transformed as if taken in daylight.
        """
        self._check_batch_validity(images, same_dim=True)
        if images[0].shape[0] < _S2_PADDING // 2 or images[0].shape[1] < _S2_PADDING // 2:
            log_and_raise(self._logger, ValueError,
                          "Minimum input image size for NightImageEnhancer is 32,32).")
        with tf.device(self._device):
            preprocessed_images, padh, padw = self._preprocessing(images)

            if self._preprocessing_size != preprocessed_images.shape[1:3]:
                self._preprocessing_size = preprocessed_images.shape[1:3]
                self._logger.info("New image shape detected, readying model for size %s",
                                  self._preprocessing_size)
                self._init_pipeline()
                self._logger.info("Model ready for image size %s", self._preprocessing_size)

            preds = self._pipeline.predict(preprocessed_images)
            preds = np.array(preds[-1][-1], np.float32)
            if padh != 0:
                preds = preds[:, padh // 2: padh // 2 - padh]
            if padw != 0:
                preds = preds[:, :, padw // 2: padw // 2 - padw]
        return np.clip(preds, 0.0, 1.0)
