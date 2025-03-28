"""
DeRain enhancer
"""
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Union

import numpy as np
import torch

from neural_de.transformations.transformation import BaseTransformation
from neural_de.external.derain.blocks import ResNetModified
from neural_de.utils.model_manager import ModelManager
from neural_de.utils.math import get_pad_value, is_scaled


# Only configuration validated : user should not modify it.
@dataclass
class DeRainConfig:
    """
    Interal configuration of the DeRain enhancer.
    """
    upsample_mode: str = "bilinear"
    ngf: int = 64
    n_blocks: int = 9
    use_dropout: bool = False
    input_nc: int = 3
    output_nc: int = 3
    padding_type: str = "reflect"


# Model loaded from Minio
_ENHANCER = "derain"
_MODEL_FILENAME = "derain_checkpoint.pth"
_DOWNLOADED_MODEL_PATH = Path(os.path.expanduser("~")) / ".neuralde" / _ENHANCER / _MODEL_FILENAME


class DeRainEnhancer(BaseTransformation):
    """
    Provides a rain removal image transformation using the GT-Rain Derain Model.

    Args:
        device: Any torch-compatible device string.
        logger: It is recommended to use the Confiance logger, obtainable with
            neural_de.utils.get_logger(...). If None, one logging with stdout will be provided.
    """
    def __init__(self, device: str = 'cpu', logger: logging.Logger = None):
        super().__init__(logger)
        self.check_device_validity(device)
        self._device = device

        # Download model if not available locally
        self._logger.info("Checking model availability...")
        self._model_manager = ModelManager(enhancer=_ENHANCER,
                                           required_model=_MODEL_FILENAME,
                                           logger=self._logger)
        self._model_manager.download_model()

        # Build the GT rain model
        self._resnet = ResNetModified(**asdict(DeRainConfig())).to(self._device)

        # Load train weights
        state_dict = torch.load(_DOWNLOADED_MODEL_PATH, map_location=self._device)["state_dict"]
        # Fix the relative path of resnet in state_dict
        state_dict = {key[key.index(".") + 1:]: val for key, val in state_dict.items()}
        self._resnet.load_state_dict(state_dict, strict=True)

        self._resnet.eval()

        # the resnet used expect inputs to be a multiple of 4
        self._ratio = 4
        self._logger.info("Trained model GT-rain loaded")

    def transform(self, images: Union[list[np.ndarray], np.ndarray]) -> np.ndarray:
        """
        Removes the rain in a batch of images. It differs from style transfer, as it does not remove
        pools and ground reflection. The outputs are as "as if the rained just stop
        falling".

        Args:
          images: Batch of images. Each image should be of a ``np.ndarray`` of target_shape *(h,w,
            channels)*. Images dimensions should be identical across one batch.
        Returns:
          The same images without rain falling on it.
        """
        self._check_batch_validity(images, same_dim=True)
        images = np.array(images, dtype=np.float32)

        # normalize only if necessary
        if not is_scaled(images[0]):
            images = images / 255
        height, width = images.shape[1:3]
        # Crop the images as Derain requires a multiple of 4

        # images = images[:, : height - height % 4, : width - width % 4, :]
        # transform to channel-first, then put the values between [-1,1]
        torch_input = torch.from_numpy(images).permute((0, 3, 1, 2)) * 2 - 1
        pad_h = get_pad_value(height, self._ratio)
        pad_w = get_pad_value(width, self._ratio)
        torch_input = torch.nn.functional.pad(torch_input, (0, pad_w, 0, pad_h),
                                              mode="reflect")
        torch_input = torch_input.to(self._device)
        # Transpose back to channel last and to [0:1]
        torch_output = (self._resnet(torch_input)[0] * 0.5 + 0.5).permute((0, 2, 3, 1))
        torch_output = torch_output[:, :height, :width]
        return torch_output.detach().cpu().numpy()
