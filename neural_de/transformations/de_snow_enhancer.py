"""
Snow removal enhancer - Prenet Based implementation
"""
import logging
import os.path
from pathlib import Path

import numpy as np
import torch
from torch.autograd import Variable

from neural_de.transformations.transformation import BaseTransformation
from neural_de.utils.model_manager import ModelManager
from neural_de.utils.math import is_scaled
from neural_de.external.prenet.networks import PReNet

_ENHANCER = "desnow"
_MODEL_FILENAME = "prenet_latest.pth"
_DOWNLOADED_MODEL_PATH = Path(os.path.expanduser("~")) / ".neuralde" / _ENHANCER / _MODEL_FILENAME


class DeSnowEnhancer(BaseTransformation):
    """
    Snow Removal Enhancer, Prenet based implementation.

    ** WARNING ** : The current method may have bad results on real images. The model had been trained
    on a simulated dataset, thus if the dataset is so different of the trained dataset, the results are not guaranteed.

    Args:
        device: Any torch-compatible device string.
        logger: It is recommended to use the Confiance logger, obtainable with
            neural_de.utils.get_logger(...). If None, one logging with stdout will be provided.
    """

    def __init__(self, device: str = "cpu", logger: logging = None):
        super().__init__(logger)
        self.check_device_validity(device)
        self._device = device
        self._prenet_iterations = 4

        # Download model if not available locally
        self._logger.info("Checking model availability...")
        self._model_manager = ModelManager(enhancer=_ENHANCER,
                                           required_model=_MODEL_FILENAME,
                                           logger=self._logger)
        self._model_manager.download_model()

        # Load model from
        self._local_model_path = str(_DOWNLOADED_MODEL_PATH)
        self._purifier = self._setup_model()
        self._logger.info('Model correctly loaded to %s', self._device)

    def _setup_model(self) -> PReNet:
        """
        Load and initialize a PreNet model trained for snow removal.
        """
        self._logger.info('Loading model. Using %s \n', self._device)
        model = PReNet(self._prenet_iterations, self._device == "cuda")
        model.load_state_dict(torch.load(self._local_model_path, map_location=self._device))
        return model.eval().to(self._device)

    def transform(self, images: np.ndarray) -> np.ndarray:
        """
        Removes the snow in a batch of images.

        **WARNING** : The current method may have bad results on real images.

        Args:
          images: Batch of images. Each image should be of a ``np.ndarray`` of target_shape *(h,w,
            channels)*. Images dimensions should be identical across one batch.
        Returns:
          The same images without snow on it.
        """
        self._check_batch_validity(images, same_dim=True)
        if isinstance(images, list):
            self._logger.info("Converting list like batch to numpy array")
            images = np.array(images)

        # Evaluate model on the batch
        batch = images.transpose(0, 3, 1, 2)
        batch = np.float32(batch)

        # normalize only if necessary
        if not is_scaled(images[0]):
            batch = batch / 255

        batch = Variable(torch.Tensor(batch))

        with torch.no_grad():
            if self._device == "cuda":
                torch.cuda.synchronize()
            model_output, _ = self._purifier(batch.to(self._device))
            model_output = torch.clamp(model_output, 0., 1.)
        model_output = np.uint8(255 * model_output.data.cpu().numpy())
        return model_output.transpose(0, 2, 3, 1)
