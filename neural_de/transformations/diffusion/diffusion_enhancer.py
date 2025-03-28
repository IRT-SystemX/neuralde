import logging
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from neural_de.transformations.diffusion.diffpure_config import DiffPureConfig, ENHANCER, MODEL_FILENAME
from neural_de.transformations.diffusion.rev_guided_diffusion import RevGuidedDiffusion
from neural_de.transformations.transformation import BaseTransformation
from neural_de.utils.model_manager import ModelManager
from neural_de.utils.math import is_scaled


class DiffusionEnhancer(BaseTransformation):
    """
    The goal of this class is to purify a batch of images, to reduce noise and to increase
    robustness against potential adversarial attacks contained in the images. The weights given in
    this librairy are adapted for an output in 256*256 format. Of course, all sizes are
    supported in input but the enhancer will resize the images to 256*256.

    Args:
         device: some steps can be computed with cpu but a gpu is highly recommended.
         config: an instance of the DiffPureConfig class. The most important attributes are: t,
          sample_step and t_delta. Higher t or sample step will lead to a stronger denoising, at
          the cost of processing time. t_delta is the quantity of noise added by the method before
          it's diffusion process : the higher, the higher the chances to remove adversarial attacks,
          at the cost of a potentiel loss of quality in the images.
          The other attributes of DiffPureConfig should be modified for a custom
          Diffusion model.
    """
    def __init__(self,
                 device: torch.DeviceObjType = None,
                 config: Optional[DiffPureConfig] = DiffPureConfig(),
                 logger: logging.Logger = None):
        super().__init__(logger)
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self._logger.info("No device provided, device inferred to be %s", device)
        self._device = device
        self._config = config
        self._model_manager = ModelManager(enhancer=ENHANCER,
                                           required_model=MODEL_FILENAME,
                                           logger=self._logger)
        self._model_manager.download_model()
        self._runner = RevGuidedDiffusion(self._config, device=self._device, logger=self._logger)
        self._runner.eval()

    def forward(self, x: torch.Tensor):
        """
        Apply the diffusion process to a tensor of images.

        Args:
            x: Tensor of batch images

        Returns:
            Tensor of images after diffusion.
        """
        x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x_re = self._runner.image_editing_sample((x - 0.5) * 2)
        return (x_re + 1) * 0.5

    def transform(self, image_batch: Union[np.ndarray, torch.Tensor]):
        """
        "Purify" (removes noise and noise-based adverserial attacks) a batch of input images by
        applying a diffusion process to the images.

        The images are resized to the diffusion model supported size (currently 256*256) :
        you may want to resize/enhance the resolution of the output images. If the input images do
        not have the same h and w, the resizing process will crop to a square image, thus losing
        some information.

        Args:
            image_batch: Batch of images to purify (numpy array or torch.Tensor).

        Returns:
            The batch of purified images (numpy array).
        """
        self._check_batch_validity(image_batch)
        image_batch = np.asarray(image_batch)
        if not is_scaled(image_batch[0]):
            image_batch = image_batch / 255
        image_batch = torch.Tensor(image_batch).permute(0, -1, -3, -2)
        return self.forward(image_batch).permute(0, 2, 3, 1).cpu().detach().numpy()
