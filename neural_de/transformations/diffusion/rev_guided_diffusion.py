import logging

import numpy as np
import torch
import torchsde

from neural_de.transformations.diffusion.diffpure_config import DiffPureConfig, CHANNEL_MULT
from neural_de.transformations.diffusion.rev_vpsde import RevVPSDE
from neural_de.utils.twe_logger import get_logger
from neural_de.transformations.diffusion.unet.unet_model import UNetModel


class RevGuidedDiffusion(torch.nn.Module):
    """
    Implements the rev-guided diffusion.

    Args:
        device: "cuda" or "cpu". Gpu is highly recommended but somme steps are available with cpu.
        config: An instance of DiffPureConfig, it has been created in the input of the
                DiffusionEhnancer class.
        logger: logger.
    """
    def __init__(self, config: DiffPureConfig, device: torch.DeviceObjType = None,
                 logger: logging.Logger = None):
        super().__init__()
        self._config = config
        self._logger = logger if logger is not None else get_logger()

        self._device = device

        self._logger.info('Building DiffPure model')
        self._logger.debug(f'Model Diffpure loaded with config : {self._config}')
        if not torch.cuda.is_available():
            self._config.use_fp16 = False
            self._logger.info("No cuda detected, the diffusion model will use cpu, "
                              "which provokes very slow inferences")

        if config.channel_mult is None:
            try:
                config.channel_mult = CHANNEL_MULT[config.image_size]
            except KeyError:
                raise NotImplementedError(f"unsupported image size: {config.image_size}")

        config.attention_resolutions = config.image_size // np.array(config.attention_resolutions).astype(np.int32)

        out_channels = 3 if not self._config.learn_sigma else 6
        self._model = UNetModel(in_channels=3, out_channels=out_channels, config=self._config,
                                logger=self._logger)
        self._logger.info(f'Loading DiffPure weights to device : {self._device}')
        self._model.load_state_dict(torch.load(self._config.weights_path,
                                               map_location=self._device))
        if self._config.use_fp16:
            self._model.convert_to_fp16()

        self._model.eval().to(self._device)
        self._rev_vpsde = RevVPSDE(model=self._model, img_shape=self._config.img_shape,
                                   logger=self._logger).to(self._device)
        self._betas = self._rev_vpsde.discrete_betas.float().to(self._device)

    def image_editing_sample(self, img: torch.Tensor):
        """
        This method apply the rev-guided diffusion to a batch of images.

        Args:
            img: Tensor (batch of images)

        Returns:
            Tensor (batch of images)
        """
        batch_size = img.shape[0]
        state_size = int(np.prod(img.shape[1:]))  # c*h*w

        img = img.to(self._device)
        x0 = img
        xs = []
        for it in range(self._config.sample_step):
            e = torch.randn_like(x0).to(self._device)
            total_noise_levels = self._config.t
            if self._config.rand_t:
                total_noise_levels = self._config.t + np.random.randint(-self._config.t_delta,
                                                                        self._config.t_delta)
            a = (1 - self._betas).cumprod(dim=0).to(self._device)
            x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()
            epsilon_dt0, epsilon_dt1 = 0, 1e-5
            t0, t1 = 1 - self._config.t * 1. / 1000 + epsilon_dt0, 1 - epsilon_dt1
            t_size = 2
            ts = torch.linspace(t0, t1, t_size).to(self._device)
            x_ = x.reshape(batch_size, -1)  # (batch_size, state_size)
            if self._config.use_bm:
                bm = torchsde.BrownianInterval(t0=t0, t1=t1, size=(batch_size, state_size),
                                               device=self._device)
                xs_ = torchsde.sdeint_adjoint(self._rev_vpsde, x_, ts, method='euler', bm=bm)
            else:
                xs_ = torchsde.sdeint_adjoint(self._rev_vpsde, x_, ts, method='euler')
            x0 = xs_[-1].view(x.shape)  # (batch_size, c, h, w)
            xs.append(x0)
        return torch.cat(xs, dim=0)
