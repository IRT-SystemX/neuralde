import logging

import torch

from neural_de.utils.twe_logger import get_logger


class RevVPSDE(torch.nn.Module):
    """
    Constructs a Variance Preserving SDE.

    Args:
        model: diffusion model
        beta_min: min value of beta for normalisation
        beta_max: max value of beta for normalisation
        N: scaling factor
        img_shape: Image dimension, channel-first.
        logger: logger (logging.Logger)
    """
    def __init__(self,
                 model: torch.nn.Module,
                 beta_min: float = 0.1,
                 beta_max: float = 20,
                 N: int = 1000,
                 img_shape: tuple = (3, 256, 256),
                 logger: logging.Logger = None):

        super().__init__()
        self._logger = logger if logger is not None else get_logger()
        self._model = model
        self._img_shape = img_shape

        self._beta_0 = beta_min
        self._beta_1 = beta_max
        self._N = N
        self._beta_range = -0.5 * (self._beta_1 - self._beta_0)

        self.noise_type = "diagonal"
        self.sde_type = "ito"
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)

    def vpsde_fn(self, t: torch.Tensor, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply variant-preserving sde to a batch of images.

        Args:
            timesteps: current timestep
            x: image batch
        """
        beta_t = self._beta_0 + t * (self._beta_1 - self._beta_0)
        drift = -0.5 * beta_t[:, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def _extract_info_from_output_tensor(self, timesteps: torch.Tensor, broadcast_shape: tuple)\
            -> torch.Tensor:
        """
        Compute and broadcast a multiplicative factor used by vpsde to obtain drift score.

        Args:
            timesteps: current timestep
            broadcast_shape: target shape
        """
        res = torch.exp(self._beta_range * timesteps**2 - self._beta_0 * timesteps)
        res = (-1. / torch.sqrt(1. - res)).float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        return res.expand(broadcast_shape)

    def rvpsde_fn(self, t: torch.Tensor, x: torch.Tensor, return_type: str = 'drift'):
        """
        Create the drift and diffusion functions for the reverse SDE

        Args:
            t: current step
            x: batch of input images
            return_type: if "drift", will apply a drift following the diffusion. If not, only the
                         diffusion will be performed.
        """
        drift, diffusion = self.vpsde_fn(t, x)

        if return_type != 'drift':
            return diffusion

        x_img = x.view(-1, *self._img_shape)
        disc_steps = (t.float() * self._N).long()
        model_output = self._model(x_img, disc_steps)
        model_output, _ = torch.split(model_output, self._img_shape[0], dim=1)
        model_output = model_output.view(x.shape[0], -1)
        score = self._extract_info_from_output_tensor(t, x.shape) * model_output
        return drift - diffusion[:, None] ** 2 * score

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Creates the drift function -f(x, 1-t) (by t' = 1 - t)
        Sdeint only support a 2D tensor (batch_size, c*h*w)

        Args:
            t: current step
            x: batch of input images
        """
        t = t.expand(x.shape[0])
        return - self.rvpsde_fn(1 - t, x, return_type='drift')

    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Create the diffusion function g(1-t) (by t' = 1 - t)
        sdeint only support a 2D tensor (batch_size, c*h*w)

        Args:
            t: current step
            x: batch of input images
        """
        t = t.expand(x.shape[0])
        diffusion = self.rvpsde_fn(1 - t, x, return_type='diffusion')
        return diffusion[:, None].expand(x.shape)
