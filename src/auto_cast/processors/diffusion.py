import math

import torch
import torch.nn as nn

from auto_cast.processors.base import Processor
from auto_cast.types import Batch, EncodedBatch, RolloutOutput, Tensor


class NoiseSchedule(nn.Module):
    """Noise Schedule Module."""

    def forward(self, t: Tensor) -> tuple[Tensor, Tensor]:
        """Get alpha and sigma for given time steps t."""
        msg = "Subclasses should implement this method."
        raise NotImplementedError(msg)
    

class LogLinearSchedule(NoiseSchedule):
    """Log-Linear Noise Schedule.
    
    Implements a log-linear schedule for alpha and sigma.
    """

    def __init__(self, sigma_min: float = 0.002, sigma_max: float = 80.0):
        super().__init__()
        self.log_sigma_min = math.log(sigma_min)
        self.log_sigma_max = math.log(sigma_max)

    def forward(self, t: Tensor) -> tuple[Tensor, Tensor]:
        alpha = torch.ones_like(t)
        sigma = torch.exp(self.log_sigma_min * (1 - t) + self.log_sigma_max * t)
        return alpha, sigma


class DiffusionProcessor(Processor):
    """Diffusion Processor."""

    def __init__(self, denoiser_nn, loss, schedule: NoiseSchedule, **kwargs):
        """Initialize the DiffusionProcessor.

        denoiser_nn: The neural network used for denoising.
        loss: The loss function.
        schedule: Noise schedule from azula.noise (e.g., LogLinearSchedule).
                  Defines how signal (alpha) and noise (sigma) scale over time.
        """
        super().__init__()
        self.denoiser_nn = denoiser_nn
        self.schedule = schedule
        self.loss_func = loss

    def map(self, x: Tensor) -> Tensor:
        """Map input window of states/times to output window using denoiser."""
        return self.denoiser_nn(x)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.map(x)

    def q_sample(self, x_0: Tensor, t:Tensor) -> Tensor:
        """Forward diffusion q(x_t | x_0).

        Sample from q(x_t|x_0) = N(alpha_t * x_0, Sigma_t^2*I)
        where alpha_t and sigma_t are obtained from the noise schedule.
        
        Args:
        x_0: clean data (B, C, H, W)
        t: time (B,)

        Returns
        -------
        x_t: noised data at t (B, C, H, W)
        """
        alpha_t, sigma_t = self.schedule(t)
        # Reshape (B,) to (B, 1, 1, 1) for broadcasting with (B, C, H, W)
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        sigma_t = sigma_t.view(-1, 1, 1, 1)
        noise = torch.randn_like(x_0)
        x_t = alpha_t * x_0 + sigma_t * noise
        return x_t
    
    def training_step(self, batch:EncodedBatch, batch_idx:int) -> Tensor:
        """Training step with diffusion loss.

        Sample random time steps and compute loss between denoised output and clean data.
        """
        x_0 = batch.encoded_output_fields  # Clean data : (B, C, H, W)
        B = x_0.size(0)

        # Sample random times in [0, 1] uniformly
        t = torch.rand(B, device=x_0.device)  # (B,)
        x_t = self.q_sample(x_0, t)  # (B, C, H, W)
        x_denoised = self.map(x_t)  # Denoised output : (B, C, H, W)
        loss = self.loss_func(x_denoised, x_0)  # loss comparing clean and denoised data
        return loss