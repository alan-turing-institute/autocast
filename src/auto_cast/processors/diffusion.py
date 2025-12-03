import math

import torch
import torch.nn as nn

from auto_cast.processors.base import Processor
from auto_cast.types import Batch, EncodedBatch, RolloutOutput, Tensor

from azula.noise import Schedule, VESchedule, VPSchedule, CosineSchedule, RectifiedSchedule
from azula.denoise import (
    Denoiser, 
    SimpleDenoiser, 
    KarrasDenoiser,
    DiracPosterior,
    GaussianPosterior
)


class DiffusionProcessor(Processor):
    """Diffusion Processor."""

    def __init__(
        self,
        backbone: nn.Module,
        schedule: Schedule,
        denoiser_type: str = 'karras',
        teacher_forcing_ratio: float = 0.0,
        stride: int = 1,
        max_rollout_steps: int = 10,
        learning_rate: float = 1e-4,
    ):

        super().__init__()
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.stride = stride
        self.max_rollout_steps = max_rollout_steps
        self.learning_rate = learning_rate
        
        # Create Azula denoiser with chosen preconditioning
        if denoiser_type == 'simple':
            self.denoiser = SimpleDenoiser(backbone=backbone, schedule=schedule)
        elif denoiser_type == 'karras':
            self.denoiser = KarrasDenoiser(backbone=backbone, schedule=schedule)
        else:
            raise ValueError(f"Unknown denoiser type: {denoiser_type}")
                
        # Store schedule for direct access
        self.schedule = schedule

    def map(self, x: Tensor) -> Tensor:
        """Map input window of states/times to output window using denoiser."""
        t = torch.zeros(x.shape[0], device=x.device)
        return self._denoise(x, t)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.map(x)
    
    def _denoise(self, x: Tensor, t: Tensor) -> Tensor:
        posterior = self.denoiser(x, t)
        return posterior.mean
    
    def q_sample(self, x_0: Tensor, t:Tensor) -> Tensor:
        """Forward diffusion q(x_t | x_0).

        Sample from q(x_t|x_0) = N(alpha_t * x_0, Sigma_t^2*I)
        where alpha_t and sigma_t are obtained from the noise schedule.
        
        Args:
        x_0: clean data (B, T, C, H, W)
        t: time (B,)

        Returns
        -------
        x_t: noised data at t (B, T,C, H, W)
        """
        alpha_t, sigma_t = self.schedule(t)
        # Reshape (B,) to (B, 1, 1, 1, 1) for broadcasting with (B, T, C, H, W)
        alpha_t = alpha_t.view(-1, 1, 1, 1, 1)
        sigma_t = sigma_t.view(-1, 1, 1, 1, 1)
        noise = torch.randn_like(x_0)
        x_t = alpha_t * x_0 + sigma_t * noise
        return x_t
    
    def training_step(self, batch:EncodedBatch, batch_idx:int) -> Tensor:
        """Training step with diffusion loss.

        Sample random time steps and compute loss between denoised output and clean data.
        """
        x_0 = batch.encoded_output_fields  # Clean data : (B, C, H, W)

        # Sample random times in [0, 1] uniformly
        t = torch.rand(x_0.size(0), device=x_0.device)  # (B,)

        # OPTION A: Use Azula's built-in weighted loss
        # loss = self.denoiser.loss(x_0, t)
        
        # OPTION B: Manual loss computation : currently the loss implemented here is the same as azula 

        # Compute weighted loss
        alpha_t, sigma_t = self.schedule(t)
        alpha_t = alpha_t.view(-1, 1, 1, 1, 1) # (B, 1, 1, 1, 1)
        sigma_t = sigma_t.view(-1, 1, 1, 1, 1) # (B, 1, 1, 1, 1)

        noise = torch.randn_like(x_0)
        x_t = alpha_t * x_0 + sigma_t * noise

        x_denoised =  self._denoise(x_t, t) # Denoised output : (B, T, C, H, W)
        w_t = (alpha_t / sigma_t) ** 2 + 1
        w_t = torch.clip(w_t, max=1e4)
        
        loss = (w_t * (x_denoised - x_0).square()).mean()
        return loss