import math

import azula
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

# Import Azula's samplers
from azula.sample import (
    Sampler,
    DDPMSampler,
    DDIMSampler,
    EulerSampler,
    HeunSampler,
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
        n_steps_output: int = 4,
    ):

        super().__init__()
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.stride = stride
        self.max_rollout_steps = max_rollout_steps
        self.learning_rate = learning_rate
        self.n_steps_output = n_steps_output
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

        # if we start from zero at every autoregressive step, 
        # the model is asked to denoise using t=0, which is a point it has never been trained on.
        self.inference_t = 1e-5
        t = torch.full((x.size(0),), self.inference_t, device=x.device)
        B, _, H, W, C = x.shape
        x_t = torch.randn(B, self.n_steps_output, H, W, C, device=x.device)
        return self._denoise(x_t, t, cond=x)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.map(x)
    
    def _denoise(self, x: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        posterior = self.denoiser(x, t, cond=cond)
        return posterior.mean
    
    # def __call__(self, *args, **kwds):
    #     return self.
    
    def training_step(self, batch: EncodedBatch, batch_idx: int) -> Tensor:
        """Training step with diffusion loss.

        Sample random time steps and compute loss between denoised output and clean data.
        """
        x_cond = batch.encoded_inputs
        x_0 = batch.encoded_output_fields  # Clean data : (B, T,C, H, W)

        # Sample random times in [0, 1] uniformly
        t = torch.rand(x_0.size(0), device=x_0.device)  # (B,)

        # OPTION A: Use Azula's built-in weighted loss
        # loss = self.denoiser.loss(x_0, t=t, cond=x_cond)
        
        # OPTION B: Manual loss computation : currently the loss implemented here is the same as azula 

        # Compute weighted loss
        alpha_t, sigma_t = self.schedule(t)
        alpha_t = alpha_t.view(-1, 1, 1, 1, 1) # (B, 1, 1, 1, 1)
        sigma_t = sigma_t.view(-1, 1, 1, 1, 1) # (B, 1, 1, 1, 1)

        noise = torch.randn_like(x_0)
        x_t = alpha_t * x_0 + sigma_t * noise

        x_denoised =  self._denoise(x_t, t, cond=x_cond) # Denoised output : (B, T, C, H, W)
        w_t = (alpha_t / sigma_t) ** 2 + 1
        w_t = torch.clip(w_t, max=1e4)
        
        loss = (w_t * (x_denoised - x_0).square()).mean()
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            batch_size=batch.encoded_inputs.shape[0]  #  proper averaging across batches
        )
        return loss
    
    def _get_sampler(self,
        x_t: Tensor,
        cond: Tensor,
        num_steps: int = 100,
        sampler: str = 'euler',
        eta: float = 0.0,
        return_trajectory: bool = False,
        silent: bool = True,
        **sampler_kwargs
    ):
        # Create appropriate Azula sampler
        if sampler == 'euler':
            azula_sampler = EulerSampler(
                denoiser=self.denoiser,
                start=1.0,
                stop=0.0,
                steps=num_steps,
                silent=silent,
                **sampler_kwargs
            )
        elif sampler == 'heun':
            azula_sampler = HeunSampler(
                denoiser=self.denoiser,
                start=1.0,
                stop=0.0,
                steps=num_steps,
                silent=silent,
                **sampler_kwargs
            )
        elif sampler == 'ddim':
            azula_sampler = DDIMSampler(
                denoiser=self.denoiser,
                eta=eta,
                start=1.0,
                stop=0.0,
                steps=num_steps,
                silent=silent,
                **sampler_kwargs
            )
        elif sampler == 'ddpm':
            azula_sampler = DDPMSampler(
                denoiser=self.denoiser,
                start=1.0,
                stop=0.0,
                steps=num_steps,
                silent=silent,
                **sampler_kwargs
            )
        else:
            raise ValueError(f"Unknown sampler: {sampler}. Choose from: 'euler', 'heun', 'ddim', 'ddpm'")

    def sample(
        self,
        x_t: Tensor,
        cond: Tensor,
        num_steps: int = 100,
        sampler: str = 'euler',
        eta: float = 0.0,
        return_trajectory: bool = False,
        silent: bool = True,
        **sampler_kwargs
    ) -> Tensor:
        """
        Generate samples via reverse diffusion using Azula's samplers.
        
        Args:
            x_t: Starting noise (B, T, C, H, W)
            cond: Conditioning input (B, T_cond, C_cond, H, W)
            num_steps: Number of denoising steps
            sampler: Type of sampler to use:
                - 'euler': Euler ODE solver (fast, deterministic)
                - 'heun': Heun's method (more accurate, deterministic)
                - 'ddim': DDIM sampler (eta controls stochasticity)
                - 'ddpm': DDPM sampler (stochastic)
            eta: Stochasticity parameter for DDIM (0=deterministic, 1=stochastic)
            return_trajectory: If True, return all intermediate steps
            silent: If True, hide progress bar
            **sampler_kwargs: Additional kwargs passed to sampler
            
        Returns:
            Generated samples (B, T, C, H, W)
            Or if return_trajectory=True: List of tensors
        """
        azula_sampler = self._get_sampler(
            x_t,
            cond,
            num_steps=num_steps,
            sampler=sampler,
            eta=eta,
            return_trajectory=return_trajectory,
            silent=silent,
            **sampler_kwargs
        )
        
        # Sample using Azula's sampler
        if return_trajectory:
            # Manually collect trajectory
            trajectory = [x_t]
            time_pairs = azula_sampler.timesteps.unfold(0, 2, 1).to(device=x_t.device)
            
            x = x_t
            for t, s in time_pairs:
                x = azula_sampler.step(x, t, s, cond=cond)
                trajectory.append(x)
            
            # Stack into single tensor , this is just for debugging and visualisation purposes 
            return torch.stack(trajectory, dim=0)  # (num_steps+1, B, T, C, H, W)
        else:
            return azula_sampler(x_t, cond=cond)  # (B, T, C, H, W)