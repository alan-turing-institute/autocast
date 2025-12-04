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

        # if we start from zero at every autoregressive step, 
        # the model is asked to denoise using t=0, which is a point it has never been trained on.
        self.inference_t = 1e-5
        t = torch.full((x.size(0),), self.inference_t, device=x.device)        
        return self._denoise(x, t)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.map(x)
    
    def _denoise(self, x: Tensor, t: Tensor) -> Tensor:
        posterior = self.denoiser(x, t)
        return posterior.mean
    
    def training_step(self, batch:EncodedBatch, batch_idx:int) -> Tensor:
        """Training step with diffusion loss.

        Sample random time steps and compute loss between denoised output and clean data.
        """
        x_0 = batch.encoded_output_fields  # Clean data : (B, T,C, H, W)

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
    
    def sample(
        self,
        x_t: Tensor,
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
        
        # Sample using Azula's sampler
        if return_trajectory:
            # Manually collect trajectory
            trajectory = [x_t]
            time_pairs = azula_sampler.timesteps.unfold(0, 2, 1).to(device=x_t.device)
            
            x = x_t
            for t, s in time_pairs:
                x = azula_sampler.step(x, t, s)
                trajectory.append(x)
            
            # Stack into single tensor , this is just for debugging and visualisation purposes 
            return torch.stack(trajectory, dim=0)  # (num_steps+1, B, T, C, H, W)
        else:
            return azula_sampler(x_t)  # (B, T, C, H, W)
        
    def rollout(self, batch: EncodedBatch) -> Tensor:
        """
        Rollout for AUTOREGRESSIVE TRAINING: Accumulates the weighted diffusion 
        loss (same as training_step) over multiple sequential prediction steps.
        """
        
        all_ground_truth = batch.encoded_output_fields
        if all_ground_truth is None:
            raise ValueError("Rollout loss calculation requires 'encoded_output_fields' (Ground Truth).")
            
        total_gt_steps = all_ground_truth.shape[1]
        total_accumulated_loss = 0.0
        
        x_input = batch.encoded_inputs
        T_out = 0 # Prediction window size
        num_successful_steps = 0
        
        for step in range(self.max_rollout_steps):
            
            # Get the prediction shape. Use self.map to get the predicted size.
            # This only needs to run once.
            if T_out == 0:
                with torch.no_grad():
                    T_out = self.map(x_input).shape[1]
            
            # --- 2. Check GT Availability ---
            start_idx = step * T_out
            end_idx = start_idx + T_out
            
            # Stop if the required GT window is out of bounds
            if end_idx > total_gt_steps:
                break
                
            step_gt = all_ground_truth[:, start_idx:end_idx] # GT for the predicted window (x_0)
            
            # --- 3. Calculate Diffusion Loss for the Current Window (Your Core Logic) ---
            
            x_0 = step_gt  # Clean data: (B, T_out, C, H, W)
            t = torch.rand(x_0.size(0), device=x_0.device) # Random time for loss
            
            # Compute weighted loss (replicating your training_step logic)
            alpha_t, sigma_t = self.schedule(t)
            # Expand dimensions to (B, 1, 1, 1, 1) or (B, 1, 1, 1, 1, 1) if using T_out > 1 for noise
            view_dims = [-1] + [1] * (x_0.dim() - 1) 
            alpha_t = alpha_t.view(view_dims)
            sigma_t = sigma_t.view(view_dims)

            noise = torch.randn_like(x_0)
            x_t = alpha_t * x_0 + sigma_t * noise

            x_denoised =  self._denoise(x_t, t) # Denoised output 
            
            w_t = (alpha_t / sigma_t) ** 2 + 1
            w_t = torch.clip(w_t, max=1e4)
            
            loss = (w_t * (x_denoised - x_0).square()).mean()
            
            total_accumulated_loss += loss
            num_successful_steps += 1
            
            # --- 4. Prepare next input (Teacher Forcing / Sliding Window) ---
            
            # Run map (without gradient tracking) to get the actual prediction for input update
            with torch.no_grad():
                pred = self.map(x_input) 

            # Select next input
            if self.training and torch.rand(1).item() < self.teacher_forcing_ratio:
                next_input = step_gt # Use GT (Teacher)
            else:
                next_input = pred # Use prediction (Autoregressive)

            # Update x_input (Sliding Window)
            if self.stride > 0 and x_input.shape[1] > self.stride:
                x_input = torch.cat([x_input[:, self.stride:], next_input], dim=1)
            else:
                x_input = next_input[:, -1:] 

        # Return the mean loss over all successful steps
        if num_successful_steps == 0:
            # Handle case where no GT was available, return a tensor loss of 0
            return torch.tensor(0.0, device=x_input.device)
            
        return total_accumulated_loss / float(num_successful_steps)
    