import torch
from azula.denoise import KarrasDenoiser, SimpleDenoiser
from azula.noise import Schedule

# Import Azula's samplers
from azula.sample import (
    DDIMSampler,
    DDPMSampler,
    EulerSampler,
    HeunSampler,
    Sampler,
)
from torch import nn

from autocast.processors.base import Processor
from autocast.types import EncodedBatch, Tensor


class DiffusionProcessor(Processor):
    """Diffusion Processor."""

    def __init__(
        self,
        backbone: nn.Module,
        schedule: Schedule,
        denoiser_type: str = "karras",
        teacher_forcing_ratio: float = 0.0,
        stride: int = 1,
        max_rollout_steps: int = 10,
        learning_rate: float = 1e-4,
        n_steps_output: int = 4,
        n_channels_out: int = 1,
        sampler_steps: int = 50,
        sampler: str = "euler",
    ):
        super().__init__()
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.stride = stride
        self.max_rollout_steps = max_rollout_steps
        self.learning_rate = learning_rate
        self.n_steps_output = n_steps_output
        self.n_channels_out = n_channels_out
        self.sampler_steps = sampler_steps
        self.sampler = sampler

        # Create Azula denoiser with chosen preconditioning
        if denoiser_type == "simple":
            self.denoiser = SimpleDenoiser(backbone=backbone, schedule=schedule)
        elif denoiser_type == "karras":
            self.denoiser = KarrasDenoiser(backbone=backbone, schedule=schedule)
        else:
            raise ValueError(f"Unknown denoiser type: {denoiser_type}")

        # Store schedule for direct access
        self.schedule = schedule

    def map(self, x: Tensor) -> Tensor:
        """Map input window of states/times to output window using denoiser."""
        dtype = x.dtype
        device = x.device
        sampler = self._get_sampler(self.sampler_steps, dtype=dtype, device=device)
        B, _, W, H, _ = x.shape

        # if we start from zero at every autoregressive step, the model is asked to
        # denoise using t=0, which is a point it has never been trained on.
        # self.inference_t = 1e-5
        # using azula sampler init to create noise at t=1
        x_1 = sampler.init(
            (B, self.n_steps_output, W, H, self.n_channels_out),
            dtype=dtype,
            device=device,
        )  # Fully noised
        return sampler(x_1, cond=x)

    def forward(self, x: Tensor) -> Tensor:
        return self.map(x)

    def _denoise(self, x: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        posterior = self.denoiser(x, t, cond=cond)
        return posterior.mean

    def loss(self, batch: EncodedBatch) -> Tensor:
        """Training step with diffusion loss.

        Sample random time steps and compute loss between denoised output and the
        clean data.
        """
        x_cond = batch.encoded_inputs
        x_0 = batch.encoded_output_fields  # Clean data : (B, T,C, H, W)

        # Sample random times in [0, 1] uniformly
        t = torch.rand(x_0.size(0), device=x_0.device)  # (B,)

        # Cannot use Azula's built-in weighted loss since ligntning calls forward
        loss = self.denoiser.loss(x_0, t=t, cond=x_cond)

        # TODO: consider an API for looking at alternative losses
        # # Compute weighted loss
        # alpha_t, sigma_t = self.schedule(t)
        # alpha_t = alpha_t.view(-1, 1, 1, 1, 1)  # (B, 1, 1, 1, 1)
        # sigma_t = sigma_t.view(-1, 1, 1, 1, 1)  # (B, 1, 1, 1, 1)

        # # Call forward in train mode to ensure gradients are tracked
        # x_denoised = self.forward(x_cond)

        # w_t = (alpha_t / sigma_t) ** 2 + 1
        # w_t = torch.clip(w_t, max=1e4)
        # loss = (w_t * (x_denoised - x_0).square()).mean()

        return loss  # noqa: RET504

    def _get_sampler(
        self,
        num_steps: int = 100,
        eta: float = 0.0,
        silent: bool = True,
        **sampler_kwargs,
    ) -> Sampler:
        sampler = self.sampler
        # Create appropriate Azula sampler
        if sampler == "euler":
            azula_sampler = EulerSampler(
                denoiser=self.denoiser,
                start=1.0,
                stop=0.0,
                steps=num_steps,
                silent=silent,
                **sampler_kwargs,
            )
        elif sampler == "heun":
            azula_sampler = HeunSampler(
                denoiser=self.denoiser,
                start=1.0,
                stop=0.0,
                steps=num_steps,
                silent=silent,
                **sampler_kwargs,
            )
        elif sampler == "ddim":
            azula_sampler = DDIMSampler(
                denoiser=self.denoiser,
                eta=eta,
                start=1.0,
                stop=0.0,
                steps=num_steps,
                silent=silent,
                **sampler_kwargs,
            )
        elif sampler == "ddpm":
            azula_sampler = DDPMSampler(
                denoiser=self.denoiser,
                start=1.0,
                stop=0.0,
                steps=num_steps,
                silent=silent,
                **sampler_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown sampler: {sampler}. Choose from: 'euler', 'heun', 'ddim',"
                "'ddpm'"
            )
        return azula_sampler

    def sample(
        self,
        x_t: Tensor,
        cond: Tensor,
        num_steps: int = 100,
        sampler: str = "euler",
        eta: float = 0.0,
        return_trajectory: bool = False,
        silent: bool = True,
        **sampler_kwargs,
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

        Returns
        -------
            Generated samples (B, T, C, H, W)
            Or if return_trajectory=True: List of tensors
        """
        azula_sampler = self._get_sampler(
            num_steps=num_steps,
            sampler=sampler,
            eta=eta,
            return_trajectory=return_trajectory,
            silent=silent,
            **sampler_kwargs,
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

            # Stack, this is just for debugging and visualisation purposes
            return torch.stack(trajectory, dim=0)  # (num_steps+1, B, T, C, H, W)
        return azula_sampler(x_t, cond=cond)  # (B, T, C, H, W)
