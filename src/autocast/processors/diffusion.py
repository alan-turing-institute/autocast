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
    zABSampler,
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
        learning_rate: float = 1e-4,
        n_steps_output: int = 4,
        n_channels_out: int = 1,
        sampler_steps: int = 50,
        sampler: str = "zab",
        sampler_start: float = 0.999,
        sampler_stop: float = 0.0,
        use_direct_prediction: bool = False,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.n_steps_output = n_steps_output
        self.n_channels_out = n_channels_out
        self.sampler_steps = sampler_steps
        self.sampler = sampler
        self.sampler_start = sampler_start
        self.sampler_stop = sampler_stop

        # Create Azula denoiser with chosen preconditioning
        if denoiser_type == "simple":
            self.denoiser = SimpleDenoiser(backbone=backbone, schedule=schedule)
        elif denoiser_type == "karras":
            self.denoiser = KarrasDenoiser(backbone=backbone, schedule=schedule)
        else:
            raise ValueError(f"Unknown denoiser type: {denoiser_type}")

        # Store schedule for direct access
        self.schedule = schedule
        # Debug/diagnostic mode: bypass iterative sampler and return a single-step
        # posterior mean prediction. This can be useful for isolating whether the
        # denoiser is learning, but it is not equivalent to full reverse sampling.
        self.use_direct_prediction = use_direct_prediction

    def map(self, x: Tensor) -> Tensor:
        """Map input window of states/times to output window using denoiser.

        By default, runs iterative reverse-time sampling.

        If use_direct_prediction=True, returns a single-step posterior mean prediction
        at t=1 (diagnostic only; not equivalent to full reverse sampling).
        """
        dtype = x.dtype
        device = x.device
        B, _, W, H, _ = x.shape

        # Iterative sampling (default)
        sampler = self._get_sampler(
            self.sampler_steps,
            dtype=dtype,
            device=device,
        )
        x_1 = sampler.init(
            (B, self.n_steps_output, W, H, self.n_channels_out),
            dtype=dtype,
            device=device,
        )  # Fully noised
        x_0 = sampler(x_1, cond=x)

        if not self.use_direct_prediction:
            return x_0

        # Diagnostic: compare to a single-step posterior mean prediction at t=1.
        t = torch.ones(B, device=device)
        posterior = self.denoiser(x_1, t, cond=x)
        return posterior.mean

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
        x_0 = batch.encoded_output_fields  # Clean data : (B, T,C, W, H)

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
        sampler: str | None = None,
        **sampler_kwargs,
    ) -> Sampler:
        sampler_name = sampler or self.sampler
        # Create appropriate Azula sampler
        if sampler_name == "euler":
            azula_sampler = EulerSampler(
                denoiser=self.denoiser,
                start=self.sampler_start,
                stop=self.sampler_stop,
                steps=num_steps,
                silent=silent,
                **sampler_kwargs,
            )
        elif sampler_name == "heun":
            azula_sampler = HeunSampler(
                denoiser=self.denoiser,
                start=self.sampler_start,
                stop=self.sampler_stop,
                steps=num_steps,
                silent=silent,
                **sampler_kwargs,
            )
        elif sampler_name == "ddim":
            azula_sampler = DDIMSampler(
                denoiser=self.denoiser,
                eta=eta,
                start=self.sampler_start,
                stop=self.sampler_stop,
                steps=num_steps,
                silent=silent,
                **sampler_kwargs,
            )
        elif sampler_name == "ddpm":
            azula_sampler = DDPMSampler(
                denoiser=self.denoiser,
                start=self.sampler_start,
                stop=self.sampler_stop,
                steps=num_steps,
                silent=silent,
                **sampler_kwargs,
            )
        elif sampler_name == "zab":
            # zABSampler integrates in u = sigma/alpha space and is often more
            # numerically stable than Euler/Heun on VP schedules when sigma/alpha
            # becomes very large.
            azula_sampler = zABSampler(
                denoiser=self.denoiser,
                order=2,
                start=self.sampler_start,
                stop=self.sampler_stop,
                steps=num_steps,
                silent=silent,
                **sampler_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown sampler: {sampler_name}. Choose from: 'euler', 'heun', "
                "'ddim', 'ddpm', 'zab'"
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
            x_t: Starting noise (B, T, C, W, H)
            cond: Conditioning input (B, T_cond, C_cond, W, H)
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
            Generated samples (B, T, C, W, H)
            Or if return_trajectory=True: List of tensors
        """
        azula_sampler = self._get_sampler(
            num_steps=num_steps,
            eta=eta,
            silent=silent,
            sampler=sampler,
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
            return torch.stack(trajectory, dim=0)  # (num_steps+1, B, T, C, W, H)
        return azula_sampler(x_t, cond=cond)  # (B, T, C, W, H)
