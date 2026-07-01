import torch
from azula.denoise import Denoiser, GaussianPosterior, KarrasDenoiser, SimpleDenoiser
from azula.noise import Schedule
from azula.sample import (
    DDIMSampler,
    DDPMSampler,
    EulerSampler,
    HeunSampler,
    Sampler,
    vABSampler,
    zABSampler,
)
from torch import nn

from autocast.processors.base import Processor
from autocast.types import EncodedBatch, Tensor


class LolaElucidatedDenoiser(Denoiser):
    """LoLA EDM-style denoiser preconditioning.

    This mirrors ``ElucidatedDenoiser`` in LoLA's ``lola/diffusion.py``. The
    main difference from Azula's local ``KarrasDenoiser`` is the modulation
    input scale: LoLA feeds ``10 * log(sigma / alpha)`` to the time embedding.
    """

    def __init__(
        self,
        backbone: nn.Module,
        schedule: Schedule,
        c_noise_scale: float = 1e1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.schedule = schedule
        self.c_noise_scale = c_noise_scale

    def forward(self, x_t: Tensor, t: Tensor, **kwargs) -> GaussianPosterior:
        alpha_t, sigma_t = self.schedule(t)

        while alpha_t.ndim < x_t.ndim:
            alpha_t, sigma_t = alpha_t[..., None], sigma_t[..., None]

        c_in = torch.rsqrt(alpha_t**2 + sigma_t**2)
        c_out = sigma_t * torch.rsqrt(alpha_t**2 + sigma_t**2)
        c_skip = alpha_t / (alpha_t**2 + sigma_t**2)
        c_noise = self.c_noise_scale * torch.log(sigma_t / alpha_t).reshape_as(t)

        mean = c_skip * x_t + c_out * self.backbone(c_in * x_t, c_noise, **kwargs)
        var = sigma_t**2 / (alpha_t**2 + sigma_t**2)

        return GaussianPosterior(mean=mean, var=var)

    def loss(self, x: Tensor, t: Tensor, **kwargs) -> Tensor:
        alpha_t, sigma_t = self.schedule(t)

        while alpha_t.ndim < x.ndim:
            alpha_t, sigma_t = alpha_t[..., None], sigma_t[..., None]

        z = torch.randn_like(x)
        x_t = alpha_t * x + sigma_t * z
        q = self(x_t, t, **kwargs)

        return ((q.mean - x).square() / q.var.detach()).mean()


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
        sampler: str = "euler",
        sampler_order: int = 2,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.n_steps_output = n_steps_output
        self.n_channels_out = n_channels_out
        self.sampler_steps = sampler_steps
        self.sampler = sampler
        # Multistep order for Adams-Bashforth samplers ("ab"/"vab")
        self.sampler_order = sampler_order

        # Create Azula denoiser with chosen preconditioning
        if denoiser_type == "simple":
            self.denoiser = SimpleDenoiser(backbone=backbone, schedule=schedule)
        elif denoiser_type == "karras":
            self.denoiser = KarrasDenoiser(backbone=backbone, schedule=schedule)
        elif denoiser_type in {"lola", "lola_elucidated"}:
            self.denoiser = LolaElucidatedDenoiser(
                backbone=backbone,
                schedule=schedule,
            )
        else:
            raise ValueError(f"Unknown denoiser type: {denoiser_type}")

        # Store schedule for direct access
        self.schedule = schedule

    def map(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        """Map input window of states/times to output window using denoiser."""
        dtype = x.dtype
        device = x.device
        sampler = self._get_sampler(self.sampler_steps, dtype=dtype, device=device)
        B, _, W, H, _ = x.shape
        sample_shape = (B, self.n_steps_output, W, H, self.n_channels_out)

        # Azula 0.7 init uses stop for descending timesteps; start directly.
        x_1 = self._init_at_sampler_start(
            sampler,
            sample_shape,
            dtype=dtype,
            device=device,
        )
        return sampler(x_1, cond=x, global_cond=global_cond)

    def _init_at_sampler_start(
        self,
        sampler: Sampler,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        start = torch.as_tensor(sampler.start, dtype=dtype, device=device)
        alpha_start, sigma_start = self.denoiser.schedule(start)
        alpha_start = alpha_start.to(dtype=dtype, device=device)
        sigma_start = sigma_start.to(dtype=dtype, device=device)

        std_start = torch.sqrt(alpha_start.square() + sigma_start.square())
        while std_start.ndim < len(shape):
            std_start = std_start[..., None]

        return std_start.expand(shape) * torch.randn(
            shape,
            dtype=dtype,
            device=device,
        )

    def forward(self, x: Tensor, global_cond: Tensor | None = None) -> Tensor:
        return self.map(x, global_cond)

    # def _denoise(self, x: Tensor, t: Tensor, cond: Tensor) -> Tensor:
    #     posterior = self.denoiser(x, t, cond=cond)
    #     return posterior.mean

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
        loss = self.denoiser.loss(x_0, t=t, cond=x_cond, global_cond=batch.global_cond)

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

        return loss

    def _get_sampler(
        self,
        num_steps: int = 100,
        sampler: str | None = None,
        eta: float = 0.0,
        silent: bool = True,
        **sampler_kwargs,
    ) -> Sampler:
        sampler_name = self.sampler if sampler is None else sampler
        # Create appropriate Azula sampler
        if sampler_name == "euler":
            azula_sampler = EulerSampler(
                denoiser=self.denoiser,
                start=1.0,
                stop=0.0,
                steps=num_steps,
                silent=silent,
                **sampler_kwargs,
            )
        elif sampler_name == "heun":
            azula_sampler = HeunSampler(
                denoiser=self.denoiser,
                start=1.0,
                stop=0.0,
                steps=num_steps,
                silent=silent,
                **sampler_kwargs,
            )
        elif sampler_name == "ddim":
            azula_sampler = DDIMSampler(
                denoiser=self.denoiser,
                eta=eta,
                start=1.0,
                stop=0.0,
                steps=num_steps,
                silent=silent,
                **sampler_kwargs,
            )
        elif sampler_name == "ddpm":
            azula_sampler = DDPMSampler(
                denoiser=self.denoiser,
                start=1.0,
                stop=0.0,
                steps=num_steps,
                silent=silent,
                **sampler_kwargs,
            )
        elif sampler_name == "ab":
            # Adams-Bashforth multistep ODE solver with noise (z) prediction.
            azula_sampler = zABSampler(
                denoiser=self.denoiser,
                order=self.sampler_order,
                start=1.0,
                stop=0.0,
                steps=num_steps,
                silent=silent,
                **sampler_kwargs,
            )
        elif sampler_name == "vab":
            # Adams-Bashforth multistep ODE solver with velocity (v) prediction.
            azula_sampler = vABSampler(
                denoiser=self.denoiser,
                order=self.sampler_order,
                start=1.0,
                stop=0.0,
                steps=num_steps,
                silent=silent,
                **sampler_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown sampler: {sampler_name}. Choose from: 'euler', 'heun', "
                "'ddim', 'ddpm', 'ab', 'vab'"
            )
        return azula_sampler

    def sample(
        self,
        x_t: Tensor,
        cond: Tensor,
        global_cond: Tensor | None = None,
        num_steps: int = 100,
        sampler: str | None = None,
        eta: float = 0.0,
        return_trajectory: bool = False,
        silent: bool = True,
        **sampler_kwargs,
    ) -> Tensor:
        """Generate samples via reverse diffusion using Azula's samplers.

        Args:
            x_t: Starting noise (B, T, C, W, H)
            cond: Conditioning input (B, T_cond, C_cond, W, H)
            global_cond: Optional non-spatial conditioning/modulation tensor.
            num_steps: Number of denoising steps
            sampler: Type of sampler to use. Defaults to the configured
                `self.sampler` when unset.
                - 'euler': Euler ODE solver (fast, deterministic)
                - 'heun': Heun's method (more accurate, deterministic)
                - 'ddim': DDIM sampler (eta controls stochasticity)
                - 'ddpm': DDPM sampler (stochastic)
                - 'ab': Adams-Bashforth ODE solver with z-prediction
                - 'vab': Adams-Bashforth ODE solver with v-prediction
            eta: Stochasticity parameter for DDIM (0=deterministic, 1=stochastic)
            return_trajectory: If True, return all intermediate steps
            silent: If True, hide progress bar
            **sampler_kwargs: Additional kwargs passed to sampler

        Returns:
            Generated samples (B, T, C, W, H)
            Or if return_trajectory=True: List of tensors
        """
        azula_sampler = self._get_sampler(
            num_steps=num_steps,
            sampler=sampler,
            eta=eta,
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
                x = azula_sampler.step(x, t, s, cond=cond, global_cond=global_cond)
                trajectory.append(x)

            # Stack, this is just for debugging and visualisation purposes
            return torch.stack(trajectory, dim=0)  # (num_steps+1, B, T, C, W, H)
        return azula_sampler(x_t, cond=cond, global_cond=global_cond)  # (B, T, C, W, H)
