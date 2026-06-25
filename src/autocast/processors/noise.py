"""Noise schedules for diffusion processors."""

import math

import torch
from azula.noise import Schedule

from autocast.types import Tensor


class NoiseSchedule(Schedule):
    """Noise Schedule Module.

    Implements Azula's ``Schedule`` interface for local schedules.
    """

    def __call__(self, t: Tensor) -> tuple[Tensor, Tensor]:
        return self.forward(t)

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


class LogLogitSchedule(NoiseSchedule):
    """LoLA log-logit noise schedule.

    Matches ``LogLogitSchedule`` in
    https://github.com/francois-rozet/lola/blob/main/lola/diffusion.py.
    """

    def __init__(
        self,
        sigma_min: float = 1e-3,
        sigma_max: float = 1e3,
        scale: float = 1.0,
        shift: float = 0.0,
    ) -> None:
        super().__init__()
        self.t_min = sigma_min / (1 + sigma_min)
        self.t_max = sigma_max / (1 + sigma_max)
        self.scale = scale
        self.shift = shift

    def forward(self, t: Tensor) -> tuple[Tensor, Tensor]:
        alpha = torch.ones_like(t)
        sigma = torch.exp(
            self.scale * torch.logit(t * (self.t_max - self.t_min) + self.t_min)
            + self.shift
        )
        return alpha, sigma
