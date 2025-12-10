# This is a the noise base schedule module
# for now we will be using Azula but if we want to avoid that dependency
# we can implement our own schedules here
# 03/12/2025
import math

import torch
from torch import nn

from autocast.types import Tensor


class NoiseSchedule(nn.Module):
    """Noise Schedule Module.

    if Azula became useful for other things we can use their base class
    instead of this base
    """

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
