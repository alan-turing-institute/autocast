import abc

import torch
from torch import nn

from autocast.types import Tensor


class NoiseInjector(nn.Module, abc.ABC):
    """Base class for noise injection strategies."""

    @abc.abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Apply noise to input tensor."""

    def _validate_std(self, std: float) -> float:
        if std <= 0.0:
            msg = "Standard deviation must be greater than 0.0 and non-negative."
            raise ValueError(msg)
        return std


class AdditiveNoiseInjector(NoiseInjector):
    """Injects additive Gaussian noise."""

    def __init__(self, std: float = 1.0):
        super().__init__()
        self.std = self._validate_std(std)

    def forward(self, x: Tensor) -> Tensor:
        return x + torch.randn_like(x) * self.std


class ConcatenatedNoiseInjector(NoiseInjector):
    """Concatenates Gaussian noise as additional channels.

    Appends `n_channels` of Gaussian noise to the last dimension (channels).
    """

    def __init__(self, n_channels: int = 1, std: float = 1.0):
        super().__init__()
        self.n_channels = n_channels
        self.std = self._validate_std(std)

    def forward(self, x: Tensor) -> Tensor:
        if self.n_channels <= 0:
            return x

        # Assume x shape is (B, ..., C) - generate noise of shape (B, ..., n_channels)
        shape = list(x.shape)
        shape[-1] = self.n_channels

        noise = torch.randn(shape, device=x.device, dtype=x.dtype) * self.std
        return torch.cat([x, noise], dim=-1)
