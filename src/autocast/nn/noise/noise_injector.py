import abc

import torch
from torch import nn

from autocast.nn.noise.source import FlowSource, _sample_checked
from autocast.types import Tensor


class NoiseInjector(nn.Module, abc.ABC):
    """Base class for noise injection strategies."""

    @abc.abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Apply noise to input tensor."""

    def get_additional_channels(self) -> int:
        """Return the number of additional channels this injector adds."""
        return 0

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
    """Concatenate sampled noise as additional channels.

    Defaults to IID Gaussian noise. An optional ``noise_source`` receives a
    reference with the input layout and ``n_channels`` on the last axis.
    Structured sources require an explicit grid, e.g. ``(B,T,Y,X,C)``.
    Sources are sampled afresh on every call; no rollout noise state is kept.
    """

    def __init__(
        self,
        n_channels: int = 1,
        std: float = 1.0,
        noise_source: FlowSource | None = None,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.std = self._validate_std(std)
        self.noise_source = noise_source

    def get_additional_channels(self) -> int:
        return self.n_channels

    def forward(self, x: Tensor) -> Tensor:
        if self.n_channels <= 0:
            return x

        # Assume x shape is (B, ..., C) - generate noise of shape (B, ..., n_channels)
        shape = list(x.shape)
        shape[-1] = self.n_channels

        if self.noise_source is None:
            noise = torch.randn(shape, device=x.device, dtype=x.dtype)
        else:
            noise = _sample_checked(self.noise_source, x.new_empty(shape))
        noise = noise * self.std
        return torch.cat([x, noise], dim=-1)
