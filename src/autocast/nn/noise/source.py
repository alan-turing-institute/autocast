"""Source distributions for flow matching."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from torch import nn

from autocast.types import Tensor


class FlowSource(nn.Module, ABC):
    """Sample initial flow states matching a reference tensor."""

    @abstractmethod
    def sample_like(self, reference: Tensor) -> Tensor:
        """Draw source states with the reference shape, device, and dtype."""


class ZeroSource(FlowSource):
    """Deterministic point source at zero."""

    def sample_like(self, reference: Tensor) -> Tensor:
        return torch.zeros_like(reference)


class IIDGaussianSource(FlowSource):
    """Independent standard Gaussian source."""

    def sample_like(self, reference: Tensor) -> Tensor:
        return torch.randn_like(reference)


def _validate_spatiotemporal(reference: Tensor) -> None:
    if reference.ndim != 5 or reference.shape[1] < 1:
        msg = "reference must have shape (B, T, Y, X, C) with T >= 1."
        raise ValueError(msg)
    if not torch.is_floating_point(reference):
        msg = "reference must be floating point."
        raise TypeError(msg)


class TemporalOUSource(FlowSource):
    """Stationary Gaussian source with an OU kernel over output time."""

    def __init__(self, *, correlation_time: float, dt: float = 1.0) -> None:
        super().__init__()
        if correlation_time <= 0.0:
            msg = "correlation_time must be positive."
            raise ValueError(msg)
        if dt <= 0.0:
            msg = "dt must be positive."
            raise ValueError(msg)
        self.correlation_time = correlation_time
        self.dt = dt

    @property
    def correlation(self) -> float:
        """Return the correlation between adjacent output frames."""
        return math.exp(-self.dt / self.correlation_time)

    def sample_like(self, reference: Tensor) -> Tensor:
        """Draw unit-variance OU samples independently over space and channel."""
        _validate_spatiotemporal(reference)
        innovations = torch.randn_like(reference)
        correlation = self.correlation
        innovation_scale = math.sqrt(max(0.0, 1.0 - correlation**2))
        samples = [innovations[:, 0]]
        for innovation in innovations[:, 1:].unbind(dim=1):
            samples.append(correlation * samples[-1] + innovation_scale * innovation)
        return torch.stack(samples, dim=1)


class SeparableGaussianSource(TemporalOUSource):
    """OU-time x periodic squared-exponential spatial Gaussian process."""

    channel_factor: Tensor | None

    def __init__(
        self,
        *,
        temporal_correlation_time: float,
        spatial_length_scale: float,
        channel_correlation: Tensor | Sequence[Sequence[float]] | None = None,
        dt: float = 1.0,
    ) -> None:
        super().__init__(correlation_time=temporal_correlation_time, dt=dt)
        if spatial_length_scale <= 0.0:
            msg = "spatial_length_scale must be positive."
            raise ValueError(msg)
        self.spatial_length_scale = spatial_length_scale
        self.register_buffer(
            "channel_factor",
            self._channel_factor(channel_correlation),
        )

    @staticmethod
    def _channel_factor(
        correlation: Tensor | Sequence[Sequence[float]] | None,
    ) -> Tensor | None:
        if correlation is None:
            return None
        matrix = torch.as_tensor(correlation, dtype=torch.float32)
        if (
            matrix.ndim != 2
            or matrix.shape[0] == 0
            or matrix.shape[0] != matrix.shape[1]
        ):
            msg = "channel_correlation must be a non-empty square matrix."
            raise ValueError(msg)
        if not torch.isfinite(matrix).all():
            msg = "channel_correlation must contain only finite values."
            raise ValueError(msg)
        if not torch.allclose(matrix, matrix.mT):
            msg = "channel_correlation must be symmetric."
            raise ValueError(msg)
        if not torch.allclose(matrix.diagonal(), torch.ones(matrix.shape[0])):
            msg = "channel_correlation must have a unit diagonal."
            raise ValueError(msg)
        factor, info = torch.linalg.cholesky_ex(matrix)
        if torch.any(info):
            msg = "channel_correlation must be positive definite."
            raise ValueError(msg)
        return factor

    def _spatial_amplitude(self, reference: Tensor) -> Tensor:
        height, width = reference.shape[2:4]
        angular_y = (
            2.0
            * math.pi
            * torch.fft.fftfreq(
                height,
                dtype=reference.dtype,
                device=reference.device,
            )
        )
        angular_x = (
            2.0
            * math.pi
            * torch.fft.fftfreq(
                width,
                dtype=reference.dtype,
                device=reference.device,
            )
        )
        squared_frequency = angular_y[:, None].square() + angular_x[None, :].square()
        power = torch.exp(-0.5 * self.spatial_length_scale**2 * squared_frequency)
        return (power / power.mean()).sqrt().to(dtype=reference.dtype)

    def sample_like(self, reference: Tensor) -> Tensor:
        """Draw a zero-mean, unit-marginal separable GP sample."""
        temporally_correlated = super().sample_like(reference)
        working = (
            temporally_correlated.float()
            if temporally_correlated.dtype in (torch.float16, torch.bfloat16)
            else temporally_correlated
        )
        spectrum = torch.fft.fft2(
            working,
            dim=(2, 3),
            norm="ortho",
        )
        amplitude = self._spatial_amplitude(working)[None, None, :, :, None]
        samples = torch.fft.ifft2(
            spectrum * amplitude,
            dim=(2, 3),
            norm="ortho",
        ).real
        if self.channel_factor is not None:
            if samples.shape[-1] != self.channel_factor.shape[0]:
                msg = (
                    "channel_correlation size must match the reference channels: "
                    f"expected {self.channel_factor.shape[0]}, got {samples.shape[-1]}."
                )
                raise ValueError(msg)
            factor = self.channel_factor.to(device=samples.device, dtype=samples.dtype)
            samples = torch.einsum("btyxc,dc->btyxd", samples, factor)
        return samples.to(dtype=reference.dtype)
