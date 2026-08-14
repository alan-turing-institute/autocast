"""Source distributions for flow matching."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import cast

import torch
from torch import nn

from autocast.nn.noise.spatial import (
    SpatialBoundary,
    gaussian_spatial_filter,
    validate_spatial_boundary,
)
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
    """OU-time x boundary-aware squared-exponential spatial Gaussian process."""

    channel_factor: Tensor | None

    def __init__(
        self,
        *,
        temporal_correlation_time: float,
        spatial_length_scale: float,
        channel_correlation: Tensor | Sequence[Sequence[float]] | None = None,
        spatial_boundaries: SpatialBoundary | Sequence[SpatialBoundary] = "periodic",
        dt: float = 1.0,
    ) -> None:
        super().__init__(correlation_time=temporal_correlation_time, dt=dt)
        if spatial_length_scale <= 0.0:
            msg = "spatial_length_scale must be positive."
            raise ValueError(msg)
        self.spatial_length_scale = spatial_length_scale
        self.spatial_boundaries = self._validate_spatial_boundaries(spatial_boundaries)
        self.register_buffer(
            "channel_factor",
            self._channel_factor(channel_correlation, self.spatial_boundaries),
        )

    @staticmethod
    def _validate_spatial_boundaries(
        boundaries: SpatialBoundary | Sequence[SpatialBoundary],
    ) -> SpatialBoundary | tuple[SpatialBoundary, ...]:
        if isinstance(boundaries, str):
            return validate_spatial_boundary(boundaries)
        validated: tuple[SpatialBoundary, ...] = tuple(
            validate_spatial_boundary(value) for value in boundaries
        )
        if not validated:
            msg = "spatial_boundaries must not be empty."
            raise ValueError(msg)
        return validated

    @staticmethod
    def _channel_factor(
        correlation: Tensor | Sequence[Sequence[float]] | None,
        boundaries: SpatialBoundary | tuple[SpatialBoundary, ...],
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
        diagonal = matrix.diagonal()
        if not torch.allclose(diagonal, torch.ones_like(diagonal)):
            msg = "channel_correlation must have a unit diagonal."
            raise ValueError(msg)
        if isinstance(boundaries, tuple):
            if len(boundaries) != matrix.shape[0]:
                msg = (
                    "spatial_boundaries size must match channel_correlation: "
                    f"expected {matrix.shape[0]}, got {len(boundaries)}."
                )
                raise ValueError(msg)
            compatible = torch.tensor(
                [[left == right for right in boundaries] for left in boundaries],
                dtype=torch.bool,
                device=matrix.device,
            )
            if torch.any(matrix.masked_select(~compatible).abs() > 1e-6):
                msg = (
                    "channel_correlation cannot couple channels with different "
                    "spatial boundaries."
                )
                raise ValueError(msg)
        factor, info = torch.linalg.cholesky_ex(matrix)
        if torch.any(info):
            msg = "channel_correlation must be positive definite."
            raise ValueError(msg)
        return factor

    def _resolved_spatial_boundaries(
        self,
        n_channels: int,
    ) -> tuple[SpatialBoundary, ...]:
        if isinstance(self.spatial_boundaries, str):
            boundary = cast("SpatialBoundary", self.spatial_boundaries)
            return (boundary,) * n_channels
        if len(self.spatial_boundaries) != n_channels:
            msg = (
                "spatial_boundaries size must match the reference channels: "
                f"expected {len(self.spatial_boundaries)}, got {n_channels}."
            )
            raise ValueError(msg)
        return self.spatial_boundaries

    def _filter_spatially(
        self,
        samples: Tensor,
        boundaries: tuple[SpatialBoundary, ...],
    ) -> Tensor:
        if all(boundary == boundaries[0] for boundary in boundaries):
            return gaussian_spatial_filter(
                samples,
                length_scale=self.spatial_length_scale,
                boundary=boundaries[0],
            )

        filtered = torch.empty_like(samples)
        for boundary in dict.fromkeys(boundaries):
            channels = [
                index
                for index, channel_boundary in enumerate(boundaries)
                if channel_boundary == boundary
            ]
            filtered[..., channels] = gaussian_spatial_filter(
                samples[..., channels],
                length_scale=self.spatial_length_scale,
                boundary=boundary,
            )
        return filtered

    def sample_like(self, reference: Tensor) -> Tensor:
        """Draw a zero-mean, unit-pooled-variance separable GP sample."""
        temporally_correlated = super().sample_like(reference)
        working = (
            temporally_correlated.float()
            if temporally_correlated.dtype in (torch.float16, torch.bfloat16)
            else temporally_correlated
        )
        boundaries = self._resolved_spatial_boundaries(working.shape[-1])
        samples = self._filter_spatially(
            working,
            boundaries,
        )
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
