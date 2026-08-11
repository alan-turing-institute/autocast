"""Affine normalization for residual forecast trajectories."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn

from autocast.types import Tensor

ResidualGranularity = Literal["channel", "lead_channel"]
Statistic = float | Sequence[float] | Sequence[Sequence[float]] | Tensor


class ResidualStandardizer(nn.Module):
    """Standardize processor-coordinate residuals using fitted statistics.

    Statistics can be shared over output time (``channel``) or vary by lead
    (``lead_channel``). Spatial dimensions are always pooled. Constructing the
    module without statistics leaves it unfitted for a training callback to
    initialize later; omitting the module from a residual processor preserves
    identity behavior. These operations are distinct from dataset normalization:
    they map between a residual and its standardized representation within the
    processor's ambient or latent coordinate system.
    """

    mean: Tensor
    scale: Tensor
    _fitted: Tensor

    _GRANULARITIES = ("channel", "lead_channel")

    def __init__(
        self,
        *,
        n_steps_output: int,
        n_channels: int,
        granularity: ResidualGranularity = "lead_channel",
        mean: Statistic | None = None,
        scale: Statistic | None = None,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if n_steps_output < 1:
            msg = "n_steps_output must be positive."
            raise ValueError(msg)
        if n_channels < 1:
            msg = "n_channels must be positive."
            raise ValueError(msg)
        if granularity not in self._GRANULARITIES:
            msg = (
                f"granularity must be one of {self._GRANULARITIES}; "
                f"got {granularity!r}."
            )
            raise ValueError(msg)
        if epsilon <= 0.0:
            msg = "epsilon must be positive."
            raise ValueError(msg)
        if (mean is None) != (scale is None):
            msg = "mean and scale must either both be provided or both be omitted."
            raise ValueError(msg)

        self.n_steps_output = n_steps_output
        self.n_channels = n_channels
        self.granularity = granularity
        self.epsilon = epsilon
        statistic_shape = self.statistic_shape
        self.register_buffer("mean", torch.zeros(statistic_shape), persistent=True)
        self.register_buffer("scale", torch.ones(statistic_shape), persistent=True)
        self.register_buffer("_fitted", torch.tensor(False), persistent=True)

        if mean is not None and scale is not None:
            self.set_statistics(mean=mean, scale=scale)

    @property
    def statistic_shape(self) -> tuple[int, ...]:
        """Shape of the stored residual statistics."""
        if self.granularity == "channel":
            return (self.n_channels,)
        return (self.n_steps_output, self.n_channels)

    @property
    def fitted(self) -> bool:
        """Whether usable residual statistics have been assigned."""
        return bool(self._fitted.item())

    def _coerce_statistic(self, value: Statistic, *, name: str) -> Tensor:
        statistic = torch.as_tensor(value, dtype=torch.float32)
        if statistic.numel() == 1:
            return statistic.reshape(1).expand(self.statistic_shape).clone()
        if tuple(statistic.shape) != self.statistic_shape:
            msg = (
                f"{name} must be scalar or have shape {self.statistic_shape}; "
                f"received {tuple(statistic.shape)}."
            )
            raise ValueError(msg)
        return statistic

    @torch.no_grad()
    def set_statistics(self, *, mean: Statistic, scale: Statistic) -> None:
        """Assign statistics, flooring zero scales for stable normalization."""
        mean_tensor = self._coerce_statistic(mean, name="mean")
        scale_tensor = self._coerce_statistic(scale, name="scale")
        if not torch.isfinite(mean_tensor).all() or not torch.isfinite(
            scale_tensor
        ).all():
            msg = "Residual statistics must contain only finite values."
            raise ValueError(msg)
        if torch.any(scale_tensor < 0.0):
            msg = "Residual scales must be non-negative."
            raise ValueError(msg)

        self.mean.copy_(mean_tensor.to(device=self.mean.device, dtype=self.mean.dtype))
        self.scale.copy_(
            scale_tensor.clamp_min(self.epsilon).to(
                device=self.scale.device,
                dtype=self.scale.dtype,
            )
        )
        self._fitted.fill_(True)

    def _validate_residual(self, residual: Tensor) -> None:
        if residual.ndim < 3:
            msg = "residual must have shape (B, T, ..., C)."
            raise ValueError(msg)
        if (
            residual.shape[1] != self.n_steps_output
            or residual.shape[-1] != self.n_channels
        ):
            msg = (
                "Residual shape does not match standardizer dimensions "
                f"(expected T={self.n_steps_output}, C={self.n_channels}; "
                f"received T={residual.shape[1]}, C={residual.shape[-1]})."
            )
            raise ValueError(msg)
        if not torch.is_floating_point(residual):
            msg = "residual must be floating point."
            raise TypeError(msg)

    def _statistics_like(self, residual: Tensor) -> tuple[Tensor, Tensor]:
        if not self.fitted:
            msg = "ResidualStandardizer must be fitted before use."
            raise RuntimeError(msg)
        self._validate_residual(residual)
        if self.granularity == "channel":
            broadcast_shape = (*([1] * (residual.ndim - 1)), self.n_channels)
        else:
            broadcast_shape = (
                1,
                self.n_steps_output,
                *([1] * (residual.ndim - 3)),
                self.n_channels,
            )
        mean = self.mean.to(device=residual.device, dtype=residual.dtype).view(
            broadcast_shape
        )
        scale = self.scale.to(device=residual.device, dtype=residual.dtype).view(
            broadcast_shape
        )
        return mean, scale

    def standardize(self, residual: Tensor) -> Tensor:
        """Map a processor-coordinate residual into standardized coordinates."""
        mean, scale = self._statistics_like(residual)
        return (residual - mean) / scale

    def unstandardize(self, standardized_residual: Tensor) -> Tensor:
        """Map a standardized residual back into processor coordinates."""
        mean, scale = self._statistics_like(standardized_residual)
        return mean + scale * standardized_residual
