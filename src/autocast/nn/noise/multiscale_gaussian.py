"""Independent periodic Gaussian fields with channel-specific spatial scales."""

import math
from collections.abc import Sequence
from numbers import Real

import torch
from torch import Tensor

from autocast.nn.noise.source import SourceDistribution, _validate_spatiotemporal
from autocast.nn.noise.spatial import multiscale_gaussian_spatial_filter


def _positive_finite(values: Sequence[float], name: str) -> tuple[float, ...]:
    if not values or any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value <= 0.0
        for value in values
    ):
        msg = f"{name} must contain finite positive numbers."
        raise ValueError(msg)
    return tuple(float(value) for value in values)


class MultiscaleGaussianSource(SourceDistribution):
    """Draw full-grid coloured Gaussian noise at prescribed radial scales.

    Samples have shape ``(B,T,Y,X,C)``. Each channel has independent Gaussian
    grid innovations, filtered with power proportional to
    ``2**(-(k / half_power_wavenumbers[c])**2)``. The filters are overlapping
    low-pass envelopes, not rings or hard bands. ``k`` is radial frequency
    in cycles per domain-length unit; domain lengths default to a unit square.

    Unit expected variance is set analytically for each supplied grid, not
    by normalizing individual draws. The zero Fourier mode is retained, so
    draws need not have zero spatial mean. Frames, members, channels and
    calls are independent. There is no autoregressive temporal state.

    The source serves either spatial AdaLN or input concatenation without
    changing those consumers. A consumer's reference determines its grid;
    the number of channels must match the number of configured cutoffs.
    All cutoffs must lie strictly below both axial grid Nyquist frequencies.
    """

    def __init__(
        self,
        half_power_wavenumbers: Sequence[float],
        domain_lengths: Sequence[float] = (1.0, 1.0),
    ) -> None:
        super().__init__()
        self.half_power_wavenumbers = _positive_finite(
            half_power_wavenumbers, "half_power_wavenumbers"
        )
        lengths = _positive_finite(domain_lengths, "domain_lengths")
        if len(lengths) != 2:
            msg = "domain_lengths must contain height and width lengths."
            raise ValueError(msg)
        self.domain_lengths = (lengths[0], lengths[1])

    def sample_like(self, reference: Tensor) -> Tensor:
        """Sample fresh Gaussian fields on the consumer's explicit spatial grid."""
        _validate_spatiotemporal(reference)
        if min(reference.shape) < 1:
            msg = "Multiscale noise requires positive batch, grid and channel sizes."
            raise ValueError(msg)
        if reference.shape[-1] != len(self.half_power_wavenumbers):
            msg = "Multiscale noise channels must match half_power_wavenumbers."
            raise ValueError(msg)
        nyquist = min(
            size / (2.0 * length)
            for size, length in zip(
                reference.shape[2:4], self.domain_lengths, strict=True
            )
        )
        if max(self.half_power_wavenumbers) >= nyquist:
            msg = (
                "Multiscale half-power wavenumbers must lie below "
                "both grid Nyquist frequencies."
            )
            raise ValueError(msg)
        work_dtype = (
            torch.float64 if reference.dtype == torch.float64 else torch.float32
        )
        innovations = torch.randn(
            reference.shape, dtype=work_dtype, device=reference.device
        )
        return multiscale_gaussian_spatial_filter(
            innovations,
            half_power_wavenumbers=self.half_power_wavenumbers,
            domain_lengths=self.domain_lengths,
        ).to(dtype=reference.dtype)
