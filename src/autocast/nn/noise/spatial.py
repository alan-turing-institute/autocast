"""Boundary-aware spatial filtering for Gaussian flow sources."""

from __future__ import annotations

import math
from typing import Literal, cast

import torch

from autocast.types import Tensor

SpatialBoundary = Literal["periodic", "neumann", "dirichlet"]

_BOUNDARIES = ("periodic", "neumann", "dirichlet")


def validate_spatial_boundary(boundary: str) -> SpatialBoundary:
    """Return a supported spatial boundary or raise a configuration error."""
    if boundary not in _BOUNDARIES:
        msg = f"spatial boundaries must be one of {_BOUNDARIES}; got {boundary!r}."
        raise ValueError(msg)
    return cast("SpatialBoundary", boundary)


def _squared_frequency(reference: Tensor) -> Tensor:
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
    return angular_y[:, None].square() + angular_x[None, :].square()


def _periodic_amplitude(reference: Tensor, length_scale: float) -> Tensor:
    power = torch.exp(-0.5 * length_scale**2 * _squared_frequency(reference))
    return (power / power.mean()).sqrt().to(dtype=reference.dtype)


def _extend_even(field: Tensor, dim: int) -> Tensor:
    return torch.cat((field, field.flip(dims=(dim,))), dim=dim)


def _extend_odd(field: Tensor, dim: int) -> Tensor:
    if field.shape[dim] < 3:
        msg = "Dirichlet filtering requires at least three points per spatial axis."
        raise ValueError(msg)
    interior = field.narrow(dim, 1, field.shape[dim] - 2)
    zero_shape = list(field.shape)
    zero_shape[dim] = 1
    zero = field.new_zeros(zero_shape)
    return torch.cat((zero, interior, zero, -interior.flip(dims=(dim,))), dim=dim)


def _extended_field(field: Tensor, boundary: SpatialBoundary) -> Tensor:
    if boundary == "neumann":
        return _extend_even(_extend_even(field, 2), 3)
    return _extend_odd(_extend_odd(field, 2), 3)


def _active_modes(size: int, boundary: SpatialBoundary) -> slice:
    if boundary == "neumann":
        return slice(0, size)
    return slice(1, size - 1)


def _basis_mode_mask(
    reference: Tensor,
    *,
    original_shape: tuple[int, int],
    boundary: SpatialBoundary,
) -> Tensor:
    frequency_y = torch.arange(reference.shape[2], device=reference.device)
    frequency_x = torch.arange(reference.shape[3], device=reference.device)
    if boundary == "neumann":
        active_y = frequency_y != original_shape[0]
        active_x = frequency_x != original_shape[1]
    else:
        active_y = (frequency_y != 0) & (frequency_y != original_shape[0] - 1)
        active_x = (frequency_x != 0) & (frequency_x != original_shape[1] - 1)
    return active_y[:, None] & active_x[None, :]


def _extended_amplitude(
    reference: Tensor,
    *,
    original_shape: tuple[int, int],
    length_scale: float,
    boundary: SpatialBoundary,
) -> Tensor:
    log_power = -0.5 * length_scale**2 * _squared_frequency(reference)
    active_y = _active_modes(original_shape[0], boundary)
    active_x = _active_modes(original_shape[1], boundary)
    active_log_power = log_power[active_y, active_x]
    log_mean_power = torch.logsumexp(active_log_power.flatten(), dim=0) - math.log(
        active_log_power.numel()
    )
    log_amplitude = 0.5 * (log_power - log_mean_power)
    if boundary == "dirichlet":
        full_points = original_shape[0] * original_shape[1]
        interior_points = (original_shape[0] - 2) * (original_shape[1] - 2)
        log_amplitude = log_amplitude + 0.5 * math.log(full_points / interior_points)
    active = _basis_mode_mask(
        reference,
        original_shape=original_shape,
        boundary=boundary,
    )
    masked_log_amplitude = torch.where(
        active,
        log_amplitude,
        torch.full_like(log_amplitude, -math.inf),
    )
    return torch.exp(masked_log_amplitude).to(dtype=reference.dtype)


def _filter_periodic(field: Tensor, length_scale: float) -> Tensor:
    spectrum = torch.fft.fft2(field, dim=(2, 3), norm="ortho")
    amplitude = _periodic_amplitude(field, length_scale)[None, None, :, :, None]
    return torch.fft.ifft2(
        spectrum * amplitude,
        dim=(2, 3),
        norm="ortho",
    ).real


def _filter_bounded(
    field: Tensor,
    *,
    length_scale: float,
    boundary: SpatialBoundary,
) -> Tensor:
    height, width = field.shape[2:4]
    extended = _extended_field(field, boundary)
    spectrum = torch.fft.fft2(extended, dim=(2, 3), norm="ortho")
    amplitude = _extended_amplitude(
        extended,
        original_shape=(height, width),
        length_scale=length_scale,
        boundary=boundary,
    )[None, None, :, :, None]
    filtered = torch.fft.ifft2(
        spectrum * amplitude,
        dim=(2, 3),
        norm="ortho",
    ).real
    return filtered[:, :, :height, :width]


def gaussian_spatial_filter(
    field: Tensor,
    *,
    length_scale: float,
    boundary: SpatialBoundary,
) -> Tensor:
    """Apply a squared-exponential filter with the selected boundary basis.

    Periodic fields use their native Fourier grid. Neumann fields use an even
    extension at the outer cell faces, equivalent to a cell-centred cosine
    basis. Dirichlet fields use an odd extension of their interior, equivalent
    to a sine basis with exact zero boundary values. Power is normalized to
    unit pooled variance on the original full grid.
    """
    if boundary == "periodic":
        return _filter_periodic(field, length_scale)
    return _filter_bounded(
        field,
        length_scale=length_scale,
        boundary=boundary,
    )
