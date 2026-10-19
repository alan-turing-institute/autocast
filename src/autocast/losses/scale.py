import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from autocast.losses.decomposition import (
    FieldComponents,
    FieldDecomposition,
    _spatial_dims,
)
from autocast.losses.spectral import _radial_frequency
from autocast.types import FieldTensor, Tensor

ScaleBoundary = Literal["reflect", "periodic", "zero"]


class SmoothingOperator(nn.Module, ABC):
    """Map a field to a smoother field with the same layout."""

    @abstractmethod
    def forward(self, tensor: FieldTensor, *, n_spatial_dims: int) -> Tensor: ...


class GaussianSmoothing(SmoothingOperator):
    """Apply separable Gaussian smoothing over spatial axes.

    ``sigma`` is measured in grid cells and the kernel radius is
    ``ceil(truncate * sigma)``.
    """

    boundary: ScaleBoundary
    kernel: Tensor

    def __init__(
        self,
        sigma: int | float,
        *,
        truncate: int | float = 4,
        boundary: ScaleBoundary = "reflect",
    ) -> None:
        super().__init__()
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"sigma must be finite and positive, got {sigma}.")
        if not math.isfinite(truncate) or truncate <= 0:
            raise ValueError(f"truncate must be finite and positive, got {truncate}.")
        if boundary not in ("reflect", "periodic", "zero"):
            raise ValueError(f"Unknown scale boundary {boundary!r}.")

        radius = math.ceil(float(truncate) * float(sigma))
        coordinates = torch.arange(-radius, radius + 1, dtype=torch.float64)
        kernel = torch.exp(-0.5 * (coordinates / float(sigma)).square())
        self.register_buffer("kernel", kernel / kernel.sum())
        self.radius = radius
        self.boundary = boundary

    def _smooth_axis(self, tensor: Tensor, dim: int) -> Tensor:
        moved = tensor.movedim(dim, -1)
        shape = moved.shape
        flattened = moved.reshape(-1, 1, shape[-1])
        if self.boundary == "periodic":
            indices = torch.arange(
                -self.radius,
                shape[-1] + self.radius,
                device=tensor.device,
            ).remainder(shape[-1])
            padded = flattened.index_select(-1, indices)
        elif self.boundary == "reflect":
            if self.radius >= shape[-1]:
                raise ValueError(
                    "Reflect padding requires each spatial axis to be longer "
                    f"than the kernel radius {self.radius}; got {shape[-1]}."
                )
            padded = F.pad(flattened, (self.radius, self.radius), mode="reflect")
        else:
            padded = F.pad(flattened, (self.radius, self.radius))

        kernel = self.kernel.to(dtype=tensor.dtype).reshape(1, 1, -1)
        smoothed = F.conv1d(padded, kernel)
        return smoothed.reshape(shape).movedim(-1, dim)

    def forward(self, tensor: FieldTensor, *, n_spatial_dims: int) -> Tensor:
        smoothed = tensor
        for dim in _spatial_dims(tensor, n_spatial_dims):
            smoothed = self._smooth_axis(smoothed, dim)
        return smoothed


class FourierLowPass(SmoothingOperator):
    """Apply an isotropic periodic Fourier low-pass filter.

    ``cutoff`` is radial frequency in cycles per grid cell and must not exceed
    the per-axis Nyquist frequency of 0.5. Diagonal corner modes can exceed
    this radius in multiple dimensions, so 0.5 is not an identity filter.
    Half and bfloat16 inputs are promoted to float32 before the FFT.
    """

    def __init__(self, cutoff: int | float) -> None:
        super().__init__()
        if not 0 < cutoff <= 0.5:
            raise ValueError(f"cutoff must be in (0, 0.5], got {cutoff}.")
        self.cutoff = float(cutoff)

    def forward(self, tensor: FieldTensor, *, n_spatial_dims: int) -> Tensor:
        spatial_dims = _spatial_dims(tensor, n_spatial_dims)
        if tensor.dtype in (torch.float16, torch.bfloat16):
            tensor = tensor.float()
        radius = _radial_frequency(tensor, spatial_dims)
        coefficients = torch.fft.fftn(tensor, dim=spatial_dims)
        filtered = torch.fft.ifftn(
            coefficients * (radius <= self.cutoff).to(device=tensor.device),
            dim=spatial_dims,
        )
        return filtered if tensor.is_complex() else filtered.real


class ScaleDecomposition(FieldDecomposition):
    """Split a field into telescoping physical-space scale bands.

    Smoothing operators must be ordered from strongest to weakest. Each named
    component is the additional detail recovered by the next operator, and the
    final residual restores the original field up to floating-point roundoff.
    """

    def __init__(
        self,
        scales: Sequence[Sequence[str | SmoothingOperator]],
        *,
        residual_name: str = "residual",
    ) -> None:
        super().__init__()
        if not scales:
            msg = "ScaleDecomposition requires at least one smoothing operator."
            raise ValueError(msg)

        # Each scale is a pair rather than a tuple specifically, because
        # configuration supplies pairs as lists. The pairing is therefore
        # checked here rather than being carried by the signature.
        names: list[str] = []
        operators: list[SmoothingOperator] = []
        for scale in scales:
            if len(scale) != 2:
                msg = (
                    "Each scale must pair a name with a smoothing operator, "
                    f"got {len(scale)} entries."
                )
                raise ValueError(msg)
            name, operator = scale
            if not isinstance(name, str) or not isinstance(operator, SmoothingOperator):
                msg = "Each scale must pair a name with a SmoothingOperator."
                raise ValueError(msg)
            names.append(name)
            operators.append(operator)

        if not residual_name or any(not name for name in names):
            msg = "Scale component names must be nonempty."
            raise ValueError(msg)
        if len({*names, residual_name}) != len(names) + 1:
            msg = "Scale component names must be unique."
            raise ValueError(msg)

        self.names = tuple(names)
        self.operators = nn.ModuleList(operators)
        self.residual_name = residual_name

    def forward(self, tensor: FieldTensor, *, n_spatial_dims: int) -> FieldComponents:
        components: FieldComponents = {}
        previous: Tensor | None = None
        for name, operator in zip(self.names, self.operators, strict=True):
            smoothed = operator(tensor, n_spatial_dims=n_spatial_dims)
            if smoothed.shape != tensor.shape:
                raise ValueError(
                    "Smoothing operators must preserve field shape; "
                    f"got {tuple(smoothed.shape)} and {tuple(tensor.shape)}."
                )
            components[name] = smoothed if previous is None else smoothed - previous
            previous = smoothed

        assert previous is not None
        components[self.residual_name] = tensor - previous
        return components
