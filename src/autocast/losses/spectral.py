import math
from collections.abc import Sequence
from itertools import pairwise
from typing import Literal

import torch

from autocast.losses.decomposition import (
    FieldComponents,
    FieldDecomposition,
    _spatial_dims,
)
from autocast.types import FieldTensor, Tensor

FourierRepresentation = Literal["complex", "magnitude", "power"]
FourierNorm = Literal["backward", "forward", "ortho"]


class FourierDecomposition(FieldDecomposition):
    """Apply a full ND Fourier transform over channels-last spatial axes.

    Inputs follow ``(B, T, S1, ..., Sd, C)`` with an optional final ensemble
    axis. The transform preserves that layout and transforms spatial axes
    only. It does not transform time, channels or ensemble members.

    Half and bfloat16 inputs are promoted to float32 on the same device before
    the FFT. Single and double precision are preserved, including complex
    inputs. The cast and transform remain differentiable.

    Args:
        representation: Return complex coefficients, their magnitudes or their
            squared magnitudes, under the component name ``fourier``.
        norm: FFT normalization. ``ortho`` scales coefficients by the reciprocal
            square root of the number of spatial cells; ``forward`` uses the
            reciprocal cell count and ``backward`` applies no scaling.
    """

    def __init__(
        self,
        representation: FourierRepresentation = "complex",
        norm: FourierNorm = "ortho",
    ) -> None:
        super().__init__()
        if representation not in ("complex", "magnitude", "power"):
            raise ValueError(f"Unknown Fourier representation {representation!r}.")
        if norm not in ("backward", "forward", "ortho"):
            raise ValueError(f"Unknown Fourier normalization {norm!r}.")
        self.representation = representation
        self.norm = norm

    def forward(self, tensor: FieldTensor, *, n_spatial_dims: int) -> FieldComponents:
        spatial_dims = _spatial_dims(tensor, n_spatial_dims)
        if tensor.dtype in (torch.float16, torch.bfloat16):
            tensor = tensor.float()
        coefficients = torch.fft.fftn(tensor, dim=spatial_dims, norm=self.norm)
        if self.representation == "magnitude":
            coefficients = coefficients.abs()
        elif self.representation == "power":
            coefficients = coefficients.abs().square()
        return {"fourier": coefficients}


class RadialFourierDecomposition(FourierDecomposition):
    """Split complex Fourier coefficients into disjoint radial bands.

    Each ``band_<index>`` retains the full field layout, with coefficients
    outside that band set to zero. This selects coefficients, not a radial
    average of their power. Frequencies beyond the final edge are omitted.
    Spatial axes, FFT normalization and precision follow FourierDecomposition.

    ``band_mean=True`` divides coefficients by the square root of the number
    of modes in their band. Summing their squared magnitudes then yields mean
    band power. This does not give a within-band mean for coefficient CRPS:
    its reduction still includes zeroed positions across the full grid.
    Band weights and the scoring objective determine the final aggregation.
    Empty bands return zero tensors. Memory grows with the number of bands.

    Args:
        edges: Finite, positive, strictly increasing upper edges in cycles per
            grid cell. The first band starts at zero. Lower edges are included;
            upper edges are excluded except for the final edge, which is
            included. A final edge of at least ``sqrt(n_spatial_dims) / 2``
            covers the entire grid, including diagonal Nyquist modes. These
            are grid-relative frequencies, not physical wavenumbers.
        norm: FFT normalization, as in FourierDecomposition.
        band_mean: Divide each nonempty band's coefficients by sqrt(mode count).
    """

    def __init__(
        self,
        edges: Sequence[int | float],
        norm: FourierNorm = "ortho",
        band_mean: bool = False,
    ) -> None:
        super().__init__(norm=norm)
        if not edges:
            msg = "RadialFourierDecomposition requires at least one band edge."
            raise ValueError(msg)
        resolved = [float(edge) for edge in edges]
        if any(not math.isfinite(edge) for edge in resolved):
            msg = f"Band edges must be finite, got {resolved}."
            raise ValueError(msg)
        if resolved[0] <= 0:
            msg = f"Band edges must be positive, got {resolved}."
            raise ValueError(msg)
        if any(later <= earlier for earlier, later in pairwise(resolved)):
            msg = f"Band edges must increase, got {resolved}."
            raise ValueError(msg)

        self.edges = tuple(resolved)
        self.band_mean = bool(band_mean)

    def _radial_frequency(self, tensor: Tensor, spatial: tuple[int, ...]) -> Tensor:
        # Geometry must not depend on prediction/target precision: rounding a
        # cutoff differently would compare different modes. Construct masks
        # in CPU float64, including for devices without float64 support.
        # Transfer masks and optional counts; field FFTs stay on their device.
        radius_squared = torch.zeros((), dtype=torch.float64, device="cpu")
        for dim in spatial:
            size = tensor.shape[dim]
            indices = torch.arange(size, dtype=torch.float64, device="cpu")
            # Divide integer bins directly: fftfreq's reciprocal multiplication
            # can put nominal edges such as 3 / 10 just above the cutoff.
            frequency = torch.where(indices < (size + 1) // 2, indices, indices - size)
            frequency = frequency / size
            shape = [1] * tensor.ndim
            shape[dim] = size
            radius_squared = radius_squared + frequency.reshape(shape).square()
        return radius_squared.sqrt()

    def forward(self, tensor: FieldTensor, *, n_spatial_dims: int) -> FieldComponents:
        coefficients = super().forward(tensor, n_spatial_dims=n_spatial_dims)["fourier"]
        spatial = _spatial_dims(tensor, n_spatial_dims)
        radius = self._radial_frequency(tensor, spatial)

        components: FieldComponents = {}
        lower = 0.0
        for index, upper in enumerate(self.edges):
            within = (radius >= lower) & (
                radius <= upper if index == len(self.edges) - 1 else radius < upper
            )
            band = coefficients * within.to(device=coefficients.device)
            if self.band_mean:
                # A coarse grid can leave a band empty.
                count = within.sum().clamp_min(1)
                band = band / count.to(device=band.device, dtype=band.real.dtype).sqrt()
            components[f"band_{index}"] = band
            lower = upper
        return components
