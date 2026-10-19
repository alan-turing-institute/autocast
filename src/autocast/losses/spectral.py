from typing import Literal

import torch

from autocast.losses.decomposition import (
    FieldComponents,
    FieldDecomposition,
    _spatial_dims,
)
from autocast.types import FieldTensor

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
