"""Spatially correlated, temporally independent Gaussian noise."""

import math
from collections.abc import Sequence
from numbers import Integral

import torch
from einops import einsum, rearrange
from torch import Tensor

from autocast.nn.noise.source import FlowSource, _validate_spatiotemporal


def _validate_wavevectors(
    wavevectors: Sequence[Sequence[int]],
) -> tuple[tuple[int, int], ...]:
    result = []
    seen = set()
    for pair in wavevectors:
        if len(pair) != 2 or any(
            isinstance(k, bool) or not isinstance(k, Integral) for k in pair
        ):
            msg = "Spatial noise wavevectors must be pairs of integers."
            raise ValueError(msg)
        vector = (int(pair[0]), int(pair[1]))
        opposite = (-vector[0], -vector[1])
        if vector == (0, 0) or vector in seen or opposite in seen:
            msg = "Spatial noise wavevectors must be nonzero and unique up to sign."
            raise ValueError(msg)
        seen.add(vector)
        result.append(vector)
    if not result:
        msg = "Spatial noise requires at least one Fourier wavevector."
        raise ValueError(msg)
    return tuple(result)


class SpatialFourierSource(FlowSource):
    """Draw periodic Gaussian fields from prescribed integer wavevectors.

    ``sample_like`` accepts ``(B, T, Y, X, C)``. Frames, members and channels
    have independent sine/cosine coefficients; only space is correlated.
    Each call samples afresh, with no temporal state. The same source can
    supply flow states, input noise or spatial AdaLN conditioning.

    Wavevectors count integer cycles in height/width order on the supplied
    grid, which may be a field grid or a patch-token grid. They must be
    distinct up to sign, nonzero and strictly below that grid's Nyquist.
    With K mode pairs there are 2*K*C Gaussian coefficients per frame.
    Marginal variance is one at every site in expectation, without
    normalizing individual draws. No trainable parameters are introduced.
    """

    def __init__(self, wavevectors: Sequence[Sequence[int]]) -> None:
        super().__init__()
        self.wavevectors = _validate_wavevectors(wavevectors)

    def sample_like(self, reference: Tensor) -> Tensor:
        """Return a fresh field matching the reference shape, device and dtype."""
        _validate_spatiotemporal(reference)
        batch, frames, height, width, channels = reference.shape
        if min(height, width, channels) < 1:
            msg = "Spatial noise requires positive spatial and channel dimensions."
            raise ValueError(msg)
        if any(
            2 * abs(kh) >= height or 2 * abs(kw) >= width for kh, kw in self.wavevectors
        ):
            msg = "Spatial noise wavevectors must lie strictly below grid Nyquist."
            raise ValueError(msg)
        # Compute phases in at least float32 even under mixed-precision training.
        work_dtype = (
            torch.float64 if reference.dtype == torch.float64 else torch.float32
        )
        kh, kw = torch.tensor(
            self.wavevectors, device=reference.device, dtype=work_dtype
        ).unbind(-1)
        h = torch.arange(height, device=reference.device, dtype=work_dtype) / height
        w = torch.arange(width, device=reference.device, dtype=work_dtype) / width
        phase = 2 * math.pi * (h[:, None, None] * kh + w[None, :, None] * kw)
        basis = torch.stack((phase.cos(), phase.sin()), dim=-1)
        coefficients = torch.randn(
            batch * frames,
            channels,
            len(self.wavevectors),
            2,
            dtype=work_dtype,
            device=reference.device,
        )
        field = einsum(coefficients, basis, "b d m s, h w m s -> b h w d")
        field = field / math.sqrt(len(self.wavevectors))
        return rearrange(field, "(b t) h w d -> b t h w d", b=batch, t=frames).to(
            dtype=reference.dtype
        )
