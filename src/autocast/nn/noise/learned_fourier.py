"""Learned spatial covariance and relative phase for shared noise innovations."""

import math

import torch
from einops import rearrange
from torch import Tensor, nn


class LearnedFourierNoise2d(nn.Module):
    """Filter one explicit noise field into block-specific spatial conditioning.

    Inputs are real ``(B, H, W, C)`` innovations. Outputs have shape
    ``(B, blocks, H, W, C)`` and reuse those innovations without drawing noise.
    A learned piecewise-linear radial response covers all non-DC modes; DC
    has a separate gain. Each block's transfer has mean squared modulus one,
    giving unit marginal variance for IID standard Gaussian inputs. Individual
    draws are never centered or standardized. Channels share filters but have
    independent innovations when the caller supplies IID noise.

    All gains start at one and phases at zero: the initial filter is IID.
    Setting the non-DC gains to zero and retaining DC gives one spatially
    constant standard Gaussian per channel, not a global multi-feature vector.
    Setting every gain to zero gives zero noise; normalization has a numerical
    floor so this degenerate setting remains finite.

    Relative phases are learned for blocks after the first. A common phase
    would leave the joint Gaussian law unchanged and is fixed by anchoring the
    first block at zero phase. Odd directional phase responses preserve real
    fields, including the self-conjugate Nyquist modes of even grids. Changing
    phase affects cross-block covariance, not a single block's radial power.

    Args:
        blocks: Number of independently parameterized block filters.
        radial_knots: Number of interpolation nodes spanning normalized radial
            frequency from zero to the grid corner. DC is handled separately.
    """

    def __init__(self, blocks: int, radial_knots: int = 16) -> None:
        super().__init__()
        for name, value, minimum in (
            ("blocks", blocks, 1),
            ("radial_knots", radial_knots, 2),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                msg = f"{name} must be an integer greater than or equal to {minimum}"
                raise ValueError(msg)
        self.blocks = blocks
        self.radial_knots = radial_knots
        self.radial_gains = nn.Parameter(torch.ones(blocks, radial_knots))
        self.dc_gain = nn.Parameter(torch.ones(blocks))
        self.phase_coefficients = (
            nn.Parameter(torch.zeros(blocks - 1, 2, radial_knots))
            if blocks > 1
            else None
        )

    def transfer(self, height: int, width: int) -> Tensor:
        """Return the normalized complex transfer for each block, ``(L,H,W)``."""
        if height < 1 or width < 1:
            msg = "Spatial noise grid dimensions must be positive."
            raise ValueError(msg)
        device = self.radial_gains.device
        dtype = (
            torch.float64 if self.radial_gains.dtype == torch.float64 else torch.float32
        )
        with torch.autocast(device_type=device.type, enabled=False):
            kh = 2 * torch.fft.fftfreq(height, device=device, dtype=dtype)
            kw = 2 * torch.fft.fftfreq(width, device=device, dtype=dtype)
            radius = (
                rearrange(kh.square(), "h -> h 1") + rearrange(kw.square(), "w -> 1 w")
            ).sqrt() / math.sqrt(2)
            coordinate = radius * (self.radial_knots - 1)
            lower = coordinate.floor().long().clamp_max(self.radial_knots - 2)
            fraction = coordinate - lower

            gains = self.radial_gains.to(dtype=dtype)
            amplitude = (
                gains[:, lower] * (1 - fraction) + gains[:, lower + 1] * fraction
            )
            amplitude = torch.where(
                radius == 0,
                rearrange(self.dc_gain.to(dtype=dtype), "l -> l 1 1"),
                amplitude,
            )
            # Normalize the expected variance, using all signed FFT modes.
            # vector_norm has a finite zero derivative at the degenerate zero
            # filter, unlike sqrt(sum(square(...))) at exactly zero.
            norm = torch.linalg.vector_norm(amplitude, dim=(-2, -1), keepdim=True)
            amplitude = amplitude * (math.sqrt(height * width) / norm.clamp_min(1e-6))

            phase = torch.zeros(self.blocks, height, width, device=device, dtype=dtype)
            if self.phase_coefficients is not None:
                coefficients = self.phase_coefficients.to(dtype=dtype)
                fields = (
                    coefficients[..., lower] * (1 - fraction)
                    + coefficients[..., lower + 1] * fraction
                ).tanh()
                if height % 2 == 0:
                    kh[height // 2] = 0
                if width % 2 == 0:
                    kw[width // 2] = 0
                relative = math.pi * (
                    fields[:, 0] * rearrange(kh, "h -> h 1")
                    + fields[:, 1] * rearrange(kw, "w -> 1 w")
                )
                phase = torch.cat((phase[:1], relative), dim=0)
            # Signed amplitudes admit exact zero gains. Expanding polar by
            # hand also allows negative amplitudes without polar's restriction.
            return amplitude * torch.complex(phase.cos(), phase.sin())

    def forward(self, innovations: Tensor) -> Tensor:
        if innovations.ndim != 4 or min(innovations.shape) < 1:
            msg = "innovations must have positive shape (B, H, W, C)."
            raise ValueError(msg)
        if not innovations.is_floating_point():
            msg = "innovations must be a real floating point tensor."
            raise TypeError(msg)
        if innovations.device != self.radial_gains.device:
            msg = "innovations and filter parameters must be on the same device."
            raise ValueError(msg)
        _, height, width, _ = innovations.shape
        dtype = torch.float64 if innovations.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=innovations.device.type, enabled=False):
            spectrum = torch.fft.fft2(
                innovations.to(dtype=dtype), dim=(1, 2), norm="ortho"
            )
            transfer = self.transfer(height, width).to(dtype=spectrum.dtype)
            filtered = rearrange(spectrum, "b h w c -> b 1 h w c") * rearrange(
                transfer, "l h w -> 1 l h w 1"
            )
            result = torch.fft.ifft2(filtered, dim=(2, 3), norm="ortho").real
        return result.to(dtype=innovations.dtype)
