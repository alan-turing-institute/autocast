"""Latent-conditioned, spatially stationary Fourier modulation."""

import math

import torch
from einops import einsum, rearrange
from torch import Tensor, nn
from torch.nn import functional as F


class FourierModulation2d(nn.Module):
    """Modulate every resolved spatial mode with a global latent per sample.

    A shared filter acts on all channels of a ``(B, H, W, C)`` field. A linear
    projection maps ``(B, mod_features)`` latents to amplitude and two directional
    phase coefficients on overlapping radial bands. The triangular bands form a
    partition of unity from DC to the grid corners; no frequencies are truncated.
    Gains lie between ``exp(-0.5)`` and ``exp(0.5)``. Directional phase coefficients
    are bounded by pi and multiply normalized height/width frequencies, allowing
    smooth phase ramps while retaining a real-compatible Fourier spectrum.

    The projection starts at zero, giving an identity filter with gradients to
    its parameters. FFTs operate only on the spatial axes, use orthonormal
    normalization, and promote float16/bfloat16 inputs to float32. Outputs retain
    the input shape and dtype. There is no timestep-dependent behavior.

    Args:
        mod_features: Number of features in each global conditioning latent.
        num_bands: Number of evenly spaced radial band centers, including 0 and 1.
    """

    def __init__(self, mod_features: int, num_bands: int = 4) -> None:
        super().__init__()
        if (
            isinstance(mod_features, bool)
            or not isinstance(mod_features, int)
            or mod_features < 1
        ):
            msg = "mod_features must be a positive integer"
            raise ValueError(msg)
        if (
            isinstance(num_bands, bool)
            or not isinstance(num_bands, int)
            or num_bands < 2
        ):
            msg = "num_bands must be an integer greater than or equal to 2"
            raise ValueError(msg)
        self.mod_features = mod_features
        self.num_bands = num_bands
        self.projection = nn.Linear(mod_features, 3 * num_bands)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: Tensor, mod: Tensor) -> Tensor:
        if x.ndim != 4 or any(size < 1 for size in x.shape):
            msg = "x must have shape (B, H, W, C) with positive dimensions"
            raise ValueError(msg)
        if mod.ndim != 2 or mod.shape != (x.shape[0], self.mod_features):
            msg = f"mod must have shape ({x.shape[0]}, {self.mod_features})"
            raise ValueError(msg)
        if not x.is_floating_point() or not mod.is_floating_point():
            msg = "x and mod must be real floating point tensors"
            raise TypeError(msg)
        if x.device != mod.device or x.device != self.projection.weight.device:
            msg = "x, mod, and module parameters must be on the same device"
            raise ValueError(msg)

        _, height, width, _ = x.shape
        fft_dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        with torch.autocast(device_type=x.device.type, enabled=False):
            # Explicit casts also support half-precision modules and conditioning
            # produced by autocast, while retaining gradients through the projection.
            coefficients = F.linear(
                mod.to(dtype=fft_dtype),
                self.projection.weight.to(dtype=fft_dtype),
                self.projection.bias.to(dtype=fft_dtype),
            )
            coefficients = rearrange(
                coefficients, "b (component band) -> b component band", component=3
            )
            kh = 2 * torch.fft.fftfreq(height, device=x.device, dtype=fft_dtype)
            kw = 2 * torch.fft.rfftfreq(width, device=x.device, dtype=fft_dtype)
            radius = (
                rearrange(kh.square(), "h -> h 1") + rearrange(kw.square(), "w -> 1 w")
            ).sqrt() / math.sqrt(2)
            centers = torch.linspace(
                0, 1, self.num_bands, device=x.device, dtype=fft_dtype
            )
            bands = (
                1
                - (rearrange(radius, "h w -> h w 1") - centers).abs()
                * (self.num_bands - 1)
            ).clamp_min(0)
            fields = einsum(
                coefficients, bands, "b component band, h w band -> b component h w"
            ).tanh()
            amplitude = (0.5 * fields[:, 0]).exp()

            # An even axis has a self-conjugate Nyquist mode: its directional
            # phase must vanish. This also preserves conjugate pairs on the DC
            # and width-Nyquist columns retained by rfft2.
            if height % 2 == 0:
                kh[height // 2] = 0
            if width % 2 == 0:
                kw[-1] = 0
            phase = math.pi * (
                fields[:, 1] * rearrange(kh, "h -> h 1")
                + fields[:, 2] * rearrange(kw, "w -> 1 w")
            )
            transfer = torch.polar(amplitude, phase)
            spectrum = torch.fft.rfft2(x.to(dtype=fft_dtype), dim=(1, 2), norm="ortho")
            result = torch.fft.irfft2(
                spectrum * rearrange(transfer, "b h w -> b h w 1"),
                s=(height, width),
                dim=(1, 2),
                norm="ortho",
            )
        return result.to(dtype=x.dtype)
