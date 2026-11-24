"""Full-grid periodic analogue of the local/global FourCastNet 3 backbone."""

from collections.abc import Sequence
from math import pi, sqrt

import torch
from einops import einsum
from torch import Tensor, nn
from torch.nn import functional as F

from autocast.processors.base import Processor
from autocast.types import EncodedBatch


class _GlobalFourierConv(nn.Module):
    def __init__(self, width: int, spectral_width: int, max_wavenumber: int):
        super().__init__()
        self.max_wavenumber = max_wavenumber
        self.lift = nn.Conv2d(width, spectral_width, 1, bias=False)
        self.project = nn.Conv2d(spectral_width, width, 1, bias=False)
        # Store real and imaginary parts separately so DDP reduces real tensors.
        k = max_wavenumber
        self.weight = nn.Parameter(
            torch.empty(2 * k + 1, k + 1, spectral_width, spectral_width, 2)
        )
        nn.init.normal_(self.weight, std=1 / sqrt(2 * spectral_width))

    def forward(self, field: Tensor) -> Tensor:
        lifted = self.lift(field)
        height, width = lifted.shape[-2:]
        k = self.max_wavenumber
        lower, upper = height // 2 - k, height // 2 + k + 1
        with torch.autocast(device_type=lifted.device.type, enabled=False):
            modes = torch.fft.rfft2(lifted.float(), norm="ortho")
            modes = torch.fft.fftshift(modes, dim=-2)[..., lower:upper, : k + 1]
            weights = torch.view_as_complex(self.weight)
            mixed = einsum(modes, weights, "b i y x, y x i o -> b o y x")
            mixed = F.pad(mixed, (0, width // 2 - k, lower, height - upper))
            mixed = torch.fft.ifftshift(mixed, dim=-2)
            # Invert the axes in sequence so the real inverse handles the
            # self-conjugate columns after transforming the vertical axis.
            mixed = torch.fft.ifft(mixed, dim=-2, norm="ortho")
            grid = torch.fft.irfft(mixed, n=width, dim=-1, norm="ortho")
        return self.project(grid.to(lifted.dtype))


class _OperatorBlock(nn.Module):
    def __init__(self, width: int, mixer: nn.Module):
        super().__init__()
        self.mixer = mixer
        self.mlp = nn.Sequential(
            nn.Conv2d(width, 2 * width, 1), nn.GELU(), nn.Conv2d(2 * width, width, 1)
        )
        self.scale = nn.Parameter(torch.full((width, 1, 1), 0.1))

    def forward(self, field: Tensor) -> Tensor:
        update = self.mlp(self.mixer(field))
        return field + self.scale.to(update.dtype) * update


class PeriodicHybridFNOProcessor(Processor[EncodedBatch]):
    """Use a global Fourier convolution followed by four local blocks, twice.

    Local convolutions wrap at the boundaries. Periodic coordinates let the
    operator represent spatially varying dynamics such as a beta plane. Noise
    is supplied alongside fields by the existing EPD input concatenation.
    """

    coordinates: Tensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_resolution: Sequence[int],
        hidden_channels: int = 128,
        spectral_channels: int = 47,
        max_wavenumber: int = 21,
    ):
        super().__init__()
        height, width = spatial_resolution
        if not 0 < max_wavenumber < min(height, width) / 2:
            msg = "The Fourier cutoff must be positive and below both Nyquist modes."
            raise ValueError(msg)
        self.spatial_resolution = (height, width)
        y, x = torch.meshgrid(
            torch.arange(height) * (2 * pi / height),
            torch.arange(width) * (2 * pi / width),
            indexing="ij",
        )
        self.register_buffer(
            "coordinates",
            torch.stack((x.sin(), x.cos(), y.sin(), y.cos())),
            persistent=False,
        )
        self.lift = nn.Conv2d(in_channels + 4, hidden_channels, 1)
        self.blocks = nn.ModuleList()
        for _ in range(2):
            self.blocks.append(
                _OperatorBlock(
                    hidden_channels,
                    _GlobalFourierConv(
                        hidden_channels, spectral_channels, max_wavenumber
                    ),
                )
            )
            for _ in range(4):
                self.blocks.append(
                    _OperatorBlock(
                        hidden_channels,
                        nn.Conv2d(
                            hidden_channels,
                            hidden_channels,
                            3,
                            padding=1,
                            padding_mode="circular",
                        ),
                    )
                )
        self.project = nn.Conv2d(hidden_channels, out_channels, 1)

    def forward(self, fields: Tensor) -> Tensor:
        if fields.shape[-2:] != self.spatial_resolution:
            msg = f"Expected spatial resolution {self.spatial_resolution}."
            raise ValueError(msg)
        coords = self.coordinates.to(fields.dtype).expand(fields.size(0), -1, -1, -1)
        state = self.lift(torch.cat((fields, coords), dim=1))
        for block in self.blocks:
            state = block(state)
        return self.project(state)

    def map(self, x: Tensor, global_cond: Tensor | None = None) -> Tensor:
        _ = global_cond  # The fixed-physics benchmark supplies no global condition.
        return self(x)

    def loss(self, batch: EncodedBatch) -> Tensor:
        return F.mse_loss(self.map(batch.encoded_inputs), batch.encoded_output_fields)
