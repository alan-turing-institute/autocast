"""Fourier mixing as an alternative to attention on the Azula patch grid."""

from collections.abc import Sequence
from math import sqrt
from typing import Any

import torch
from azula.nn.vit import ViT
from einops import einsum, rearrange
from torch import Tensor, nn

from autocast.processors.azula_vit import AzulaViTProcessor


class _FourierPatchMixer(nn.Module):
    """Use a channel bottleneck and a full-spectrum periodic convolution."""

    def __init__(
        self, channels: int, spectral_channels: int, grid_shape: tuple[int, int]
    ):
        super().__init__()
        self.grid_shape = grid_shape
        height, width = grid_shape
        self.in_proj = nn.Linear(channels, spectral_channels, bias=False)
        # Real/imaginary pairs give DDP real tensors to reduce.
        self.weight = nn.Parameter(
            torch.empty(height, width // 2 + 1, spectral_channels, spectral_channels, 2)
        )
        nn.init.normal_(self.weight, std=1 / sqrt(2 * spectral_channels))
        self.out_proj = nn.Linear(spectral_channels, channels, bias=False)

    def forward(self, x: Tensor, _pos: Tensor | None, mask: Tensor | None) -> Tensor:
        """Replace the self-attention call in an unmasked Azula block."""
        if mask is not None:
            msg = "Fourier patch mixing requires an unmasked periodic grid."
            raise ValueError(msg)
        height, width = self.grid_shape
        x = self.in_proj(x)
        grid = rearrange(x, "b (h w) c -> b c h w", h=height, w=width)
        # FFT and complex contractions run in float32 under mixed precision.
        with torch.autocast(device_type=grid.device.type, enabled=False):
            modes = torch.fft.rfft2(grid.float(), norm="ortho")
            weight = torch.view_as_complex(self.weight.float())
            mixed_modes = einsum(modes, weight, "b i h w, h w i o -> b o h w")
            mixed = torch.fft.irfft2(mixed_modes, s=self.grid_shape, norm="ortho")
        return self.out_proj(rearrange(mixed.to(x.dtype), "b c h w -> b (h w) c"))


class AdaLNPatchFNOProcessor(AzulaViTProcessor):
    """Use periodic Fourier mixing with the ViT's patching and AdaLN blocks.

    The processor is configured for a fixed two-dimensional field resolution.
    It retains the absolute position embedding, MLPs, adaptive normalization,
    residual gates and global or per-token noise interface. Relative attention
    biases and rotary embeddings are specific to the replaced mixer.
    """

    def __init__(
        self,
        *,
        spectral_channels: int,
        spatial_resolution: Sequence[int],
        **kwargs: Any,
    ):
        if (
            kwargs.get("graph_hops") is not None
            or kwargs.get("modulation_mode", "adaln") != "adaln"
        ):
            msg = "Fourier patch mixing requires dense blocks with AdaLN."
            raise ValueError(msg)
        # Generic EPD setup also supplies the flow-processor channel alias.
        kwargs.pop("n_channels_out", None)
        super().__init__(spatial_resolution=spatial_resolution, **kwargs)
        height, width = spatial_resolution
        patch_h, patch_w = self.patch_size
        if height % patch_h or width % patch_w:
            msg = "The configured resolution must be divisible by patch_size."
            raise ValueError(msg)
        self.field_shape = (height, width)
        grid_shape = (height // patch_h, width // patch_w)
        backbone = self.model.backbone
        if not isinstance(backbone, ViT):
            msg = "Fourier patch mixing requires the Azula ViT block layout."
            raise TypeError(msg)
        channels = backbone.in_proj.out_features
        if not 1 <= spectral_channels <= channels:
            msg = "spectral_channels must be between one and the hidden width."
            raise ValueError(msg)
        for block in backbone.blocks:
            block.msa = _FourierPatchMixer(channels, spectral_channels, grid_shape)

    def forward(
        self,
        x: Tensor,
        x_noise: Tensor | None = None,
        global_cond: Tensor | None = None,
    ) -> Tensor:
        """Forecast the configured grid using the existing AdaLN interface."""
        if x.ndim in (4, 5):
            shape = x.shape[-2:] if x.ndim == 4 else x.shape[2:4]
            if shape != self.field_shape:
                msg = f"Expected field grid {self.field_shape}, got {tuple(shape)}."
                raise ValueError(msg)
        return super().forward(x, x_noise, global_cond)
