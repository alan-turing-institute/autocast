import math
from collections.abc import Sequence
from typing import cast

import torch
from azula.nn.layers import ConvNd, Unpatchify
from einops import rearrange
from torch import nn

from auto_cast.decoders.base import Decoder
from auto_cast.nn import ResBlock
from auto_cast.nn.dc_utils import build_sample_block
from auto_cast.types import TensorBCTSPlus, TensorBTSPlusC


class DCDecoder(Decoder):
    """Deep Compressed (DC) decoder module.

    Progressively upsamples from latent representation back to original spatial
    dimensions using residual blocks with optional attention.

    Parameters
    ----------
    in_channels: int
        Number of input (latent) channels.
    out_channels: int
        Number of output channels.
    hid_channels: Sequence[int]
        Number of channels at each depth level.
    hid_blocks: Sequence[int]
        Number of residual blocks at each depth level.
    kernel_size: int | Sequence[int]
        Kernel size for convolutions.
    stride: int | Sequence[int]
        Stride for upsampling operations.
    pixel_shuffle: bool
        Whether to use pixel shuffling or nearest upsampling.
    norm: str
        Type of normalization ('layer' or 'group').
    attention_heads: dict[int, int] | None
        Dict mapping depth index to number of attention heads.
    ffn_factor: int
        Channel expansion factor in FFN blocks.
    spatial: int
        Number of spatial dimensions (2 for 2D, 3 for 3D).
    patch_size: int | Sequence[int]
        Patch size for unpatchifying at the end.
    periodic: bool
        Whether spatial dimensions are periodic (use circular padding).
    dropout: float | None
        Dropout rate.
    checkpointing: bool
        Whether to use gradient checkpointing.
    identity_init: bool
        Initialize up/downsampling convolutions as identity.

    Notes
    -----
    Based on the implementation from:
    - Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models
    (Chen et al., 2024), https://arxiv.org/abs/2410.10733v1
    - Lost in Latent Space: An Empirical Study of Latent Diffusion Models for Physics
    Emulation (Rozet et al., 2024), https://arxiv.org/abs/2507.02608,
    https://github.com/PolymathicAI/lola

    """

    decoder_model: nn.Module

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_channels: Sequence[int] = (64, 128, 256),
        hid_blocks: Sequence[int] = (3, 3, 3),
        kernel_size: int | Sequence[int] = 3,
        stride: int | Sequence[int] = 2,
        pixel_shuffle: bool = True,
        norm: str = "layer",
        attention_heads: dict[int, int] | None = None,
        ffn_factor: int = 1,
        spatial: int = 2,
        patch_size: int | Sequence[int] = 1,
        periodic: bool = False,
        dropout: float | None = None,
        checkpointing: bool = False,
        identity_init: bool = True,
    ) -> None:
        super().__init__()
        self.latent_dim = in_channels
        self.output_channels = out_channels
        attention_heads = attention_heads or {}
        assert len(hid_blocks) == len(hid_channels)

        # Normalize to sequences
        kernel_size = (
            [kernel_size] * spatial if isinstance(kernel_size, int) else kernel_size
        )
        stride = [stride] * spatial if isinstance(stride, int) else stride
        patch_size = (
            [patch_size] * spatial if isinstance(patch_size, int) else patch_size
        )

        kwargs = {
            "kernel_size": tuple(kernel_size),
            "padding": tuple(k // 2 for k in kernel_size),
            "padding_mode": "circular" if periodic else "zeros",
        }

        self.unpatch = Unpatchify(patch_shape=tuple(patch_size))

        # Build decoder from deepest to shallowest
        self.ascent = nn.ModuleList()
        for i, num_blocks in reversed(list(enumerate(hid_blocks))):
            blocks = nn.ModuleList()

            # Initial projection from latent at deepest level
            if i + 1 == len(hid_blocks):
                blocks.append(
                    ConvNd(
                        in_channels,
                        hid_channels[i],
                        spatial=spatial,
                        identity_init=identity_init,
                        **kwargs,
                    )
                )

            # Add residual blocks
            for _ in range(num_blocks):
                blocks.append(
                    ResBlock(
                        hid_channels[i],
                        norm=norm,
                        attention_heads=attention_heads.get(i),
                        ffn_factor=ffn_factor,
                        spatial=spatial,
                        dropout=dropout,
                        checkpointing=checkpointing,
                        **kwargs,
                    )
                )

            # Upsampling to next level (except at shallowest)
            if i > 0:
                blocks.append(
                    build_sample_block(
                        hid_channels[i],
                        hid_channels[i - 1],
                        stride,
                        pixel_shuffle,
                        spatial,
                        identity_init,
                        upsample=True,
                        **kwargs,
                    )
                )
            else:
                # Final projection to output channels at shallowest level
                blocks.append(
                    ConvNd(
                        hid_channels[i],
                        math.prod(patch_size) * out_channels,
                        spatial=spatial,
                        **kwargs,
                    )
                )

            self.ascent.append(blocks)

        self.decoder_model = self.ascent

    def postprocess(self, decoded: TensorBTSPlusC) -> TensorBTSPlusC:
        return rearrange(decoded, "B C ... -> B ... C")

    def forward(self, z: TensorBCTSPlus) -> TensorBTSPlusC:
        """Forward pass through decoder (for direct tensor input).

        Parameters
        ----------
        z: TensorBCTSPlus
            Latent tensor with shape (B, C_i, L_1, ..., L_N).

        Returns
        -------
        TensorBTSPlusC
            Decoded tensor with shape (B, L_1 x 2^D, ..., L_N x 2^D, C_o).

        """
        x = z
        for blocks in self.ascent:
            for block in cast(nn.ModuleList, blocks):
                x = block(x)
        x = self.unpatch(x)
        return self.postprocess(x)

    def decode(self, z: TensorBTSPlusC) -> TensorBTSPlusC:
        """Decode latent tensor with time dimension back to original space.

        Parameters
        ----------
        z: Tensor
            Latent tensor with shape (B, T, spatial..., C_i) where C_i is last dim.

        Returns
        -------
        Tensor
            Decoded tensor with shape (B, T, spatial_expanded..., C_o).

        """
        outputs = []
        for idx in range(z.shape[1]):
            x = z[:, idx, ...]
            x = rearrange(x, "B ... C -> B C ...")
            x = self.forward(x)
            outputs.append(x)
        return torch.stack(outputs, dim=1)

    def __call__(self, z: TensorBTSPlusC) -> TensorBTSPlusC:
        return self.decode(z)
