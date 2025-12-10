import math
from collections.abc import Sequence
from typing import cast

import torch
from azula.nn.layers import ConvNd, Patchify
from einops import rearrange
from torch import nn

from autocast.encoders.base import Encoder
from autocast.nn import ResBlock
from autocast.nn.dc_utils import build_sample_block
from autocast.types import Batch, TensorBTSC


class DCEncoder(Encoder):
    """Deep Compressed (DC) encoder module.

    Progressively downsamples input to latent representation using residual blocks
    with optional attention.

    Parameters
    ----------
    in_channels: int
        Number of input channels.
    out_channels: int
        Number of output (latent) channels.
    hid_channels: Sequence[int]
        Number of channels at each depth level.
    hid_blocks: Sequence[int]
        Number of residual blocks at each depth level.
    kernel_size: int | Sequence[int]
        Kernel size for convolutions.
    stride: int | Sequence[int]
        Stride for downsampling operations.
    pixel_shuffle: bool
        Whether to use pixel shuffling (patchify) for downsampling.
    norm: str
        Type of normalization ('layer' or 'group').
    attention_heads: dict[int, int] | None
        Dict mapping depth index to number of attention heads.
    ffn_factor: int
        Channel expansion factor in FFN blocks.
    spatial: int
        Number of spatial dimensions (2 for 2D, 3 for 3D).
    patch_size: int | Sequence[int]
        Patch size for patchifying at the start.
    periodic: bool
        Whether spatial dimensions are periodic (use circular padding).
    dropout: float | None
        Dropout rate.
    checkpointing: bool
        Whether to use gradient checkpointing.
    identity_init: bool
        Initialize down/upsampling convolutions as identity.

    Notes
    -----
    Based on the implementation from:
    - Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models
    (Chen et al., 2024), https://arxiv.org/abs/2410.10733v1
    - Lost in Latent Space: An Empirical Study of Latent Diffusion Models for Physics
    Emulation (Rozet et al., 2024), https://arxiv.org/abs/2507.02608,
    https://github.com/PolymathicAI/lola
    """

    encoder_model: nn.Module

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

        self.patch = Patchify(patch_shape=tuple(patch_size))
        self.latent_dim = out_channels
        self.input_channels = in_channels

        # Build encoder from shallowest to deepest
        self.descent = nn.ModuleList()

        for i, num_blocks in enumerate(hid_blocks):
            blocks = nn.ModuleList()

            # Downsampling from previous level (except at first level)
            if i > 0:
                blocks.append(
                    build_sample_block(
                        hid_channels[i - 1],
                        hid_channels[i],
                        stride,
                        pixel_shuffle,
                        spatial,
                        identity_init,
                        **kwargs,
                    )
                )
            else:
                # Initial projection at shallowest level
                blocks.append(
                    ConvNd(
                        math.prod(patch_size) * in_channels,
                        hid_channels[i],
                        spatial=spatial,
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

            # Final projection to latent at deepest level
            if i + 1 == len(hid_blocks):
                blocks.append(
                    ConvNd(
                        hid_channels[i],
                        out_channels,
                        spatial=spatial,
                        identity_init=identity_init,
                        **kwargs,
                    )
                )

            self.descent.append(blocks)

        self.encoder_model = self.descent

    def preprocess(self, batch: Batch) -> Batch:
        x = batch.input_fields
        x = rearrange(x, "B T ... C -> B C T ...")
        return Batch(
            input_fields=x,
            output_fields=batch.output_fields,
            constant_scalars=batch.constant_scalars,
            constant_fields=batch.constant_fields,
        )

    def encode(self, batch: Batch) -> TensorBTSC:
        """Encode input batch to latent representation.

        Parameters
        ----------
        batch: Batch
            Input batch containing input_fields with shape (B, T, spatial..., C_i).

        Returns
        -------
        Tensor
            Encoded latent tensor with shape (B, T, spatial_reduced..., C_o).

        """
        batch = self.preprocess(batch)
        outputs = []
        for idx in range(batch.input_fields.shape[2]):  # loop over time dimension
            x = batch.input_fields[:, :, idx, ...]
            x = self.patch(x)
            for blocks in self.descent:
                for block in cast(nn.ModuleList, blocks):  # ModuleList in construction
                    x = block(x)
            outputs.append(x)

        # Stack outputs along time dimension
        stacked = torch.stack(outputs, dim=1)  # (B, T, C, spatial...)
        return rearrange(stacked, "B T C ... -> B T ... C")

    def forward(self, x: TensorBTSC) -> TensorBTSC:
        """Forward pass through encoder (for direct tensor input).

        Parameters
        ----------
        x: TensorBTSC
            Input tensor with shape (B, T, spatial..., C_i).

        Returns
        -------
        TensorBTSC
            Encoded latent tensor.

        """
        x = self.patch(x)

        for blocks in self.descent:
            for block in cast(nn.ModuleList, blocks):
                x = block(x)
        return x

    def __call__(self, batch: Batch) -> TensorBTSC:
        return self.encode(batch)
