import math
from collections.abc import Sequence
from typing import cast

from azula.nn.layers import ConvNd, Patchify
from einops import rearrange
from torch import Tensor, nn

from autocast.encoders.base import EncoderWithCond
from autocast.nn import ResBlock
from autocast.nn.dc_utils import build_sample_block
from autocast.types import Batch, TensorBTSC


class DCEncoder(EncoderWithCond):
    """Deep Compressed (DC) encoder module.

    Progressively downsamples input to latent representation using residual blocks
    with optional attention.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output (latent) channels.
        hid_channels: Number of channels at each depth level.
        hid_blocks: Number of residual blocks at each depth level.
        kernel_size: Kernel size for convolutions.
        stride: Stride for downsampling operations.
        pixel_shuffle: Whether to use pixel shuffling (patchify) for downsampling.
        norm: Type of normalization ('layer' or 'group').
        attention_heads: Dict mapping depth index to number of attention heads.
        ffn_factor: Channel expansion factor in FFN blocks.
        spatial: Number of spatial dimensions (2 for 2D, 3 for 3D).
        patch_size: Patch size for patchifying at the start.
        periodic: Whether spatial dimensions are periodic (use circular padding).
        dropout: Dropout rate.
        checkpointing: Whether to use gradient checkpointing.
        identity_init: Initialize down/upsampling convolutions as identity.
        ffn_out_scale: Optional multiplicative scale applied to each ResBlock
            FFN output conv.
        saturation: Optional latent saturation mode. Supported: {"softclip2",
            "softclip", "tanh", "arcsinh", "rmsnorm"}.
        saturation_scale: Saturation scale B used by soft clipping/tanh variants.

    Note:
        Based on the implementation from:
        - Deep Compression Autoencoder for Efficient High-Resolution Diffusion Models
        (Chen et al., 2024), https://arxiv.org/abs/2410.10733v1
        - Lost in Latent Space: An Empirical Study of Latent Diffusion Models
          for Physics Emulation (Rozet et al., 2024),
          https://arxiv.org/abs/2507.02608, https://github.com/PolymathicAI/lola
    """

    encoder_model: nn.Module
    channel_axis: int = -1

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
        ffn_out_scale: float | None = None,
        saturation: str | None = None,
        saturation_scale: float = 5.0,
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
        self.latent_channels = out_channels
        self.input_channels = in_channels
        self.saturation = saturation
        self.saturation_scale = saturation_scale

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
                        ffn_out_scale=ffn_out_scale,
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

    def _saturate(self, z: Tensor) -> Tensor:
        if self.saturation is None:
            return z

        mode = self.saturation.lower()
        b = self.saturation_scale

        if mode == "softclip2":
            return z * (1 + (z / b).pow(2)).rsqrt()
        if mode == "softclip":
            return z / (1 + z.abs() / b)
        if mode == "tanh":
            return (z / b).tanh() * b
        if mode == "arcsinh":
            return z.arcsinh()
        if mode == "rmsnorm":
            return z * (z.square().mean(dim=1, keepdim=True) + 1e-5).rsqrt()

        msg = f"Unknown saturation mode: {self.saturation}"
        raise ValueError(msg)

    def encode(self, batch: Batch) -> TensorBTSC:
        """Encode input batch to latent representation.

        Args:
            batch: Input batch containing input_fields with shape
                (B, T, spatial..., C_i).

        Returns:
            Encoded latent tensor with shape (B, T, spatial_reduced..., C_o).
        """
        return self.encode_tensor(batch.input_fields)

    def encode_tensor(self, x: TensorBTSC) -> TensorBTSC:
        """Forward pass through encoder (for direct tensor input).

        Args:
            x: Input tensor with shape ``(B, T, spatial..., C_i)``.

        Returns:
            Encoded latent tensor.
        """
        b, t, *_, _ = x.shape
        # Concatenate batch and time for processing
        x = rearrange(x, "B T ... C -> (B T) C ...")

        def _heavy(x_chunk: TensorBTSC) -> TensorBTSC:
            x_chunk = self.patch(x_chunk)
            for blocks in self.descent:
                for block in cast(nn.ModuleList, blocks):  # ModuleList in construction
                    x_chunk = block(x_chunk)
            return self._saturate(x_chunk)

        x = self._chunked_apply(_heavy, x)
        return rearrange(
            x, "(B T) C ... -> B T ... C", B=b, T=t, C=self.latent_channels
        )
