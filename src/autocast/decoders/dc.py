import math
from collections.abc import Sequence
from typing import cast

import torch
from azula.nn.layers import ConvNd, Unpatchify
from einops import rearrange
from torch import nn

from autocast.decoders.base import Decoder
from autocast.nn import ResBlock
from autocast.nn.dc_utils import build_sample_block
from autocast.types import TensorBTSC


class DCDecoder(Decoder):
    """Deep Compressed (DC) decoder module.

    Progressively upsamples from latent representation back to original spatial
    dimensions using residual blocks with optional attention.

    Args:
        in_channels: Number of input (latent) channels.
        out_channels: Number of output channels.
        hid_channels: Number of channels at each depth level.
        hid_blocks: Number of residual blocks at each depth level.
        kernel_size: Kernel size for convolutions.
        stride: Stride for upsampling operations.
        pixel_shuffle: Whether to use pixel shuffling or nearest upsampling.
        norm: Type of normalization ('layer' or 'group').
        attention_heads: Dict mapping depth index to number of attention heads.
        ffn_factor: Channel expansion factor in FFN blocks.
        spatial: Number of spatial dimensions (2 for 2D, 3 for 3D).
        patch_size: Patch size for unpatchifying at the end.
        periodic: Whether spatial dimensions are periodic (use circular padding).
        dropout: Dropout rate.
        checkpointing: Whether to use gradient checkpointing.
        identity_init: Initialize up/downsampling convolutions as identity.
        ffn_out_scale: Optional multiplicative scale applied to each ResBlock
            FFN output conv.
        out_channels_multiplier: Multiplier applied to the final output channel
            count. With multiplier M, the decoder produces M * out_channels
            channels instead of out_channels. Used by heteroscedastic outputs
            (M=2 for packed (mu, log_var)). Default 1 preserves the standard
            behaviour.

    Note:
        Based on the implementation from:
        - Deep Compression Autoencoder for Efficient High-Resolution Diffusion
          Models (Chen et al., 2024), https://arxiv.org/abs/2410.10733v1
        - Lost in Latent Space: An Empirical Study of Latent Diffusion Models
          for Physics Emulation (Rozet et al., 2024),
          https://arxiv.org/abs/2507.02608, https://github.com/PolymathicAI/lola

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
        ffn_out_scale: float | None = None,
        out_channels_multiplier: int = 1,
    ) -> None:
        super().__init__()
        self.latent_channels = in_channels
        self.out_channels_multiplier = out_channels_multiplier
        self.output_channels = out_channels * out_channels_multiplier
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
                        ffn_out_scale=ffn_out_scale,
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
                        math.prod(patch_size) * self.output_channels,
                        spatial=spatial,
                        **kwargs,
                    )
                )

            self.ascent.append(blocks)

        self.decoder_model = self.ascent

        if out_channels_multiplier > 1:
            final_block_list = cast(nn.ModuleList, self.ascent[-1])
            self._zero_extra_channel_bias(
                final_conv=final_block_list[-1],
                base_channels=out_channels,
                patch_volume=math.prod(patch_size),
            )

    @staticmethod
    def _zero_extra_channel_bias(
        final_conv: nn.Module, base_channels: int, patch_volume: int
    ) -> None:
        """Zero-init the bias for the extra output channels.

        Used when ``out_channels_multiplier > 1`` so the bias for the extra
        output channels (e.g. log-variance under Gaussian NLL) starts at 0.
        The conv weights themselves remain at their default init, so the
        initial extra-channel output is centred at 0 in expectation rather
        than identically 0. The conv output is laid out as ``(Z, *patch)``
        with ``Z`` slow-varying (see ``azula.nn.layers.Unpatchify``), so final
        channels ``[base_channels, M * base_channels)`` map to bias indices
        ``[base_channels * patch_volume :]``.

        Args:
            final_conv: The final projection conv module.
            base_channels: The un-multiplied output channel count.
            patch_volume: Product of the patch size across spatial dims.
        """
        bias = getattr(final_conv, "bias", None)
        if not isinstance(bias, nn.Parameter):
            msg = (
                "DCDecoder final projection has no bias parameter; cannot "
                "zero-init the extra output channels required when "
                "out_channels_multiplier > 1."
            )
            raise RuntimeError(msg)
        with torch.no_grad():
            bias[base_channels * patch_volume :].zero_()

    def decode(self, z: TensorBTSC) -> TensorBTSC:
        """Decode latent tensor with time dimension back to original space.

        Args:
            z: Latent tensor with shape (B, T, spatial..., C_i) where C_i is last dim.

        Returns:
            Decoded tensor with shape (B, T, spatial_expanded..., C_o).
        """
        b, t, *_ = z.shape
        z = rearrange(z, "B T ... C -> (B T) C ...")

        def _heavy(z_chunk: TensorBTSC) -> TensorBTSC:
            for blocks in self.ascent:
                for block in cast(nn.ModuleList, blocks):
                    z_chunk = block(z_chunk)
            return self.unpatch(z_chunk)

        z = self._chunked_apply(_heavy, z)
        z = rearrange(z, "(B T) C ... -> B T ... C", B=b, T=t, C=self.output_channels)
        return z
