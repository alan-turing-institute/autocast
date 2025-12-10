"""Utilities for DCAE encoder/decoder construction."""

import math
from collections.abc import Sequence

from azula.nn.layers import ConvNd, Patchify, Unpatchify
from torch import nn


def build_sample_block(
    in_channels: int,
    out_channels: int,
    stride: Sequence[int],
    pixel_shuffle: bool,
    spatial: int,
    identity_init: bool,
    upsample: bool = False,
    **kwargs,
) -> nn.Module:
    """Build up/downsampling block (patchify/unpatchify or strided/nearest)."""
    if upsample:
        if pixel_shuffle:
            return nn.Sequential(
                ConvNd(
                    in_channels,
                    out_channels * math.prod(stride),
                    spatial=spatial,
                    identity_init=identity_init,
                    **kwargs,
                ),
                Unpatchify(patch_shape=tuple(stride)),
            )
        return nn.Sequential(
            nn.Upsample(scale_factor=tuple(stride), mode="nearest"),
            ConvNd(
                in_channels,
                out_channels,
                spatial=spatial,
                identity_init=identity_init,
                **kwargs,
            ),
        )
    # Downsample
    if pixel_shuffle:
        return nn.Sequential(
            Patchify(patch_shape=tuple(stride)),
            ConvNd(
                in_channels * math.prod(stride),
                out_channels,
                spatial=spatial,
                identity_init=identity_init,
                **kwargs,
            ),
        )
    return ConvNd(
        in_channels,
        out_channels,
        spatial=spatial,
        stride=stride,
        identity_init=identity_init,
        **kwargs,
    )
