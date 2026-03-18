from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn as nn
from torchvision.models.swin_transformer import SwinTransformer

from autocast.processors.base import Processor
from autocast.types import EncodedBatch, Tensor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_decoder(
    in_channels: int,
    out_channels: int,
    total_stride: int,
    hidden_channels: int,
) -> nn.Module:
    """Build a transposed-convolution decoder that upsamples by ``total_stride``.

    The decoder uses a sequence of 2× upsample steps followed by a final
    1×1 projection to ``out_channels``.
    """
    layers: list[nn.Module] = []
    c = in_channels
    stride = total_stride
    while stride > 1:
        step = min(stride, 2)          # upsample 2× at a time
        c_next = max(hidden_channels, out_channels)
        layers += [
            nn.ConvTranspose2d(c, c_next, kernel_size=step, stride=step, bias=False),
            nn.GroupNorm(max(1, c_next // 16), c_next),
            nn.GELU(),
        ]
        c = c_next
        stride //= step
    layers.append(nn.Conv2d(c, out_channels, kernel_size=1, bias=True))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

_SwinVariant = Literal["tiny", "small", "base", "nano"]  

# Configs taken from torchvision source.
# embed_dim, depths, num_heads, window_size
_SWIN_CONFIGS: dict[_SwinVariant, dict] = {
    "tiny": dict(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        stochastic_depth_prob=0.2,
    ),
    "small": dict(
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        stochastic_depth_prob=0.3,
    ),
    "base": dict(
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[7, 7],
        stochastic_depth_prob=0.5,
    ),
    "nano": dict(
        embed_dim=128,           # Hidden dim = 128
        depths=[6],              # 6 layers in ONE stage (stays at 128 dim)
        num_heads=[4],           # Single stage needs only one head count
        window_size=[7, 7],
        stochastic_depth_prob=0.1,
    ),
}

class SwinProcessor(Processor[EncodedBatch]):
    """Dense Swin Transformer Processor.

    Wraps the torchvision ``SwinTransformer`` backbone as a pixel-to-pixel
    mapping suitable for spatiotemporal field prediction.

    The backbone's patch-embedding stem is replaced so that arbitrary
    ``in_channels`` are accepted. The classification head is discarded. A
    transposed-convolution decoder upsamples the encoder output back to the
    original spatial resolution and projects to ``out_channels``.

    Parameters
    ----------
    in_channels:
        Number of input channels.
    out_channels:
        Number of output channels.
    spatial_resolution:
        ``(H, W)`` of the input field. Both dimensions must be divisible by
        the patch stride (default 4 for all Swin variants).
    variant:
        Swin model size: ``"tiny"``, ``"small"``, or ``"base"``.
    patch_size:
        Patch size for the stem convolution. Default is ``4``, matching the
        original Swin paper.
    decoder_hidden_channels:
        Number of intermediate channels in the decoder. Defaults to the
        backbone's final-stage channel count.
    loss_func:
        Loss function. Defaults to ``nn.MSELoss()``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_resolution: Sequence[int],
        variant: _SwinVariant = "tiny",
        patch_size: int = 4,
        decoder_hidden_channels: int | None = None,
        loss_func: nn.Module | None = None,
    ) -> None:
        super().__init__()

        if len(spatial_resolution) != 2:
            raise ValueError(
                "SwinProcessor only supports 2-D spatial inputs "
                f"(got spatial_resolution={spatial_resolution})."
            )
        h, w = spatial_resolution
        if h % patch_size != 0 or w % patch_size != 0:
            raise ValueError(
                f"spatial_resolution {spatial_resolution} must be divisible by "
                f"patch_size={patch_size}."
            )

        cfg = _SWIN_CONFIGS[variant]
        embed_dim: int = cfg["embed_dim"]
        depths: list[int] = cfg["depths"]

        # ------------------------------------------------------------------
        # Build the torchvision SwinTransformer backbone.
        # We use in_channels=3 as a placeholder; the stem is replaced below.
        # num_classes=0 removes the classification head (returns features).
        # ------------------------------------------------------------------
        self.backbone = SwinTransformer(
            patch_size=[patch_size, patch_size],
            embed_dim=embed_dim,
            depths=depths,
            num_heads=cfg["num_heads"],
            window_size=cfg["window_size"],
            mlp_ratio=4.0,
            dropout=0.0,
            attention_dropout=0.0,
            stochastic_depth_prob=cfg["stochastic_depth_prob"],
            num_classes=0,   # removes the head → forward returns (B, C) pooled
        )

        # ------------------------------------------------------------------
        # Replace the patch-embedding stem so it accepts `in_channels`.
        #
        # torchvision's SwinTransformer.features[0] is the stem:
        #   Sequential(
        #     [0] Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size),
        #     [1] Permute([0, 2, 3, 1]),   # B C H W -> B H W C
        #     [2] LayerNorm(embed_dim),
        #   )
        # We only need to swap the Conv2d.
        # ------------------------------------------------------------------
        stem_block = self.backbone.features[0]
        old_conv: nn.Conv2d = stem_block[0]
        stem_block[0] = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            bias=old_conv.bias is not None,
        )

        # ------------------------------------------------------------------
        # Determine the output channel count of the backbone.
        # After 4 stages the channel count is embed_dim * 2^(n_stages-1).
        # ------------------------------------------------------------------
        n_stages = len(depths)
        backbone_out_channels = embed_dim * (2 ** (n_stages - 1))

        # ------------------------------------------------------------------
        # Build decoder.
        # The backbone reduces spatial resolution by patch_size (via the stem)
        # and by 2× for each merging step between stages (n_stages - 1 times).
        # Total spatial downsampling = patch_size * 2^(n_stages-1).
        # ------------------------------------------------------------------
        total_stride = patch_size * (2 ** (n_stages - 1))
        dec_hidden = decoder_hidden_channels or backbone_out_channels

        self.decoder = _make_decoder(
            in_channels=backbone_out_channels,
            out_channels=out_channels,
            total_stride=total_stride,
            hidden_channels=dec_hidden,
        )

        self.loss_func = loss_func or nn.MSELoss()
        self._backbone_out_channels = backbone_out_channels
        self._total_stride = total_stride

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, x: Tensor) -> Tensor:
        """Run the Swin backbone and return spatial feature maps.

        torchvision's SwinTransformer.features produces a sequence of feature
        tensors in (B, H', W', C) layout; the final element is the last-stage
        output. We permute to (B, C, H', W') for the decoder.
        """
        # features is an nn.Sequential; iterate to get final output.
        feat = x
        for layer in self.backbone.features:
            feat = layer(feat)
        # feat: (B, H', W', C)  — SwinTransformer keeps spatial-last internally
        return feat.permute(0, 3, 1, 2).contiguous()  # (B, C, H', W')

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: ``(B, C_in, H, W) -> (B, C_out, H, W)``."""
        feat = self._encode(x)          # (B, C_backbone, H/s, W/s)
        return self.decoder(feat)       # (B, C_out, H, W)

    def map(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        """Processor interface: map encoded inputs to predictions."""
        _ = global_cond  # unused — Swin does not currently consume global cond
        return self(x)

    def loss(self, batch: EncodedBatch) -> Tensor:
        """Compute the loss for a training batch."""
        pred = self.map(batch.encoded_inputs, batch.global_cond)
        return self.loss_func(pred, batch.encoded_output_fields)