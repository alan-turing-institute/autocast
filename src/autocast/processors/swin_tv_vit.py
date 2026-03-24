from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import nn
from torchvision.models.swin_transformer import SwinTransformer

from autocast.nn.noise.conditional_layer_norm import ConditionalLayerNorm
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

    The decoder uses a sequence of 2x upsample steps followed by a final
    1x1 projection to ``out_channels``.
    """
    layers: list[nn.Module] = []
    c = in_channels
    stride = total_stride
    while stride > 1:
        step = min(stride, 2)  # upsample 2x at a time
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


class _NoiseConditionedLayerNorm(nn.Module):
    """Adapter to use ConditionalLayerNorm in modules expecting LayerNorm(x)."""

    def __init__(self, base_norm: nn.LayerNorm, n_noise_channels: int) -> None:
        super().__init__()
        self.norm = ConditionalLayerNorm(
            normalized_shape=torch.Size(base_norm.normalized_shape),
            n_noise_channels=n_noise_channels,
            eps=base_norm.eps,
            elementwise_affine=False,
            bias=base_norm.bias is not None,
        )
        self._x_noise: Tensor | None = None

    def set_noise(self, x_noise: Tensor | None) -> None:
        self._x_noise = x_noise

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x, x_noise=self._x_noise)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class SwinTVProcessor(Processor[EncodedBatch]):
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
        the patch stride (default 4).
    embed_dim:
        Patch embedding channel dimension used by the Swin backbone.
    depths:
        Number of transformer blocks in each Swin stage.
    num_heads:
        Attention heads per stage. Must match ``depths`` length.
    window_size:
        Swin local attention window size.
    stochastic_depth_prob:
        Stochastic depth probability for the Swin backbone.
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
        embed_dim: int = 96,
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        window_size: Sequence[int] = (7, 7),
        stochastic_depth_prob: float = 0.2,
        patch_size: int = 4,
        decoder_hidden_channels: int | None = None,
        loss_func: nn.Module | None = None,
        n_noise_channels: int | None = None,
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

        depths = list(depths)
        num_heads = list(num_heads)
        window_size = list(window_size)
        if len(depths) != len(num_heads):
            raise ValueError(
                "`depths` and `num_heads` must have the same length "
                f"(got {len(depths)} and {len(num_heads)})."
            )

        # ------------------------------------------------------------------
        # Build the torchvision SwinTransformer backbone.
        # We use in_channels=3 as a placeholder; the stem is replaced below.
        # num_classes=0 removes the classification head (returns features).
        # ------------------------------------------------------------------
        self.backbone = SwinTransformer(
            patch_size=[patch_size, patch_size],
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=4.0,
            dropout=0.0,
            attention_dropout=0.0,
            stochastic_depth_prob=stochastic_depth_prob,
            num_classes=0,  # removes the head → forward returns (B, C) pooled
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
        stem_block = cast(nn.Sequential, self.backbone.features[0])
        old_conv = cast(nn.Conv2d, stem_block[0])
        kernel_size = cast(tuple[int, int], old_conv.kernel_size)
        stride = cast(tuple[int, int], old_conv.stride)
        stem_block[0] = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=old_conv.bias is not None,
        )

        self._conditioned_norms: list[_NoiseConditionedLayerNorm] = []
        if n_noise_channels is not None:
            for layer in self.backbone.features:
                if not isinstance(layer, nn.Sequential):
                    continue
                for block in layer:
                    if not hasattr(block, "norm1") or not hasattr(block, "norm2"):
                        continue

                    norm1 = block.norm1
                    norm2 = block.norm2
                    if isinstance(norm1, nn.LayerNorm):
                        conditioned_norm1 = _NoiseConditionedLayerNorm(
                            norm1, n_noise_channels
                        )
                        block.norm1 = conditioned_norm1
                        self._conditioned_norms.append(conditioned_norm1)
                    if isinstance(norm2, nn.LayerNorm):
                        conditioned_norm2 = _NoiseConditionedLayerNorm(
                            norm2, n_noise_channels
                        )
                        block.norm2 = conditioned_norm2
                        self._conditioned_norms.append(conditioned_norm2)

        # ------------------------------------------------------------------
        # Determine the output channel count of the backbone.
        # After 4 stages the channel count is embed_dim * 2^(n_stages-1).
        # ------------------------------------------------------------------
        n_stages = len(depths)
        backbone_out_channels = embed_dim * (2 ** (n_stages - 1))

        # ------------------------------------------------------------------
        # Build decoder.
        # The backbone reduces spatial resolution by patch_size (via the stem)
        # and by 2x for each merging step between stages (n_stages - 1 times).
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
        self.n_noise_channels = n_noise_channels
        self._backbone_out_channels = backbone_out_channels
        self._total_stride = total_stride

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, x: Tensor, x_noise: Tensor | None = None) -> Tensor:
        """Run the Swin backbone and return spatial feature maps.

        torchvision's SwinTransformer.features produces a sequence of feature
        tensors in (B, H', W', C) layout; the final element is the last-stage
        output. We permute to (B, C, H', W') for the decoder.
        """
        for conditioned_norm in self._conditioned_norms:
            conditioned_norm.set_noise(x_noise)

        # features is an nn.Sequential; iterate to get final output.
        feat = x
        for layer in self.backbone.features:
            feat = layer(feat)
        # feat: (B, H', W', C)  — SwinTransformer keeps spatial-last internally
        return feat.permute(0, 3, 1, 2).contiguous()  # (B, C, H', W')

    def _decode(self, feat: Tensor) -> Tensor:
        return self.decoder(feat)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, x: Tensor, x_noise: Tensor | None = None) -> Tensor:
        """Forward pass: ``(B, C_in, H, W) -> (B, C_out, H, W)``."""
        feat = self._encode(x, x_noise=x_noise)  # (B, C_backbone, H/s, W/s)
        return self._decode(feat)  # (B, C_out, H, W)

    def map(self, x: Tensor, global_cond: Tensor | None) -> Tensor:
        """Processor interface: map encoded inputs to predictions."""
        _ = global_cond  # unused — Swin does not currently consume global cond
        if self.n_noise_channels is None:
            noise = None
        else:
            noise = torch.randn(
                x.shape[0], self.n_noise_channels, dtype=x.dtype, device=x.device
            )
        return self(x, noise)

    def loss(self, batch: EncodedBatch) -> Tensor:
        """Compute the loss for a training batch."""
        pred = self.map(batch.encoded_inputs, batch.global_cond)
        return self.loss_func(pred, batch.encoded_output_fields)
