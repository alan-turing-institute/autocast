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
    hidden_dim: int,
    decoder_depths: Sequence[int],
    decoder_num_heads: Sequence[int],
    patch_size: int,
    groups: int,
) -> nn.Module:
    """Build a hierarchical decoder compatible with Swin encoder stage layout.

    Decoder stages are parameterized by ``decoder_depths`` and perform
    stage-wise feature refinement plus 2x upsampling between stages.
    A final transposed projection with ``patch_size`` restores full resolution.
    """
    if len(decoder_depths) != len(decoder_num_heads):
        raise ValueError(
            "`decoder_depths` and `decoder_num_heads` must have the same length "
            f"(got {len(decoder_depths)} and {len(decoder_num_heads)})."
        )

    n_stages = len(decoder_depths)
    stage_channels = [hidden_dim * (2 ** (n_stages - i - 1)) for i in range(n_stages)]

    def _group_count(channels: int) -> int:
        grp = min(groups, channels)
        while grp > 1 and channels % grp != 0:
            grp -= 1
        return max(grp, 1)

    layers: list[nn.Module] = []
    c = in_channels
    for i, depth in enumerate(decoder_depths):
        c_stage = stage_channels[i]
        for _ in range(depth):
            layers += [
                nn.Conv2d(c, c_stage, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(_group_count(c_stage), c_stage),
                nn.GELU(),
            ]
            c = c_stage

        if i < n_stages - 1:
            c_next = stage_channels[i + 1]
            layers += [
                nn.ConvTranspose2d(c, c_next, kernel_size=2, stride=2, bias=False),
                nn.GroupNorm(_group_count(c_next), c_next),
                nn.GELU(),
            ]
            c = c_next

    if patch_size > 1:
        layers += [
            nn.ConvTranspose2d(
                c,
                c,
                kernel_size=patch_size,
                stride=patch_size,
                bias=False,
            ),
            nn.GroupNorm(_group_count(c), c),
            nn.GELU(),
        ]
    layers.append(nn.Conv2d(c, out_channels, kernel_size=1, bias=True))
    return nn.Sequential(*layers)


class _NoiseConditionedLayerNorm(nn.Module):
    """Adapter to use ConditionalLayerNorm in modules expecting LayerNorm(x)."""

    def __init__(
        self,
        base_norm: nn.LayerNorm,
        n_noise_channels: int,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.norm = ConditionalLayerNorm(
            normalized_shape=torch.Size(base_norm.normalized_shape),
            n_noise_channels=n_noise_channels,
            eps=base_norm.eps,
            elementwise_affine=False,
            bias=base_norm.bias is not None,
        )
        if zero_init and self.norm.gamma is not None and self.norm.beta is not None:
            nn.init.zeros_(self.norm.gamma.weight)
            nn.init.ones_(self.norm.gamma.bias)
            nn.init.zeros_(self.norm.beta.weight)
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
    hidden_dim:
        Patch embedding channel dimension used by the Swin backbone.
    encoder_depths:
        Number of transformer blocks in each Swin stage.
    encoder_num_heads:
        Attention heads per stage. Must match ``depths`` length.
    decoder_depths:
        Number of conv refinement blocks in each decoder stage.
    decoder_num_heads:
        Decoder attention-head placeholder for config compatibility.
    window_size:
        Swin local attention window size.
    drop_path:
        Stochastic depth probability for the Swin backbone.
    patch_size:
        Patch size for the stem convolution. Default is ``4``, matching the
        original Swin paper.
    groups:
        Group count used in decoder GroupNorm layers.
    use_ada_ln:
        Enables noise-conditioned LayerNorm replacement in Swin blocks.
    zero_init:
        Initializes conditional affine projections to identity transform.
    loss_func:
        Loss function. Defaults to ``nn.MSELoss()``.
    """

    def __init__(  # noqa: PLR0912, PLR0915
        self,
        in_channels: int,
        out_channels: int,
        spatial_resolution: Sequence[int],
        hidden_dim: int = 96,
        encoder_depths: Sequence[int] = (2, 2, 6, 2),
        encoder_num_heads: Sequence[int] = (3, 6, 12, 24),
        decoder_depths: Sequence[int] = (2, 6, 2, 2),
        decoder_num_heads: Sequence[int] = (24, 12, 6, 3),
        patch_size: int = 4,
        window_size: Sequence[int] = (7, 7),
        drop_path: float = 0.0,
        groups: int = 12,
        loss_func: nn.Module | None = None,
        n_noise_channels: int | None = None,
        use_ada_ln: bool = True,
        zero_init: bool = True,
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

        encoder_depths = list(encoder_depths)
        encoder_num_heads = list(encoder_num_heads)
        decoder_depths = list(decoder_depths)
        decoder_num_heads = list(decoder_num_heads)
        window_size = list(window_size)

        if len(window_size) != 2:
            raise ValueError(
                "`window_size` must have exactly 2 values for 2-D Swin "
                f"(got {window_size})."
            )
        if any(ws <= 0 for ws in window_size):
            raise ValueError(
                f"`window_size` values must be positive (got {window_size})."
            )

        if len(encoder_depths) != len(encoder_num_heads):
            raise ValueError(
                "`encoder_depths` and `encoder_num_heads` must have the same "
                f"length (got {len(encoder_depths)} and {len(encoder_num_heads)})."
            )
        if len(decoder_depths) != len(decoder_num_heads):
            raise ValueError(
                "`decoder_depths` and `decoder_num_heads` must have the same "
                f"length (got {len(decoder_depths)} and {len(decoder_num_heads)})."
            )
        if len(encoder_depths) != len(decoder_depths):
            raise ValueError(
                "`encoder_depths` and `decoder_depths` must have the same "
                f"length (got {len(encoder_depths)} and {len(decoder_depths)})."
            )
        for i, heads in enumerate(encoder_num_heads):
            stage_dim = hidden_dim * (2**i)
            if stage_dim % heads != 0:
                raise ValueError(
                    "Each encoder stage dim must be divisible by its head count "
                    f"(stage={i}, dim={stage_dim}, heads={heads})."
                )

        # ------------------------------------------------------------------
        # Build the torchvision SwinTransformer backbone.
        # We use in_channels=3 as a placeholder; the stem is replaced below.
        # num_classes=0 removes the classification head (returns features).
        # ------------------------------------------------------------------
        self.backbone = SwinTransformer(
            patch_size=[patch_size, patch_size],
            embed_dim=hidden_dim,
            depths=encoder_depths,
            num_heads=encoder_num_heads,
            window_size=window_size,
            mlp_ratio=4.0,
            dropout=0.0,
            attention_dropout=0.0,
            stochastic_depth_prob=drop_path,
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
        if n_noise_channels is not None and use_ada_ln:
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
                            norm1, n_noise_channels, zero_init=zero_init
                        )
                        block.norm1 = conditioned_norm1
                        self._conditioned_norms.append(conditioned_norm1)
                    if isinstance(norm2, nn.LayerNorm):
                        conditioned_norm2 = _NoiseConditionedLayerNorm(
                            norm2, n_noise_channels, zero_init=zero_init
                        )
                        block.norm2 = conditioned_norm2
                        self._conditioned_norms.append(conditioned_norm2)

        # ------------------------------------------------------------------
        # Determine the output channel count of the backbone.
        # After 4 stages the channel count is embed_dim * 2^(n_stages-1).
        # ------------------------------------------------------------------
        n_stages = len(encoder_depths)
        backbone_out_channels = hidden_dim * (2 ** (n_stages - 1))

        self.decoder = _make_decoder(
            in_channels=backbone_out_channels,
            out_channels=out_channels,
            hidden_dim=hidden_dim,
            decoder_depths=decoder_depths,
            decoder_num_heads=decoder_num_heads,
            patch_size=patch_size,
            groups=groups,
        )

        self.loss_func = loss_func or nn.MSELoss()
        self.n_noise_channels = n_noise_channels
        self._backbone_out_channels = backbone_out_channels
        self._total_stride = patch_size * (2 ** (n_stages - 1))

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
