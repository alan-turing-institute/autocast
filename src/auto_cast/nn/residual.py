from typing import Any

from azula.nn.attention import MultiheadSelfAttention
from azula.nn.layers import ConvNd, LayerNorm
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .spatial_attention import SpatialAttentionWrapper


class Residual(nn.Sequential):
    """Residual wrapper that adds input to output.

    This wrapper can be used to add a residual connection around any sequence
    of layers.
    """

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass with residual connection.

        Parameters
        ----------
        input: Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Input + output from sequential layers.

        """
        return input + super().forward(input)


class ResBlock(nn.Module):
    """Residual block with normalization, optional attention, and FFN.

    Parameters
    ----------
    channels: int
        Number of channels.
    norm: str
        Type of normalization ('layer' or 'group').
    groups: int
        Number of groups for GroupNorm.
    attention_heads: int | None
        Number of attention heads (None for no attention).
    ffn_factor: int
        Channel expansion factor in FFN.
    spatial: int
        Number of spatial dimensions.
    dropout: float | None
        Dropout rate.
    checkpointing: bool
        Whether to use gradient checkpointing.
    **kwargs: Any
        Additional arguments for convolution layers.

    """

    def __init__(
        self,
        channels: int,
        norm: str = "layer",
        groups: int = 16,
        attention_heads: int | None = None,
        ffn_factor: int = 1,
        spatial: int = 2,
        dropout: float | None = None,
        checkpointing: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        self.checkpointing = checkpointing

        # Normalization
        if norm == "layer":
            self.norm = LayerNorm(dim=-spatial - 1)
        elif norm == "group":
            self.norm = nn.GroupNorm(
                num_groups=min(groups, channels),
                num_channels=channels,
            )
        else:
            msg = f"Unknown normalization type: {norm}"
            raise NotImplementedError(msg)

        # Attention (optional)
        if attention_heads is None:
            self.attn = nn.Identity()
        else:
            # Wrap attention to handle spatial dimensions
            attn_module = MultiheadSelfAttention(
                channels, attention_heads=attention_heads
            )
            self.attn = SpatialAttentionWrapper(attn_module, spatial)

        # FFN
        self.ffn = nn.Sequential(
            ConvNd(channels, ffn_factor * channels, spatial=spatial, **kwargs),
            nn.SiLU(),
            nn.Identity() if dropout is None else nn.Dropout(dropout),
            ConvNd(ffn_factor * channels, channels, spatial=spatial, **kwargs),
        )

        # TODO: check if required
        # # Initialize last layer with small weights for stability
        # if hasattr(self.ffn[-1], "weight"):
        #     self.ffn[-1].weight.data *= 1e-2

    def _forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual connection.

        Parameters
        ----------
        x: Tensor
            Input tensor with shape (B, C, L_1, ..., L_N).

        Returns
        -------
        Tensor
            Output tensor with shape (B, C, L_1, ..., L_N).

        """
        y = self.norm(x)
        y = self.attn(y)
        y = self.ffn(y)
        return x + y

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with optional gradient checkpointing.

        Parameters
        ----------
        x: Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor with same shape as input.

        """
        if self.checkpointing:
            result = checkpoint(self._forward, x, use_reentrant=False)
            assert isinstance(result, Tensor)
            return result
        return self._forward(x)


__all__ = ["ResBlock", "Residual", "SpatialAttentionWrapper"]
