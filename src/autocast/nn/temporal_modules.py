"""Temporal processing modules for sequential data.

This module provides temporal processing methods that can be used
in temporal backbone architectures (UNet, ViT, etc.) for handling sequences
in the format (B, T, W, H, C).
"""

from einops import rearrange
from torch import nn

from autocast.types import TensorBTSC


class TemporalAttention(nn.Module):
    """Temporal self-attention that processes the time dimension.

    Each spatial location independently attends to its temporal history.
    """

    def __init__(
        self,
        channels: int,
        attention_heads: int = 8,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        self.channels = channels
        self.attention_heads = attention_heads

        # Project to higher dimension if needed
        if hidden_dim is None:
            # Ensure divisible by attention_heads
            hidden_dim = max(
                channels,
                attention_heads * ((channels + attention_heads - 1) // attention_heads),
            )

        self.hidden_dim = hidden_dim

        # Project input to hidden dimension
        self.proj_in = (
            nn.Linear(channels, hidden_dim) if channels != hidden_dim else nn.Identity()
        )

        # Use GroupNorm for spatial dimensions
        self.norm = nn.GroupNorm(min(8, hidden_dim), hidden_dim)

        # Multi-head attention over temporal dimension
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=attention_heads, batch_first=True
        )

        # Project back to original dimension
        self.proj_out = (
            nn.Linear(hidden_dim, channels) if channels != hidden_dim else nn.Identity()
        )

    def forward(self, x: TensorBTSC) -> TensorBTSC:
        """Apply temporal self-attention.

        Args:
            x: (B, T, W, H, C)

        Returns
        -------
            (B, T, W, H, C) with temporal attention applied
        """
        B, _, W, H, _ = x.shape

        # Project to hidden dimension: (B, T, W, H, C) -> (B, T, W, H, hidden_dim)
        x_proj = self.proj_in(x)

        # Normalize spatially
        x_norm = rearrange(x_proj, "b t w h c -> b c t w h")
        x_norm = self.norm(x_norm)
        x_norm = rearrange(x_norm, "b c t w h -> b t w h c")

        # Reshape to (B*W*H, T, hidden_dim) for temporal attention
        x_flat = rearrange(x_norm, "b t w h c -> (b w h) t c")

        # Apply temporal self-attention
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)

        # Reshape back
        attn_out = rearrange(attn_out, "(b w h) t c -> b t w h c", b=B, w=W, h=H)

        # Project back to original dimension
        attn_out = self.proj_out(attn_out)

        # Residual connection
        return x + attn_out


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network using 1D convolutions over time.

    Applies causal 1D convolutions along the temporal dimension independently
    at each spatial location.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int | None = None,
        kernel_size: int = 3,
        num_layers: int = 2,
        dilation_base: int = 2,
    ):
        """Initialize Temporal ConvNet.

        Args:
            channels: Number of input/output channels
            hidden_channels: Hidden dimension (defaults to channels)
            kernel_size: Size of temporal convolution kernel
            num_layers: Number of TCN layers
            dilation_base: Base for exponential dilation (dilation = base^layer)
        """
        super().__init__()
        self.channels = channels
        hidden_channels = hidden_channels or channels

        layers = []
        for i in range(num_layers):
            dilation = dilation_base**i
            padding = (kernel_size - 1) * dilation  # Causal padding

            layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        channels if i == 0 else hidden_channels,
                        hidden_channels,
                        kernel_size=kernel_size,
                        padding=padding,
                        dilation=dilation,
                    ),
                    nn.GroupNorm(min(8, hidden_channels), hidden_channels),
                    nn.GELU(),
                )
            )

        self.layers = nn.ModuleList(layers)

        # Output projection
        self.proj_out = (
            nn.Conv1d(hidden_channels, channels, kernel_size=1)
            if hidden_channels != channels
            else nn.Identity()
        )

    def forward(self, x: TensorBTSC) -> TensorBTSC:
        """Apply temporal convolutions.

        Args:
            x: (B, T, W, H, C)

        Returns
        -------
            (B, T, W, H, C) with temporal convolutions applied
        """
        B, T, W, H, _ = x.shape

        # Reshape to (B*W*H, C, T) for 1D conv
        x_flat = rearrange(x, "b t w h c -> (b w h) c t")

        # Apply TCN layers
        for layer in self.layers:
            residual = x_flat
            x_flat = layer(x_flat)
            # Remove extra padding from causal convolution
            x_flat = x_flat[..., :T]
            # Residual connection (only if dimensions match)
            if x_flat.shape == residual.shape:
                x_flat = x_flat + residual

        # Output projection
        x_flat = self.proj_out(x_flat)

        # Reshape back
        return rearrange(x_flat, "(b w h) c t -> b t w h c", b=B, w=W, h=H)

